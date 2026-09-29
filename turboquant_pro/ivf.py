# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""IVF coarse-partition layer: sublinear search over chunks of the v3 scan.

The base :class:`~turboquant_pro.adc_index.ADCIndex` scans every code (``O(N)``).
IVF (inverted file) clusters the corpus into ``nlist`` cells and scans only the
cells that can contain the answer. Here a cell **is** a chunk of the ADC index:
rows are sorted by cell at build, and a search is one ``search_chunks`` call over
the probed cells with a per-(query, cell) constant, on the kernel when it is
compiled and on the numpy path otherwise.

**Residual coding (default).** The coarse quantizer is a plain k-means in PCA
coordinates; a row is coded as the direction of ``x_p - c`` plus ``||x_p - c||``.
The residual is shorter than the row, so the same bits carry less error, and
because the rotation is global the query's lookup table serves every cell: the
centroid enters only through the constant ``q_proj . c``. RaBitQ's IVF form does
the same, and the public comparison measured it worth 4 to 24 points of
single-pass recall at 1 bit over flat coding (``docs/PREREG_rabitq_public.md``,
interim 2026-09-17). ``residual=False`` keeps the plain codes and only partitions.

**Probe order and stop.** Cells are ordered by an upper bound on the score any
of their rows can reach (``(qbias + q_proj . c + ||q|| R_c) * vr_max``, ``R_c``
the cell's largest residual norm). A fixed ``nprobe`` scans that many; the
adaptive stop (``nprobe=None``) probes best-first and stops when the next bound
cannot beat the incumbent k-th score. The admissible bound is exact but loose
in high dimension; **weighted A-star** (``radius_scale < 1``) shrinks the radius
to prune more for a bounded loss of recall. ``benchmarks/bench_ivf.py`` measures
the trade-off.

This is the single-node substrate; the hierarchical coarse quantizer and the
sharded IVF path (``sharded_index.py``) build on the functions below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import scorer as _scorer
from .adc_index import ADCIndex, _normalize
from .metrics import COSINE
from .pca import PCAMatryoshka
from .telemetry import trace as _trace


def _resolve_device(device: str):
    """Return ``(xp, to_host)`` for ``device`` — NumPy on CPU, CuPy on GPU. The GPU
    path is where the ``O(N·nlist)`` coarse-quantizer matmul belongs at scale (a
    single GV100/A10 does the assignment ~1000x faster than NumPy)."""
    if device in ("gpu", "cuda"):
        try:
            import cupy as cp
        except ImportError as e:  # pragma: no cover - optional GPU extra
            raise ImportError(
                "device='gpu' needs CuPy — pip install cupy-cuda12x"
            ) from e
        return cp, cp.asnumpy
    return np, (lambda a: a)


def _kmeans_unit(
    x: np.ndarray,
    k: int,
    iters: int,
    rng: np.random.Generator,
    block: int = 100_000,
    device: str = "cpu",
) -> np.ndarray:
    """Spherical k-means: cluster unit vectors by cosine (== nearest centroid dot).

    ``x`` is assumed L2-normalized. Returns ``k`` unit centroids. The per-iteration
    assignment (the matmul-heavy step) honours ``device``; the small update stays on
    the host.
    """
    n = len(x)
    c = x[rng.choice(n, size=k, replace=False)].copy()
    c = _normalize(c)
    for _ in range(iters):
        assign = _assign(x, c, block, device=device)
        new = np.zeros_like(c)
        counts = np.bincount(assign, minlength=k)
        np.add.at(new, assign, x)
        empty = counts == 0
        new[~empty] /= counts[~empty, None]
        if empty.any():  # reseed dead cells on random points
            new[empty] = x[rng.choice(n, size=int(empty.sum()), replace=False)]
        c = _normalize(new)
    return c.astype(np.float32)


def _assign(
    x: np.ndarray, c: np.ndarray, block: int = 100_000, device: str = "cpu"
) -> np.ndarray:
    """Nearest-centroid (max dot) assignment, blocked to bound memory. ``device='gpu'``
    runs the ``(block × nlist)`` matmul + argmax on the GPU (centroids resident, rows
    streamed in blocks), returning host cell ids."""
    xp, to_host = _resolve_device(device)
    out = np.empty(len(x), dtype=np.int64)
    cg = xp.asarray(c)
    for s in range(0, len(x), block):
        xb = xp.asarray(x[s : s + block])
        out[s : s + block] = to_host(xp.argmax(xb @ cg.T, axis=1)).astype(np.int64)
    return out


def adc_score_rows(adc, rows: np.ndarray, q_rot_i: np.ndarray, qbias_i: float):
    """Exact ADC score of specific ``rows`` for one query — the same score the
    brute-force scan computes, restricted to a candidate subset. Shared by
    :class:`IVFIndex` and the sharded IVF path, and routed through
    :func:`~turboquant_pro.adc_index.score_block` so the metric cannot differ
    between the exhaustive and probed paths."""
    from .adc_index import score_block

    cc = adc._cent[np.asarray(adc._codes[rows])]
    return score_block(
        getattr(adc, "_metric", "cosine"),
        (cc @ q_rot_i)[None, :],
        np.asarray([qbias_i], dtype=np.float32),
        np.asarray(adc._cnorm[rows]),
        np.asarray(adc._vrnorm[rows]),
    )[0]


def inverted_lists(assign: np.ndarray, nlist: int):
    """Group row positions by cell. Returns (offsets[nlist+1], members[n]) so a
    cell's rows are ``members[offsets[c]:offsets[c+1]]`` — the IVF posting lists."""
    order = np.argsort(assign, kind="stable").astype(np.int64)
    offsets = np.searchsorted(assign[order], np.arange(nlist + 1)).astype(np.int64)
    return offsets, order


# --------------------------------------------------------------------------- #
# Hierarchical (IVF-of-IVF) coarse quantizer                                   #
# --------------------------------------------------------------------------- #
# A flat ``nlist`` quantizer costs ``O(N·nlist)`` to assign and, at 1T, gets coarse
# (few centroids per unit of the sphere). A two-level quantizer fixes both: fit ``n1``
# **top** centroids, then ``sub_nlist`` **leaf** centroids *within each* top cell.
# Leaf id is ``top*sub_nlist + sub`` — so a top cell owns the contiguous leaf block
# ``[top*sub_nlist : (top+1)*sub_nlist]`` (rectangular layout). This buys three things:
#   * assignment drops to ``O(N·(n1 + sub_nlist))`` (top-assign, then sub within top);
#   * the coarse quantizer is finer at the same build cost (the 1T quality fix);
#   * probes cluster into a few *top* cells, so a query touches contiguous leaves and,
#     placed top-cell -> server, only a few servers (the routing / locality win the 1B
#     run pointed at — random-access cost, not compute, was the wall).


def build_hierarchical_quantizer(
    train_dirs: np.ndarray,
    top_nlist: int,
    sub_nlist: int,
    iters: int,
    rng: np.random.Generator,
    block: int = 100_000,
    device: str = "cpu",
):
    """Fit a two-level quantizer on unit directions. Returns
    ``(top_centroids[n1,d], leaf_centroids[n1*sub_nlist,d], leaf_top[n1*sub_nlist])``.

    Leaf centroids are laid out top-major (leaves of top ``t`` are the contiguous block
    ``[t*sub_nlist:(t+1)*sub_nlist]``), so downstream code can treat them as a flat
    ``nlist = n1*sub_nlist`` quantizer while still recovering the hierarchy from
    ``leaf_top`` (or the rectangular arithmetic ``leaf // sub_nlist``)."""
    dim = train_dirs.shape[1]
    top = _kmeans_unit(train_dirs, top_nlist, iters, rng, block=block, device=device)
    tassign = _assign(train_dirs, top, block=block, device=device)
    leaves = np.zeros((top_nlist * sub_nlist, dim), dtype=np.float32)
    leaf_top = np.repeat(np.arange(top_nlist), sub_nlist).astype(np.int64)
    for t in range(top_nlist):
        pts = train_dirs[tassign == t]
        base = t * sub_nlist
        if len(pts) >= sub_nlist:
            sub = _kmeans_unit(pts, sub_nlist, iters, rng, block=block, device=device)
        elif len(pts) > 0:  # too few to cluster: seed with the points, pad w/ the top
            sub = np.repeat(top[t][None], sub_nlist, axis=0).astype(np.float32)
            sub[: len(pts)] = _normalize(pts.astype(np.float32))
        else:  # empty top cell: degenerate leaves collapse onto the top centroid
            sub = np.repeat(top[t][None], sub_nlist, axis=0).astype(np.float32)
        leaves[base : base + sub_nlist] = sub
    return top, _normalize(leaves), leaf_top


def _assign_hier(
    x: np.ndarray,
    top: np.ndarray,
    leaves: np.ndarray,
    sub_nlist: int,
    block: int | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Hierarchical nearest-leaf assignment: pick the best top cell, then the best leaf
    *within* it. ``O(N·(n1 + sub_nlist))`` vs the flat ``O(N·n1·sub_nlist)``. Rows are
    streamed in blocks; the per-block sub gather is ``(block, sub_nlist, d)``, so the
    block is bounded to keep that tensor small."""
    xp, to_host = _resolve_device(device)
    dim = leaves.shape[1]
    if block is None:
        block = max(1024, 40_000_000 // max(sub_nlist * dim, 1))
    out = np.empty(len(x), dtype=np.int64)
    topg = xp.asarray(top)
    leavesg = xp.asarray(leaves)
    ar = xp.arange(sub_nlist)
    for s in range(0, len(x), block):
        xb = xp.asarray(x[s : s + block])
        t = xp.argmax(xb @ topg.T, axis=1)  # (b,) best top cell
        idx = t[:, None] * sub_nlist + ar[None, :]  # (b, sub_nlist) that top's leaves
        subc = leavesg[idx]  # (b, sub_nlist, d)
        sub = xp.argmax(xp.einsum("bd,bsd->bs", xb, subc), axis=1)  # best leaf in-top
        out[s : s + block] = to_host(t * sub_nlist + sub).astype(np.int64)
    return out


def probed_leaves_hier(
    q_dir: np.ndarray,
    top: np.ndarray,
    leaf_centroids: np.ndarray,
    leaf_radius: np.ndarray,
    sub_nlist: int,
    nprobe: int,
    top_probe: int,
    radius_scale: float = 0.5,
    bound: str = "weighted",
):
    """Two-level probe selection. For each query: score the ``n1`` top cells, keep the
    best ``top_probe``, then rank only the leaves under those tops by the weighted-A*
    bound and return the best ``nprobe`` leaves. Returns ``(probed[nq,nprobe],
    top_sel[nq,top_probe])`` — the probed leaf ids and the selected top cells (the
    latter is what a router scatters on). Restricting to ``top_probe`` tops is exactly
    what makes the probe local: the ``nprobe`` leaves live in ``<= top_probe`` cells."""
    q_dir = np.asarray(q_dir, dtype=np.float32)
    if q_dir.ndim == 1:
        q_dir = q_dir[None]
    nq = len(q_dir)
    n1 = len(top)
    top_probe = min(top_probe, n1)
    top_cos = q_dir @ top.T  # (nq, n1)
    top_sel = np.argsort(-top_cos, axis=1)[:, :top_probe]  # (nq, top_probe) best tops
    beta = 1.0 if bound == "admissible" else float(radius_scale)
    probed = np.empty((nq, nprobe), dtype=np.int64)
    for i in range(nq):
        # candidate leaves = the contiguous blocks of the selected tops
        leaf_ids = (top_sel[i][:, None] * sub_nlist + np.arange(sub_nlist)).ravel()
        cos = q_dir[i] @ leaf_centroids[leaf_ids].T
        theta = np.arccos(np.clip(cos, -1.0, 1.0))
        ub = np.cos(np.maximum(0.0, theta - beta * leaf_radius[leaf_ids]))
        sel = leaf_ids[np.argsort(-ub)[:nprobe]]
        if len(sel) < nprobe:  # pad (few tops): repeat last so shape stays rectangular
            sel = np.concatenate([sel, np.full(nprobe - len(sel), sel[-1])])
        probed[i] = sel
    return probed, top_sel


@dataclass
class ProbeStats:
    """Per-search diagnostics: how much of the corpus the query actually touched."""

    cells_probed: int
    rows_scanned: int
    rows_total: int

    @property
    def scan_fraction(self) -> float:
        return self.rows_scanned / max(self.rows_total, 1)


def _kmeans_points(
    x: np.ndarray, k: int, iters: int, rng: np.random.Generator, block: int = 100_000
) -> np.ndarray:
    """Plain k-means in PCA coordinates (the coarse quantizer of a residual-coded
    index). Empty cells are reseeded on random points."""
    n = len(x)
    c = x[rng.choice(n, size=k, replace=False)].astype(np.float32).copy()
    for _ in range(iters):
        assign = _assign_points(x, c, block)
        new = np.zeros_like(c)
        counts = np.bincount(assign, minlength=k)
        np.add.at(new, assign, x)
        empty = counts == 0
        new[~empty] /= counts[~empty, None]
        if empty.any():
            new[empty] = x[rng.choice(n, size=int(empty.sum()), replace=False)]
        c = new.astype(np.float32)
    return c


def _assign_points(x: np.ndarray, c: np.ndarray, block: int = 100_000) -> np.ndarray:
    """Nearest centroid in L2, blocked:
    ``argmin ||x - c||^2 = argmax x.c - ||c||^2 / 2``."""
    half = 0.5 * (c * c).sum(axis=1).astype(np.float32)
    out = np.empty(len(x), dtype=np.int64)
    # the (block x nlist) score matrix is the peak: 100k rows against 4,096 centroids
    # is 1.6 GiB, which killed the 2 GiB pods of the public comparison's small arms
    block = max(1024, min(int(block), 25_000_000 // max(len(c), 1)))
    for s in range(0, len(x), block):
        xb = np.asarray(x[s : s + block], dtype=np.float32)
        out[s : s + block] = np.argmax(xb @ c.T - half[None, :], axis=1)
    return out


class IVFIndex:
    """Coarse-partitioned ADC index on the v3 scan: every cell is one chunk.

    Rows are sorted by cell at build, so a cell is one chunk of the underlying
    :class:`ADCIndex` and a search is a single :meth:`ADCIndex.search_chunks`
    call over the probed cells. With ``residual=True`` (the default) a row is
    coded relative to its cell's centroid in PCA coordinates: the codes describe
    the direction of ``x_p - c`` and the stored norm is ``||x_p - c||``, which is
    smaller than ``||x_p||``, so the same bits carry less error. The lookup table
    does not depend on the centroid (the rotation is global), so the only cost
    at scan time is the per-(query, cell) constant ``q_rot . rotate(c)``.

    ``nprobe`` an int probes that many nearest cells (classic IVF). ``nprobe=None``
    is the adaptive best-first stop: cells are ordered by an upper bound on the
    score any of their rows can reach, and probing stops when the next bound
    cannot beat the incumbent k-th score. ``bound="admissible"`` uses the exact
    bound (never prunes a cell that could win; scans a lot in high dimension);
    ``bound="weighted"`` shrinks each cell's radius by ``radius_scale`` (weighted
    A*), pruning more for a bounded loss of recall.
    """

    def __init__(
        self,
        adc: ADCIndex,
        centroids: np.ndarray,
        members: np.ndarray,
        offsets: np.ndarray,
        radius: np.ndarray,
        vr_range: np.ndarray,
        ang_radius: np.ndarray,
        originals: np.ndarray | None,
        residual: bool,
    ):
        self._adc = adc
        self._c = np.asarray(centroids, dtype=np.float32)  # (nlist, d') PCA coordinates
        self._c_rot = np.ascontiguousarray(adc._coder.rotate(self._c), dtype=np.float32)
        self._members = np.asarray(members, dtype=np.int64)  # original row per position
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._radius = np.asarray(radius, dtype=np.float32)  # max ||x_p - c|| per cell
        self._vr = np.asarray(vr_range, dtype=np.float32)  # (nlist, 2) min/max vrnorm
        self._ang_radius = np.asarray(ang_radius, dtype=np.float32)
        self._originals = originals
        self._residual = bool(residual)
        self._n = int(len(members))
        self.last_scorer: dict | None = None  # the last search's scorer provenance

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        embeddings: np.ndarray,
        *,
        output_dim: int | None = None,
        bits: int = 4,
        bit_schedule: list[tuple[int, int]] | None = None,
        nlist: int | None = None,
        seed: int = 42,
        whiten: bool = False,
        train_cap: int = 200_000,
        kmeans_iters: int = 12,
        keep_originals: bool = True,
        residual: bool = True,
        block: int = 100_000,
    ) -> IVFIndex:
        """Build from a corpus in RAM: fit the PCA on its first ``train_cap`` rows,
        then :meth:`from_blocks` over it."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n, dim), got {x.shape}")
        n, dim = x.shape
        out = dim if output_dim is None else min(int(output_dim), dim)
        pca = PCAMatryoshka(input_dim=dim, output_dim=out, whiten=whiten)
        pca.fit(x[: min(n, train_cap)])
        rng = np.random.default_rng(seed)
        train = x if n <= train_cap else x[rng.choice(n, size=train_cap, replace=False)]
        ivf = cls.from_blocks(
            pca,
            lambda: (x[s : s + block] for s in range(0, n, block)),
            n=n,
            train=train,
            bits=bits,
            bit_schedule=bit_schedule,
            nlist=nlist,
            seed=seed,
            kmeans_iters=kmeans_iters,
            residual=residual,
            block=block,
        )
        ivf._originals = x if keep_originals else None
        return ivf

    @classmethod
    def from_blocks(
        cls,
        pca: PCAMatryoshka,
        blocks,
        *,
        n: int,
        train: np.ndarray,
        bits: int = 4,
        bit_schedule: list[tuple[int, int]] | None = None,
        nlist: int | None = None,
        seed: int = 42,
        kmeans_iters: int = 12,
        residual: bool = True,
        block: int = 100_000,
    ) -> IVFIndex:
        """Build from a corpus streamed as row blocks, in two passes.

        ``pca`` is fitted; ``blocks()`` returns a fresh iterable of ``(rows, dim)``
        float32 arrays covering the ``n`` rows in order, and is called twice: once to
        assign every row to a cell, once to code the rows. ``train`` (rows of the
        input space) fits the coarse quantizer. Nothing larger than the per-cell
        code buffers and one block is held, so a 10M-row corpus builds on a pod
        that cannot hold it in float32.
        """
        rng = np.random.default_rng(seed)
        if bit_schedule is None:
            pipeline = pca.with_quantizer(bits=bits, seed=seed)
        else:
            pipeline = pca.with_weighted_quantizer(bit_schedule=bit_schedule, seed=seed)
        adc = ADCIndex(pipeline)
        if nlist is None:  # FAISS-style sqrt(N), clamped to something sane
            nlist = int(np.clip(round(np.sqrt(n)), 1, max(1, n)))
        nlist = min(int(nlist), n)
        centroids = _kmeans_points(adc.project(train), nlist, kmeans_iters, rng, block)

        # pass 1: a cell for every row
        assign = np.empty(n, dtype=np.int64)
        s = 0
        for rows in blocks():
            xp = adc.project(rows)
            assign[s : s + len(xp)] = _assign_points(xp, centroids, block)
            s += len(xp)
        if s != n:
            raise ValueError(f"blocks covered {s} rows, expected {n}")
        offsets, members = inverted_lists(assign, nlist)

        # pass 2: code every row against its cell, buffered per cell
        buf_codes = [[] for _ in range(nlist)]
        buf_cnorm = [[] for _ in range(nlist)]
        buf_vr = [[] for _ in range(nlist)]
        buf_segw = [[] for _ in range(nlist)]
        radius = np.zeros(nlist, dtype=np.float32)
        ang = np.zeros(nlist, dtype=np.float32)
        vr_lo = np.full(nlist, np.inf, dtype=np.float32)
        vr_hi = np.full(nlist, -np.inf, dtype=np.float32)
        c_unit = centroids / np.maximum(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-30
        )
        s = 0
        for rows in blocks():
            xp = adc.project(rows)
            a = assign[s : s + len(xp)]
            order = np.argsort(a, kind="stable")
            bounds = np.searchsorted(a[order], np.arange(nlist + 1))
            for c in range(nlist):
                sel = order[bounds[c] : bounds[c + 1]]
                if not len(sel):
                    continue
                sub = xp[sel]
                if residual:
                    codes, cnorm, vrnorm, segw = adc.encode_residuals(sub, centroids[c])
                else:
                    cnorm = np.linalg.norm(sub, axis=1).astype(np.float32)
                    codes, segw, cc = adc._coder.encode(
                        sub / np.maximum(cnorm[:, None], 1e-30)
                    )
                    vrnorm = adc._recon_inverse_norm(cc, cnorm)
                buf_codes[c].append(codes)
                buf_cnorm[c].append(cnorm)
                buf_vr[c].append(vrnorm)
                if segw is not None:
                    buf_segw[c].append(segw)
                r = float(np.linalg.norm(sub - centroids[c], axis=1).max())
                if not residual:  # the bound then needs ||x_p|| itself
                    r += float(np.linalg.norm(centroids[c]))
                radius[c] = max(radius[c], r)
                vr_lo[c] = min(vr_lo[c], float(vrnorm.min()))
                vr_hi[c] = max(vr_hi[c], float(vrnorm.max()))
                u = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-30)
                ang[c] = max(
                    ang[c], float(np.arccos(np.clip(u @ c_unit[c], -1.0, 1.0)).max())
                )
            s += len(xp)
        # members must list rows in the order the buffers hold them: block order
        # within a cell, which is what the stable argsort of ``assign`` gives
        for c in range(nlist):
            d = adc.dim
            codes = (
                np.concatenate(buf_codes[c])
                if buf_codes[c]
                else np.zeros((0, d), np.uint8)
            )
            cnorm = (
                np.concatenate(buf_cnorm[c])
                if buf_cnorm[c]
                else np.zeros(0, np.float32)
            )
            vrnorm = np.concatenate(buf_vr[c]) if buf_vr[c] else np.zeros(0, np.float32)
            segw = None
            if adc._coder.nseg > 1:
                segw = (
                    np.concatenate(buf_segw[c])
                    if buf_segw[c]
                    else np.zeros((0, adc._coder.nseg), np.float32)
                )
            adc.add_coded(codes, cnorm, vrnorm, segw)
            buf_codes[c] = buf_cnorm[c] = buf_vr[c] = buf_segw[c] = None
        vr_range = np.stack(
            [
                np.where(np.isfinite(vr_lo), vr_lo, 0.0),
                np.where(np.isfinite(vr_hi), vr_hi, 0.0),
            ],
            axis=1,
        )
        return cls(
            adc, centroids, members, offsets, radius, vr_range, ang, None, residual
        )

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def _cell_terms(self, q_rot: np.ndarray, qbias: np.ndarray):
        """Per-(query, cell) scan constants: ``ip = q_rot . rotate(c)`` is
        ``q_proj . c``, the centroid's share of ``q . recon`` under residual
        coding, so the constant is ``qbias + ip``; without residual coding the
        centroid carries no part of the score and the constant is ``qbias``."""
        ip = q_rot @ self._c_rot.T  # (nq, nlist)
        if self._residual:
            biases = qbias[:, None] + ip
        else:
            biases = np.broadcast_to(qbias[:, None], ip.shape)
        return np.ascontiguousarray(biases, dtype=np.float32), ip

    def _bounds(self, beta: float, q_rot: np.ndarray):
        """Per-cell upper bound on the cosine any row of the cell can reach.

        Every row of cell ``c`` lies within the cell's angular radius ``phi_c`` of
        the centroid direction, so its angle to the query is at least
        ``theta_qc - phi_c`` and its cosine at most ``cos(max(0, theta_qc - beta *
        phi_c))``; ``beta = 1`` is the admissible bound, ``beta < 1`` the weighted
        A-star heuristic. The bound is on the PCA-space cosine, which is what the
        cosine score is up to the mean term and the reconstruction norm; the
        admissible bound is widened by two percent so it also covers the kernel's
        uint8 table rounding.
        """
        qd = q_rot / np.maximum(np.linalg.norm(q_rot, axis=1, keepdims=True), 1e-30)
        cd = self._c_rot / np.maximum(
            np.linalg.norm(self._c_rot, axis=1, keepdims=True), 1e-30
        )
        theta = np.arccos(np.clip(qd @ cd.T, -1.0, 1.0))  # (nq, nlist)
        ub = np.cos(np.maximum(0.0, theta - beta * self._ang_radius[None, :]))
        if beta >= 1.0:
            ub = ub + 0.02
        return ub.astype(np.float32)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        *,
        nprobe: int | None = None,
        rerank: int = 0,
        bound: str = "weighted",
        radius_scale: float = 0.5,
        max_cells: int | None = None,
        return_stats: bool = False,
        mode: str | None = None,
    ):
        """Top-``k`` per query; see the class docstring for ``nprobe`` and ``bound``.

        ``mode`` chooses the scan's scorer as in :meth:`ADCIndex.search`
        (``"exact"`` the numpy reference, ``"fast"`` the kernel where it can
        run); the search's provenance is left in :attr:`last_scorer`.

        The adaptive stop (``nprobe=None``) compares a bound on the cosine with
        the incumbent k-th score, which is meaningful only when the score is a
        cosine. Under ``inner_product`` or ``l2`` it is refused; a fixed
        ``nprobe`` still works, with cells ordered by the same angular bound
        and scored in the index's metric.
        """
        if nprobe is None and self._adc._metric != COSINE:
            raise ValueError(
                "adaptive probing (nprobe=None) bounds a cosine and cannot stop "
                f"a {self._adc._metric!r} search correctly; pass nprobe"
            )
        mode = _scorer.resolve_mode(mode, False)
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        nq = len(q)
        tr = _trace.begin(
            "IVFIndex.search",
            q,
            self._trace_identity(),
            k=k,
            nprobe=nprobe,
            adaptive=nprobe is None or None,
            rerank=rerank,
            bound=bound,
            radius_scale=None if bound == "admissible" else radius_scale,
            max_cells=max_cells,
            mode=mode,
        )
        q_rot, qbias = self._adc._query_terms(q)
        biases, _ = self._cell_terms(q_rot, qbias)
        beta = 1.0 if bound == "admissible" else float(radius_scale)
        ub = self._bounds(beta, q_rot)  # (nq, nlist)
        order = np.argsort(-ub, axis=1)
        nlist = len(self._c)
        cap = nlist if max_cells is None else min(int(max_cells), nlist)
        kk = k * max(rerank, 1) if rerank else k
        if tr:
            tr.lap("encode")  # query terms, per-cell constants and cell bounds

        if nprobe is not None:
            p = min(int(nprobe), cap)
            probes = order[:, :p].astype(np.int32)
            pos, sc = self._adc.search_chunks(
                q_rot, probes, np.take_along_axis(biases, probes, axis=1), kk, mode=mode
            )
            probed = [p] * nq
        else:
            pos = np.full((nq, kk), -1, dtype=np.int64)
            sc = np.full((nq, kk), -1e30, dtype=np.float32)
            probed = []
            for i in range(nq):
                pos[i], sc[i], m = self._probe_adaptive(
                    q_rot[i], order[i], ub[i], biases[i], kk, cap, mode
                )
                probed.append(m)
        counts = np.diff(self._offsets)
        stats = [
            ProbeStats(int(m), int(counts[order[i, :m]].sum()), self._n)
            for i, m in enumerate(probed)
        ]
        ids = np.where(pos >= 0, self._members[np.maximum(pos, 0)], -1)
        scores = np.where(pos >= 0, sc, np.nan).astype(np.float32)
        kernel = mode == _scorer.FAST and self._adc._kernel_scan()
        self.last_scorer = _scorer.provenance(
            mode,
            self._adc.kernel_scorer if kernel else _scorer.EXACT_FLOAT,
            self._adc._fallback_reason() if mode == _scorer.FAST else None,
            kernel=self._adc._kernel,
            rerank_width=kk if (rerank and self._originals is not None) else 0,
            rerank_basis="originals",
        )
        if tr:
            tr.set(scan_path="kernel" if kernel else "numpy", scorer=self.last_scorer)
            tr.lap(
                "scan",
                candidates=int(kk),
                rows=int(round(np.mean([s.rows_scanned for s in stats]))),
                rows_total=int(self._n),
                cells_probed=int(round(np.mean([s.cells_probed for s in stats]))),
                nlist=int(nlist),
                scan_fraction=float(np.mean([s.scan_fraction for s in stats])),
            )
        if rerank and self._originals is not None:
            approx_ids, approx_sc = ids, scores
            ids, first_exact = self._adc._rerank(
                ids, q, self._originals, k, first_scores=True
            )
            scores = np.full((nq, k), np.nan, dtype=np.float32)
            if tr:
                tr.lap("rerank", candidates=int(kk), basis="originals")
                tr.results(approx_ids, approx_sc, ids, first_exact, k=k)
        else:
            ids, scores = ids[:, :k], scores[:, :k]
            if tr:
                tr.results(ids, scores, k=k)
        if tr:
            tr.finish()
        if return_stats:
            return ids, scores, stats
        return ids, scores

    def _probe_adaptive(self, q_rot_i, order, ub, biases, kk, cap, mode=None):
        """Best-first probing for one query in growing rounds: the probed prefix of
        ``order`` doubles until the next cell's bound cannot beat the incumbent
        k-th score (or ``cap`` cells are probed). Returns (positions, scores, cells)."""
        m = 1
        pos = sc = None
        while True:
            m = min(m, cap)
            probes = order[None, :m].astype(np.int32)
            pos, sc = self._adc.search_chunks(
                q_rot_i[None, :], probes, biases[None, order[:m]], kk, mode=mode
            )
            filled = int((pos[0] >= 0).sum())
            kth = float(sc[0, kk - 1]) if filled >= kk else -np.inf
            if m >= cap or ub[order[m]] <= kth:
                return pos[0], sc[0], m
            m *= 2

    def _rerank(self, ids, q, k):
        # ids are rows of the originals, so the flat index's metric-exact rerank
        # applies unchanged; one definition of what reranking means.
        return self._adc._rerank(ids, q, self._originals, k)

    def _trace_identity(self) -> dict:
        """What a trace needs to name this index (no payload)."""
        return {
            "kind": "IVFIndex",
            "rows": int(self._n),
            "dim": int(self._adc.dim),
            "metric": self._adc._metric,
            "nlist": int(len(self._c)),
            "residual": self._residual,
            "stored_bytes_per_row": int(self._adc.stored_bytes_per_row),
            "kernel": bool(self._adc.uses_kernel),
        }

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        counts = np.diff(self._offsets)
        return {
            "n_rows": int(self._n),
            "nlist": int(len(self._c)),
            "dim_coarse": int(self._c.shape[1]),
            "residual": self._residual,
            "cell_min": int(counts.min()),
            "cell_max": int(counts.max()),
            "cell_mean": float(counts.mean()),
            "empty_cells": int((counts == 0).sum()),
            "radius_mean": float(self._radius.mean()),
            "radius_mean_deg": float(np.degrees(self._ang_radius.mean())),
            "index_bytes_per_row": round(self._adc.nbytes / max(self._n, 1), 1),
            "stored_bytes_per_row": int(self._adc.stored_bytes_per_row),
        }
