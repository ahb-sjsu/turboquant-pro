# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""IVF coarse-partition layer — sublinear compressed-domain search (experimental).

The base :class:`~turboquant_pro.adc_index.ADCIndex` scans every code (``O(N)``).
That is fine to a few million rows and, sharded + memory-mapped, to a billion; but
trillion-scale *serving* needs a query to touch a small fraction of the corpus. IVF
(inverted file) does that: cluster the corpus into ``nlist`` cells, and at query time
scan only the cells that can contain the answer.

**Coarse space = the quantized direction.** Everything is derived from the codes the
ADC index already stores: a row's direction is ``normalize(cent[codes])``. We k-means
those unit directions into ``nlist`` centroids, so a cell groups rows the ADC scorer
would score similarly. No new per-row storage — the partition is reconstructable from
the codes.

**Probe order + stop = the kernel of A\\*.** Each cell carries its angular radius
``r`` (the largest angle from its centroid to any member). On the unit sphere the
triangle inequality gives, for a query at angle ``theta`` from the centroid, an
*admissible* upper bound on any member's cosine: ``cos(max(0, theta - r))`` — no
member can score higher. Order cells by that bound (best-first), keep the incumbent
k-th surrogate score, and **stop as soon as the next cell's bound cannot beat it**.
The stop is provably safe (the heuristic never underestimates), so it returns the
exact top-k of the coarse surrogate.

The honest catch, measured (``benchmarks/bench_ivf.py``): in high dimension cells are
angularly *wide* (radius ~65 deg at 32-d), so the worst-case bound is near 1 for most
cells and the admissible stop prunes almost nothing — it can approach a full scan.
The fix is **weighted A\\*** (ε-admissible): shrink the radius by ``beta =
radius_scale`` to inflate the heuristic and prune more, trading a bounded amount of
recall for a large scan reduction. ``beta`` sweeps smoothly from exact (``beta=1``,
provable, ~full scan) to aggressive (``beta=0.25``: recall ~0.80 at ~3% scan, ~30x
fewer rows); ``beta=0.5`` sits near recall ~1.0 at ~20% scan. Fixed ``nprobe`` gives
the same tradeoff on a manual dial (nprobe=32 -> recall ~0.96 at ~10% scan). Final
ranking rescoring the probed candidates with the exact ADC score (and optional fp32
rerank) matches the brute-force ADC top-k up to the recall reported above. Acceptance
is recall of the shortlist, never reconstruction cosine.

This is the single-node substrate; it composes with sharding (per-shard IVF, the same
best-first order across shards) and memmap (a probe gathers only its cells' rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adc_index import ADCIndex, _normalize
from .pca import PCAMatryoshka


def _kmeans_unit(
    x: np.ndarray, k: int, iters: int, rng: np.random.Generator, block: int = 100_000
) -> np.ndarray:
    """Spherical k-means: cluster unit vectors by cosine (== nearest centroid dot).

    ``x`` is assumed L2-normalized. Returns ``k`` unit centroids.
    """
    n = len(x)
    c = x[rng.choice(n, size=k, replace=False)].copy()
    c = _normalize(c)
    for _ in range(iters):
        assign = _assign(x, c, block)
        new = np.zeros_like(c)
        counts = np.bincount(assign, minlength=k)
        np.add.at(new, assign, x)
        empty = counts == 0
        new[~empty] /= counts[~empty, None]
        if empty.any():  # reseed dead cells on random points
            new[empty] = x[rng.choice(n, size=int(empty.sum()), replace=False)]
        c = _normalize(new)
    return c.astype(np.float32)


def _assign(x: np.ndarray, c: np.ndarray, block: int = 100_000) -> np.ndarray:
    """Nearest-centroid (max dot) assignment, blocked to bound memory."""
    out = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), block):
        out[s : s + block] = np.argmax(x[s : s + block] @ c.T, axis=1)
    return out


@dataclass
class ProbeStats:
    """Per-search diagnostics: how much of the corpus the query actually touched."""

    cells_probed: int
    rows_scanned: int
    rows_total: int

    @property
    def scan_fraction(self) -> float:
        return self.rows_scanned / max(self.rows_total, 1)


class IVFIndex:
    """Coarse-partitioned ADC index with best-first, early-terminating probing.

    Experimental. Build with :meth:`create`; search with a fixed ``nprobe`` (classic
    IVF) or, by default, the adaptive A\\*-style stop (``nprobe=None``).
    """

    def __init__(
        self,
        adc: ADCIndex,
        centroids: np.ndarray,
        assign: np.ndarray,
        radius: np.ndarray,
        originals: np.ndarray | None,
    ):
        self._adc = adc
        self._c = centroids  # (nlist, d') unit
        self._assign = assign  # (N,) cell per row
        self._radius = radius  # (nlist,) angular radius (radians)
        self._originals = originals
        self._n = len(assign)
        # inverted lists: rows grouped by cell (contiguous, via argsort on assignment)
        order = np.argsort(assign, kind="stable")
        bounds = np.searchsorted(assign[order], np.arange(len(centroids) + 1))
        self._members = order.astype(np.int64)
        self._offsets = bounds.astype(np.int64)
        # quantized unit directions (source of truth = the codes)
        self._dir = _normalize(adc._cent[adc._codes].astype(np.float32))

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
        nlist: int | None = None,
        seed: int = 42,
        whiten: bool = False,
        train_cap: int = 200_000,
        kmeans_iters: int = 12,
        keep_originals: bool = True,
    ) -> IVFIndex:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n, dim), got {x.shape}")
        n, dim = x.shape
        out = dim if output_dim is None else min(int(output_dim), dim)
        rng = np.random.default_rng(seed)

        pca = PCAMatryoshka(input_dim=dim, output_dim=out, whiten=whiten)
        pca.fit(x[: min(n, train_cap)])
        adc = ADCIndex(pca.with_quantizer(bits=bits)).add(x)

        # Coarse quantizer over the quantized unit directions.
        d = _normalize(adc._cent[adc._codes].astype(np.float32))
        if nlist is None:  # FAISS-style sqrt(N), clamped to something sane
            nlist = int(np.clip(round(np.sqrt(n)), 1, max(1, n)))
        nlist = min(nlist, n)
        train = d if n <= train_cap else d[rng.choice(n, size=train_cap, replace=False)]
        centroids = _kmeans_unit(train, nlist, kmeans_iters, rng)
        assign = _assign(d, centroids)

        # Per-cell angular radius: the largest angle from centroid to any member.
        dots = np.einsum("ij,ij->i", d, centroids[assign])
        ang = np.arccos(np.clip(dots, -1.0, 1.0))
        radius = np.zeros(nlist, dtype=np.float32)
        np.maximum.at(radius, assign, ang)
        return cls(adc, centroids, assign, radius, x if keep_originals else None)

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
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
    ):
        """Top-``k`` per query.

        ``nprobe`` an int → probe that many best-first cells (classic IVF).
        ``nprobe=None`` → adaptive best-first stop: probe cells in descending
        upper-bound order and stop once the next cell's bound can't beat the
        incumbent k-th neighbour. The bound is ``cos(max(0, theta - beta*r))`` with
        ``beta = radius_scale`` — **weighted A\\***:

        * ``bound="admissible"`` (beta forced to 1) — worst-case radius, provably the
          coarse-exact top-k, but ``r`` is large in high dimension so it prunes little
          (can approach a full scan). Use when exactness matters.
        * ``bound="weighted"`` (default, ``radius_scale`` in [0,1]) — shrinking the
          radius inflates the heuristic and prunes more, trading a little recall for a
          large scan reduction. ``radius_scale`` is the adaptive-``nprobe`` knob.
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        q_rot, qbias = self._adc._query_terms(q)
        qdir = _normalize(q_rot)  # cosine surrogate lives in rotated-direction space
        cos_qc = qdir @ self._c.T  # (nq, nlist)
        theta = np.arccos(np.clip(cos_qc, -1.0, 1.0))
        beta = 1.0 if bound == "admissible" else float(radius_scale)
        ub = np.cos(np.maximum(0.0, theta - beta * self._radius[None, :]))
        order = np.argsort(-ub, axis=1)  # best-first: descending (weighted) bound

        ids = np.full((len(q), k), -1, dtype=np.int64)
        scores = np.full((len(q), k), np.nan, dtype=np.float32)
        stats: list[ProbeStats] = []
        cap = max_cells if max_cells is not None else len(self._c)
        for i in range(len(q)):
            cand, cells, scanned = self._probe_one(
                qdir[i], order[i], ub[i], k, nprobe, cap
            )
            stats.append(ProbeStats(cells, scanned, self._n))
            if not len(cand):
                continue
            ci, cs = self._score_candidates(cand, q_rot[i], qbias[i], k)
            if rerank and self._originals is not None:
                ci = self._rerank_one(ci, q[i], k)
            m = len(ci)
            ids[i, :m] = ci
            scores[i, :m] = cs[:m] if not rerank else np.nan
        if return_stats:
            return ids, scores, stats
        return ids, scores

    def _probe_one(self, qdir_i, order, stop_i, k, nprobe, cap):
        """Best-first cell probing for one query. Returns (candidate rows, #cells,
        #rows scanned). Adaptive stop when the next cell's stop value <= incumbent."""
        cand_parts: list[np.ndarray] = []
        scanned = 0
        cells = 0
        kth = -np.inf  # incumbent k-th surrogate cosine
        best = np.empty(0, dtype=np.float32)  # running top-k surrogate cosines
        fixed = nprobe is not None
        for c in order[:nprobe] if fixed else order:
            if not fixed and stop_i[c] <= kth:
                break  # no better cell remains under this stop rule
            s, e = self._offsets[c], self._offsets[c + 1]
            if e <= s:
                continue
            rows = self._members[s:e]
            cand_parts.append(rows)
            cells += 1
            scanned += len(rows)
            if not fixed:  # maintain incumbent k-th surrogate score for the stop test
                best = np.concatenate([best, self._dir[rows] @ qdir_i])
                if len(best) > k:
                    best = np.partition(best, len(best) - k)[-k:]
                if len(best) >= k:
                    kth = float(best.min())
            if cells >= cap:
                break
        cand = np.concatenate(cand_parts) if cand_parts else np.empty(0, np.int64)
        return cand, cells, scanned

    def _score_candidates(self, cand, q_rot_i, qbias_i, k):
        """Exact ADC score over probed candidate rows; return top-k (ids, scores)."""
        cc = self._adc._cent[self._adc._codes[cand]]  # (m, d')
        adc = cc @ q_rot_i
        sc = (qbias_i + self._adc._cnorm[cand] * adc) * self._adc._vrnorm[cand]
        kk = min(k, len(cand))
        top = np.argpartition(-sc, kk - 1)[:kk]
        top = top[np.argsort(-sc[top])]
        return cand[top], sc[top]

    def _rerank_one(self, cand_ids, q_i, k):
        s = self._originals[cand_ids] @ q_i
        return cand_ids[np.argsort(-s)[:k]]

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        counts = np.diff(self._offsets)
        return {
            "n_rows": int(self._n),
            "nlist": int(len(self._c)),
            "dim_coarse": int(self._c.shape[1]),
            "cell_min": int(counts.min()),
            "cell_max": int(counts.max()),
            "cell_mean": float(counts.mean()),
            "empty_cells": int((counts == 0).sum()),
            "radius_mean_deg": float(np.degrees(self._radius.mean())),
        }
