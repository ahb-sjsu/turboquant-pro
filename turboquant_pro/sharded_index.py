# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Sharded vector index — one logical index across many TQIX files.

A single :class:`~turboquant_pro.index.TQEIndex` scales to as many vectors as one
file conveniently holds; beyond that, :class:`ShardedIndex` splits the corpus into
shards that **share one PCA basis** (so their compressed-domain scores are directly
comparable) and a small JSON manifest that ties them together. Search fans out over
the shards — each opened **memory-mapped**, so the whole set never has to fit in RAM
— and merges the per-shard top-k into a global top-k. This is the path to
billion-scale, memmap-searched indexes, and the shards are independent so the fan-out
parallelizes trivially across cores or machines.

    ShardedIndex.create(embeddings, "corpus.shards", shard_size=1_000_000)
    sh = ShardedIndex.open("corpus.shards/manifest.json")
    ids, scores = sh.search(queries, k=10, rerank=10)

Each shard is itself a complete, self-describing TQIX index (it carries its own copy
of the shared basis — a few tens of KB), so a shard can also be opened standalone.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .adc_index import _normalize, score_block
from .index import TQEIndex
from .ivf import (
    _assign,
    _assign_hier,
    _kmeans_unit,
    build_hierarchical_quantizer,
    inverted_lists,
    probed_leaves_hier,
)
from .rerank_tier import rerank_candidates

MANIFEST_SCHEMA = "turboquant-pro/index-shards"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ShardRef:
    path: str  # relative to the manifest directory
    n_rows: int
    id_min: int
    id_max: int


class ShardedIndex:
    """A fan-out search index over many single-basis TQIX shards."""

    def __init__(
        self, manifest_path: str, mmap: bool = True, max_open_shards: int = 128
    ):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"not a shard manifest: {manifest.get('schema')!r}")
        self._dir = os.path.dirname(os.path.abspath(manifest_path))
        self._metric = manifest["metric"]
        self._n_rows = int(manifest["n_rows"])
        self._shards = [
            ShardRef(s["path"], s["n_rows"], s["id_min"], s["id_max"])
            for s in manifest["shards"]
        ]
        self._mmap = mmap
        self._manifest_path = os.path.abspath(manifest_path)
        # Bounded cache of open shards. Each mmap-opened shard holds several file
        # descriptors (one per section), so eagerly opening every shard blows past
        # the process fd limit at scale (200 shards x ~6 fds > default 1024). Keep at
        # most ``max_open_shards`` open, evicting FIFO — a popped TQEIndex has no other
        # refs, so its memmaps (and their fds) are released at once.
        self._open: dict[int, TQEIndex] = {}
        self._max_open = max(1, int(max_open_shards))
        # Optional IVF coarse layer (added by build_ivf, loaded lazily on first use).
        self._ivf_meta = manifest.get("ivf")
        self._ivf_centroids: np.ndarray | None = None
        self._ivf_radius: np.ndarray | None = None
        # Optional hierarchical (IVF-of-IVF) layer: top centroids + leaf->top map, so a
        # query probes a few top cells then the leaves within them (locality/routing).
        self._ivf_hier = bool(self._ivf_meta and self._ivf_meta.get("hierarchical"))
        self._ivf_top: np.ndarray | None = None
        self._ivf_leaf_top: np.ndarray | None = None
        self._ivf_sub_nlist = int(self._ivf_meta["sub_nlist"]) if self._ivf_hier else 0

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        embeddings: np.ndarray,
        out_dir: str,
        *,
        shard_size: int = 1_000_000,
        output_dim: int | None = None,
        bits: int = 3,
        seed: int = 42,
        rotation: str = "qr",
        whiten: bool = False,
        metric: str = "cosine",
        keep_originals: bool = True,
        train_cap: int = 200_000,
    ) -> ShardedIndex:
        """Fit one PCA basis, then write the corpus as ``shard_size``-row shards.

        Every shard reuses the same fitted basis (fit once on the first
        ``train_cap`` rows), so cross-shard scores are comparable. Ids are global
        and contiguous across shards. Convenience wrapper over
        :meth:`create_streaming` for a corpus that already fits in one array; for a
        corpus too large to hold in RAM, feed :meth:`create_streaming` an iterator.
        """
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n, dim), got {x.shape}")
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        n = len(x)

        def _blocks():
            for s in range(0, n, shard_size):
                yield x[s : s + shard_size]

        return cls.create_streaming(
            _blocks(),
            out_dir,
            output_dim=output_dim,
            bits=bits,
            seed=seed,
            rotation=rotation,
            whiten=whiten,
            metric=metric,
            keep_originals=keep_originals,
            train_cap=train_cap,
            shard_size=shard_size,
        )

    @classmethod
    def create_streaming(
        cls,
        blocks,
        out_dir: str,
        *,
        output_dim: int | None = None,
        bits: int = 3,
        seed: int = 42,
        rotation: str = "qr",
        whiten: bool = False,
        metric: str = "cosine",
        keep_originals: bool = True,
        train_cap: int = 200_000,
        shard_size: int | None = None,
    ) -> ShardedIndex:
        """Build a sharded index from an iterable of row-blocks — **one shard per
        block** — without ever holding the whole corpus in RAM.

        This is the ingest counterpart to memory-mapped search: a corpus of a
        billion vectors need never be materialized as one array (or one file). The
        first block fits the shared PCA basis (on its first ``train_cap`` rows);
        every block is quantized into a shard reusing that basis, and ids are
        assigned globally and contiguously in iteration order. Each block should be
        sized to fit comfortably in RAM (e.g. ``shard_size`` rows); peak build
        memory is one block plus its quantized shard, independent of the corpus.
        """
        os.makedirs(out_dir, exist_ok=True)
        shards: list[dict] = []
        pca = None
        first: TQEIndex | None = None
        offset = 0

        def _write(idx: TQEIndex, i: int, id_min: int) -> None:
            path = f"shard_{i:05d}.tqe"
            idx.save(os.path.join(out_dir, path))
            shards.append(
                {
                    "path": path,
                    "n_rows": idx.n_rows,
                    "id_min": id_min,
                    "id_max": id_min + idx.n_rows - 1,
                }
            )

        for i, block in enumerate(blocks):
            chunk = np.asarray(block, dtype=np.float32)
            if chunk.ndim != 2:
                raise ValueError(f"each block must be 2-D (n, dim), got {chunk.shape}")
            if i == 0:
                # First block fits the basis and becomes shard 0.
                first = TQEIndex.create(
                    chunk,
                    output_dim=output_dim,
                    bits=bits,
                    seed=seed,
                    rotation=rotation,
                    whiten=whiten,
                    metric=metric,
                    keep_originals=keep_originals,
                    ids=np.arange(len(chunk), dtype=np.int64),
                    train_cap=train_cap,
                )
                pca = first._pca
                _write(first, 0, 0)
            else:
                idx = TQEIndex(
                    pca,
                    bits=first._bits,
                    seed=first._seed,
                    rotation=first._rotation,
                    metric=first._metric,
                    fit_retained_var=first._fit_retained_var,
                )
                idx._append(
                    chunk,
                    np.arange(offset, offset + len(chunk), dtype=np.int64),
                    keep_originals=keep_originals,
                )
                _write(idx, i, offset)
            offset += len(chunk)

        if not shards:
            raise ValueError("create_streaming requires at least one non-empty block")

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "n_shards": len(shards),
            "n_rows": offset,
            "metric": metric,
            "shard_size": shard_size if shard_size is not None else shards[0]["n_rows"],
            "shards": shards,
        }
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return cls(manifest_path, mmap=True)

    # ------------------------------------------------------------------ #
    # Distributed build — shard 0 fits the basis, workers build the rest #
    # ------------------------------------------------------------------ #
    @classmethod
    def write_shard(
        cls,
        out_dir: str,
        block: np.ndarray,
        shard_index: int,
        id_offset: int = 0,
        *,
        ids: np.ndarray | None = None,
        basis_from: str | None = None,
        output_dim: int | None = None,
        bits: int = 3,
        seed: int = 42,
        rotation: str = "qr",
        whiten: bool = False,
        metric: str = "cosine",
        keep_originals: bool = True,
        train_cap: int = 200_000,
    ) -> dict:
        """Build **one** shard independently — the unit of a parallel/distributed
        build. Shard 0 fits the shared PCA basis and must be written first; every other
        shard reads that basis from ``shard_00000.tqe`` (read-only), so any number of
        workers or pods can build the remaining shards concurrently, each writing its
        own file. Returns the shard's manifest entry — collect them across workers and
        pass to :meth:`finalize_manifest`.

        Ids are ``ids`` if given (arbitrary global ids, e.g. cell-grouped rows for
        distributed routing) else ``arange(id_offset, id_offset+len(block))``. The basis
        comes from shard 0, or from ``basis_from`` (a shard file) if given — so
        cell-grouped shards can reuse an existing index's basis (keeping cells stable);
        the config kwargs apply to shard 0 only. Build time drops from serial-days to
        ``total / n_workers``.
        """
        os.makedirs(out_dir, exist_ok=True)
        chunk = np.asarray(block, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"block must be 2-D (n, dim), got {chunk.shape}")
        if ids is None:
            ids = np.arange(id_offset, id_offset + len(chunk), dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)
        if shard_index == 0 and basis_from is None:
            idx = TQEIndex.create(
                chunk,
                output_dim=output_dim,
                bits=bits,
                seed=seed,
                rotation=rotation,
                whiten=whiten,
                metric=metric,
                keep_originals=keep_originals,
                ids=ids,
                train_cap=train_cap,
            )
        else:
            src = basis_from or os.path.join(out_dir, "shard_00000.tqe")
            base = TQEIndex.open(src, mmap=True)
            idx = TQEIndex(
                base._pca,
                bits=base._bits,
                seed=base._seed,
                rotation=base._rotation,
                metric=base._metric,
                fit_retained_var=base._fit_retained_var,
            )
            idx._append(chunk, ids, keep_originals=keep_originals)
        path = f"shard_{shard_index:05d}.tqe"
        idx.save(os.path.join(out_dir, path))
        return {
            "path": path,
            "n_rows": idx.n_rows,
            "id_min": int(ids.min()) if len(ids) else 0,
            "id_max": int(ids.max()) if len(ids) else -1,
        }

    @classmethod
    def finalize_manifest(
        cls,
        out_dir: str,
        shard_metas: list[dict],
        *,
        metric: str = "cosine",
        shard_size: int | None = None,
    ) -> ShardedIndex:
        """Assemble the manifest from independently-built shard metas (from
        :meth:`write_shard`) and open the index — the coordinator step of a parallel
        build. Metas may arrive in any order; they are sorted by id."""
        metas = sorted(shard_metas, key=lambda m: m["id_min"])
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "n_shards": len(metas),
            "n_rows": sum(m["n_rows"] for m in metas),
            "metric": metric,
            "shard_size": (
                shard_size
                if shard_size is not None
                else (metas[0]["n_rows"] if metas else 0)
            ),
            "shards": metas,
        }
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return cls(manifest_path, mmap=True)

    @classmethod
    def open(
        cls, manifest_path: str, mmap: bool = True, max_open_shards: int = 128
    ) -> ShardedIndex:
        """Open a sharded index from its manifest (shards are opened lazily, at most
        ``max_open_shards`` held open at once to bound file descriptors)."""
        return cls(manifest_path, mmap=mmap, max_open_shards=max_open_shards)

    # ------------------------------------------------------------------ #
    # IVF coarse layer (opt-in, additive)                                #
    # ------------------------------------------------------------------ #
    def build_ivf(
        self,
        nlist: int | None = None,
        *,
        train_cap: int = 200_000,
        kmeans_iters: int = 12,
        seed: int = 42,
        device: str = "cpu",
        centroids: np.ndarray | None = None,
        hierarchical: bool = False,
        top_nlist: int | None = None,
        sub_nlist: int | None = None,
    ) -> ShardedIndex:
        """Add an IVF coarse-partition layer over the already-built shards, in place.

        Fits **one global coarse quantizer** (k-means on the quantized directions of
        the first shard — the same "fit once" pattern as the shared PCA basis), assigns
        every shard's rows to cells, and writes per-shard inverted lists as sidecars
        (memory-mappable ``shard_XXXXX.ivf.{off,memb}.npy``) plus the global
        centroids/radii. Afterwards
        ``search(nprobe=...)`` probes only the best cells instead of scanning every
        row. Opt-in and additive — the shard files themselves are untouched, so this
        can be run against an existing sharded index.

        ``device='gpu'`` runs k-means + the per-shard assignment on the GPU (CuPy). The
        assignment is ``O(N*nlist)``, the build wall at scale; a single GPU does it
        ~1000x faster than NumPy, making a fine ``nlist`` affordable at a billion rows.
        ``centroids`` (an ``(nlist, out_dim)`` array) skips the fit and uses a supplied
        quantizer, so every server in a distributed index can share one global coarse
        layer (required for cross-node cell routing).

        ``hierarchical=True`` builds a **two-level (IVF-of-IVF)** quantizer instead:
        ``top_nlist`` top cells, each split into ``sub_nlist`` leaf cells (leaves stored
        top-major, leaf id ``top*sub_nlist+sub``). Assignment drops to
        ``O(N*(top_nlist+sub_nlist))``, the quantizer is finer at the same build cost,
        and probes cluster into a few top cells — so a query reads contiguous leaves and
        (placed top-cell -> server) touches few servers. ``top_nlist``/``sub_nlist``
        default to ``~sqrt(nlist)`` each. Mutually exclusive with ``centroids=``.
        """
        if hierarchical and centroids is not None:
            raise ValueError("pass either centroids= or hierarchical=, not both")
        rng = np.random.default_rng(seed)
        n = self._n_rows
        adc0 = self._get_shard(0)._adc
        top = leaf_top = None

        if hierarchical:
            base_nlist = nlist if nlist is not None else int(round(np.sqrt(n)))
            base_nlist = int(np.clip(base_nlist, 1, n))
            if sub_nlist is None:
                sub_nlist = max(2, int(round(np.sqrt(base_nlist))))
            if top_nlist is None:
                top_nlist = max(1, int(round(base_nlist / sub_nlist)))
            top_nlist = int(min(top_nlist, n))
            sub_nlist = int(max(1, sub_nlist))
            nlist = top_nlist * sub_nlist
            block0 = max(1024, 50_000_000 // max(nlist, 1))
            d0 = _normalize(adc0._cent[adc0._codes].astype(np.float32))
            train = (
                d0
                if len(d0) <= train_cap
                else d0[rng.choice(len(d0), size=train_cap, replace=False)]
            )
            top, centroids, leaf_top = build_hierarchical_quantizer(
                train,
                top_nlist,
                sub_nlist,
                kmeans_iters,
                rng,
                block=block0,
                device=device,
            )
        elif centroids is not None:
            centroids = np.asarray(centroids, dtype=np.float32)
            nlist = len(centroids)
        else:
            if nlist is None:
                nlist = int(np.clip(round(np.sqrt(n)), 1, n))
            nlist = min(nlist, n)
            block0 = max(1024, 50_000_000 // max(nlist, 1))
            # Fit the coarse centroids once, on shard 0's quantized directions.
            d0 = _normalize(adc0._cent[adc0._codes].astype(np.float32))
            train = (
                d0
                if len(d0) <= train_cap
                else d0[rng.choice(len(d0), size=train_cap, replace=False)]
            )
            centroids = _kmeans_unit(
                train, nlist, kmeans_iters, rng, block=block0, device=device
            )
        block = max(1024, 50_000_000 // max(nlist, 1))  # bound assign memory

        # Assign every shard, accumulate per-cell angular radius, write posting lists.
        radius = np.zeros(nlist, dtype=np.float32)
        for i in range(len(self._shards)):
            adc = self._get_shard(i)._adc
            d = _normalize(adc._cent[adc._codes].astype(np.float32))
            if hierarchical:
                cells = _assign_hier(d, top, centroids, sub_nlist, device=device)
            else:
                cells = _assign(d, centroids, block=block, device=device)
            dots = np.einsum("ij,ij->i", d, centroids[cells])
            np.maximum.at(radius, cells, np.arccos(np.clip(dots, -1.0, 1.0)))
            offsets, members = inverted_lists(cells, nlist)
            # Members are row positions *within this shard*, so uint32 always
            # suffices (shards are far below 4.3B rows) — half the sidecar bytes
            # of int64. Readers take the dtype from the .npy itself, so old
            # int64 sidecars keep working.
            if len(members) == 0 or int(members.max()) < 2**32:
                members = members.astype(np.uint32)
            # Two .npy files (not one .npz) so the large member list is memory-mappable
            # at search time — a probe reads only the cell slices it needs, sequential
            # within a cell (inverted-list members are ascending). Offsets are tiny.
            # Name the sidecar after the shard's own file (not the loop index): after
            # finalize_manifest sorts shards by id, self._shards[i].path need not be
            # shard_{i}.tqe, and the reader (_ivf_scan_shards) keys off the path too.
            base = os.path.join(
                self._dir, os.path.splitext(self._shards[i].path)[0] + ".ivf"
            )
            np.save(base + ".off.npy", offsets)
            np.save(base + ".memb.npy", members)
        np.save(os.path.join(self._dir, "coarse_centroids.npy"), centroids)
        np.save(os.path.join(self._dir, "coarse_radius.npy"), radius)

        ivf_meta = {
            "nlist": int(nlist),
            "centroids": "coarse_centroids.npy",
            "radius": "coarse_radius.npy",
        }
        if hierarchical:
            np.save(os.path.join(self._dir, "coarse_top.npy"), top)
            np.save(os.path.join(self._dir, "coarse_leaf_top.npy"), leaf_top)
            ivf_meta.update(
                hierarchical=True,
                top_nlist=int(top_nlist),
                sub_nlist=int(sub_nlist),
                top="coarse_top.npy",
                leaf_top="coarse_leaf_top.npy",
            )

        with open(self._manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["ivf"] = ivf_meta
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self._ivf_meta = manifest["ivf"]
        self._ivf_centroids, self._ivf_radius = centroids, radius
        self._ivf_hier = hierarchical
        if hierarchical:
            self._ivf_top, self._ivf_leaf_top = top, leaf_top
            self._ivf_sub_nlist = int(sub_nlist)
        return self

    def _load_ivf(self):
        if self._ivf_centroids is None:
            self._ivf_centroids = np.load(
                os.path.join(self._dir, self._ivf_meta["centroids"])
            )
            self._ivf_radius = np.load(
                os.path.join(self._dir, self._ivf_meta["radius"])
            )
        if self._ivf_hier and self._ivf_top is None:
            self._ivf_top = np.load(os.path.join(self._dir, self._ivf_meta["top"]))
            self._ivf_leaf_top = np.load(
                os.path.join(self._dir, self._ivf_meta["leaf_top"])
            )
        return self._ivf_centroids, self._ivf_radius

    @property
    def has_ivf(self) -> bool:
        return self._ivf_meta is not None

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def _get_shard(self, i: int) -> TQEIndex:
        """Return shard ``i``, opening it (mmap) if needed. Keeps at most
        ``_max_open`` shards open, evicting the oldest to bound file descriptors."""
        shard = self._open.get(i)
        if shard is None:
            if len(self._open) >= self._max_open:
                self._open.pop(next(iter(self._open)))  # FIFO evict -> frees its fds
            shard = TQEIndex.open(
                os.path.join(self._dir, self._shards[i].path), mmap=self._mmap
            )
            self._open[i] = shard
        return shard

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        rerank: int = 0,
        block: int | None = None,
        nprobe: int | None = None,
        radius_scale: float = 0.5,
        bound: str = "weighted",
        workers: int = 1,
        top_probe: int | None = None,
        rerank_store=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Global top-``k`` external ids per query across all shards.

        Each shard returns its own top-``k`` (scores are comparable — shared basis
        + metric); the merge keeps the best ``k`` overall. ``rerank`` reranks
        within each shard against that shard's originals before the merge. Shards are
        opened lazily and at most ``max_open_shards`` are held open at once, so the
        fan-out scales to arbitrarily many shards without exhausting file descriptors.

        If ``nprobe`` is given **and** an IVF layer exists (:meth:`build_ivf`), the
        search is sublinear: the global best ``nprobe`` cells are selected once
        (weighted-A\\* order, ``radius_scale``/``bound``) and only those cells' rows
        are scored across shards, rather than scanning every row. ADC-scored (no
        rerank in the IVF path yet). ``workers`` > 1 parallelizes the IVF shard fan-out
        across a thread pool (the shards are independent) — throughput on storage where
        reads don't serialize (NVMe / block). For a **hierarchical** IVF layer,
        ``top_probe`` caps how many top cells a query may open (default derived from
        ``nprobe``/``sub_nlist``); fewer top cells = more locality, fewer servers hit.

        ``rerank_store`` turns on the **tiered rerank**: the IVF pass returns a wide
        ``k*rerank`` shortlist, then those candidates' *original* vectors are fetched
        from the cold tier (an ``NpyOriginalStore`` or any ``fetch(ids)`` callable) and
        exactly re-scored — breaking the ADC recall ceiling with a read bounded to the
        shortlist. Needs ``nprobe`` and ``rerank>0``.
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        if nprobe is not None and self._ivf_meta is not None:
            if rerank and rerank_store is not None:
                depth = k * rerank  # widen the ADC shortlist before exact rescoring
                cand_ids, _ = self._ivf_search(
                    q, depth, nprobe, radius_scale, bound, workers, top_probe
                )
                return rerank_candidates(
                    q, cand_ids, k, rerank_store, metric=self._metric
                )
            return self._ivf_search(
                q, k, nprobe, radius_scale, bound, workers, top_probe
            )
        ids_parts, sc_parts = [], []
        for i in range(len(self._shards)):
            ids, sc = self._get_shard(i).search(q, k=k, rerank=rerank, block=block)
            ids_parts.append(ids)
            sc_parts.append(sc)
        ids = np.concatenate(ids_parts, axis=1)  # (nq, n_shards * k)
        sc = np.concatenate(sc_parts, axis=1)
        sc_f = np.where(np.isfinite(sc), sc, -np.inf)
        order = np.argsort(-sc_f, axis=1)[:, :k]
        out_ids = np.take_along_axis(ids, order, axis=1)
        out_sc = np.take_along_axis(sc, order, axis=1)
        missing = ~np.isfinite(np.take_along_axis(sc_f, order, axis=1))
        out_ids[missing] = -1
        out_sc[missing] = np.nan
        return out_ids, out_sc

    def _ivf_search(self, q, k, nprobe, radius_scale, bound, workers=1, top_probe=None):
        """Sublinear IVF search: pick the best ``nprobe`` cells globally, then score
        only those cells' rows across shards (posting lists), merged to a global top-k.

        Locality-optimized for local storage (NVMe): the fan-out is grouped **by
        cell**, so each probed cell's members and codes are read once per shard —
        sequentially, since inverted-list members are ascending — and scored against
        the whole batch of queries that probe that cell in a single matmul, rather than
        re-gathering the cell once per query with many tiny matmuls. Member lists are
        memory-mapped (``.memb.npy``), so a probe faults in only the cell slices it
        touches. Memory stays O(nq·k) plus one cell; the running per-query top-k merge
        is exact (a global top-k row is necessarily in its own cell's top-k).

        ``workers`` > 1 fans the (independent) shards out across a thread pool: each
        worker scans a disjoint shard subset into its own partial top-k, then the
        partials merge — exact, and real throughput on storage where reads don't
        serialize (the gather + matmul release the GIL)."""
        cent, radius = self._load_ivf()
        q_rot, qbias = self._get_shard(0)._adc._query_terms(q)  # shared basis/pipeline
        qdir = _normalize(q_rot)
        if self._ivf_hier:
            # Two-level probe: pick the best top cells, then the best leaves within them
            # (so the nprobe leaves cluster in a few top cells -> contiguous reads).
            if top_probe is None:
                top_probe = max(
                    1, int(np.ceil(nprobe / max(self._ivf_sub_nlist, 1))) * 2
                )
            probed, _ = probed_leaves_hier(
                qdir,
                self._ivf_top,
                cent,
                radius,
                self._ivf_sub_nlist,
                nprobe,
                top_probe,
                radius_scale=radius_scale,
                bound=bound,
            )
        else:
            theta = np.arccos(np.clip(qdir @ cent.T, -1.0, 1.0))
            beta = 1.0 if bound == "admissible" else float(radius_scale)
            ub = np.cos(np.maximum(0.0, theta - beta * radius[None, :]))
            probed = np.argsort(-ub, axis=1)[:, :nprobe]  # (nq, nprobe) best cells
        nq = len(q)

        # Invert query→cells into cell→queries: read/score each cell once per shard
        # against the batch of queries that probe it, not once per (query, cell).
        cell_qs: dict[int, list[int]] = {}
        for qi in range(nq):
            for c in probed[qi].tolist():
                cell_qs.setdefault(c, []).append(qi)

        n = len(self._shards)
        if workers and workers > 1 and n > 1:
            groups = [list(range(w, n, workers)) for w in range(min(workers, n))]
            with ThreadPoolExecutor(max_workers=len(groups)) as ex:
                partials = list(
                    ex.map(
                        lambda g: self._ivf_scan_shards(
                            g, q_rot, qbias, cell_qs, nq, k
                        ),
                        groups,
                    )
                )
            best_ids, best_sc = self._merge_partials(partials, nq, k)
        else:
            best_ids, best_sc = self._ivf_scan_shards(
                range(n), q_rot, qbias, cell_qs, nq, k
            )

        out_sc = np.where(np.isfinite(best_sc), best_sc, np.nan).astype(np.float32)
        out_ids = np.where(best_sc > -np.inf, best_ids, -1)
        return out_ids, out_sc

    def _ivf_scan_shards(self, shard_indices, q_rot, qbias, cell_qs, nq, k):
        """Score the probed cells over a subset of shards into a partial per-query
        top-k. Opens each shard freshly (no shared cache → thread-safe) and releases it
        before the next, so open fds stay bounded per worker (~one shard's worth)."""
        best_ids = np.full((nq, k), -1, dtype=np.int64)
        best_sc = np.full((nq, k), -np.inf, dtype=np.float32)
        for si in shard_indices:
            shard = TQEIndex.open(
                os.path.join(self._dir, self._shards[si].path), mmap=self._mmap
            )
            # Sidecar name follows the shard's own file (global index), not the loop
            # position — so a server holding a shard *subset* reads the right sidecars.
            base = os.path.join(
                self._dir, os.path.splitext(self._shards[si].path)[0] + ".ivf"
            )
            offsets = np.load(base + ".off.npy")
            members = np.load(
                base + ".memb.npy", mmap_mode="r"
            )  # only cell slices read
            ids_g = np.asarray(shard._ids)
            adc = shard._adc
            for c, qs in cell_qs.items():
                s, e = int(offsets[c]), int(offsets[c + 1])
                if e <= s:
                    continue
                rows = np.asarray(members[s:e])  # ascending → sequential code reads
                cc = adc._cent[adc._codes[rows]]  # one gather serves every query on c
                qa = np.asarray(qs)
                adc_m = cc @ q_rot[qa].T  # (m, |qs|) batched over the cell's queries
                # (m, |qs|) -> score_block wants (nq, m); transpose either side.
                sc_m = score_block(
                    self._metric,
                    adc_m.T,
                    qbias[qa],
                    np.asarray(adc._cnorm[rows]),
                    np.asarray(adc._vrnorm[rows]),
                ).T
                kk = min(k, len(rows))
                part = np.argpartition(-sc_m, kk - 1, axis=0)[:kk]  # top-kk per query
                for j, qi in enumerate(qs):
                    tp = part[:, j]
                    msc = np.concatenate([best_sc[qi], sc_m[tp, j]])
                    mids = np.concatenate([best_ids[qi], ids_g[rows[tp]]])
                    order = np.argsort(-msc)[:k]
                    best_sc[qi], best_ids[qi] = msc[order], mids[order]
            del members, shard  # release this shard's fds before the next
        return best_ids, best_sc

    @staticmethod
    def _merge_partials(partials, nq, k):
        """Merge per-worker partial top-k arrays into the global top-k (exact: a global
        top-k row is in its worker's shard-subset top-k)."""
        best_ids = np.full((nq, k), -1, dtype=np.int64)
        best_sc = np.full((nq, k), -np.inf, dtype=np.float32)
        for pids, psc in partials:
            for qi in range(nq):
                msc = np.concatenate([best_sc[qi], psc[qi]])
                mids = np.concatenate([best_ids[qi], pids[qi]])
                order = np.argsort(-msc)[:k]
                best_sc[qi], best_ids[qi] = msc[order], mids[order]
        return best_ids, best_sc

    def close(self) -> None:
        """Drop the cached open shards (releasing their memory maps / fds)."""
        self._open.clear()

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_shards(self) -> int:
        return len(self._shards)

    def stats(self) -> dict:
        return {
            "n_shards": self.n_shards,
            "n_rows": self._n_rows,
            "metric": self._metric,
            "mmap": self._mmap,
            "shard_rows": [s.n_rows for s in self._shards],
        }
