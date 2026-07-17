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
from dataclasses import dataclass

import numpy as np

from .index import TQEIndex

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

    def __init__(self, manifest_path: str, mmap: bool = True):
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
        self._open: list[TQEIndex] | None = None

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

    @classmethod
    def open(cls, manifest_path: str, mmap: bool = True) -> ShardedIndex:
        """Open a sharded index from its manifest (shards are opened lazily)."""
        return cls(manifest_path, mmap=mmap)

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def _ensure_open(self) -> list[TQEIndex]:
        if self._open is None:
            self._open = [
                TQEIndex.open(os.path.join(self._dir, s.path), mmap=self._mmap)
                for s in self._shards
            ]
        return self._open

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        rerank: int = 0,
        block: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Global top-``k`` external ids per query across all shards.

        Each shard returns its own top-``k`` (scores are comparable — shared basis
        + metric); the merge keeps the best ``k`` overall. ``rerank`` reranks
        within each shard against that shard's originals before the merge.
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        shards = self._ensure_open()
        ids_parts, sc_parts = [], []
        for shard in shards:
            ids, sc = shard.search(q, k=k, rerank=rerank, block=block)
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

    def close(self) -> None:
        """Drop the cached open shards (releasing their memory maps)."""
        self._open = None

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
