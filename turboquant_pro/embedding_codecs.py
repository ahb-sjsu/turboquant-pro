# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Embedding codecs for the planner, registered through the plugin interface.

The planner (:mod:`turboquant_pro.planner`) chooses among codecs in the plugin
registry, and until this module the registry held no codec for the
``embedding`` target, so a retrieval plan could only abstain. Each codec here
is the configuration the RaBitQ public campaign ran
(``benchmarks/rabitq_public/cell.py``), with the campaign's stored-byte
accounting, so a plan and the campaign's exhaustive grid price a
configuration identically:

- ``tq_embedding``: :class:`PCAMatryoshka` to ``out_dim``, then TurboQuant
  scalar codes at ``bits``. Bytes: packed codes ``ceil(d' b / 8)`` plus one
  fp32 norm.
- ``faiss_pq`` / ``faiss_opq``: faiss ``PQ{m}x8`` / ``OPQ{m},PQ{m}x8``, inner
  product. Bytes: ``m``.
- ``faiss_rabitq``: faiss flat ``RaBitQ{b}``, inner product, unquantized
  queries (``qb = 0``). Bytes: faiss ``code_size``, the codes and the
  per-vector correction factors.

Shared structures (the PCA basis and rotation, codebooks, the OPQ rotation, a
faiss index's trained tables) live under the container's ``shared`` attribute,
which :func:`turboquant_pro.planner.container_bytes` reports separately and
does not charge per vector, the campaign's rule.

**Search.** A codec may expose ``search(container, queries, n)`` returning the
ids of its top-``n`` candidates under its own score estimator. The retrieval
consumer uses it when present and falls back to exact search over the
decompressed rows otherwise. For ``tq_embedding``, PQ and OPQ the two agree,
because their asymmetric distance is the inner product with the decoded
vector. RaBitQ's is not: its estimator uses per-vector correction factors that
decoding does not reproduce, so it is scored through its own search, as the
campaign scored it.

faiss is an optional dependency. Without it the faiss codecs still register;
building one raises ``ImportError`` and the planner records the codec as
``unsupported`` with that reason instead of dropping it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .plugins import TARGET_EMBEDDING, PluginSpec, register

__all__ = [
    "EmbeddingContainer",
    "TQEmbeddingCodec",
    "FaissCodec",
    "tq_bytes_per_vector",
    "register_embedding_codecs",
]

TQ_BITS = (2, 3, 4)
RABITQ_BITS = (1, 2, 3, 4, 5)
TRAIN_ROWS = 100_000  # PCA fit, as the campaign
FAISS_TRAIN_ROWS = 200_000  # PQ / OPQ / RaBitQ training, as the campaign


@dataclass
class EmbeddingContainer:
    """Per-vector arrays are stored bytes; ``shared`` is not charged per vector."""

    codes: np.ndarray
    norms: np.ndarray | None = None
    shape: tuple = ()
    shared: dict = field(default_factory=dict)


def tq_bytes_per_vector(out_dim: int, bits: int) -> int:
    return int(math.ceil(out_dim * bits / 8)) + 4


def _as_rows(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return a.reshape(-1, a.shape[-1])


def _out_dims(d: int) -> list:
    """Full dimension, and on 1024-d and wider arms d/4 and d/2 (the campaign)."""
    return [d] + ([d // 4, d // 2] if d >= 1024 else [])


# ------------------------------------------------------------------ #
# turboquant-pro                                                      #
# ------------------------------------------------------------------ #


class TQEmbeddingCodec:
    """PCA (optionally truncated) followed by TurboQuant scalar quantization."""

    def __init__(
        self,
        bits: int = 4,
        out_dim: int | None = None,
        seed: int = 0,
        train_rows: int = TRAIN_ROWS,
        dim: int | None = None,
        **_: Any,
    ) -> None:
        if int(bits) not in TQ_BITS:
            raise ValueError(f"tq_embedding supports bits {TQ_BITS}, got {bits}")
        self.bits = int(bits)
        self.out_dim = None if out_dim is None else int(out_dim)
        self.seed = int(seed)
        self.train_rows = int(train_rows)
        self.dim = None if dim is None else int(dim)

    def capabilities(self) -> dict:
        caps: dict = {"bit_widths": list(TQ_BITS), "requires_calibration": False}
        if self.dim:
            # The campaign's grid: full dimension at 2-4 bits, and on wide arms
            # d/4 and d/2 at 3 and 4 bits.
            caps["configs"] = [{"bits": b, "out_dim": self.dim} for b in TQ_BITS] + [
                {"bits": b, "out_dim": od}
                for od in _out_dims(self.dim)[1:]
                for b in (3, 4)
            ]
            caps["bytes_per_vector"] = tq_bytes_per_vector(
                self.out_dim or self.dim, self.bits
            )
        return caps

    def compress(self, x: np.ndarray, **_: Any) -> EmbeddingContainer:
        from .pca import PCAMatryoshka

        rows = _as_rows(x)
        d = rows.shape[1]
        out_dim = self.out_dim or d
        pca = PCAMatryoshka(input_dim=d, output_dim=out_dim)
        pca.fit(rows[: self.train_rows])
        pipe = pca.with_quantizer(bits=self.bits, seed=self.seed)
        comp = pipe.compress_batch(rows)
        codes = np.frombuffer(b"".join(c.packed_bytes for c in comp), dtype=np.uint8)
        codes = codes.reshape(len(comp), -1).copy()
        norms = np.asarray([c.norm for c in comp], dtype=np.float32)
        return EmbeddingContainer(
            codes=codes,
            norms=norms,
            shape=tuple(np.asarray(x).shape),
            shared={"pipeline": pipe},
        )

    def decompress(self, c: EmbeddingContainer) -> np.ndarray:
        from .pca import PCACompressedEmbedding

        pipe = c.shared["pipeline"]
        comp = [
            PCACompressedEmbedding(
                packed_bytes=c.codes[i].tobytes(),
                norm=float(c.norms[i]),
                pca_dim=pipe.output_dim,
                bits=pipe.bits,
            )
            for i in range(c.codes.shape[0])
        ]
        return pipe.decompress_batch(comp).astype(np.float32).reshape(c.shape)


# ------------------------------------------------------------------ #
# faiss                                                               #
# ------------------------------------------------------------------ #


def _faiss():
    try:
        import faiss  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "faiss codecs need faiss (pip install faiss-cpu); "
            "the planner records this codec as unsupported without it"
        ) from e
    return faiss


def pq_m_options(d: int) -> list:
    """Sub-quantizer counts that divide d, from d/8 up to d/2 (1-4 bytes per 8 dims)."""
    return [m for m in (d // 8, d // 4, d // 2) if m >= 1 and d % m == 0]


class FaissCodec:
    """A faiss standalone codec: ``PQ``, ``OPQ`` or flat ``RaBitQ``, inner product."""

    KINDS = ("pq", "opq", "rabitq")

    def __init__(
        self,
        kind: str,
        m: int | None = None,
        bits: int | None = None,
        train_rows: int = FAISS_TRAIN_ROWS,
        seed: int = 0,
        threads: int | None = None,
        dim: int | None = None,
        **_: Any,
    ) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown faiss codec kind {kind!r}")
        self.faiss = _faiss()
        self.kind = kind
        self.dim = None if dim is None else int(dim)
        if kind == "rabitq":
            b = 1 if bits is None else int(bits)
            if b not in RABITQ_BITS:
                raise ValueError(f"faiss_rabitq supports bits {RABITQ_BITS}, got {b}")
            self.bits = b
            self.m = None
        else:
            if m is None and self.dim:
                m = pq_m_options(self.dim)[-1]
            self.m = None if m is None else int(m)
            self.bits = None
        self.train_rows = int(train_rows)
        self.seed = int(seed)
        self.threads = threads

    def _spec(self, d: int) -> str:
        if self.kind == "rabitq":
            return "RaBitQ" if self.bits == 1 else f"RaBitQ{self.bits}"
        m = self.m or pq_m_options(d)[-1]
        if d % m:
            raise ValueError(f"PQ needs m dividing d; m={m}, d={d}")
        return f"PQ{m}x8" if self.kind == "pq" else f"OPQ{m},PQ{m}x8"

    def capabilities(self) -> dict:
        caps: dict = {"requires_calibration": True}
        if self.kind == "rabitq":
            caps["bit_widths"] = list(RABITQ_BITS)
        if self.dim:
            caps["configs"] = (
                [{"bits": b} for b in RABITQ_BITS]
                if self.kind == "rabitq"
                else [{"m": m} for m in pq_m_options(self.dim)]
            )
            # An untrained index already knows its code size: exact, no data.
            faiss = self.faiss
            probe = faiss.index_factory(
                self.dim, self._spec(self.dim), faiss.METRIC_INNER_PRODUCT
            )
            caps["bytes_per_vector"] = int(probe.sa_code_size())
        return caps

    def _build(self, rows: np.ndarray):
        faiss = self.faiss
        if self.threads:
            faiss.omp_set_num_threads(int(self.threads))
        d = rows.shape[1]
        index = faiss.index_factory(d, self._spec(d), faiss.METRIC_INNER_PRODUCT)
        if self.kind == "rabitq":
            faiss.downcast_index(index).qb = 0  # unquantized queries, as the campaign
        else:
            # index_factory turns on polysemous training for "PQ{m}x8": a simulated
            # annealing over code ids that only matters to Hamming-filtered search.
            # ADC scores are invariant to it, and it costs seconds per
            # sub-quantizer regardless of n, so a planner evaluating on samples
            # would spend most of its time there.
            pq_index = faiss.downcast_index(
                index.index if self.kind == "opq" else index
            )
            pq_index.do_polysemous_training = False
        rng = np.random.default_rng(self.seed)
        n_train = min(self.train_rows, rows.shape[0])
        train = rows[np.sort(rng.permutation(rows.shape[0])[:n_train])]
        index.train(train)
        return index

    def compress(self, x: np.ndarray, **_: Any) -> EmbeddingContainer:
        rows = _as_rows(x)
        index = self._build(rows)
        codes = np.ascontiguousarray(index.sa_encode(rows), dtype=np.uint8)
        index.add(rows)
        return EmbeddingContainer(
            codes=codes, shape=tuple(np.asarray(x).shape), shared={"index": index}
        )

    def decompress(self, c: EmbeddingContainer) -> np.ndarray:
        index = c.shared["index"]
        out = np.asarray(index.sa_decode(c.codes), dtype=np.float32)
        return out.reshape(c.shape)

    def search(self, c: EmbeddingContainer, queries: np.ndarray, n: int) -> np.ndarray:
        """Top-``n`` ids under the codec's own estimator (RaBitQ's corrections)."""
        index = c.shared["index"]
        _, ids = index.search(_as_rows(queries), int(n))
        return ids


# ------------------------------------------------------------------ #
# Registration                                                        #
# ------------------------------------------------------------------ #


def register_embedding_codecs(*, overwrite: bool = False) -> list:
    specs = [
        PluginSpec(
            name="tq_embedding",
            factory=lambda **cfg: TQEmbeddingCodec(**cfg),
            targets=frozenset({TARGET_EMBEDDING}),
            tier="beta",
            description="PCA + TurboQuant scalar codes; bytes = codes + fp32 norm",
        ),
        PluginSpec(
            name="faiss_pq",
            factory=lambda **cfg: FaissCodec("pq", **cfg),
            targets=frozenset({TARGET_EMBEDDING}),
            tier="experimental",
            description="faiss PQ{m}x8, inner product; bytes = m",
        ),
        PluginSpec(
            name="faiss_opq",
            factory=lambda **cfg: FaissCodec("opq", **cfg),
            targets=frozenset({TARGET_EMBEDDING}),
            tier="experimental",
            description="faiss OPQ{m},PQ{m}x8, inner product; bytes = m",
        ),
        PluginSpec(
            name="faiss_rabitq",
            factory=lambda **cfg: FaissCodec("rabitq", **cfg),
            targets=frozenset({TARGET_EMBEDDING}),
            tier="experimental",
            description="faiss flat RaBitQ, qb=0, inner product; bytes = code_size",
        ),
    ]
    out = []
    for s in specs:
        try:
            out.append(register(s, overwrite=overwrite))
        except ValueError:
            pass  # already registered
    return out
