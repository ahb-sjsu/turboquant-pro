"""Fast compressed-domain search over PCA-Matryoshka + TurboQuant codes.

:class:`ADCIndex` stores the same compact codes produced by
:meth:`PCAMatryoshkaPipeline.compress_batch` and searches them with an
asymmetric-distance (ADC) scan that reproduces the pipeline's 768-d
reconstruct-cosine ranking *exactly*, but ~8x faster, using an optional AVX2
kernel (:mod:`turboquant_pro._adc`) with a correct numpy fallback.

Example
-------
::

    pca = PCAMatryoshka(input_dim=768, output_dim=256).fit(train)
    index = ADCIndex(pca.with_quantizer(bits=3)).add(corpus)
    idx, scores = index.search(queries, k=10)          # fast, compressed
    idx = index.search(queries, k=10, rerank=5, originals=corpus)  # exact rerank

The math: for a query ``q`` and a stored vector reconstructed to the original
space ``recon = inverse_transform(norm * unrotate(cent[codes]))``,

    cos(q, recon) = (q.mean + norm * sum_j rotate(q_proj)[j] * cent[code[j]])
                    / ||recon||

with ``q_proj = q @ components^T`` and ``||recon||`` precomputed per vector at
build time. The ADC sum is what the kernel evaluates over the packed codes.
"""

from __future__ import annotations

import numpy as np

from . import _adc
from .pca import PCAMatryoshkaPipeline


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


class ADCIndex:
    """Compressed ADC search index built from a fitted PCA-Matryoshka pipeline."""

    def __init__(self, pipeline: PCAMatryoshkaPipeline):
        pca = pipeline.pca
        if not pca.is_fitted:
            raise ValueError("pipeline.pca must be fitted before building an ADCIndex")
        self._pca = pca
        self._tq = pipeline.quantizer
        self._cent = np.asarray(self._tq.centroids, dtype=np.float32)
        self._mean = np.asarray(pca._mean, dtype=np.float32)
        self._comp = np.asarray(pca._components, dtype=np.float32)  # (out, in)
        mp = (self._comp @ self._mean).astype(np.float32)
        self._mp_rot = np.ascontiguousarray(
            self._tq._rotate(mp[None, :])[0], dtype=np.float32
        )
        self._mean_sq = float(self._mean @ self._mean)
        self._kernel = _adc.load()
        self._codes: np.ndarray | None = None
        self._cnorm: np.ndarray | None = None
        self._vrnorm: np.ndarray | None = None

    @property
    def uses_kernel(self) -> bool:
        """True if the compiled AVX2 kernel is in use (else numpy fallback)."""
        return self._kernel is not None

    @property
    def size(self) -> int:
        return 0 if self._codes is None else len(self._codes)

    def add(self, embeddings: np.ndarray) -> ADCIndex:
        """Compress ``embeddings`` (n, input_dim) and index them for fast search."""
        x = np.asarray(embeddings, dtype=np.float32)
        xp = np.asarray(self._pca.transform(x), dtype=np.float32)
        cnorm = np.linalg.norm(xp, axis=1).astype(np.float32)
        rotated = self._tq._rotate(xp / np.maximum(cnorm[:, None], 1e-30))
        codes = np.searchsorted(self._tq.boundaries, rotated).astype(np.uint8)
        cc = self._cent[codes]
        m_n = (cc @ self._mp_rot).astype(np.float32)
        s2 = (cc * cc).sum(axis=1).astype(np.float32)
        recon_n2 = cnorm**2 * s2 + 2.0 * cnorm * m_n + self._mean_sq
        self._codes = np.ascontiguousarray(codes)
        self._cnorm = cnorm
        self._vrnorm = (1.0 / np.sqrt(np.maximum(recon_n2, 1e-30))).astype(np.float32)
        return self

    def _query_terms(self, queries: np.ndarray):
        qn = _normalize(np.asarray(queries, dtype=np.float32))
        qt = (qn @ self._comp.T).astype(np.float32)
        q_rot = np.ascontiguousarray(self._tq._rotate(qt), dtype=np.float32)
        qbias = np.ascontiguousarray(qn @ self._mean, dtype=np.float32)
        return q_rot, qbias

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        rerank: int = 0,
        originals: np.ndarray | None = None,
    ):
        """Return ``(indices, scores)`` for the top-``k`` matches per query.

        If ``rerank > 0`` and ``originals`` (the fp32 corpus) is given, the top
        ``k * rerank`` ADC candidates are rescored by exact inner product and
        the best ``k`` returned (indices only).
        """
        if self._codes is None:
            raise RuntimeError("index is empty; call add() first")
        q_rot, qbias = self._query_terms(queries)
        kk = k * max(rerank, 1) if rerank else k
        if self._kernel is not None:
            idx, sc = self._kernel.search(
                self._codes,
                q_rot,
                self._cent,
                self._cnorm,
                self._vrnorm,
                qbias,
                kk,
                True,
            )
        else:
            idx, sc = self._search_numpy(q_rot, qbias, kk)
        if rerank and originals is not None:
            return self._rerank(idx, queries, originals, k)
        return idx[:, :k], sc[:, :k]

    def _search_numpy(self, q_rot, qbias, kk):
        cc = self._cent[self._codes]  # (N, d')
        adc = q_rot @ cc.T  # (nq, N)
        scores = (qbias[:, None] + self._cnorm[None, :] * adc) * self._vrnorm[None, :]
        idx = np.argpartition(-scores, min(kk, scores.shape[1] - 1), axis=1)[:, :kk]
        srt = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-srt, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        return idx.astype(np.int64), np.take_along_axis(scores, idx, axis=1)

    def _rerank(self, cand, queries, originals, k):
        q = np.asarray(queries, dtype=np.float32)
        originals = np.asarray(originals, dtype=np.float32)
        out = np.full((len(q), k), -1, dtype=np.int64)
        for i in range(len(q)):
            c = cand[i][cand[i] >= 0]
            if len(c) == 0:
                continue
            s = originals[c] @ q[i]
            top = c[np.argsort(-s)[:k]]
            out[i, : len(top)] = top
        return out
