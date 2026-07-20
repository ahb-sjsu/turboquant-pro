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


def score_block(
    metric: str,
    adc: np.ndarray,
    qbias: np.ndarray,
    cnorm: np.ndarray,
    vrnorm: np.ndarray,
) -> np.ndarray:
    """Scores (higher is better) for one block, shared by every scan path.

    ``adc`` is ``(nq, m)`` = ``q_rot @ cent[codes].T``; ``qbias`` is ``(nq,)``;
    ``cnorm``/``vrnorm`` are ``(m,)`` per stored row.

    ``q.recon = qbias + cnorm * adc`` in both metrics — what differs is only
    what is done with it:

    * **cosine** — divide by ``||recon||``, which is stored directly as
      ``vrnorm = 1/||recon||``.
    * **l2** — rank by ``-||q-recon||^2 = 2 q.recon - ||recon||^2`` after
      dropping the per-query constant ``||q||^2``. ``||recon||^2`` is
      recovered as ``1/vrnorm^2``, so L2 needs **no extra stored bytes**.

    Defining this once is deliberate: the flat, blocked, IVF and sharded paths
    must not be able to disagree about what a score is.
    """
    inner = qbias[:, None] + cnorm[None, :] * adc
    if metric == "l2":
        recon_sq = 1.0 / np.maximum(np.asarray(vrnorm, dtype=np.float32), 1e-30) ** 2
        return 2.0 * inner - recon_sq[None, :]
    return inner * vrnorm[None, :]


class ADCIndex:
    """Compressed ADC search index built from a fitted PCA-Matryoshka pipeline.

    Recommendation: build from an **unwhitened** PCA (``whiten=False``, the default).
    Whitening equalizes the PCA modes, which lets low-variance components inject
    angular noise into the quantized direction and measurably lowers retrieval recall
    (~0.82 -> 0.71 recall@10 on LaBSE at 384-d / 3-bit). ``whiten=True`` is supported
    and now scored correctly, but it is a worse operating point for search.
    """

    def __init__(self, pipeline: PCAMatryoshkaPipeline, metric: str = "cosine"):
        pca = pipeline.pca
        if not pca.is_fitted:
            raise ValueError("pipeline.pca must be fitted before building an ADCIndex")
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")
        self._metric = metric
        self._pca = pca
        self._tq = pipeline.quantizer
        self._cent = np.asarray(self._tq.centroids, dtype=np.float32)
        self._mean = np.asarray(pca._mean, dtype=np.float32)
        self._comp = np.asarray(pca._components, dtype=np.float32)  # (out, in)
        self._mp = (self._comp @ self._mean).astype(np.float32)  # (out,), un-rotated
        self._mp_rot = np.ascontiguousarray(
            self._tq._rotate(self._mp[None, :])[0], dtype=np.float32
        )
        self._mean_sq = float(self._mean @ self._mean)
        # Whitening awareness. ``PCAMatryoshka.transform`` scales each component by
        # ``1/sqrt(eigenvalue)`` when ``whiten=True``; the reconstruction (and hence
        # the cosine the ADC scorer targets) is in the *un-whitened* original space,
        # so both the query pairing and the reconstruction norm need the per-component
        # ``sqrt(eigenvalue)`` factor. Without it (the pre-fix behaviour) the DB was
        # whitened but the query was not, silently mis-scoring. Kept exact for
        # ``whiten=False`` (``sqrt_eig`` unused). Note: whitening *degrades* retrieval
        # recall (it equalizes PCA modes); ``whiten=False`` is recommended for search.
        self._whiten = bool(getattr(pca, "whiten", False))
        if self._whiten:
            eig = np.asarray(pca._eigenvalues, dtype=np.float32)
            self._sqrt_eig = np.sqrt(np.maximum(eig, 1e-12)).astype(np.float32)
        else:
            self._sqrt_eig = None
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
        """Compress ``embeddings`` (n, input_dim) and index them for fast search.

        Successive calls **accumulate**: later batches are appended to the index
        rather than replacing it, so ``index.add(a).add(b)`` holds both. Returned
        search indices are positional into the concatenation order.
        """
        x = np.asarray(embeddings, dtype=np.float32)
        xp = np.asarray(self._pca.transform(x), dtype=np.float32)
        cnorm = np.linalg.norm(xp, axis=1).astype(np.float32)
        rotated = self._tq._rotate(xp / np.maximum(cnorm[:, None], 1e-30))
        codes = np.searchsorted(self._tq.boundaries, rotated).astype(np.uint8)
        cc = self._cent[codes]
        if self._whiten:
            # Reconstruction lives in the un-whitened space: un-rotate the codes to
            # PCA coordinates, undo the 1/sqrt(eig) scale, then measure the norm.
            uw = (self._tq._unrotate(cc) * self._sqrt_eig).astype(np.float32)
            s2 = (uw * uw).sum(axis=1).astype(np.float32)
            m_n = (uw @ self._mp).astype(np.float32)
        else:
            # Rotation-invariant fast path: ||unrotate(cc)|| == ||cc|| and
            # unrotate(cc)·mp == cc·rotate(mp) == cc·mp_rot, no un-rotation needed.
            s2 = (cc * cc).sum(axis=1).astype(np.float32)
            m_n = (cc @ self._mp_rot).astype(np.float32)
        recon_n2 = cnorm**2 * s2 + 2.0 * cnorm * m_n + self._mean_sq
        codes = np.ascontiguousarray(codes)
        vrnorm = (1.0 / np.sqrt(np.maximum(recon_n2, 1e-30))).astype(np.float32)
        if self._codes is None:
            self._codes = codes
            self._cnorm = cnorm
            self._vrnorm = vrnorm
        else:
            # Append: build up the index across successive add() calls.
            self._codes = np.ascontiguousarray(np.concatenate([self._codes, codes]))
            self._cnorm = np.concatenate([self._cnorm, cnorm])
            self._vrnorm = np.concatenate([self._vrnorm, vrnorm])
        return self

    def _query_terms(self, queries: np.ndarray):
        """Per-query terms for the ADC sum.

        Under ``cosine`` the query is normalized (the score is a cosine, so the
        query's magnitude is irrelevant and dividing it out early is cheapest).
        Under ``l2`` it must **not** be: ``-||q-recon||^2`` expands to
        ``2 q.recon - ||recon||^2`` (dropping the per-query constant
        ``||q||^2``), and that inner product is with the *un-normalized* query.
        """
        q = np.asarray(queries, dtype=np.float32)
        qn = q if self._metric == "l2" else _normalize(q)
        qt = (qn @ self._comp.T).astype(np.float32)
        if self._whiten:
            # Match the un-whitened reconstruction: the DB codes carry the whitened
            # projection, so the query pairing must restore the sqrt(eig) factor.
            qt = qt * self._sqrt_eig
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
        # The compiled kernel implements the cosine score only; l2 takes the
        # numpy path, which is exact (and identical in ranking to the blocked
        # and IVF paths, which share score_block).
        if self._metric == "l2":
            idx, sc = self._search_numpy(q_rot, qbias, kk)
        elif self._kernel is not None:
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
        cc = self._cent[np.asarray(self._codes)]  # (N, d')
        adc = q_rot @ cc.T  # (nq, N)
        scores = score_block(self._metric, adc, qbias, self._cnorm, self._vrnorm)
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
