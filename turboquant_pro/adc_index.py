"""Fast compressed-domain search over PCA-Matryoshka + TurboQuant codes.

:class:`ADCIndex` stores the same compact codes produced by
:meth:`PCAMatryoshkaPipeline.compress_batch` and searches them with an
asymmetric-distance (ADC) scan that reproduces the pipeline's 768-d
reconstruct-cosine ranking *exactly*, but ~8x faster, using an optional AVX2
kernel (:mod:`turboquant_pro._adc`) with a correct numpy fallback.

Example
-------
::

    pca = PCAMatryoshka(input_dim=768, output_dim=256)
    pca.fit(train)                                      # returns a PCAFitResult
    index = ADCIndex(pca.with_quantizer(bits=3)).add(corpus)
    mips = ADCIndex(pca.with_quantizer(bits=3), metric="inner_product").add(corpus)
    idx, scores = index.search(queries, k=10)          # fast, compressed
    idx = index.search(queries, k=10, rerank=5, originals=corpus)  # exact rerank

The math: for a query ``q`` and a stored vector reconstructed to the original
space ``recon = inverse_transform(norm * unrotate(cent[codes]))``,

    cos(q, recon) = (q.mean + norm * sum_j rotate(q_proj)[j] * cent[code[j]])
                    / ||recon||

with ``q_proj = q @ components^T`` and ``||recon||`` precomputed per vector at
build time. The ADC sum is what the kernel evaluates over the packed codes.

Storage (v3, 2026-09-17). Codes live in the kernel's blocked nibble layout
(:class:`~turboquant_pro.packed_codes.BlockedCodes`) from the moment they are
computed: half a byte per code, packed once, never repacked at search. The
index is a list of **chunks**, one per :meth:`add` batch, each scanned in place;
an IVF cell is the same object with a centroid. ``_codes`` remains readable as
an ``(N, d)`` uint8 array for the paths that reconstruct or gather rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _adc
from .metrics import COSINE, INNER_PRODUCT, L2, check_metric, exact_scores
from .packed_codes import BlockedCodes, PackedCodes
from .pca import EigenweightedPipeline, PCAMatryoshkaPipeline
from .telemetry import trace as _trace

TABLE = 16  # symbol-table stride of the kernel (pshufb needs 16 entries)


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

    ``q.recon = qbias + cnorm * adc`` in every metric (with ``q`` unit under
    cosine, as given otherwise). The three metrics are three functions of that
    inner product and ``||recon||``:

    * **inner_product**: ``q.recon`` itself.
    * **cosine**: divide by ``||recon||``, which is stored directly as
      ``vrnorm = 1/||recon||``.
    * **l2**: rank by ``-||q-recon||^2 = 2 q.recon - ||recon||^2`` after
      dropping the per-query constant ``||q||^2``. ``||recon||^2`` is
      recovered as ``1/vrnorm^2``.

    None of them needs a stored byte beyond the codes, ``cnorm`` and
    ``vrnorm``. Defining this once is deliberate: the flat, blocked, IVF and
    sharded paths must not be able to disagree about what a score is.
    """
    inner = qbias[:, None] + cnorm[None, :] * adc
    if metric == INNER_PRODUCT:
        return inner
    if metric == L2:
        recon_sq = 1.0 / np.maximum(np.asarray(vrnorm, dtype=np.float32), 1e-30) ** 2
        return 2.0 * inner - recon_sq[None, :]
    return inner * vrnorm[None, :]


# --------------------------------------------------------------------------- #
# Coders: what a pipeline's quantizer looks like to the scan                   #
# --------------------------------------------------------------------------- #


class _UniformCoder:
    """One rotation and one Lloyd-Max table for every dim (``PCAMatryoshkaPipeline``).

    Presents the general shape the kernel scans — per-dim symbol tables, dims
    grouped into segments with per-row weights — with one segment and no
    weights, so a uniform index costs exactly what it did before.
    """

    def __init__(self, pipeline: PCAMatryoshkaPipeline):
        self._tq = pipeline.quantizer
        self.d = int(pipeline.output_dim)
        self.cent = np.asarray(self._tq.centroids, dtype=np.float32)
        self.nsym = np.full(self.d, len(self.cent), dtype=np.int32)
        self.segs = np.asarray([0, self.d], dtype=np.int32)
        self.tables = np.zeros((self.d, TABLE), dtype=np.float32)
        s = min(len(self.cent), TABLE)
        self.tables[:, :s] = self.cent[None, :s]
        self.kernel_codes = len(self.cent) <= TABLE

    @property
    def nseg(self) -> int:
        return 1

    def rotate(self, x: np.ndarray) -> np.ndarray:
        return self._tq._rotate(x)

    def unrotate(self, y: np.ndarray) -> np.ndarray:
        return self._tq._unrotate(y)

    def encode(self, unit: np.ndarray):
        """Unit directions (n, d) in PCA space -> ``(codes, segw, cc)``.

        ``cc`` is the rotated reconstruction of the direction (weights applied),
        what the norm terms and the numpy scan need.
        """
        rotated = self.rotate(unit)
        codes = np.searchsorted(self._tq.boundaries, rotated).astype(np.uint8)
        return np.ascontiguousarray(codes), None, self.cent[codes]

    def reconstruct(self, codes: np.ndarray, segw: np.ndarray | None) -> np.ndarray:
        """Rotated reconstruction of stored rows from their codes (weights applied)."""
        return self.cent[np.asarray(codes)]

    @property
    def stored_bytes_per_row(self) -> int:
        """Codes packed at their width plus one float32 norm."""
        return -(-self.d * int(self._tq.bits) // 8) + 4


class _SegmentedCoder:
    """One rotation and one table per segment (``EigenweightedPipeline``).

    The unit direction is split into the pipeline's contiguous segments. Each
    segment's sub-vector is scaled to unit norm, rotated by that segment's own
    rotation and quantized with that segment's Lloyd-Max table; the norm it was
    scaled by, its **energy fraction** (the fractions' squares sum to one), is
    stored per row as one byte (``round(f * 255)``) and is the per-segment weight
    the kernel applies. One byte per segment buys widths that follow the spectrum.
    """

    def __init__(self, pipeline: EigenweightedPipeline):
        self._segments = list(pipeline.segments)  # (offset, n, bits, tq)
        self.d = int(pipeline.output_dim)
        if sum(n for _, n, _, _ in self._segments) != self.d:
            raise ValueError("segments do not cover output_dim")
        self.cent = None  # no single table: use ``tables``
        self.segs = np.asarray(
            [off for off, _, _, _ in self._segments] + [self.d], dtype=np.int32
        )
        self.nsym = np.zeros(self.d, dtype=np.int32)
        self.tables = np.zeros((self.d, TABLE), dtype=np.float32)
        for off, n, _bits, tq in self._segments:
            cent = np.asarray(tq.centroids, dtype=np.float32)
            self.nsym[off : off + n] = len(cent)
            self.tables[off : off + n, : min(len(cent), TABLE)] = cent[None, :TABLE]
        self.kernel_codes = bool(self.nsym.max() <= TABLE)

    @property
    def nseg(self) -> int:
        return len(self._segments)

    def _per_segment(self, x: np.ndarray, fn) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        for off, n, _, tq in self._segments:
            out[..., off : off + n] = fn(tq, x[..., off : off + n])
        return out

    def rotate(self, x: np.ndarray) -> np.ndarray:
        return self._per_segment(
            np.asarray(x, dtype=np.float32), lambda tq, v: tq._rotate(v)
        )

    def unrotate(self, y: np.ndarray) -> np.ndarray:
        return self._per_segment(
            np.asarray(y, dtype=np.float32), lambda tq, v: tq._unrotate(v)
        )

    def encode(self, unit: np.ndarray):
        unit = np.asarray(unit, dtype=np.float32)
        n_rows = len(unit)
        codes = np.empty((n_rows, self.d), dtype=np.uint8)
        segw = np.empty((n_rows, self.nseg), dtype=np.float32)
        cc = np.empty((n_rows, self.d), dtype=np.float32)
        for g, (off, n, _, tq) in enumerate(self._segments):
            sub = unit[:, off : off + n]
            f = np.linalg.norm(sub, axis=1).astype(np.float32)
            # the weight is stored as one byte, so score with what is stored
            f8 = (np.round(f * 255.0) / 255.0).astype(np.float32)
            rotated = tq._rotate(sub / np.maximum(f[:, None], 1e-30))
            c = np.searchsorted(tq.boundaries, rotated).astype(np.uint8)
            codes[:, off : off + n] = c
            segw[:, g] = f8
            cent = np.asarray(tq.centroids, dtype=np.float32)
            cc[:, off : off + n] = f8[:, None] * cent[c]
        return np.ascontiguousarray(codes), segw, cc

    def reconstruct(self, codes: np.ndarray, segw: np.ndarray | None) -> np.ndarray:
        codes = np.asarray(codes)
        if segw is None:
            raise ValueError("a segmented index needs its per-row segment weights")
        cc = self.tables[np.arange(self.d)[None, :], codes]  # (n, d) table values
        w = np.repeat(np.asarray(segw, np.float32), np.diff(self.segs), axis=1)
        return (cc * w).astype(np.float32)

    @property
    def stored_bytes_per_row(self) -> int:
        """Each segment's codes packed at its width, one float32 norm, one byte
        of energy fraction per segment."""
        codes = sum(-(-n * int(b) // 8) for _, n, b, _ in self._segments)
        return codes + 4 + self.nseg


def _coder_for(pipeline):
    if isinstance(pipeline, PCAMatryoshkaPipeline):
        return _UniformCoder(pipeline)
    if isinstance(pipeline, EigenweightedPipeline):
        return _SegmentedCoder(pipeline)
    raise TypeError(
        "ADCIndex needs a PCAMatryoshkaPipeline or an EigenweightedPipeline, "
        f"got {type(pipeline).__name__}"
    )


# --------------------------------------------------------------------------- #
# Chunks                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _Chunk:
    codes: object  # BlockedCodes (kernel-scannable) or any (n, d) uint8 array-like
    n: int

    @property
    def scannable(self) -> bool:
        return isinstance(self.codes, BlockedCodes)


class _ChunkedCodes:
    """Read-only ``(N, d)`` uint8 view over several chunks (the legacy ``_codes``)."""

    def __init__(self, chunks: list[_Chunk], dim: int):
        self._chunks = chunks
        self._offsets = np.cumsum([0] + [c.n for c in chunks]).astype(np.int64)
        self.shape = (int(self._offsets[-1]), int(dim))
        self.dtype = np.dtype(np.uint8)
        self.ndim = 2

    def __len__(self) -> int:
        return self.shape[0]

    def astype(self, dtype, copy: bool = True) -> np.ndarray:
        return np.asarray(self).astype(dtype)

    def _gather(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        n = self.shape[0]
        rows = np.where(rows < 0, rows + n, rows)
        if rows.size and (rows.min() < 0 or rows.max() >= n):
            raise IndexError(f"row index out of range for {n} rows")
        out = np.empty((len(rows), self.shape[1]), dtype=np.uint8)
        which = np.searchsorted(self._offsets, rows, side="right") - 1
        for c in np.unique(which):
            sel = which == c
            local = rows[sel] - self._offsets[c]
            out[sel] = np.asarray(self._chunks[c].codes[local])
        return out

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, slice):
            return self._gather(np.arange(*key.indices(self.shape[0])))
        if isinstance(key, (int, np.integer)):
            return self._gather(np.asarray([int(key)]))[0]
        idx = np.asarray(key)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return self._gather(idx)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        parts = [np.asarray(c.codes) for c in self._chunks]
        out = (
            np.concatenate(parts)
            if parts
            else np.empty((0, self.shape[1]), dtype=np.uint8)
        )
        return out if dtype is None else out.astype(dtype)


class _Parts:
    """A per-row float array kept as the parts ``add`` produced, concatenated lazily."""

    def __init__(self):
        self.parts: list[np.ndarray] = []
        self._cat: np.ndarray | None = None

    def append(self, a: np.ndarray) -> None:
        self.parts.append(a)
        self._cat = None

    def set(self, a) -> None:
        self.parts = [] if a is None else [np.asarray(a, dtype=np.float32)]
        self._cat = None

    @property
    def array(self) -> np.ndarray | None:
        if not self.parts:
            return None
        if self._cat is None:
            self._cat = (
                self.parts[0] if len(self.parts) == 1 else np.concatenate(self.parts)
            )
        return self._cat


# --------------------------------------------------------------------------- #
# The index                                                                    #
# --------------------------------------------------------------------------- #


class ADCIndex:
    """Compressed ADC search index built from a fitted PCA-Matryoshka pipeline.

    Accepts a :class:`PCAMatryoshkaPipeline` (one width for every dim) or an
    :class:`EigenweightedPipeline` (widths that follow the spectrum, scanned as
    weighted segments; see :mod:`turboquant_pro.spectrum`). The segmented form
    has no single centroid table, so the paths that reconstruct rows through
    ``_cent`` (``TQEIndex``, ``ShardedIndex``, ``IVFIndex``) take the uniform
    form only.

    Recommendation: build from an **unwhitened** PCA (``whiten=False``, the default).
    Whitening equalizes the PCA modes, which lets low-variance components inject
    angular noise into the quantized direction and measurably lowers retrieval recall
    (~0.82 -> 0.71 recall@10 on LaBSE at 384-d / 3-bit). ``whiten=True`` is supported
    and now scored correctly, but it is a worse operating point for search.

    ``metric`` is one of :data:`turboquant_pro.metrics.METRICS`. Every metric
    scores the query against the reconstruction in the input space. Use
    ``"inner_product"`` when magnitudes carry ranking information: a
    maximum-inner-product consumer, or queries and rows that went through
    different linear maps (a two-map consumer basis), where cosine discards a
    scale the score depends on.
    """

    def __init__(self, pipeline: PCAMatryoshkaPipeline, metric: str = COSINE):
        pca = pipeline.pca
        if not pca.is_fitted:
            raise ValueError("pipeline.pca must be fitted before building an ADCIndex")
        self._metric = check_metric(metric)
        self._pca = pca
        self._coder = _coder_for(pipeline)
        self._tq = getattr(pipeline, "quantizer", None)  # None when segmented
        self._cent = self._coder.cent  # (S,) for a uniform pipeline, else None
        self._mean = np.asarray(pca._mean, dtype=np.float32)
        self._comp = np.asarray(pca._components, dtype=np.float32)  # (out, in)
        self._mp = (self._comp @ self._mean).astype(np.float32)  # (out,), un-rotated
        self._mp_rot = np.ascontiguousarray(
            self._coder.rotate(self._mp[None, :])[0], dtype=np.float32
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
        self._chunks: list[_Chunk] = []
        self._cnorm_parts = _Parts()
        self._vrnorm_parts = _Parts()
        self._segw_parts = _Parts()
        self._freq = None
        self._freq_key = None
        self._unit_scale: np.ndarray | None = None  # inner product's kernel scale

    # ------------------------------------------------------------------ #
    # Storage                                                         #
    # ------------------------------------------------------------------ #
    @property
    def uses_kernel(self) -> bool:
        """True if the compiled AVX2 kernel is in use (else numpy fallback)."""
        return self._kernel is not None

    @property
    def size(self) -> int:
        return sum(c.n for c in self._chunks)

    @property
    def dim(self) -> int:
        return self._coder.d

    @property
    def stored_bytes_per_row(self) -> int:
        """Bytes a stored row costs on disk: packed codes, one float32 norm, and for
        a segmented pipeline one byte of energy fraction per segment. The
        accounting the public comparison uses (every per-vector scalar counted)."""
        return int(self._coder.stored_bytes_per_row)

    @property
    def nbytes(self) -> int:
        """Bytes the stored rows occupy in RAM (codes, norms, segment weights)."""
        total = 0
        for c in self._chunks:
            total += getattr(c.codes, "nbytes", 0) or np.asarray(c.codes).nbytes
        for p in (self._cnorm_parts, self._vrnorm_parts, self._segw_parts):
            total += sum(int(a.nbytes) for a in p.parts)
        return total

    def _store(self, codes) -> _Chunk:
        """Wrap codes for a chunk: blocked for the kernel when they fit in 4 bits and
        live in RAM; a memory-mapped or packed store is kept as it is (numpy path)."""
        if isinstance(codes, BlockedCodes):
            return _Chunk(codes, len(codes))
        if isinstance(codes, (PackedCodes, np.memmap)) or not self._coder.kernel_codes:
            return _Chunk(codes, len(codes))
        return _Chunk(BlockedCodes.from_codes(codes, self._kernel), len(codes))

    @property
    def _codes(self):
        if not self._chunks:
            return None
        if len(self._chunks) == 1:
            return self._chunks[0].codes
        return _ChunkedCodes(self._chunks, self.dim)

    @_codes.setter
    def _codes(self, value) -> None:
        self._chunks = [] if value is None else [self._store(value)]
        self._freq_key = None

    @property
    def _cnorm(self):
        return self._cnorm_parts.array

    @_cnorm.setter
    def _cnorm(self, value) -> None:
        self._cnorm_parts.set(value)

    @property
    def _vrnorm(self):
        return self._vrnorm_parts.array

    @_vrnorm.setter
    def _vrnorm(self, value) -> None:
        self._vrnorm_parts.set(value)

    @property
    def _segw(self):
        return self._segw_parts.array

    @_segw.setter
    def _segw(self, value) -> None:
        self._segw_parts.set(value)

    def add(self, embeddings: np.ndarray) -> ADCIndex:
        """Compress ``embeddings`` (n, input_dim) and index them for fast search.

        Successive calls **accumulate**: later batches are appended to the index
        rather than replacing it, so ``index.add(a).add(b)`` holds both. Returned
        search indices are positional into the concatenation order. Each call
        becomes one chunk, packed once into the kernel's layout.
        """
        x = np.asarray(embeddings, dtype=np.float32)
        xp = np.asarray(self._pca.transform(x), dtype=np.float32)
        cnorm = np.linalg.norm(xp, axis=1).astype(np.float32)
        codes, segw, cc = self._coder.encode(xp / np.maximum(cnorm[:, None], 1e-30))
        vrnorm = self._recon_inverse_norm(cc, cnorm)
        self._chunks.append(self._store(codes))
        self._cnorm_parts.append(cnorm)
        self._vrnorm_parts.append(vrnorm)
        if segw is not None:
            self._segw_parts.append(np.ascontiguousarray(segw, dtype=np.float32))
        self._freq_key = None
        return self

    def _recon_inverse_norm(self, cc: np.ndarray, cnorm: np.ndarray) -> np.ndarray:
        """``1 / ||recon||`` per row from the rotated reconstruction ``cc`` of its
        direction and its norm."""
        if self._whiten:
            # Reconstruction lives in the un-whitened space: un-rotate the codes to
            # PCA coordinates, undo the 1/sqrt(eig) scale, then measure the norm.
            uw = (self._coder.unrotate(cc) * self._sqrt_eig).astype(np.float32)
            s2 = (uw * uw).sum(axis=1).astype(np.float32)
            m_n = (uw @ self._mp).astype(np.float32)
        else:
            # Rotation-invariant fast path: ||unrotate(cc)|| == ||cc|| and
            # unrotate(cc)·mp == cc·rotate(mp) == cc·mp_rot, no un-rotation needed.
            s2 = (cc * cc).sum(axis=1).astype(np.float32)
            m_n = (cc @ self._mp_rot).astype(np.float32)
        recon_n2 = cnorm**2 * s2 + 2.0 * cnorm * m_n + self._mean_sq
        return (1.0 / np.sqrt(np.maximum(recon_n2, 1e-30))).astype(np.float32)

    def _recon_inverse_norm_xp(self, xp_hat: np.ndarray) -> np.ndarray:
        """``1 / ||recon||`` per row from a reconstruction given in PCA coordinates
        (the residual-coded form ``centroid + rho * unrotate(cent[codes])``)."""
        xp_hat = np.asarray(xp_hat, dtype=np.float32)
        if self._whiten:
            xp_hat = (xp_hat * self._sqrt_eig).astype(np.float32)
        s2 = (xp_hat * xp_hat).sum(axis=1).astype(np.float32)
        m_n = (xp_hat @ self._mp).astype(np.float32)
        recon_n2 = s2 + 2.0 * m_n + self._mean_sq
        return (1.0 / np.sqrt(np.maximum(recon_n2, 1e-30))).astype(np.float32)

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA coordinates ``xp`` of ``embeddings``, what the codes describe."""
        x = np.asarray(embeddings, dtype=np.float32)
        return np.asarray(self._pca.transform(x), dtype=np.float32)

    def encode_residuals(self, xp: np.ndarray, centroid: np.ndarray):
        """Code ``xp - centroid`` (PCA coordinates) as direction codes plus norm.

        Returns ``(codes, cnorm, vrnorm, segw)`` for :meth:`add_coded`: ``cnorm`` is
        the residual's norm and ``vrnorm`` the inverse norm of the reconstruction
        ``centroid + cnorm * unrotate(cent[codes])``. A chunk coded this way is
        scanned with the per-(query, chunk) constant ``qbias + q_rot . rotate(c)``;
        the lookup table is the same for every centroid, so residual coding costs
        nothing at scan time.
        """
        xp = np.asarray(xp, dtype=np.float32)
        c = np.asarray(centroid, dtype=np.float32).reshape(1, -1)
        r = xp - c
        rho = np.linalg.norm(r, axis=1).astype(np.float32)
        codes, segw, cc = self._coder.encode(r / np.maximum(rho[:, None], 1e-30))
        xp_hat = c + rho[:, None] * self._coder.unrotate(cc)
        return codes, rho, self._recon_inverse_norm_xp(xp_hat), segw

    def add_coded(
        self,
        codes: np.ndarray,
        cnorm: np.ndarray,
        vrnorm: np.ndarray,
        segw: np.ndarray | None = None,
    ) -> ADCIndex:
        """Append rows already coded (by :meth:`encode_residuals` or by another
        index built on the same pipeline) as one chunk. An empty batch still
        adds a chunk, so chunk ids can stand for cells."""
        codes = np.asarray(codes, dtype=np.uint8)
        if codes.ndim != 2 or codes.shape[1] != self.dim:
            raise ValueError(f"codes must be (n, {self.dim}), got {codes.shape}")
        n = len(codes)
        if len(cnorm) != n or len(vrnorm) != n:
            raise ValueError("cnorm and vrnorm must have one entry per row")
        if (segw is None) != (self._coder.nseg == 1):
            raise ValueError("segment weights are required exactly when segmented")
        self._chunks.append(self._store(codes))
        self._cnorm_parts.append(np.ascontiguousarray(cnorm, dtype=np.float32))
        self._vrnorm_parts.append(np.ascontiguousarray(vrnorm, dtype=np.float32))
        if segw is not None:
            self._segw_parts.append(np.ascontiguousarray(segw, dtype=np.float32))
        self._freq_key = None
        return self

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def _query_terms(self, queries: np.ndarray):
        """Per-query terms for the ADC sum.

        Under ``cosine`` the query is normalized (the score is a cosine, so the
        query's magnitude is irrelevant and dividing it out early is cheapest).
        Under ``inner_product`` and ``l2`` it must **not** be: both score
        ``q.recon`` with the query as given (``l2`` after expanding
        ``-||q-recon||^2`` to ``2 q.recon - ||recon||^2`` and dropping the
        per-query constant ``||q||^2``).
        """
        q = np.asarray(queries, dtype=np.float32)
        qn = _normalize(q) if self._metric == COSINE else q
        qt = (qn @ self._comp.T).astype(np.float32)
        if self._whiten:
            # Match the un-whitened reconstruction: the DB codes carry the whitened
            # projection, so the query pairing must restore the sqrt(eig) factor.
            qt = qt * self._sqrt_eig
        q_rot = np.ascontiguousarray(self._coder.rotate(qt), dtype=np.float32)
        qbias = np.ascontiguousarray(qn @ self._mean, dtype=np.float32)
        return q_rot, qbias

    def _kernel_scan(self) -> bool:
        """The compiled kernel scans this index: cosine or inner product, every
        chunk blocked."""
        return (
            self._kernel is not None
            and self._metric in (COSINE, INNER_PRODUCT)
            and hasattr(self._kernel, "search_chunks")
            and all(c.scannable for c in self._chunks)
        )

    def _row_scale(self) -> np.ndarray:
        """The per-row factor the kernel multiplies ``q.recon`` by.

        The kernel scores ``(bias + cnorm * lookup_sum) * scale[n]``, which is
        :func:`score_block` for cosine with ``scale = vrnorm``, and for inner
        product with ``scale = 1``. So inner product is the cosine scan with a
        unit denominator: no second kernel, and the pruned scan's bound, which
        is monotone in the lookup sum for any positive scale, stays valid.
        """
        if self._metric == COSINE:
            return self._vrnorm
        if self._unit_scale is None or len(self._unit_scale) != self.size:
            self._unit_scale = np.ones(self.size, dtype=np.float32)
        return self._unit_scale

    def _kernel_chunks(self):
        blocks = [c.codes.blocked for c in self._chunks]
        ns = np.asarray([c.n for c in self._chunks], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(np.int64)
        return blocks, ns, offsets

    def search_chunks(
        self,
        q_rot: np.ndarray,
        probes: np.ndarray,
        biases: np.ndarray,
        k: int,
        use_simd: bool = True,
    ):
        """Scan of the chunks ``probes[q]`` names for each query (``-1`` pads), with
        the per-(query, chunk) constant ``biases``; the flat and IVF paths share
        it. Returns ``(indices, scores)`` with indices positional in chunk order.
        Uses the kernel when it can scan this index, else the numpy path."""
        if not self._kernel_scan():
            return self._search_chunks_numpy(q_rot, probes, biases, k)
        blocks, ns, offsets = self._kernel_chunks()
        return self._kernel.search_chunks(
            blocks,
            ns,
            offsets,
            np.ascontiguousarray(q_rot, dtype=np.float32),
            self._coder.tables,
            self._coder.nsym,
            self._coder.segs,
            self._cnorm,
            self._row_scale(),
            self._segw,
            np.ascontiguousarray(probes, dtype=np.int32),
            np.ascontiguousarray(biases, dtype=np.float32),
            int(k),
            bool(use_simd),
        )

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        rerank: int = 0,
        originals: np.ndarray | None = None,
        prune: tuple[float, float] | None = None,
    ):
        """Return ``(indices, scores)`` for the top-``k`` matches per query.

        If ``rerank > 0`` and ``originals`` (the fp32 corpus) is given, the top
        ``k * rerank`` ADC candidates are rescored exactly in the index's metric
        (:func:`turboquant_pro.metrics.exact_scores`) and the best ``k`` returned
        (indices only).

        ``prune=(prefix_fraction, z)`` (experimental) uses the compiled two-pass
        scan: it sums the first ``prefix_fraction`` of the dims for every vector,
        extrapolates the rest from that prefix, and finishes only vectors whose
        upper bound can reach the top-k. Returned scores are exact; a true top-k
        vector can be pruned with a probability that shrinks as ``z`` grows. It
        applies to the cosine and inner-product metrics with the kernel
        compiled, one chunk, and a uniform quantizer of at most 4 bits;
        otherwise the unpruned path runs.
        The per-query survivor counts of the last pruned search are kept in
        ``last_survivors``.
        """
        if not self._chunks:
            raise RuntimeError("index is empty; call add() first")
        tr = _trace.begin(
            "ADCIndex.search",
            queries,
            self._trace_identity(),
            k=k,
            rerank=rerank,
            prune=None if prune is None else list(prune),
        )
        q_rot, qbias = self._query_terms(queries)
        if tr:
            tr.lap("encode")
        kk = k * max(rerank, 1) if rerank else k
        path = "kernel"
        if not self._kernel_scan():
            path = "numpy"
            # The kernel scores cosine and inner product (_row_scale); l2, codes
            # above 4 bits and memory-mapped stores take the numpy path, which is
            # exact (and identical in ranking to the blocked, IVF and sharded
            # paths, which share score_block).
            idx, sc = self._search_numpy(q_rot, qbias, kk)
        elif (
            prune is not None
            and hasattr(self._kernel, "search_pruned")
            and len(self._chunks) == 1
            and self._coder.nseg == 1
            and self.dim >= 2
        ):
            path = "kernel_pruned"
            d = self.dim
            m = min(d - 1, max(1, round(prune[0] * d)))
            chunk = self._chunks[0]
            idx, sc, self.last_survivors = self._kernel.search_pruned(
                chunk.codes.blocked,
                chunk.n,
                q_rot,
                self._coder.tables,
                self._coder.nsym,
                self._cnorm,
                self._row_scale(),
                qbias,
                self.code_frequencies(),
                kk,
                m,
                float(prune[1]),
            )
        else:
            nc = len(self._chunks)
            probes = np.broadcast_to(np.arange(nc, dtype=np.int32), (len(q_rot), nc))
            biases = np.broadcast_to(qbias[:, None], (len(q_rot), nc))
            idx, sc = self.search_chunks(q_rot, probes, biases, kk)
        if tr:
            tr.set(scan_path=path)
            tr.lap("scan", candidates=int(kk), rows=int(self.size))
        if rerank and originals is not None:
            if not tr:
                return self._rerank(idx, queries, originals, k)
            out, first_exact = self._rerank(
                idx, queries, originals, k, first_scores=True
            )
            tr.lap("rerank", candidates=int(kk))
            tr.results(idx, sc, out, first_exact, k=k)
            tr.finish()
            return out
        if tr:
            tr.results(idx[:, :k], sc[:, :k], k=k)
            tr.finish()
        return idx[:, :k], sc[:, :k]

    def _trace_identity(self) -> dict:
        """What a trace needs to name the index it ran against (no payload)."""
        return {
            "kind": "ADCIndex",
            "rows": int(self.size),
            "dim": int(self.dim),
            "metric": self._metric,
            "stored_bytes_per_row": int(self.stored_bytes_per_row),
            "kernel": bool(self.uses_kernel),
        }

    def code_frequencies(self) -> np.ndarray:
        """(d', 16) float32: how often each symbol occurs in each dim of the index."""
        key = (self.size, len(self._chunks))
        if self._freq_key != key:
            freq = np.zeros((self.dim, TABLE), dtype=np.float64)
            for rows in self._iter_rows():
                for j in range(self.dim):
                    freq[j] += np.bincount(rows[:, j], minlength=TABLE)[:TABLE]
            self._freq = (freq / max(self.size, 1)).astype(np.float32)
            self._freq_key = key
        return self._freq

    def _iter_rows(self, block: int = 65536):
        """Unpacked ``(rows, d)`` uint8 codes, chunk by chunk, in row blocks that
        bound memory; yields ``(codes_block)`` in index order."""
        for c in self._chunks:
            for s in range(0, c.n, block):
                yield np.asarray(c.codes[s : min(s + block, c.n)])

    def _search_chunks_numpy(self, q_rot, probes, biases, k):
        """Exact float scan of probed chunks, one query at a time."""
        probes = np.asarray(probes, dtype=np.int64)
        biases = np.asarray(biases, dtype=np.float32)
        nq = len(q_rot)
        offsets = np.concatenate([[0], np.cumsum([c.n for c in self._chunks])]).astype(
            np.int64
        )
        segw = self._segw
        cnorm, vrnorm = self._cnorm, self._vrnorm
        out_ix = np.full((nq, k), -1, dtype=np.int64)
        out_sc = np.full((nq, k), -1e30, dtype=np.float32)
        for i in range(nq):
            parts_ix, parts_sc = [], []
            for p, c in enumerate(probes[i]):
                if c < 0 or self._chunks[c].n == 0:
                    continue
                s, e = offsets[c], offsets[c + 1]
                rows = np.asarray(self._chunks[c].codes[0 : e - s])
                cc = self._coder.reconstruct(rows, None if segw is None else segw[s:e])
                adc = q_rot[i][None, :] @ cc.T
                sc = score_block(
                    self._metric, adc, biases[i, p : p + 1], cnorm[s:e], vrnorm[s:e]
                )[0]
                parts_ix.append(np.arange(s, e, dtype=np.int64))
                parts_sc.append(sc.astype(np.float32))
            if not parts_ix:
                continue
            ix = np.concatenate(parts_ix)
            sc = np.concatenate(parts_sc)
            m = min(k, len(ix))
            top = np.argpartition(-sc, m - 1)[:m]
            top = top[np.argsort(-sc[top], kind="stable")]
            out_ix[i, :m] = ix[top]
            out_sc[i, :m] = sc[top]
        return out_ix, out_sc

    def _search_numpy(self, q_rot, qbias, kk):
        """Exact float scan over every chunk, blockwise, with a running top-``kk``."""
        nq = len(q_rot)
        segw = self._segw
        cnorm, vrnorm = self._cnorm, self._vrnorm
        kk = min(kk, self.size)
        best_sc = np.full((nq, 0), -np.inf, dtype=np.float32)
        best_ix = np.full((nq, 0), -1, dtype=np.int64)
        s = 0
        for rows in self._iter_rows():
            e = s + len(rows)
            cc = self._coder.reconstruct(rows, None if segw is None else segw[s:e])
            adc = q_rot @ cc.T
            sc = score_block(self._metric, adc, qbias, cnorm[s:e], vrnorm[s:e])
            ix = np.broadcast_to(np.arange(s, e, dtype=np.int64), (nq, e - s))
            csc = np.concatenate([best_sc, sc.astype(np.float32)], axis=1)
            cix = np.concatenate([best_ix, ix], axis=1)
            keep = min(kk, csc.shape[1])
            part = np.argpartition(-csc, keep - 1, axis=1)[:, :keep]
            best_sc = np.take_along_axis(csc, part, axis=1)
            best_ix = np.take_along_axis(cix, part, axis=1)
            s = e
        order = np.argsort(-best_sc, axis=1, kind="stable")
        return (
            np.take_along_axis(best_ix, order, axis=1),
            np.take_along_axis(best_sc, order, axis=1),
        )

    def _rerank(self, cand, queries, originals, k, first_scores: bool = False):
        """Reorder each query's candidates by the exact score in this index's
        metric. A raw dot product would be the inner-product order under every
        metric, which reorders a cosine or l2 index whose rows are not unit.
        ``first_scores`` also returns the first query's exact top-k scores (the
        query trace keeps them; the public return stays ids only)."""
        q = np.asarray(queries, dtype=np.float32)
        originals = np.asarray(originals, dtype=np.float32)
        out = np.full((len(q), k), -1, dtype=np.int64)
        first = []
        for i in range(len(q)):
            c = cand[i][cand[i] >= 0]
            if len(c) == 0:
                continue
            s = exact_scores(q[i : i + 1], originals[c], self._metric)[0]
            order = np.argsort(-s, kind="stable")[:k]
            out[i, : len(order)] = c[order]
            if i == 0:
                first = s[order].tolist()
        return (out, first) if first_scores else out
