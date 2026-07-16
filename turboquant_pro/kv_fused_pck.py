# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Fused decode over per-channel keys with dense-sparse outliers (M4 reference).

The M0--M3 fused path (:mod:`turboquant_pro.kv_fused`) decodes PolarQuant codes
-- the *values* format. The recommended *keys* format is
:class:`~turboquant_pro.per_channel_kv.PerChannelKV` (per-channel asym-NF4,
optional top-2% fp16 outliers, three zero-point modes), which until now went
through decompress-then-attend. This module is the NumPy/CuPy reference for
fusing it: keys only enter attention through ``q . k``, so every piece of the
per-channel format becomes a term of the score and nothing is ever scattered
into a reconstructed K:

    score_s = q.mu(h)                    # bias: all zero-point modes fold here
            + sum_j (q_j a_j) grid[c_js]  # dense: per-channel weights, 16-entry grid
            + sum_{j in outliers(s)} q_j Delta_(s,j)   # sparse CSR deltas

with ``Delta = v_fp16 - dequant(code)`` built once from the container
(token-major CSR: selection is channel-major, consumption is token-major).
Per-channel keys live in the original post-RoPE basis -- **no rotation** -- so
the raw decode query is used directly. Values stay PolarQuant (the M1/M2
kernels); the two merge through the shared online-softmax partials.

See ``docs/DESIGN_fused_kv_decode.md`` section 8. Exactness gate:
``fused_decode_pck`` == attention over ``PerChannelKV.decompress`` keys and
dequantized PolarQuant values (tests/test_kv_fused_pck.py).
"""

from __future__ import annotations

import numpy as np

from .kv_fused import _hot_partials, merge_partials
from .per_channel_kv import _NF4, CompressedPerChannelKV, PerChannelKV, _unpack_indices

__all__ = [
    "PreparedPCKBlock",
    "build_outlier_csr",
    "pck_key_scores",
    "pck_cold_partials",
    "fused_decode_pck",
]


def _codes(c: CompressedPerChannelKV) -> np.ndarray:
    B, H, S, D = c.shape
    if c.packed:
        return _unpack_indices(c.indices, B * H * S * D, c.bits).reshape(B, H, S, D)
    return c.indices


def _grid_params(q_kv: PerChannelKV, c: CompressedPerChannelKV):
    """Per-channel (mu, table_weight) so dequant = mu + weight * grid[code].

    Returns ``mu (H, D)``, ``weight (H, D)``, ``grid (levels,)`` for the NF4
    family, or ``mu = zero (H, D)``, ``weight = scale (H, D)``, ``grid =
    arange(levels)`` for uniform -- one formula for every mode. B must be 1
    (decode cache blocks are (1, H, S, D)).
    """
    B, H, S, D = c.shape
    if B != 1:
        raise ValueError("fused decode operates on single-batch blocks (B=1)")
    if c.nf4_scale is not None:
        weight = c.nf4_scale[0]  # (H, D)
        if c.zp_mode == "bias":
            if q_kv.k_bias is None:
                raise ValueError(
                    'container uses zero_point="bias"; pass the PerChannelKV '
                    "built with the same k_bias"
                )
            mu = q_kv.rope_averaged_bias(
                q_kv.k_bias, c.rope_theta, D, c.position_start, S
            )
        elif c.zp_mode == "sparse":
            dc = q_kv.dc_channel_mask(c.rope_theta, D, S)
            mu = np.zeros((H, D), dtype=np.float32)
            mu[:, dc] = c.nf4_mean[0]
        elif c.nf4_mean is not None:
            mu = c.nf4_mean[0]
        else:  # symmetric NF4
            mu = np.zeros((H, D), dtype=np.float32)
        return mu.astype(np.float32), weight.astype(np.float32), _NF4
    if c.levels is not None:
        raise NotImplementedError(
            "fused per-channel decode supports uniform and NF4-family grids; "
            "data-fit quantile (nuq) tables are decompress-then-attend"
        )
    # uniform asymmetric: dequant = zero + scale * code
    mu = c.zero[0, :, 0, :]  # (H, D)
    weight = c.scale[0, :, 0, :]
    grid = np.arange(2**c.bits, dtype=np.float32)
    return mu.astype(np.float32), weight.astype(np.float32), grid


def build_outlier_csr(q_kv: PerChannelKV, c: CompressedPerChannelKV):
    """Token-major CSR of outlier score deltas.

    The container stores outliers channel-major (top-``outlier_frac`` per
    channel) as flat indices + fp16 values that *overwrite* the dense dequant.
    The fused form needs them token-major and as deltas against the dense
    dequant, so the dense pass stays branch-free and the sparse pass is pure
    addition. Returns ``(row_ptr (H, S+1) int32, cols (nnz,) uint16,
    deltas (nnz,) float32)`` or ``None`` when the container has no outliers.
    """
    if c.outlier_idx is None:
        return None
    B, H, S, D = c.shape
    if B != 1:
        raise ValueError("fused decode operates on single-batch blocks (B=1)")
    mu, weight, grid = _grid_params(q_kv, c)
    codes = _codes(c)
    flat = c.outlier_idx.astype(np.int64)
    h = (flat // (S * D)) % H
    s = (flat // D) % S
    j = flat % D
    dense = mu[h, j] + weight[h, j] * grid[codes[0, h, s, j]]
    deltas = c.outlier_val.astype(np.float32) - dense

    order = np.lexsort((j, s, h))  # token-major within head
    h, s, j, deltas = h[order], s[order], j[order], deltas[order]
    row = h * S + s
    counts = np.bincount(row, minlength=H * S)
    row_ptr = np.zeros(H * S + 1, dtype=np.int32)
    np.cumsum(counts, out=row_ptr[1:])
    return row_ptr.reshape(-1)[: H * S + 1], j.astype(np.uint16), deltas


def _is_cupy(xp):
    """True when ``xp`` is the CuPy module (so the K2 kernels are usable)."""
    return getattr(xp, "__name__", "") == "cupy"


class PreparedPCKBlock:
    """Query-independent prepared form of one cold (K, V) page for the fused path.

    A cold page is immutable once flushed, so everything that does not depend on
    the decode query -- the unpacked key codes, the grid parameters ``(mu,
    weight, grid)``, the token-major outlier CSR, and the PolarQuant value
    codes/norms -- is built exactly once (here) and reused for every decode
    step. Per call only the O(H*D) query projections remain: ``w = q * weight``
    and ``bias = q . mu``. This is the amortization the per-call wrapper
    (:func:`turboquant_pro.kv_kernel.fused_decode_pck_cuda` without a prepared
    block) pays on every decode step.

    With ``xp=cupy`` the arrays live on the device and :meth:`partials` runs the
    fused CUDA kernel; with ``xp=numpy`` it runs the reference einsum. Memory
    note: key/value codes are held bit-unpacked (one uint8 per code) -- the
    same materialization the kernel consumes; the packed container remains the
    storage of record.
    """

    def __init__(
        self,
        key_quantizer: PerChannelKV,
        key_container: CompressedPerChannelKV,
        vcodes,
        norm_v,
        xp=np,
    ):
        self.xp = xp
        _, H, S, D = key_container.shape
        self.H, self.S, self.D = H, S, D
        mu, weight, grid = _grid_params(key_quantizer, key_container)
        self.mu = xp.asarray(mu, dtype=xp.float32)
        self.weight = xp.asarray(weight, dtype=xp.float32)
        self.grid = xp.ascontiguousarray(xp.asarray(grid, dtype=xp.float32))
        self.kcodes = xp.ascontiguousarray(
            xp.asarray(_codes(key_container)[0], dtype=xp.uint8)
        )
        csr = build_outlier_csr(key_quantizer, key_container)
        if csr is None:
            row_ptr = np.zeros(H * S + 1, dtype=np.int32)
            cols = np.zeros(0, dtype=np.uint16)
            deltas = np.zeros(0, dtype=np.float32)
        else:
            row_ptr, cols, deltas = csr
        # expanded host index form for the reference correction (np.add.at)
        self._rows_h = np.repeat(np.arange(H * S), np.diff(row_ptr))
        self._heads_h = self._rows_h // S
        self._cols_h = cols.astype(np.int64)
        self._deltas_h = deltas.astype(np.float64)
        # CSR form the kernel consumes
        self.row_ptr = xp.ascontiguousarray(xp.asarray(row_ptr, dtype=xp.int32))
        self.cols = xp.ascontiguousarray(xp.asarray(cols, dtype=xp.uint16))
        self.deltas = xp.ascontiguousarray(xp.asarray(deltas, dtype=xp.float32))
        self.vcodes = xp.ascontiguousarray(xp.asarray(vcodes, dtype=xp.uint8))
        self.norm_v = xp.ascontiguousarray(xp.asarray(norm_v, dtype=xp.float32))

    def key_scores(self, q):
        """Fused key scores (H, S) from the cached arrays (reference path)."""
        xp = self.xp
        qf = xp.asarray(q, dtype=xp.float32)
        w = qf * self.weight
        bias = (qf * self.mu).sum(axis=1)
        scores = xp.einsum("hd,hsd->hs", w, self.grid[self.kcodes]) + bias[:, None]
        if self._cols_h.size:
            qh = np.asarray(q, dtype=np.float64)
            contrib = qh[self._heads_h, self._cols_h] * self._deltas_h
            corr = np.zeros(self.H * self.S, dtype=np.float64)
            np.add.at(corr, self._rows_h, contrib)
            scores = scores + xp.asarray(corr.reshape(self.H, self.S), dtype=xp.float32)
        return scores

    def partials(self, q, tq, scale):
        """Unnormalized (m, l, acc) for this page: the CUDA kernel on GPU
        (``xp=cupy``), the reference einsum on CPU. Merges with hot/other-page
        partials via :func:`turboquant_pro.kv_fused.merge_partials`."""
        xp = self.xp
        if _is_cupy(xp):
            from .kv_kernel import pck_block_partials_cuda

            return pck_block_partials_cuda(q, self, tq, scale=scale)
        from .kv_fused import _rot_matrices

        _, pi, cent = _rot_matrices(tq, xp, like=xp.asarray(q))
        sc = self.key_scores(q) * scale
        m = xp.amax(sc, axis=1)
        e = xp.exp(sc - m[:, None])
        acc_code = xp.einsum("hs,hsd->hd", e * self.norm_v, cent[self.vcodes])
        return m, e.sum(axis=1), acc_code @ pi


def pck_key_scores(q, q_kv: PerChannelKV, c: CompressedPerChannelKV, xp=np):
    """Fused per-channel key scores (H, S): bias + dense grid sum + CSR deltas.

    On CuPy the dense sum and the outlier deltas run as the Volta K2 kernels
    (:mod:`turboquant_pro.volta_kernels`), reading the codes directly with no
    ``(H,S,D)`` fp32 intermediate; on NumPy the einsum reference runs. Both are
    exact to fp32 (``tests/test_volta_k2.py``)."""
    B, H, S, D = c.shape
    mu, weight, grid = _grid_params(q_kv, c)
    codes = _codes(c)[0]  # (H, S, D)
    from .kv_fused import _align_device

    qf = xp.asarray(q, dtype=xp.float32)
    w = qf * _align_device(xp.asarray(weight), qf)  # per-channel table weights
    bias = (qf * _align_device(xp.asarray(mu), qf)).sum(axis=1)  # zero-point
    g = _align_device(xp.asarray(grid, dtype=xp.float32), qf)

    if _is_cupy(xp):
        from .volta_kernels import apply_outlier_csr, k2_key_scores

        scores = k2_key_scores(xp.asarray(codes), w, bias, g)
        csr = build_outlier_csr(q_kv, c)
        if csr is not None:
            row_ptr, cols, deltas = csr
            scores = apply_outlier_csr(
                scores,
                xp.asarray(row_ptr),
                xp.asarray(cols),
                xp.asarray(deltas),
                qf,
            )
        return scores

    codes_d = _align_device(xp.asarray(codes), qf)
    scores = xp.einsum("hd,hsd->hs", w, g[codes_d]) + bias[:, None]
    csr = build_outlier_csr(q_kv, c)
    if csr is not None:
        row_ptr, cols, deltas = csr
        rows = np.repeat(np.arange(H * S), np.diff(np.asarray(row_ptr)))
        heads = rows // S
        contrib = np.asarray(q, dtype=np.float64)[
            heads, np.asarray(cols, dtype=np.int64)
        ] * np.asarray(deltas, dtype=np.float64)
        corr = np.zeros(H * S, dtype=np.float64)
        np.add.at(corr, rows, contrib)
        scores = scores + xp.asarray(corr.reshape(H, S), dtype=xp.float32)
    return scores


def pck_cold_partials(q, q_kv, kc, vcodes, norm_v, tq, scale, xp=np):
    """Unnormalized (m, l, acc) over a cold block: per-channel keys, PolarQuant
    values -- the M4 analogue of ``kv_fused._cold_partials``."""
    from .kv_fused import _rot_matrices

    _, pi, cent = _rot_matrices(tq, xp, like=xp.asarray(q))
    sc = pck_key_scores(q, q_kv, kc, xp) * scale
    m = xp.amax(sc, axis=1)
    e = xp.exp(sc - m[:, None])
    lsum = e.sum(axis=1)
    wv = e * xp.asarray(norm_v, dtype=xp.float32)
    if _is_cupy(xp):
        from .volta_kernels import value_accum

        acc_code = value_accum(xp.asarray(vcodes), wv, cent)
    else:
        acc_code = xp.einsum("hs,hsd->hd", wv, cent[vcodes])
    return m, lsum, acc_code @ pi


def fused_decode_pck(
    q,
    hot_k,
    hot_v,
    key_quantizer,
    key_container,
    vcodes,
    norm_v,
    tq,
    xp=np,
):
    """Full decode step: fp16 hot window + per-channel-coded cold keys with
    PolarQuant values, merged by online softmax. Exact vs decompress-then-attend."""
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    parts = []
    if key_container is not None and key_container.shape[2] > 0:
        parts.append(
            pck_cold_partials(
                q, key_quantizer, key_container, vcodes, norm_v, tq, scale, xp
            )
        )
    if hot_k is not None and hot_k.shape[1] > 0:
        parts.append(_hot_partials(q, hot_k, hot_v, scale, xp))
    if not parts:
        raise ValueError("no keys: provide hot and/or cold")
    return merge_partials(parts, xp)
