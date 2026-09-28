"""Weight codecs: round-to-nearest (Part III), and GPTQ and AWQ (Part III-c).

``rtn`` is Part III's codec: asymmetric min/max grid per group of 128 input columns,
deterministic and data-free. ``gptq`` and ``awq`` are the error-compensating codecs of
``docs/PREREG_weights_codec_allocation.md``. All three share one grid (``_grid`` and
``_qdq``), so that each reduces to ``rtn`` bit for bit at its identity setting (gate G0):
``gptq`` with ``H = I`` and no damping, ``awq`` with ``alpha = 0``. A difference between
codecs is then the mechanism's, not a difference in grids, rounding or grouping.

Both read only the input second moment ``S = E[x x^T]`` of the matrix (Part III's
``tables.Accumulator`` statistic) and, for AWQ, the per-channel mean ``|x|``: GPTQ's
Hessian is proportional to ``S``, and the output error on the calibration inputs is exactly
``E ||D x||^2 = tr(D S D^T)``, so neither needs raw activations.
"""

from __future__ import annotations

import torch

GROUP = 128
LEVELS = (2, 3, 4, 5, 6, 8)


def rtn(w: torch.Tensor, bits: int, group: int = GROUP) -> torch.Tensor:
    """Quantize-dequantize ``w`` (out, in) at ``bits`` with a min/max grid per group of
    ``group`` input columns. Computed in float32; returns float32."""
    out, inp = w.shape
    if inp % group:
        raise ValueError(f"in-dimension {inp} is not a multiple of the group {group}")
    x = w.float().reshape(out, inp // group, group)
    lo, scale = _grid(x, bits)
    return _qdq(x, lo, scale, bits).reshape(out, inp)


def _grid(x: torch.Tensor, bits):
    """(minimum, step) of the min/max grid over the last axis. ``bits`` is an int, or a
    per-row tensor of widths broadcastable against the grid (stacked GPTQ)."""
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    return lo, (hi - lo).clamp_min(1e-12) / _qmax(bits)


def _qmax(bits):
    return 2**bits - 1 if isinstance(bits, int) else (2.0**bits - 1.0)


def _qdq(x: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor, bits):
    """Quantize-dequantize on a given grid: the one rounding rule of every codec."""
    q = _qmax(bits)
    r = ((x - lo) / scale).round()
    r = r.clamp(0, q) if isinstance(q, int) else torch.minimum(r.clamp_min(0), q)
    return r * scale + lo


def gptq(
    w: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    group: int = GROUP,
    damp: float = 0.01,
    block: int = GROUP,
) -> torch.Tensor:
    """GPTQ (Frantar et al. 2022), one-shot: quantize columns in natural order on the
    min/max grid of each group (fixed when the group starts, from the weights as updated
    so far), and push each column's rounding error into the columns not yet quantized
    through the inverse Hessian's Cholesky factor. ``H`` is the input second moment
    (in, in); ``damp`` adds that fraction of its mean diagonal. Returns float32."""
    return gptq_stack([w], H, [bits], group, damp, block)[0]


def gptq_stack(
    ws: list,
    H: torch.Tensor,
    bits: list,
    group: int = GROUP,
    damp: float = 0.01,
    block: int = GROUP,
) -> list:
    """GPTQ of several matrices, or of one matrix at several widths, that share one
    input second moment ``H``, in one pass. Given ``H`` every row is quantized
    independently, so stacking rows changes nothing but the size of each step; the
    stack keeps a GPU busy where one small matrix would leave it launch-bound. Returns
    the float32 results in the order given."""
    inp = ws[0].shape[1]
    if any(w.shape[1] != inp for w in ws) or len(bits) != len(ws):
        raise ValueError(
            "stacked matrices must share the input dimension, one width each"
        )
    if inp % group or block % group:
        raise ValueError("in-dimension and block must be multiples of the group")
    W = torch.cat([w.float() for w in ws])
    rows = torch.cat(
        [
            torch.full((w.shape[0], 1), float(b), device=W.device)
            for w, b in zip(ws, bits)
        ]
    )
    H = H.float().clone()
    dead = torch.diagonal(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    if damp > 0:
        H += damp * torch.diagonal(H).mean() * torch.eye(inp, device=H.device)
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True
    )
    Q = torch.zeros_like(W)
    for i1 in range(0, inp, block):
        i2 = min(i1 + block, inp)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        lo = scale = None
        for i in range(i2 - i1):
            if i % group == 0:
                lo, scale = _grid(W1[:, i : i + group], rows)
            col = W1[:, i]
            q = _qdq(col, lo[:, 0], scale[:, 0], rows[:, 0])
            Q[:, i1 + i] = q
            err = (col - q) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return list(torch.split(Q, [w.shape[0] for w in ws]))


AWQ_ALPHAS = tuple(i / 19 for i in range(20))  # 20 values in [0, 1], 0 and 1 included


def output_error(d: torch.Tensor, S: torch.Tensor) -> float:
    """Mean squared output error on the calibration inputs: E ||D x||^2 = tr(D S D^T)."""
    d = d.double()
    return float(((d @ S.double()) * d).sum())


def awq(
    w: torch.Tensor,
    S: torch.Tensor,
    absmean: torch.Tensor,
    bits: int,
    group: int = GROUP,
    alphas=AWQ_ALPHAS,
):
    """AWQ (Lin et al. 2023): scale input channels by ``s = mean|x|^alpha`` (normalized
    so that sqrt(max s * min s) = 1), quantize ``W diag(s)`` with RTN and undo the scale,
    choosing ``alpha`` from ``alphas`` by the output error on the calibration inputs
    (``output_error``); the first of equal errors wins. Returns (weights, alpha)."""
    best = None
    a = absmean.float().clamp_min(1e-8)
    for alpha in alphas:
        if alpha == 0:
            wq = rtn(w, bits, group)  # s = 1 exactly: no scaling arithmetic at all
        else:
            s = a.pow(alpha).clamp_min(1e-4)
            s = s / (s.max() * s.min()).sqrt()
            wq = rtn(w.float() * s, bits, group) / s
        e = output_error(wq - w.float(), S)
        if best is None or e < best[0]:
            best = (e, wq, alpha)
    return best[1], best[2]


def stored_bits(n_params: int, bits: int, group: int = GROUP) -> float:
    """Codes plus two fp16 per group (min and scale): the matched quantity of a rate."""
    return n_params * bits + (n_params // group) * 32
