"""Encode a model's weights with a ``tqp.weight_plan/1`` (``tqp plan encode-weights``).

The codec is GPTQ, one-shot, on the min/max grid of every group of 128 input columns,
as registered and measured in Part III-c (``docs/PREREG_weights_codec_allocation.md``,
results in ``benchmarks/RESULTS_weights_codec_allocation.md``). Its product consequence
is the one this module implements: plan with the one diagonal-Fisher cost table and
encode with GPTQ. Per-codec cost tables bought nothing (C3 failed with C1a holding), so
the planner is unchanged and only the encoder moves.

The functions below are the harness codec (``benchmarks/weight_observer/quant.py``)
ported line for line, and :func:`encode_model` reproduces the harness's encoding path
(``codec_run.encode_group``): the same calibration windows, the same input second moment
``S`` taken from the full-precision model, the same shared-input stacks, each encoded at
every width in ``LEVELS`` and the planned width kept. A test pins the port and the path
bit for bit against the harness, so the product writes the weights the study measured.

``rtn`` is the data-free baseline the plan schema also names. Both codecs write the
dequantized weights back into the model in its own dtype: what the study measured. The
packing of codes and scales into the stored format the plan's byte budget counts is not
done here.

Needs torch; transformers for :func:`encode_model`'s callers only.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

GROUP = 128
LEVELS = (2, 3, 4, 5, 6, 8)
DAMP = 0.01
CODECS = ("gptq", "rtn")
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SHARED_INPUT = (
    ("q_proj", "k_proj", "v_proj"),
    ("o_proj",),
    ("gate_proj", "up_proj"),
    ("down_proj",),
)
N_CALIB, SEQ = 128, 1024
GROUP_LAYERS = 2
# Gemma-2 soft-caps its attention logits and the sdpa path drops the cap.
EAGER_ONLY = ("gemma2",)


# ----------------------------------------------------------------------------- codec


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
    per-row tensor of widths. The step is a tensor-by-tensor division: on CUDA a
    division by a Python scalar is a multiplication by its reciprocal, not correctly
    rounded."""
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    q = torch.as_tensor(_qmax(bits), dtype=x.dtype, device=x.device)
    return lo, (hi - lo).clamp_min(1e-12) / q


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
    damp: float = DAMP,
    block: int = GROUP,
) -> torch.Tensor:
    """GPTQ (Frantar et al. 2022), one-shot: quantize columns in natural order on the
    min/max grid of each group (fixed when the group starts), and push each column's
    rounding error into the columns not yet quantized through the inverse Hessian's
    Cholesky factor. ``H`` is the input second moment (in, in); ``damp`` adds that
    fraction of its mean diagonal. Returns float32."""
    return gptq_stack([w], H, [bits], group, damp, block)[0]


def gptq_stack(
    ws: list,
    H: torch.Tensor,
    bits: list,
    group: int = GROUP,
    damp: float = DAMP,
    block: int = GROUP,
) -> list:
    """GPTQ of several matrices, or of one matrix at several widths, that share one
    input second moment ``H``, in one pass. Returns the float32 results in order."""
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


# ----------------------------------------------------------------------------- model


def attention(model_type: str) -> str:
    """The attention kernel the study measured with: eager where sdpa is not the
    architecture's own computation, sdpa elsewhere."""
    return "eager" if model_type in EAGER_ONLY else "sdpa"


def linear_modules(model) -> dict:
    """{name: nn.Linear} for every decoder projection, in layer order; the names a
    weight cost table and a weight plan use (``layers.<i>.self_attn.q_proj``)."""
    out = {}
    for li, layer in enumerate(model.model.layers):
        for part, holder in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for n in LINEAR:
                m = getattr(holder, n, None)
                if m is not None:
                    out[f"layers.{li}.{part}.{n}"] = m
    return out


def calibration_windows(tok, text: str, n: int = N_CALIB, seq: int = SEQ) -> list:
    """The first ``n`` disjoint ``seq``-token windows of ``text``, each (1, seq)."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    ws = [ids[i : i + seq] for i in range(0, (len(ids) // seq) * seq, seq)][:n]
    return [w[None] for w in ws]


def windows_sha(ws: list) -> str:
    """sha256 of the windows' int64 token ids in order (the harness's ``ids_sha``)."""
    h = hashlib.sha256()
    for w in ws:
        h.update(np.asarray(w.reshape(-1), dtype=np.int64).tobytes())
    return h.hexdigest()


def units(names: list) -> list:
    """The matrices of some layers, in sets that read the same input (one ``S``)."""
    out = []
    for layer in sorted({int(n.split(".")[1]) for n in names}):
        for kinds in SHARED_INPUT:
            u = [
                n
                for n in names
                if int(n.split(".")[1]) == layer and n.split(".")[-1] in kinds
            ]
            if u:
                out.append(u)
    return out


class _Stop(Exception):
    pass


@torch.no_grad()
def input_stats(model, modules: dict, calib: list, stop_after=None) -> dict:
    """{matrix: E[x x^T]} of each module's input over the calibration windows, float32
    (accumulated in float64). ``stop_after``, a module, ends each forward pass once it
    has run: nothing later can change the inputs already seen."""
    sums, handles = {}, []

    def hook(name):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            s = sums[name]
            s[0] += x.T @ x
            s[2] += x.shape[0]

        return f

    def stop(mod, inp, out):
        raise _Stop

    for name, m in modules.items():
        i = m.weight.shape[1]
        sums[name] = [torch.zeros(i, i, device=m.weight.device, dtype=torch.float64)]
        sums[name] += [None, 0]
        handles.append(m.register_forward_hook(hook(name)))
    if stop_after is not None:
        handles.append(stop_after.register_forward_hook(stop))
    try:
        for ids in calib:
            try:
                model(ids.to(model.device), use_cache=False)
            except _Stop:
                pass
    finally:
        for h in handles:
            h.remove()
    return {n: (s[0] / s[2]).float() for n, s in sums.items()}


def check_plan(plan: dict, modules: dict) -> dict:
    """The plan's {matrix: bits}, refused unless it names exactly this model's
    matrices, each at an encodable width."""
    bits = plan["bits"]
    missing = sorted(set(modules) - set(bits))
    extra = sorted(set(bits) - set(modules))
    if missing or extra:
        raise ValueError(
            f"plan does not match the model: {len(missing)} model matrices unplanned "
            f"{missing[:3]}, {len(extra)} planned matrices absent {extra[:3]}"
        )
    bad = {n: b for n, b in bits.items() if b not in LEVELS}
    if bad:
        raise ValueError(f"widths outside {LEVELS}: {dict(list(bad.items())[:3])}")
    return bits


@torch.no_grad()
def encode_model(
    model, plan: dict, calib: list | None, codec: str = "gptq", log=None
) -> dict:
    """Write ``plan``'s widths into ``model`` in place with ``codec``; return a summary.

    GPTQ is one-shot: every ``S`` comes from the full-precision model. Layer groups are
    encoded last to first, so when a group's inputs are collected every earlier layer is
    still full precision and the inputs are exactly the unquantized model's, with no
    second copy of the model held. Each shared-input set is one GPTQ stack at every
    width in ``LEVELS``, as in the harness, and the planned width is kept."""
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}; expected one of {CODECS}")
    mods = linear_modules(model)
    bits = check_plan(plan, mods)
    if codec == "gptq" and not calib:
        raise ValueError("gptq needs calibration windows")
    layers = list(model.model.layers)
    groups = [
        list(range(s, min(len(layers), s + GROUP_LAYERS)))
        for s in range(0, len(layers), GROUP_LAYERS)
    ]
    for grp in reversed(groups):
        names = [n for n in mods if int(n.split(".")[1]) in grp]
        S = None
        if codec == "gptq":
            S = input_stats(model, {n: mods[n] for n in names}, calib, layers[grp[-1]])
        for unit in units(names):
            ws = {n: mods[n].weight.float() for n in unit}
            if codec == "gptq":
                flat = [(n, b) for n in unit for b in LEVELS]
                res = gptq_stack(
                    [ws[n] for n, _ in flat], S[unit[0]], [b for _, b in flat]
                )
                enc = {(n, b): q for (n, b), q in zip(flat, res)}
                for n in unit:
                    mods[n].weight.copy_(enc[(n, bits[n])].to(mods[n].weight.dtype))
                del res, enc
            else:
                for n in unit:
                    q = rtn(ws[n], bits[n])
                    mods[n].weight.copy_(q.to(mods[n].weight.dtype))
        del S
        if log:
            log(f"encoded layers {grp[0]}-{grp[-1]} ({len(names)} matrices)")
    hist: dict = {}
    for b in bits.values():
        hist[b] = hist.get(b, 0) + 1
    return {
        "codec": codec,
        "matrices": len(bits),
        "bits_histogram": {str(b): n for b, n in sorted(hist.items())},
    }
