"""Composable key-coding stages for the KV harness (observer-advantage Part II).

The shipped key path is ``codebook(keys)`` in native channels, with a sink and
per-channel fp16 outliers (``tq_paper_lb_shard.qdq_key_block``). This module puts
three independent stages around that codebook, each selected by one env var, so
every combination runs through the same harness and the same codebooks:

  KEY_BASIS   the coordinates the codebook sees, fitted per (layer, KV head)
      native     identity (the shipped method; the other knobs still apply)
      P          eigenvectors of the key second moment S (reconstruction basis)
      O          observer basis, singular values split evenly between the two
                 maps (Part I's registered O, ``observer_advantage.cell.balanced_o``)
      Oey        observer basis, Eckart-Young split (consumer_basis's O)
      O_foreign  balanced O with C from the NEXT KV group's queries (wrong reader)
      R          seeded random orthogonal rotation (data-free, zero stored bytes)
      H          seeded random-sign normalized Hadamard (QuaRot-style, zero bytes)
  BASIS_FIT   where S and C come from
      prefill    this prompt's own settled keys and the queries reading them
                 (calibration-free; the README's whole-sequence estimate)
      calib      once per model from BASIS_CALIB_N WikiText-2 train sequences
  KEY_ALLOC   bits per coded coordinate, total fixed at D * KEY_BITS per head
      uniform    KEY_BITS everywhere (the shipped allocation)
      read       water-filling on E[(A q)_j^2] * Var((B k)_j): the logit error
      key        water-filling on Var((B k)_j) alone: reconstruction error
      head       water-filling on w_j * Var(k_j), w_j = sum_i sum_t a_it^2 q_ij^2
                 ||W_O (v_t - o_i)||^2: the error the head's output sees through the
                 softmax, the values and W_O (Part V; AATC's key term, arXiv 2608.14191).
                 Native channels only.
      fisher     water-filling on F_j * Var(k_j), F_j = E[(dl/dk_tj)^2] with labels
                 sampled from the model: the error the model's output sees (Part V).
                 Native channels only, calibrated only (it needs a backward pass).
  BYTE_MATCH  0 | 1: add fp16 outliers worth exactly the bytes a dense basis
              stores, so a native arm can be compared at matched stored bytes.
  KEY_JITTER  0 | 1: move every settled key element one fp16 ulp up or down
              (seeded) before coding. An inconsequential perturbation: the arm
              measures how far the endpoints move under fp16 rounding alone,
              the noise floor of every comparison (runs are bit-deterministic,
              so a plain repeat would measure nothing).

Keys are coded as ``z = B k`` and the cache holds
``k_hat = k + B^-1 (Q(z) - z)``, which is ``B^-1 Q(z)`` in exact arithmetic; every
logit is then ``(B^-T q) . Q(B k)``, so at full dimension a basis changes only what
the codebook sees. The residual form carries only the quantization error through
the inverse, so with the codebook replaced by the identity every basis returns the
native keys bit for bit (gate G0 of the registration); the direct form moved a few
near-zero entries by one fp16 ulp, which fp16 inference amplified to ~0.1%
perplexity (smoke, 2026-09-24). Sink tokens and outliers are kept fp16 in the coded
coordinates, so their residual is zero.

Stored-byte accounting (``account``) is per (layer, KV head): code bits, codebook
metadata, outliers at (16-bit value + 16-bit index), a dense basis at D*D fp16
(P, O, Oey, O_foreign; R and H regenerate from a seed), and 4 bits per
coordinate for a non-uniform allocation.
"""

from __future__ import annotations

import math
import os

import torch

KEY_BASIS = os.environ.get("KEY_BASIS", "native")
BASIS_FIT = os.environ.get("BASIS_FIT", "prefill")
KEY_ALLOC = os.environ.get("KEY_ALLOC", "uniform")
BYTE_MATCH = int(os.environ.get("BYTE_MATCH", "0"))
KEY_JITTER = int(os.environ.get("KEY_JITTER", "0"))
BASIS_SEED = int(os.environ.get("BASIS_SEED", "0"))
ALLOC_BMIN = int(os.environ.get("ALLOC_BMIN", "1"))
ALLOC_BMAX = int(os.environ.get("ALLOC_BMAX", "8"))
BASIS_CALIB = os.environ.get("BASIS_CALIB", "")
BASIS_CALIB_N = int(os.environ.get("BASIS_CALIB_N", "16"))
PART5_CALIB_SEQ = int(os.environ.get("PART5_CALIB_SEQ", "1024"))  # head / fisher calibration
FISHER_SEED = int(os.environ.get("FISHER_SEED", "20261013"))  # sampled labels
FISHER_ACCUM = False

BASES = ("native", "P", "O", "Oey", "O_foreign", "R", "H")
FITS = ("prefill", "calib")
ALLOCS = ("uniform", "read", "key", "head", "fisher")
NATIVE_ONLY = ("head", "fisher")  # weights defined per native key channel
DENSE = ("P", "O", "Oey", "O_foreign")  # bases that must be stored per sequence
NEEDS_QUERIES = ("O", "Oey", "O_foreign")
# Same value as consumer_basis.arms.EIG_FLOOR (test_key_coding checks it): the
# relative floor on eigenvalues raised to negative powers.
EIG_FLOOR = 1e-6
OUTLIER_BITS = 32  # fp16 value + 16-bit position


def active() -> bool:
    """True when any stage departs from the shipped key path."""
    return (KEY_BASIS != "native" or KEY_ALLOC != "uniform" or bool(BYTE_MATCH)
            or bool(KEY_JITTER))


def needs_queries() -> bool:
    return KEY_BASIS in NEEDS_QUERIES or KEY_ALLOC in ("read", "head")


def needs_values() -> bool:
    """The head-output weight reads the values and W_O (prefill fit, or its calibration)."""
    return KEY_ALLOC == "head"


def validate(codebook: str, prerope: int, noquant: int) -> None:
    """Refuse configurations the stages cannot honour, loudly (errata 2026-08-15)."""
    if KEY_BASIS not in BASES:
        raise SystemExit(f"unknown KEY_BASIS={KEY_BASIS!r}; expected one of {BASES}")
    if BASIS_FIT not in FITS:
        raise SystemExit(f"unknown BASIS_FIT={BASIS_FIT!r}; expected one of {FITS}")
    if KEY_ALLOC not in ALLOCS:
        raise SystemExit(f"unknown KEY_ALLOC={KEY_ALLOC!r}; expected one of {ALLOCS}")
    if noquant or not active():
        return
    if prerope:
        raise SystemExit(
            "key-coding stages act on post-RoPE keys, where q.k is the logit; "
            "PREROPE=1 has no single read operator per head"
        )
    if KEY_ALLOC != "uniform" and codebook in ("nf4", "nf4a", "kvquant"):
        raise SystemExit(
            f"KEY_ALLOC={KEY_ALLOC} needs a codebook defined at every bit width; "
            f"CODEBOOK={codebook} is 4-bit only (or calibrated per channel)"
        )
    if KEY_ALLOC in NATIVE_ONLY and KEY_BASIS != "native":
        raise SystemExit(f"KEY_ALLOC={KEY_ALLOC} is defined per native channel; KEY_BASIS must be native")
    if KEY_ALLOC == "fisher" and BASIS_FIT != "calib":
        raise SystemExit("KEY_ALLOC=fisher needs a backward pass, so BASIS_FIT must be calib")
    if (BASIS_FIT == "calib" and KEY_BASIS in ("native", "R", "H")
            and KEY_ALLOC not in ("read", "head", "fisher")):
        raise SystemExit(f"BASIS_FIT=calib has nothing to fit for KEY_BASIS={KEY_BASIS}")


def config() -> dict:
    return {
        "key_basis": KEY_BASIS, "basis_fit": BASIS_FIT, "key_alloc": KEY_ALLOC,
        "byte_match": BYTE_MATCH, "key_jitter": KEY_JITTER, "basis_seed": BASIS_SEED,
        "alloc_bmin": ALLOC_BMIN, "alloc_bmax": ALLOC_BMAX,
        "basis_calib_n": BASIS_CALIB_N if BASIS_FIT == "calib" else 0,
        "part5_calib_seq": PART5_CALIB_SEQ if KEY_ALLOC in NATIVE_ONLY and BASIS_FIT == "calib" else 0,
        "fisher_seed": FISHER_SEED if KEY_ALLOC == "fisher" else 0,
    }


# ------------------------------------------------------------------ #
# Second moments                                                     #
# ------------------------------------------------------------------ #


def key_moment(k: torch.Tensor) -> torch.Tensor:
    """(B, H, n, D) keys -> (H, D, D) uncentered second moment, float64."""
    kk = k.double().transpose(0, 1).reshape(k.shape[1], -1, k.shape[3])
    return kk.transpose(1, 2) @ kk / kk.shape[1]


def query_moment(q: torch.Tensor, h_kv: int) -> torch.Tensor:
    """(B, Hq, T, D) queries -> (H_kv, D, D): pooled over each KV head's readers.

    Query head i reads KV head i // (Hq / H_kv), the GQA layout of Llama,
    Mistral and Qwen2 (``repeat_kv``)."""
    b, hq, t, d = q.shape
    g = hq // h_kv
    qq = q.double().reshape(b, h_kv, g, t, d).transpose(0, 1).reshape(h_kv, -1, d)
    return qq.transpose(1, 2) @ qq / qq.shape[1]


# ------------------------------------------------------------------ #
# Bases (batched over heads, float64)                                #
# ------------------------------------------------------------------ #


def _eigh_desc(m):
    w, v = torch.linalg.eigh((m + m.transpose(-1, -2)) / 2)
    return w.flip(-1), v.flip(-1)


def _mat_pow(m, p):
    w, v = _eigh_desc(m)
    w = torch.maximum(w, EIG_FLOOR * w[..., :1])
    return (v * w.unsqueeze(-2) ** p) @ v.transpose(-1, -2)


def observer_maps(s: torch.Tensor, c: torch.Tensor, balanced: bool = True):
    """(query_map A, key_map B), each (H, D, D), with (A q).(B k) ~ q.k.

    With C^1/2 S^1/2 = U L V^T: Eckart-Young A = (C^-1/2 U)^T, B = L V^T S^-1/2;
    balanced A = L^1/2 U^T C^-1/2, B = L^1/2 V^T S^-1/2. The formulas of
    consumer_basis.run.bases and observer_advantage.cell.balanced_o, batched."""
    u, lam, vt = torch.linalg.svd(_mat_pow(c, 0.5) @ _mat_pow(s, 0.5))
    a = (_mat_pow(c, -0.5) @ u).transpose(-1, -2)
    b = lam.unsqueeze(-1) * vt @ _mat_pow(s, -0.5)
    if balanced:
        r = lam.sqrt().unsqueeze(-1)
        a, b = r * a, b / r
    return a, b


def _hadamard(d: int) -> torch.Tensor:
    if d & (d - 1):
        raise SystemExit(f"KEY_BASIS=H needs a power-of-two head_dim, got {d}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(d)


def seeded_basis(kind: str, h: int, d: int, layer: int) -> torch.Tensor:
    """(H, D, D) data-free orthogonal basis, reproducible from (seed, layer, head)."""
    out = []
    for hi in range(h):
        g = torch.Generator().manual_seed(BASIS_SEED * 1_000_003 + layer * 1009 + hi)
        if kind == "R":
            qm, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
            out.append(qm * torch.sign(torch.diagonal(r)))
        else:
            signs = torch.randint(0, 2, (d,), generator=g).double() * 2 - 1
            out.append(_hadamard(d) * signs)
    return torch.stack(out)


def key_map(kind: str, s, c, layer: int, h: int, d: int) -> torch.Tensor:
    """(H, D, D) float64 map B for this layer's KV heads."""
    if kind == "native":
        return torch.eye(d, dtype=torch.float64).expand(h, d, d).clone()
    if kind in ("R", "H"):
        return seeded_basis(kind, h, d, layer)
    if kind == "P":
        return _eigh_desc(s)[1].transpose(-1, -2)
    if kind == "O_foreign":
        if h < 2:
            raise SystemExit("KEY_BASIS=O_foreign needs at least two KV heads")
        c = c.roll(-1, dims=0)  # KV head h is coded for the readers of head h+1
    return observer_maps(s, c, balanced=(kind != "Oey"))[1]


# ------------------------------------------------------------------ #
# Allocation                                                         #
# ------------------------------------------------------------------ #


def water_fill(cost: torch.Tensor, mean_bits: int) -> torch.Tensor:
    """Integer bits (H, D) minimising sum_j cost_j 4^-b_j at sum_j b_j = D*mean_bits.

    Greedy marginal allocation, exact for this separable convex integer program:
    every step gives one bit to the coordinate whose error falls most."""
    h, d = cost.shape
    bits = torch.full((h, d), ALLOC_BMIN, dtype=torch.long, device=cost.device)
    budget = d * (mean_bits - ALLOC_BMIN)
    if budget < 0 or mean_bits > ALLOC_BMAX:
        raise SystemExit(f"KEY_BITS={mean_bits} outside [{ALLOC_BMIN}, {ALLOC_BMAX}]")
    rows = torch.arange(h, device=cost.device)
    for _ in range(budget):
        gain = cost * 0.25 ** bits.double() * 0.75
        gain = torch.where(bits >= ALLOC_BMAX, torch.full_like(gain, -1.0), gain)
        bits[rows, gain.argmax(1)] += 1
    return bits


def allocation(z: torch.Tensor, a_map, c, mean_bits: int):
    """(H, D) bits for coded keys z (B, H, n, D), or None for uniform."""
    if KEY_ALLOC == "uniform":
        return None
    var = z.double().var(dim=2, unbiased=False).mean(0)  # (H, D), per coded coordinate
    if KEY_ALLOC == "key":
        return water_fill(var, mean_bits)
    if KEY_ALLOC in NATIVE_ONLY:  # c is the (H, D) head or Fisher weight, native channels
        return water_fill(var * c.to(var.device, var.dtype), mean_bits)
    read = torch.diagonal(a_map @ c @ a_map.transpose(-1, -2), dim1=-2, dim2=-1)
    return water_fill(var * read, mean_bits)


# ------------------------------------------------------------------ #
# Part V: the head-output consumer                                   #
# ------------------------------------------------------------------ #

_WO = {}  # layer -> o_proj weight (hidden, Hq * D), registered by the harness
HEAD_CHUNK = int(os.environ.get("HEAD_CHUNK", "512"))  # queries per attention block


def register_output_projections(model) -> None:
    """Record every layer's o_proj weight (a reference, no copy) for KEY_ALLOC=head."""
    for li, layer in enumerate(model.model.layers):
        _WO[li] = layer.self_attn.o_proj.weight


def head_weight(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, wo: torch.Tensor,
                n: int) -> torch.Tensor:
    """(H_kv, D) head-output read weight of the first ``n`` keys' channels.

    q (B, Hq, T, D) post-RoPE queries, k and v (B, H_kv, T, D) the full prefill keys and
    values, wo (hidden, Hq * D). For query head g reading KV head g // (Hq / H_kv),
    attention a_it = softmax_t(q_i . k_t / sqrt(D)) under the causal mask, output
    o_i = sum_t a_it v_t, and

        w_j = sum_g sum_i sum_{t < n} a_it^2 q_ij^2 ||W_O^g (v_t - o_i)||^2,

    which is D times the expected energy of the head's output error per unit of key
    noise variance in channel j, with errors independent across keys and channels.
    The squared norm uses M = W_O^gT W_O^g (D x D), so no hidden-size tensor is formed,
    and queries are taken in blocks of HEAD_CHUNK."""
    b, hq, t, d = q.shape
    hkv = k.shape[1]
    grp = hq // hkv
    w = torch.zeros(hkv, d, dtype=torch.float64, device=k.device)
    scale = 1.0 / math.sqrt(d)
    pos = torch.arange(t, device=k.device)
    for g in range(hq):
        h = g // grp
        wg = wo[:, g * d : (g + 1) * d].float()
        m = wg.T @ wg  # (D, D)
        kk, vv = k[:, h].float(), v[:, h].float()  # (B, T, D)
        vmv = ((vv @ m) * vv).sum(-1)  # (B, T)
        for s0 in range(0, t, HEAD_CHUNK):
            s1 = min(t, s0 + HEAD_CHUNK)
            qi = q[:, g, s0:s1].float()  # (B, c, D)
            logit = (qi @ kk.transpose(1, 2)) * scale  # (B, c, T)
            future = pos[None, None, :] > pos[s0:s1][None, :, None]
            a = torch.softmax(logit.masked_fill(future, float("-inf")), dim=-1)
            o = a @ vv  # (B, c, D)
            om = o @ m  # (B, c, D)
            # ||W (v_t - o_i)||^2 = v_t M v_t - 2 v_t M o_i + o_i M o_i
            res = (vmv[:, None, :n] - 2.0 * (om @ vv[:, :n].transpose(1, 2))
                   + (om * o).sum(-1, keepdim=True))
            c = (a[:, :, :n] ** 2 * res.clamp_min(0)).sum(-1)  # (B, c)
            w[h] += (c[..., None].double() * qi.double() ** 2).sum((0, 1))
    return w


# ------------------------------------------------------------------ #
# Calibration moments (BASIS_FIT=calib)                              #
# ------------------------------------------------------------------ #

_CAL = {}  # layer -> [S_sum, C_sum, count]
ACCUMULATING = False


def accumulate_weight(layer: int, w: torch.Tensor, count: int) -> None:
    """BASIS_FIT=calib for head and fisher: sum an (H, D) weight over calibration
    sequences; ``count`` is the number of positions it summed over."""
    w = w.double().cpu()
    if layer not in _CAL:
        _CAL[layer] = [torch.zeros_like(w), None, 0]
    _CAL[layer][0] += w
    _CAL[layer][2] += count


def accumulate(layer: int, k: torch.Tensor, q) -> None:
    s = key_moment(k).cpu()
    c = query_moment(q, k.shape[1]).cpu() if q is not None else None
    if layer not in _CAL:
        _CAL[layer] = [torch.zeros_like(s), None if c is None else torch.zeros_like(c), 0]
    _CAL[layer][0] += s
    if c is not None:
        _CAL[layer][1] += c
    _CAL[layer][2] += 1


def save_calibration(path: str) -> None:
    torch.save({li: (v[0] / v[2], None if v[1] is None else v[1] / v[2])
                for li, v in _CAL.items()}, path)


_LOADED = None


def calibration(layer: int):
    global _LOADED
    if _LOADED is None:
        _LOADED = torch.load(BASIS_CALIB, map_location="cpu")
    return _LOADED[layer]


# ------------------------------------------------------------------ #
# The coded key block                                                #
# ------------------------------------------------------------------ #


def extra_outlier_frac(n: int, d: int, key_bits: int) -> float:
    """Outlier fraction whose net stored bits equal one dense D x D fp16 basis."""
    if not BYTE_MATCH or n <= 0:
        return 0.0
    return 16.0 * d / (n * (OUTLIER_BITS - key_bits))


def account(n: int, h: int, d: int, key_bits: int, group: int, out_frac: float,
            sink: int, meta_per_group: int) -> dict:
    """Stored bits of one layer's settled keys under the active stages."""
    k_out = max(1, int(round(n * out_frac))) if out_frac > 0 else 0
    kept = min(sink, n) + k_out  # per channel (upper bound: sink and outliers may overlap)
    code = n * h * d * key_bits
    meta = math.ceil(n / group) * h * d * meta_per_group * 16
    outl = kept * h * d * (OUTLIER_BITS - key_bits)
    basis = h * d * d * 16 if KEY_BASIS in DENSE else 0
    alloc = h * d * 4 if KEY_ALLOC != "uniform" else 0
    return {"n": n, "elems": n * h * d, "code": code, "meta": meta, "outliers": outl, "basis": basis,
            "alloc": alloc, "total": code + meta + outl + basis + alloc}


def jitter(k: torch.Tensor, layer: int) -> torch.Tensor:
    """Every element moved one fp16 ulp up or down, seeded by (BASIS_SEED, layer)."""
    if k.element_size() != 2:
        raise SystemExit(f"KEY_JITTER needs 16-bit keys, got {k.dtype}")
    g = torch.Generator(device=k.device).manual_seed(BASIS_SEED * 7919 + layer)
    up = torch.rand(k.shape, generator=g, device=k.device) < 0.5
    # On the magnitude bits of a 16-bit float, +-1 is one ulp away from or towards
    # zero. Zero can only move away (a -1 would wrap to a NaN pattern) and the
    # largest finite magnitude only towards zero (a +1 would reach inf).
    bits = k.contiguous().view(torch.int16).int() & 0xFFFF
    sign, mag = bits & 0x8000, bits & 0x7FFF
    top = 0x7BFF if k.dtype == torch.float16 else 0x7F7F  # largest finite
    step = torch.where(up, 1, -1)
    step = torch.where(mag == 0, 1, torch.where(mag >= top, -1, step))
    out = sign | (mag + step)
    return torch.where(out >= 0x8000, out - 0x10000, out).to(torch.int16).view(k.dtype)


def code_keys(k: torch.Tensor, q, layer: int, quantize, key_bits: int, kv_full=None):
    """Code settled keys k (B, H_kv, n, D) through the active stages.

    ``quantize(z, bits)`` is the harness's codebook + sink + outlier path applied
    to coded keys z; ``bits`` is None (uniform) or an (H, D) integer tensor.
    Returns the reconstructed keys in k's dtype."""
    b, h, n, d = k.shape
    if KEY_ALLOC in NATIVE_ONLY:
        # Part V: native channels; weights from the head output or the Fisher table.
        if BASIS_FIT == "calib":
            w = calibration(layer)[0].to(k.device)
        else:
            if q is None or kv_full is None or layer not in _WO:
                raise RuntimeError(f"layer {layer}: KEY_ALLOC=head needs queries, values and W_O")
            w = head_weight(q, kv_full[0], kv_full[1], _WO[layer], n)
        bits = allocation(k.double(), None, w, key_bits)
        z32 = k.float()
        return (k.double() + (quantize(z32, bits).float() - z32).double()).to(k.dtype)
    if BASIS_FIT == "calib" and (KEY_BASIS in DENSE or KEY_ALLOC == "read"):
        s, c = calibration(layer)
        s, c = s.to(k.device), (None if c is None else c.to(k.device))
    else:
        s = key_moment(k) if KEY_BASIS in ("P",) + NEEDS_QUERIES or KEY_ALLOC != "uniform" else None
        c = query_moment(q, h) if needs_queries() else None
    if needs_queries() and c is None:
        raise RuntimeError(f"layer {layer}: no queries captured for KEY_BASIS={KEY_BASIS}")
    bmap = key_map(KEY_BASIS, s, c, layer, h, d).to(k.device)  # (H, D, D)
    binv = torch.linalg.inv(bmap)
    z = torch.einsum("hij,bhnj->bhni", bmap, k.double())
    bits = allocation(z, binv.transpose(-1, -2), c, key_bits) if KEY_ALLOC != "uniform" else None
    z32 = z.float()
    resid = (quantize(z32, bits).float() - z32).double()
    return (k.double() + torch.einsum("hij,bhnj->bhni", binv, resid)).to(k.dtype)
