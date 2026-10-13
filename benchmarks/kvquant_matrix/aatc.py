"""AATC, attention-aware transform coding of the KV cache, reimplemented for Part V.

Laus, Verdun, Wang, Calmon and Krahmer, arXiv 2608.14191 (August 2026); no public code
was found, so this follows the paper's section 4, Algorithm 1 and Appendix "Experimental
Setting" (docs/PREREG_full_consumer_kv.md, section 3). What the paper fixes and this does:

  transform   per layer and head group, k = h B with B = S^(1/2) P^T from the whitened
              SVD of W_K (W_V for values); by the paper's Proposition (Appendix) P and S
              are the eigenvectors and the square-root eigenvalues of the keys' second
              moment K^T K / T, which is how it is computed here, from WikiText-2. For a
              model whose key projection has a bias (Qwen2.5) the moment includes it.
              A head group holds 512 channels: the paper's G = 1 for Qwen2.5-7B (4 KV
              heads of 128) and G = 2 for Llama-3.1-8B (8 KV heads); so G = 2 for
              Mistral-7B and G = 8 for Llama-2-7B. Keys are coded before RoPE (the cache
              holds h; RoPE acts on the reconstructed key).
  weights     key channel c of layer l: w = mean over calibration tokens and the group's
              query heads of (B[:, head] q)_c^2, pre-RoPE queries; value channel c:
              w = sum over the query heads reading it of ||(B^V W_O)_c||^2. sigma^2 is
              the channel's mean square on the calibration set. Both are divided by
              their layer's 95th percentile before the global allocation.
  allocation  integer widths 0..16 by reverse water filling over every (layer, channel)
              of the unprotected layers, minimising sum w sigma^2 4^(-b) at the budget
              (exact: the marginal gains are decreasing, so the budget's largest gains
              are an optimal allocation). Layers 0-2 are protected: their key channels
              get the average plus one bit and their value channels the average.
  quantizer   per-channel scale s_c = 99th percentile of |h_c| on the calibration set;
              per token and head group alpha = max_c |h_c / s_c|; step
              Delta = 2 alpha s_c / 2^b; codes clamp(floor(h / Delta + 1/2),
              -2^(b-1), 2^(b-1) - 1); a channel with 0 bits is reconstructed as 0.
  window      the first SINK tokens and the last HOT tokens stay in fp16 (the paper's
              4 and 128). The harness codes the settled prefill once, as it does for
              every arm; the paper flushes the window in blocks of 16 during decoding.
  calibration 128 sequences of 2048 tokens from FineWeb sample-10BT and 128 from
              OpenR1-Math-220k (the staged text, part5_aatc_text.py) for sigma^2, w and
              s; the paper does not say how many WikiText-2 tokens fit the transform:
              64 windows of 2048 here.

Env: AATC ("" off, "k" keys, "kv" keys and values), AATC_KEY_BITS and AATC_VAL_BITS
(average code bits per channel), AATC_TRANSFORM (1; 0 = native channels, the reported
``aatc_native``), AATC_VAR_ONLY (1 = the paper's var-only ablation), AATC_CALIB (the
calibration file), AATC_TEXT (the staged calibration text, jsonl).
"""

from __future__ import annotations

import json
import os

import torch

AATC = os.environ.get("AATC", "")
KEY_BITS = float(os.environ.get("AATC_KEY_BITS", "0"))
VAL_BITS = float(os.environ.get("AATC_VAL_BITS", "0"))
TRANSFORM = int(os.environ.get("AATC_TRANSFORM", "1"))
VAR_ONLY = int(os.environ.get("AATC_VAR_ONLY", "0"))
CALIB = os.environ.get("AATC_CALIB", "")
TEXT = os.environ.get("AATC_TEXT", "")
BMAX = 16
PROTECT = 3
GROUP_DIMS = 512
SCALE_PCT = 0.99
NORM_PCT = 0.95
HIST_BINS = 2048
TRANSFORM_WINDOWS, SEQ = 64, 2048
EIG_FLOOR = 1e-10  # relative: an eigenvalue below this x the largest is held at it


def active() -> bool:
    return AATC in ("k", "kv")


def codes_keys() -> bool:
    return active()


def codes_values() -> bool:
    return AATC == "kv"


def validate() -> None:
    if AATC not in ("", "k", "kv"):
        raise SystemExit(f"unknown AATC={AATC!r}; expected '', 'k' or 'kv'")
    if not active():
        return
    if not 0 < KEY_BITS <= BMAX or (codes_values() and not 0 < VAL_BITS <= BMAX):
        raise SystemExit(f"AATC needs average bits in (0, {BMAX}]: keys {KEY_BITS}, values {VAL_BITS}")
    if not CALIB:
        raise SystemExit("AATC needs AATC_CALIB")


def config() -> dict:
    return {"aatc": AATC, "key_bits": KEY_BITS, "val_bits": VAL_BITS if codes_values() else 0,
            "transform": TRANSFORM, "var_only": VAR_ONLY, "bmax": BMAX, "protect": PROTECT,
            "group_dims": GROUP_DIMS}


def groups(hkv: int, d: int) -> int:
    """Head groups of a layer: GROUP_DIMS channels each (whole KV heads)."""
    c = hkv * d
    if c % GROUP_DIMS:
        raise SystemExit(f"{hkv} KV heads of {d} do not split into groups of {GROUP_DIMS}")
    return c // GROUP_DIMS


# ------------------------------------------------------------------ transform


def fit_transform(m2: torch.Tensor):
    """(G, C, C) second moments -> (B, B^-1, S): k = h B, h = k B^-1, E[h_c^2] = S_c."""
    lam, p = torch.linalg.eigh((m2 + m2.transpose(-1, -2)) / 2)
    lam, p = lam.flip(-1), p.flip(-1)  # descending
    lam = lam.clamp_min(EIG_FLOOR * lam[..., :1])
    s = lam.sqrt()  # singular values of the whitened weight
    b = s.sqrt().unsqueeze(-1) * p.transpose(-1, -2)  # S^(1/2) P^T
    binv = p / s.sqrt().unsqueeze(-2)  # P S^(-1/2)
    return b, binv, s


def identity_transform(g: int, c: int, device=None):
    eye = torch.eye(c, dtype=torch.float64, device=device).expand(g, c, c).clone()
    return eye, eye.clone()


# ------------------------------------------------------------------ allocation


def allocate(costs: list, avg: float, bonus: int, protect: int = PROTECT, bmax: int = BMAX):
    """Integer widths per layer: protected layers get round(avg) + bonus per channel, the
    rest share what remains of round(avg x channels) by reverse water filling on
    ``costs`` (per layer, (C,), already normalised), minimising sum c 4^(-b)."""
    sizes = [c.numel() for c in costs]
    budget = round(avg * sum(sizes))
    fixed = max(0, min(bmax, round(avg) + bonus))
    out = [None] * len(costs)
    for li in range(min(protect, len(costs))):
        out[li] = torch.full((sizes[li],), fixed, dtype=torch.long)
        budget -= fixed * sizes[li]
    free = [li for li in range(len(costs)) if out[li] is None]
    if not free:
        return out
    if budget < 0:
        raise SystemExit(f"AATC budget {avg} leaves nothing after the protected layers")
    c = torch.cat([costs[li].double().cpu().clamp_min(0) for li in free])
    if budget > c.numel() * bmax:
        raise SystemExit(f"AATC budget {avg} exceeds {bmax} bits on every channel")
    # gain of the (b+1)-th bit: c (4^-b - 4^-(b+1)) = 0.75 c 4^-b, decreasing in b, so the
    # budget's largest gains form a prefix per channel: an optimal integer allocation.
    gains = 0.75 * c[:, None] * torch.pow(4.0, -torch.arange(bmax, dtype=torch.float64))[None]
    flat = gains.flatten()
    # stable order: by gain, ties by (channel, bit) so lower bits of a channel come first
    order = torch.argsort(-flat, stable=True)
    take = torch.zeros_like(flat, dtype=torch.long)
    take[order[:budget]] = 1
    bits = take.view(-1, bmax).sum(1)
    k = 0
    for li in free:
        out[li] = bits[k : k + sizes[li]]
        k += sizes[li]
    return out


def normalised(x: torch.Tensor) -> torch.Tensor:
    """A layer's weights divided by their 95th percentile (the paper's scale fix)."""
    p = torch.quantile(x.double().flatten(), NORM_PCT)
    return x.double() / p.clamp_min(1e-30)


# ------------------------------------------------------------------ quantizer


def quantize(h: torch.Tensor, bits: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """h (..., C) one head group's coefficients, bits and s (C,): the paper's
    token-scaled symmetric uniform quantizer, dequantized."""
    hf = h.float()
    sf = s.float().to(h.device)
    bf = bits.to(h.device)
    alpha = (hf / sf).abs().amax(-1, keepdim=True).clamp_min(1e-12)
    step = 2 * alpha * sf / torch.pow(2.0, bf.float())
    lo = -torch.pow(2.0, (bf - 1).clamp_min(0).float())
    hi = torch.pow(2.0, (bf - 1).clamp_min(0).float()) - 1
    q = torch.floor(hf / step + 0.5).clamp(lo, hi)
    return torch.where(bf > 0, q * step, torch.zeros_like(hf))


# ------------------------------------------------------------------ calibration


class _Stats:
    """Per layer accumulators of one calibration pass."""

    def __init__(self):
        self.m2 = {}  # (layer, kind) -> (G, C, C) float64
        self.sq = {}  # (layer, kind) -> (G*C,) float64 sum of h^2
        self.n = {}
        self.wq = {}  # layer -> (G*C,) sum of squared query coefficients
        self.amax = {}
        self.hist = {}


def _heads(model):
    cfg = model.config
    hq = cfg.num_attention_heads
    hkv = getattr(cfg, "num_key_value_heads", None) or hq
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // hq
    return hq, hkv, d


def _per_group(x: torch.Tensor, g: int) -> torch.Tensor:
    """(B, T, G*C) -> (B*T, G, C)."""
    return x.reshape(-1, g, x.shape[-1] // g)


def calibrate(model, transform_windows, rd_windows, path: str) -> dict:
    """Fit and save the calibration: transform from ``transform_windows``, then sigma^2,
    the query weights and the 99th percentiles from ``rd_windows`` (lists of (1, T)
    token tensors). Nothing is quantized while it runs."""
    hq, hkv, d = _heads(model)
    g = groups(hkv, d)
    c = hkv * d // g
    grp = hq // hkv
    layers = model.model.layers
    dev = next(model.parameters()).device
    st = _Stats()
    T = {}  # (layer, kind) -> (B, Binv, S)
    hooks = []

    def run(windows):
        with torch.no_grad():
            for ids in windows:
                model(input_ids=ids.to(dev), use_cache=False)

    # pass A: the transform
    def moment(li, kind):
        def hook(mod, inp, out):
            x = _per_group(out.detach().double(), g)
            m = torch.einsum("ngi,ngj->gij", x, x)
            st.m2[(li, kind)] = st.m2.get((li, kind), 0) + m
            st.n[(li, kind)] = st.n.get((li, kind), 0) + x.shape[0]
        return hook

    if TRANSFORM:
        for li, layer in enumerate(layers):
            hooks.append(layer.self_attn.k_proj.register_forward_hook(moment(li, "k")))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(moment(li, "v")))
        try:
            run(transform_windows)
        finally:
            for h in hooks:
                h.remove()
            hooks.clear()
        for key, m in st.m2.items():
            T[key] = fit_transform(m / st.n[key])
    else:
        for li in range(len(layers)):
            for kind in ("k", "v"):
                b, binv = identity_transform(g, c, dev)
                T[(li, kind)] = (b, binv, None)

    def coeffs(li, kind, out):
        b, binv, _ = T[(li, kind)]
        x = _per_group(out.detach().double(), g)
        return torch.einsum("ngi,gij->ngj", x, binv.to(x.device))  # (N, G, C)

    # pass B: sigma^2, query weights, max |h|
    def stats(li, kind):
        def hook(mod, inp, out):
            h = coeffs(li, kind, out).reshape(-1, g * c)
            st.sq[(li, kind)] = st.sq.get((li, kind), 0) + (h**2).sum(0)
            st.n[("rd", li, kind)] = st.n.get(("rd", li, kind), 0) + h.shape[0]
            a = h.abs().amax(0)
            st.amax[(li, kind)] = torch.maximum(st.amax.get((li, kind), a), a)
        return hook

    def qweight(li):
        def hook(mod, inp, out):
            q = out.detach().double().reshape(-1, hq, d)  # (N, Hq, D), pre-RoPE
            b = T[(li, "k")][0].to(q.device)  # (G, C, C): k_group = h_group @ B
            hpg = hkv // g
            w = torch.zeros(g, c, dtype=torch.float64, device=q.device)
            for r in range(hq):
                kv = r // grp
                gi, j = divmod(kv, hpg)
                coef = q[:, r] @ b[gi][:, j * d : (j + 1) * d].T  # (N, C)
                w[gi] += (coef**2).sum(0)
            st.wq[li] = st.wq.get(li, 0) + w.flatten()
        return hook

    for li, layer in enumerate(layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(qweight(li)))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(stats(li, "k")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(stats(li, "v")))
    try:
        run(rd_windows)
    finally:
        for h in hooks:
            h.remove()
        hooks.clear()

    # pass C: histograms of |h| / max for the 99th percentile
    def histo(li, kind):
        def hook(mod, inp, out):
            h = coeffs(li, kind, out).reshape(-1, g * c).abs()
            top = st.amax[(li, kind)].clamp_min(1e-30)
            idx = (h / top * HIST_BINS).long().clamp(0, HIST_BINS - 1)
            flat = idx + torch.arange(g * c, device=h.device)[None] * HIST_BINS
            cnt = torch.bincount(flat.flatten(), minlength=g * c * HIST_BINS)
            st.hist[(li, kind)] = st.hist.get((li, kind), 0) + cnt.view(g * c, HIST_BINS)
        return hook

    for li, layer in enumerate(layers):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(histo(li, "k")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(histo(li, "v")))
    try:
        run(rd_windows)
    finally:
        for h in hooks:
            h.remove()

    out = {"meta": {"hq": hq, "hkv": hkv, "d": d, "groups": g, "transform": TRANSFORM,
                    "transform_tokens": sum(int(w.numel()) for w in transform_windows) if TRANSFORM else 0,
                    "rd_tokens": sum(int(w.numel()) for w in rd_windows)},
           "layers": []}
    for li, layer in enumerate(layers):
        rec = {}
        for kind in ("k", "v"):
            b, binv, _ = T[(li, kind)]
            sig2 = st.sq[(li, kind)] / st.n[("rd", li, kind)]
            s = percentile(st.hist[(li, kind)], st.amax[(li, kind)], SCALE_PCT)
            if kind == "k":
                w = st.wq[li] / st.n[("rd", li, kind)]
            else:
                w = value_weight(b, layer.self_attn.o_proj.weight, hq, hkv, d, g)
            rec[kind] = {"B": b.cpu(), "Binv": binv.cpu(), "sig2": sig2.cpu(), "w": w.cpu(),
                         "s": s.cpu()}
        out["layers"].append(rec)
    if path:
        torch.save(out, path)
    return out


def percentile(hist: torch.Tensor, top: torch.Tensor, p: float) -> torch.Tensor:
    """Per channel p-quantile of |h| from (C, bins) counts over [0, top], interpolated
    linearly within the bin."""
    h = hist.double()
    cum = h.cumsum(1)
    tot = cum[:, -1:].clamp_min(1)
    target = p * tot
    i = torch.searchsorted(cum, target).clamp(max=hist.shape[1] - 1)  # first bin reaching it
    prev = torch.where(i > 0, cum.gather(1, (i - 1).clamp_min(0)), torch.zeros_like(target))
    inbin = h.gather(1, i).clamp_min(1)
    frac = ((target - prev) / inbin).clamp(0, 1)
    width = top.double()[:, None] / hist.shape[1]
    return ((i.double() + frac) * width).squeeze(1).clamp_min(1e-12)


def value_weight(b: torch.Tensor, wo: torch.Tensor, hq: int, hkv: int, d: int, g: int):
    """w_c = sum over query heads r reading the group of ||(B^V[:, head(r)] W_O^r^T)_c||^2,
    (G*C,): the output energy of a unit error in value coefficient c."""
    c = hkv * d // g
    hpg = hkv // g
    grp = hq // hkv
    w = torch.zeros(g, c, dtype=torch.float64, device=b.device)
    wo = wo.detach().double().to(b.device)
    for r in range(hq):
        kv = r // grp
        gi, j = divmod(kv, hpg)
        m = b[gi][:, j * d : (j + 1) * d] @ wo[:, r * d : (r + 1) * d].T  # (C, hidden)
        w[gi] += (m**2).sum(1)
    return w.flatten()


# ------------------------------------------------------------------ inference


class Coder:
    """The fitted coder of one model: per layer and kind, B, B^-1, widths and scales."""

    def __init__(self, cal: dict, key_bits: float, val_bits: float, var_only: bool = False):
        m = cal["meta"]
        self.g = m["groups"]
        self.c = m["hkv"] * m["d"] // self.g
        self.layers = cal["layers"]
        self.bits = {}
        for kind, avg, bonus in (("k", key_bits, 1), ("v", val_bits, 0)):
            if avg <= 0:
                continue
            costs = []
            for rec in self.layers:
                r = rec[kind]
                cost = normalised(r["sig2"]) if var_only else normalised(r["w"]) * normalised(r["sig2"])
                costs.append(cost)
            for li, b in enumerate(allocate(costs, avg, bonus)):
                self.bits[(li, kind)] = b

    def code(self, x: torch.Tensor, li: int, kind: str) -> torch.Tensor:
        """x (B, n, G*C) projection outputs of the tokens to code -> reconstruction."""
        r = self.layers[li][kind]
        dev = x.device
        h = torch.einsum("bngi,gij->bngj", x.double().reshape(*x.shape[:2], self.g, self.c),
                         r["Binv"].to(dev))
        bits = self.bits[(li, kind)].view(self.g, self.c)
        s = r["s"].view(self.g, self.c)
        hq = torch.stack([quantize(h[:, :, gi], bits[gi], s[gi]) for gi in range(self.g)], 2)
        y = torch.einsum("bngj,gji->bngi", hq.double(), r["B"].to(dev))
        return y.reshape(x.shape).to(x.dtype)

    def code_bits(self, li: int, kind: str) -> int:
        """Code bits of one token of one layer."""
        return int(self.bits[(li, kind)].sum())


_CODER = None
_HOOKS = []


def install(model, cal: dict, sink: int, hot: int, account, enabled=lambda: True) -> None:
    """Forward hooks on k_proj (and v_proj) coding the settled prefill tokens
    [sink, T - hot) of every layer, before RoPE; ``account(li, n_coded, n_sink, coder)``
    records the stored bits of each layer's keys. ``enabled()`` is read at every call
    (the harness passes its NOQUANT switch, which the KL reference pass turns off)."""
    global _CODER
    _CODER = Coder(cal, KEY_BITS, VAL_BITS if codes_values() else 0, bool(VAR_ONLY))
    remove()

    def hook(li, kind):
        def f(mod, inp, out):
            t = out.shape[1]
            n = t - hot
            if n <= sink or not enabled():
                return out
            y = out.clone()
            y[:, sink:n] = _CODER.code(out[:, sink:n], li, kind)
            if kind == "k":
                account(li, n - sink, sink, _CODER)
            return y
        return f

    for li, layer in enumerate(model.model.layers):
        _HOOKS.append(layer.self_attn.k_proj.register_forward_hook(hook(li, "k")))
        if codes_values():
            _HOOKS.append(layer.self_attn.v_proj.register_forward_hook(hook(li, "v")))


def remove() -> None:
    for h in _HOOKS:
        h.remove()
    _HOOKS.clear()


def account(n_coded: int, n_sink: int, hkv: int, d: int, coder: Coder, li: int) -> dict:
    """Stored bits of one layer's settled keys (the harness's account() fields): the
    codes, one fp16 alpha per token and head group, the sink in fp16. The per-model
    transform, scales and widths are reported separately (model_bits)."""
    elems = (n_coded + n_sink) * hkv * d
    code = n_coded * coder.code_bits(li, "k")
    meta = n_coded * coder.g * 16
    outl = n_sink * hkv * d * 16
    return {"n": n_coded + n_sink, "elems": elems, "code": code, "meta": meta, "outliers": outl,
            "basis": 0, "alloc": 0, "total": code + meta + outl}


def model_bits(cal: dict) -> dict:
    """Bits AATC stores once per model, per kind: B (fp16), scales (fp16), widths (5 bits)."""
    out = {}
    for kind in ("k", "v"):
        tot = 0
        for rec in cal["layers"]:
            r = rec[kind]
            tot += (r["B"].numel() if TRANSFORM else 0) * 16 + r["s"].numel() * (16 + 5)
        out[kind] = tot
    return out


def load_text(path: str) -> list:
    """The staged RD calibration text: jsonl rows {"source", "text"}."""
    return [json.loads(x)["text"] for x in open(path, encoding="utf-8")]


def windows(tok, texts: list, n: int, seq: int = SEQ) -> list:
    """n windows of seq tokens: texts tokenized and packed end to end (EOS between)."""
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    out, buf = [], []
    for t in texts:
        buf += tok(t).input_ids + [eos]
        while len(buf) >= seq and len(out) < n:
            out.append(torch.tensor(buf[:seq])[None])
            buf = buf[seq:]
        if len(out) >= n:
            return out
    raise SystemExit(f"calibration text gave {len(out)} of {n} windows of {seq} tokens")

