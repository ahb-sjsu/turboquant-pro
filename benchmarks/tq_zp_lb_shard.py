#!/usr/bin/env python3
"""Zero-point-variant LongBench runner (adapted from tq_paper_lb_shard.py).

Adds two CODEBOOK values to the published harness, changing ONLY the
zero-point of asym-NF4 keys:
  nf4a_bias   -- mu from the RoPE-averaged k_proj bias over each token
                 group's true positions: weights + config only, ZERO
                 calibration (falls back to nf4a if the model has no bias)
  nf4a_sparse -- per-group calibrated mean masked to the channels whose
                 rotary wavelength exceeds the settled length (config-
                 identified DC channels; ~1/3 less zero-point metadata)
Everything else (grouping, outliers, sink, hot window, values) is the
published NF4A configuration, for direct comparability.

Original docstring follows.

Paper experiment runner: KV-key quantization quality across models/tasks/methods.

Unlike ``tq_enh_lb_shard.py`` (a faithful but slow simulation that re-quantizes the
settled window every decode step), this implements the **deployable** method:
**quantize the prefill KV once** and freeze it.

  * On the first cache update of each layer (= the prefill, the long prompt), the
    settled region ``[0 : T-HOT]`` is quantized one time: group-wise per-channel keys
    (NF4 or uniform) with the first ``SINK`` tokens kept fp16 and the top-``OUT_FRAC``
    magnitude entries **per channel** (global over the prefill) kept fp16.
  * The most recent ``HOT`` prefill tokens and **all generated tokens** stay fp16.

For LongBench (max_new_tokens <= HOT) generated tokens never leave the hot window, so
the cache quantizes exactly once per layer -> ~fp16 speed, and quality matches the
slow per-step simulation (verified on Llama-2-7B). Values use per-token uniform.

Config via env; sharded by SHARD_ID / NUM_SHARDS. Model via MODEL (HF id) and
MODEL_KEY (LongBench config key, e.g. ``llama2-7b-chat-4k``) for max-length/prompts.
"""

import json
import math
import os

import torch

# Volta (GV100) host: the SDPA math kernel materializes the full attention
# matrix (8 GB at ~14k tokens) and OOMs on long qasper samples; the
# memory-efficient kernel handles 16k tokens in <0.5 GB. Force it.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# The mem-efficient kernel rejects GQA-packed calls (enable_gqa=True) on this
# torch build, and with math disabled that raises "Invalid backend". Expand KV
# heads before dispatch (0.4 GB at 12k tokens vs 8 GB for the math kernel).
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa_expand_gqa(q, k, v, *args, **kwargs):
    kwargs.pop("enable_gqa", None)
    if k.dim() == 4 and q.dim() == 4 and k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return _orig_sdpa(q, k, v, *args, **kwargs)


torch.nn.functional.scaled_dot_product_attention = _sdpa_expand_gqa

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = os.environ.get("MODEL", "NousResearch/Llama-2-7b-chat-hf")
MODEL_KEY = os.environ.get("MODEL_KEY", "llama2-7b-chat-4k")
LBROOT = os.environ.get("LBROOT", "/root/LongBench/LongBench")
DATADIR = os.environ.get("DATADIR", "/root/lb_data/data")
DATASETS = os.environ.get("DATASETS", "trec,triviaqa,qasper").split(",")

KB = int(os.environ.get("KEY_BITS", "4"))
VB = int(os.environ.get("VAL_BITS", "4"))
G = int(os.environ.get("GROUP", "32"))
HOT = int(os.environ.get("HOT", "128"))  # fp16 recent window (== RESID)
SINK = int(os.environ.get("SINK", "0"))
OUT_FRAC = float(os.environ.get("OUTLIER_FRAC", "0.0"))
NUQ = int(os.environ.get("NUQ", "0"))
NOQUANT = int(os.environ.get("NOQUANT", "0"))
# Tier-1 levers:
#  CODEBOOK: uniform | nf4 | quantile | kmeans  (key codebook)
#    quantile/kmeans = ONLINE per-channel codebook fit to THIS prefill (calibration-free,
#    data-optimal -- the both-worlds answer to KVQuant's offline Fisher K-means).
#  PREROPE: quantize keys in pre-RoPE space (RoPE smears per-channel structure).
CODEBOOK = os.environ.get("CODEBOOK", "nf4" if NUQ else "uniform")
PREROPE = int(os.environ.get("PREROPE", "0"))
KMEANS_ITERS = int(os.environ.get("KMEANS_ITERS", "8"))
TAG = os.environ.get("TAG", "v0")
SHARD = int(os.environ["SHARD_ID"])
NSH = int(os.environ["NUM_SHARDS"])
CHAT = int(os.environ.get("CHAT", "1"))  # wrap prompt in the model chat template
_MAXGEN = int(
    os.environ.get("MAXGEN", "0")
)  # >0 overrides per-task max_new_tokens (ablation)
OUT = f"{os.environ.get('OUTROOT', '/root')}/out_{TAG}"
os.makedirs(OUT, exist_ok=True)

NF4 = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)



# ---- zero-point variant machinery (weights + config only) -------------------
ZP: dict = {}


def _zp_theta(cfg) -> float:
    v = getattr(cfg, "rope_theta", None)
    if v:
        return float(v)
    for attr in ("rope_parameters", "rope_scaling"):
        d = getattr(cfg, attr, None)
        if isinstance(d, dict) and d.get("rope_theta"):
            return float(d["rope_theta"])
    return 10000.0


def zp_init(model):
    """Per-layer k_proj bias + cos/sin cumsum tables (cum[i] = sum over p < i)."""
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    half = hd // 2
    theta = _zp_theta(cfg)
    inv = theta ** (-2.0 * torch.arange(half, dtype=torch.float64) / hd)
    P = MAXLEN + 8192
    ang = torch.arange(P, dtype=torch.float64)[:, None] * inv[None, :]
    zc = torch.zeros(1, half, dtype=torch.float64)
    ZP["cum_cos"] = torch.cat([zc, torch.cos(ang).cumsum(0)]).float()
    ZP["cum_sin"] = torch.cat([zc, torch.sin(ang).cumsum(0)]).float()
    ZP["half"], ZP["hd"] = half, hd
    ZP["wl"] = (2.0 * math.pi / inv).float()
    n_kv = cfg.num_key_value_heads
    bs = []
    for lay in model.model.layers:
        b = lay.self_attn.k_proj.bias
        bs.append(None if b is None else b.detach().float().cpu().view(n_kv, hd))
    ZP["bias"] = bs
    print(
        f"[zp] theta={theta:g} hd={hd} bias={'yes' if bs[0] is not None else 'NO'}",
        flush=True,
    )


def _group_pos_means(n, g):
    """Mean cos/sin per token group over the group's true positions [start, end)."""
    Tg = (n + g - 1) // g
    starts = torch.arange(Tg, dtype=torch.long) * g
    ends = torch.clamp(starts + g, max=n)
    cnt = (ends - starts).float().unsqueeze(1)
    mc = (ZP["cum_cos"][ends] - ZP["cum_cos"][starts]) / cnt
    ms = (ZP["cum_sin"][ends] - ZP["cum_sin"][starts]) / cnt
    return mc, ms  # (Tg, half)


def _quant_nf4a_bias_group(x, g, nf4, li):
    """asym-NF4 with mu = RoPE-averaged k_proj bias (zero calibration)."""
    b = ZP["bias"][li]
    if b is None:  # bias-free model: identical to calibrated nf4a
        return _quant_nf4a_group(x, g, nf4)
    B, H, n, D = x.shape
    half = ZP["half"]
    mc, ms = _group_pos_means(n, g)
    c_full = torch.cat([mc, mc], dim=1)  # (Tg, D), rotate_half layout
    s_full = torch.cat([ms, ms], dim=1)
    rh = torch.cat([-b[:, half:], b[:, :half]], dim=1)  # rotate_half(b), (H, D)
    mu = b[None, :, :] * c_full[:, None, :] + rh[None, :, :] * s_full[:, None, :]
    mu = mu.permute(1, 0, 2)[None, :, :, None, :].to(x.device, x.dtype)
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    xc = xg - mu
    amax = xc.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xc / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax + mu
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _quant_nf4a_sparse_group(x, g, nf4):
    """asym-NF4 with calibrated mu only on config-identified DC channels."""
    B, H, n, D = x.shape
    wl = torch.cat([ZP["wl"], ZP["wl"]])[:D]
    mask = (wl > float(n)).view(1, 1, 1, 1, D).to(x.device, x.dtype)
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mu = xg.mean(3, keepdim=True) * mask
    xc = xg - mu
    amax = xc.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xc / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax + mu
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


# LongBench model max-context (truncation) and the chat-wrap style per model family.
MODEL_MAXLEN = {
    "llama2-7b-chat-4k": 3500,
    "llama2-13b-chat-4k": 3500,
    "mistral-7b-instruct": 31500,
    "qwen2.5-7b-instruct": 31500,
}
MAXLEN = int(os.environ.get("MAXLEN", str(MODEL_MAXLEN.get(MODEL_KEY, 3500))))


def _group_pad(x, g):
    n = x.shape[2]
    pad = (g - n % g) % g
    if pad:
        x = torch.cat([x, x[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
    return x, n


def _quant_uniform_group(x, bits, g):
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mn = xg.amin(3, keepdim=True)
    mx = xg.amax(3, keepdim=True)
    qm = 2**bits - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    xq = ((xg - mn) / sc).round().clamp(0, qm) * sc + mn
    return xq.reshape(B, H, Tg * g, D)[:, :, :n0, :]


def _quant_nf4_group(x, g, nf4):
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    amax = xg.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xg / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _quant_nf4a_group(x, g, nf4):
    """Asymmetric / zero-point NF4: subtract the per-group per-channel MEAN (the DC offset
    that wastes symmetric NF4's codes), NF4-quantize the centered residual, add the mean
    back. Keeps NF4's nonlinear level placement AND handles offset KV distributions -> aims
    to be robust across MHA (Llama) and high-GQA (Qwen) models with one codebook.
    Metadata: per-group mean + absmax (2 fp16 scalars/group/channel vs 1 for symmetric).
    """
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mu = xg.mean(3, keepdim=True)
    xc = xg - mu
    amax = xc.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xc / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax + mu
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _kmeans_1d(x, k, iters):
    """Per-row 1-D k-means (Lloyd). x: (R, n) -> centroids (R, k). Init = quantiles."""
    qs = torch.linspace(0.0, 1.0, k, device=x.device, dtype=x.dtype)
    cent = torch.quantile(x, qs, dim=1).T.contiguous()  # (R, k)
    for _ in range(iters):
        idx = (x.unsqueeze(-1) - cent.unsqueeze(1)).abs().argmin(-1)  # (R, n)
        oh = torch.nn.functional.one_hot(idx, k).to(x.dtype)  # (R, n, k)
        cnt = oh.sum(1)  # (R, k)
        s = torch.einsum("rn,rnk->rk", x, oh)  # (R, k)
        new = torch.where(cnt > 0, s / cnt.clamp_min(1), cent)
        cent = new
    return cent


def _quant_perchannel_codebook(x, bits, mode):
    """Online per-channel non-uniform codebook fit to this block (quantile or kmeans).

    Fit on a token subsample (codebook is robust to it); assign via searchsorted on the
    sorted codebook -> O(n log nlev), not O(n*nlev).
    """
    B, H, n, D = x.shape
    nlev = 2**bits
    xt = x.permute(0, 1, 3, 2).reshape(-1, n).float()  # (B*H*D, n) per-channel rows
    sub = xt[:, :: max(1, n // 1024)][:, :1024] if n > 1536 else xt
    if mode == "kmeans":
        cent = _kmeans_1d(sub, nlev, KMEANS_ITERS)  # (R, nlev)
    else:  # quantile
        qs = torch.linspace(0.0, 1.0, nlev, device=x.device, dtype=torch.float32)
        cent = torch.quantile(sub, qs, dim=1).T  # (R, nlev)
    cent = cent.sort(dim=1).values.contiguous()
    pos = torch.searchsorted(cent, xt.contiguous()).clamp(1, nlev - 1)
    left = torch.gather(cent, 1, pos - 1)
    right = torch.gather(cent, 1, pos)
    idx = torch.where((xt - left).abs() <= (right - xt).abs(), pos - 1, pos)
    deq = torch.gather(cent, 1, idx)  # (R, n)
    return deq.reshape(B, H, D, n).permute(0, 1, 3, 2).to(x.dtype)


def qdq_key_block(x, li=None):
    """Quantize the full settled key block once (global per-channel outliers + sink)."""
    B, H, n, D = x.shape
    if CODEBOOK == "nf4a_bias" and KB == 4:
        hq = _quant_nf4a_bias_group(x, G, NF4, li)
    elif CODEBOOK == "nf4a_sparse" and KB == 4:
        hq = _quant_nf4a_sparse_group(x, G, NF4)
    elif CODEBOOK == "nf4" and KB == 4:
        hq = _quant_nf4_group(x, G, NF4)
    elif CODEBOOK == "nf4a" and KB == 4:
        hq = _quant_nf4a_group(x, G, NF4)
    elif CODEBOOK in ("quantile", "kmeans"):
        hq = _quant_perchannel_codebook(x, KB, CODEBOOK)
    else:
        hq = _quant_uniform_group(x, KB, G)
    keep = torch.zeros(B, H, n, D, dtype=torch.bool, device=x.device)
    if SINK > 0:
        keep[:, :, : min(SINK, n), :] = True
    if OUT_FRAC > 0:
        k = max(1, int(round(n * OUT_FRAC)))
        absn = x.abs()
        thr = absn.kthvalue(n - k + 1, dim=2, keepdim=True).values
        keep |= absn >= thr
    return torch.where(keep, x, hq).to(x.dtype)


def qdq_val_block(x):
    mn = x.amin(3, keepdim=True)
    mx = x.amax(3, keepdim=True)
    qm = 2**VB - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    return (((x - mn) / sc).round().clamp(0, qm) * sc + mn).to(x.dtype)


_orig_update = DynamicCache.update


def _patched_update(self, k, v, li, cache_kwargs=None):
    fk, fv = _orig_update(self, k, v, li, cache_kwargs)
    if NOQUANT:
        return fk, fv
    # Quantize the settled prefill region ONCE, IN PLACE in the stored cache tensor.
    # `fk` is `self.key_cache[li]`; later decode steps cat new fp16 tokens onto it, so
    # the quantized settled region persists with zero per-step overhead (~fp16 speed).
    if not hasattr(self, "_qdone"):
        self._qdone = set()
    if li not in self._qdone:
        T = fk.shape[2]
        n = max(0, T - HOT)
        if n > 0:
            if (
                not PREROPE
            ):  # post-RoPE keys quantized here; pre-RoPE done in the rope hook
                fk[:, :, :n, :] = qdq_key_block(fk[:, :, :n, :], li)
            fv[:, :, :n, :] = qdq_val_block(fv[:, :, :n, :])
        self._qdone.add(li)
    return fk, fv


DynamicCache.update = _patched_update


def _make_prerope_hook(orig):
    def _hook(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if not NOQUANT:
            T = k.shape[2]
            if T > HOT:  # prefill: quantize settled keys once, pre-RoPE
                n = T - HOT
                k = k.clone()
                k[:, :, :n, :] = qdq_key_block(k[:, :, :n, :])
        return orig(q, k, cos, sin, position_ids, unsqueeze_dim)

    return _hook


def _install_prerope():
    """Quantize keys in **pre-RoPE** space: hook ``apply_rotary_pos_emb`` so the settled
    prefill keys are quantized before the position rotation is applied (RoPE smears the
    per-channel structure that per-channel quantization relies on).

    HF defines ``apply_rotary_pos_emb`` per model family and the attention forward looks
    it up from the module globals at call time, so monkeypatching the module attribute is
    sufficient. Patch every family we benchmark that is importable in this transformers.
    """
    mods = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
    ]
    patched = []
    import importlib

    for mp in mods:
        try:
            m = importlib.import_module(mp)
        except Exception:
            continue
        if hasattr(m, "apply_rotary_pos_emb"):
            m.apply_rotary_pos_emb = _make_prerope_hook(m.apply_rotary_pos_emb)
            patched.append(mp.split(".")[-2])
    print(f"[prerope] patched: {patched}", flush=True)


if PREROPE:
    _install_prerope()


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def build_chat(tok, prompt):
    if not CHAT:
        return prompt
    if "llama2" in MODEL_KEY:
        return f"[INST]{prompt}[/INST]"
    try:  # Mistral / Qwen / others: use the tokenizer chat template
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def main():
    print(
        f"[shard {SHARD}/{NSH}] TAG={TAG} MODEL={MODEL_KEY} KB={KB} HOT={HOT} "
        f"SINK={SINK} OUT={OUT_FRAC} NUQ={NUQ} NOQUANT={NOQUANT}",
        flush=True,
    )
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config.use_cache = True
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            config=config,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .cuda()
        .eval()
    )
    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL, trust_remote_code=True, use_fast=True
        )
    except (
        Exception
    ) as e:  # some fast tokenizers need a newer tokenizers lib; fall back
        print(
            f"[tok] fast load failed ({repr(e)[:80]}); using slow tokenizer", flush=True
        )
        tok = AutoTokenizer.from_pretrained(
            MODEL, trust_remote_code=True, use_fast=False
        )
    if CODEBOOK.startswith("nf4a_"):
        zp_init(model)
    device = torch.device("cuda:0")
    d2p = json.load(open(f"{LBROOT}/config/dataset2prompt.json"))
    d2m = json.load(open(f"{LBROOT}/config/dataset2maxlen.json"))
    no_chat = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    for dataset in DATASETS:
        data = load_jsonl(f"{DATADIR}/{dataset}.jsonl")
        pf = d2p[dataset]
        mg = int(d2m[dataset])
        fo = open(f"{OUT}/{dataset}.{SHARD}.jsonl", "w")
        for gi, o in enumerate(data):
            if gi % NSH != SHARD:
                continue
            if _MAXGEN > 0:
                mg = (
                    _MAXGEN  # override the LongBench per-task max_new_tokens (ablation)
                )
            prompt = pf.format(**o)
            tp = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tp) > MAXLEN:
                h = MAXLEN // 2
                prompt = tok.decode(tp[:h], skip_special_tokens=True) + tok.decode(
                    tp[-h:], skip_special_tokens=True
                )
            if dataset not in no_chat:
                prompt = build_chat(tok, prompt)
            inp = tok(prompt, truncation=False, return_tensors="pt").to(device)
            cl = inp.input_ids.shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=mg,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.eos_token_id,
                )[0]
            pred = tok.decode(out[cl:], skip_special_tokens=True)
            fo.write(
                json.dumps(
                    {
                        "idx": gi,
                        "pred": pred,
                        "answers": o["answers"],
                        "all_classes": o["all_classes"],
                    }
                )
                + "\n"
            )
            fo.flush()
        fo.close()
        print(f"[shard {SHARD}] {dataset} done", flush=True)
    print(f"SHARD_{SHARD}_DONE", flush=True)


if __name__ == "__main__":
    main()
