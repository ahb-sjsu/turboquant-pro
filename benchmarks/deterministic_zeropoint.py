# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Deterministic zero-points for asym-NF4 keys: how little calibration works?

Follow-up to RESULTS_rope_offsets.md (the key DC offset is RoPE-frequency-
structured, 96-99% of its mass in wavelength>window channels). Five key
quantizers at 4-bit NF4, identical except for the zero-point mu, evaluated
by WikiText-2 perplexity (the metric that catches key failures; values stay
fp16 to isolate the key effect):

  sym        mu = 0                       (symmetric NF4: collapse reference)
  calib      mu = per-window channel mean (the shipped asym-NF4)
  sparse     mu = window mean ONLY on DC channels (wavelength > window,
               identified from the config alone) -- tests "config decides
               WHERE, calibration decides the value" (~33% less metadata
               at theta=1e6)
  offline    mu = static per-channel means calibrated ONCE on a held-out
               wikitext TRAIN sample -- no per-window recompute
  bias       mu = k_proj.bias pushed through the position-averaged rotation
               (RoPE-averaged bias) -- fully deterministic from weights +
               config, ZERO calibration data

Also reports Spearman(measured window mean, RoPE-averaged bias) per model:
does the bias explain the offset?

Run on a GPU host:  python benchmarks/deterministic_zeropoint.py
Results in RESULTS_rope_offsets.md were produced on the 2x-GV100 host.
Models: Qwen2.5-1.5B/7B-Instruct (Qwen has k_proj bias; Mistral does not,
so the bias variant is Qwen-specific by construction).
"""

from __future__ import annotations

import gc
import json
import math
import os

import numpy as np
import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

# QLoRA NF4 codebook (inlined from turboquant_pro.per_channel_kv._NF4 so the
# script is self-contained on hosts with older library versions)
_NF4 = [
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

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
SEQ, NCH, NCAL = 512, 3, 2
OUT = os.environ.get("ZP_OUT", "item4b_result.json")
dev = "cuda"

NF4_LEVELS = torch.tensor(_NF4, dtype=torch.float32)


def theta_of(cfg) -> float:
    v = getattr(cfg, "rope_theta", None)
    if v:
        return float(v)
    for attr in ("rope_parameters", "rope_scaling"):
        d = getattr(cfg, attr, None)
        if isinstance(d, dict) and d.get("rope_theta"):
            return float(d["rope_theta"])
    return 10000.0


def spearman(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wikitext(split: str, n: int, tok):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    txt = "\n\n".join(t for t in ds["text"] if len(t) > 160)
    ids = tok(txt, return_tensors="pt").input_ids[0]
    return [ids[i * SEQ : (i + 1) * SEQ] for i in range(n)]


def nf4_quant(k: torch.Tensor, mu) -> torch.Tensor:
    """4-bit NF4 per channel (reduce over tokens, dim=2) with zero-point mu."""
    r = k.float() - mu
    amax = r.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    x = r / amax
    lv = NF4_LEVELS.to(k.device)
    idx = (x.unsqueeze(-1) - lv).abs().argmin(dim=-1)
    return (mu + amax * lv[idx]).to(k.dtype)


def two_tier(out, orig, s=4, h=64):
    h = min(h, orig.shape[2] // 4)
    out = out.clone()
    out[:, :, :s, :] = orig[:, :, :s, :]
    out[:, :, orig.shape[2] - h :, :] = orig[:, :, orig.shape[2] - h :, :]
    return out


STATE: dict = {}
_orig_rope = qwen2.apply_rotary_pos_emb


def patched_rope(q, k, cos, sin, unsqueeze_dim=1):
    q2, k2 = _orig_rope(q, k, cos, sin, unsqueeze_dim)
    layer = STATE["call"] % STATE["n_layers"]
    STATE["call"] += 1
    if STATE.get("capture") is not None:
        STATE["capture"][layer] += k2[0].float().mean(dim=1).cpu()  # (n_kv, hd)
        STATE["ncap"][layer] += 1
        return q2, k2
    fn = STATE.get("keyfn")
    if fn is not None:
        kq = two_tier(fn(k2, layer), k2)
        STATE["errs"].append(((kq - k2).norm() / k2.norm().clamp_min(1e-8)).item())
        return q2, kq
    return q2, k2


qwen2.apply_rotary_pos_emb = patched_rope


@torch.no_grad()
def ppl_run(model, chunks, label, keyfn=None):
    STATE.update(keyfn=keyfn, errs=[], call=0, capture=None)
    losses = [
        model(
            input_ids=c.unsqueeze(0).to(dev), labels=c.unsqueeze(0).to(dev)
        ).loss.item()
        for c in chunks
    ]
    ppl = math.exp(sum(losses) / len(losses))
    e = STATE["errs"]
    rm = sum(e) / len(e) if e else 0.0
    print(f"  {label:34s} ppl={ppl:10.3f}  key_recon mean={rm:.3f}", flush=True)
    return ppl, rm


def rope_averaged_bias(model, n_layers, n_kv, hd, theta):
    """mu_hat[l,h,c] = mean over positions of RoPE(k_proj.bias) -- config+weights only."""
    half = hd // 2
    inv_freq = theta ** (-2.0 * torch.arange(half, dtype=torch.float64) / hd)
    pos = torch.arange(SEQ, dtype=torch.float64)
    ang = pos[:, None] * inv_freq[None, :]  # (SEQ, half)
    c_mean = torch.cos(ang).mean(dim=0)  # (half,)
    s_mean = torch.sin(ang).mean(dim=0)
    c_full = torch.cat([c_mean, c_mean]).float()  # rotate_half layout
    s_full = torch.cat([s_mean, s_mean]).float()
    out = torch.zeros(n_layers, n_kv, hd)
    layers = model.model.layers
    for li, lay in enumerate(layers):
        b = lay.self_attn.k_proj.bias
        if b is None:
            return None
        b = b.detach().float().cpu().view(n_kv, hd)
        rh = torch.cat([-b[:, hd // 2 :], b[:, : hd // 2]], dim=1)  # rotate_half(b)
        out[li] = b * c_full[None, :] + rh * s_full[None, :]
    return out  # (L, n_kv, hd)


def analyze(name: str) -> dict:
    tok = AutoTokenizer.from_pretrained(name)
    model = (
        AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
        .to(dev)
        .eval()
    )
    cfg = model.config
    n_layers, n_kv = cfg.num_hidden_layers, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    theta = theta_of(cfg)
    half = hd // 2
    wl = 2 * math.pi / (theta ** (-2.0 * np.arange(half) / hd))
    dc_mask = torch.from_numpy(np.tile(wl, 2)[:hd] > SEQ).view(1, 1, 1, hd).to(dev)
    print(
        f"{name}: layers={n_layers} kv={n_kv} hd={hd} theta={theta:g} "
        f"DC channels={float(dc_mask.float().mean()):.1%}",
        flush=True,
    )

    test = wikitext("test", NCH, tok)
    train = wikitext("train", NCAL, tok)

    # Offline calibration pass (train split): per-layer per-channel key means.
    STATE.update(
        keyfn=None,
        call=0,
        n_layers=n_layers,
        capture=[torch.zeros(n_kv, hd) for _ in range(n_layers)],
        ncap=[0] * n_layers,
    )
    with torch.no_grad():
        for c in train:
            model(input_ids=c.unsqueeze(0).to(dev))
    offline = torch.stack(
        [s / max(n, 1) for s, n in zip(STATE["capture"], STATE["ncap"])]
    )  # (L, n_kv, hd)
    STATE["capture"] = None

    bias_mu = rope_averaged_bias(model, n_layers, n_kv, hd, theta)
    sp_bias = (
        spearman(offline.numpy(), bias_mu.numpy()) if bias_mu is not None else None
    )
    off_dev = offline.to(dev)
    bias_dev = bias_mu.to(dev) if bias_mu is not None else None

    variants = {
        "fp16 (no key quant)": None,
        "sym-NF4 K4": lambda k, layer: nf4_quant(k, 0.0),
        "calib asym-NF4 K4 (shipped)": lambda k, layer: nf4_quant(
            k, k.float().mean(dim=2, keepdim=True)
        ),
        "sparse asym (DC channels only)": lambda k, layer: nf4_quant(
            k, k.float().mean(dim=2, keepdim=True) * dc_mask
        ),
        "offline static mu (train-calib)": lambda k, layer: nf4_quant(
            k, off_dev[layer].view(1, n_kv, 1, hd)
        ),
    }
    if bias_dev is not None:
        variants["bias-derived mu (zero calib)"] = lambda k, layer: nf4_quant(
            k, bias_dev[layer].view(1, n_kv, 1, hd)
        )

    rows = {}
    for label, fn in variants.items():
        ppl, rm = ppl_run(model, test, label, fn)
        rows[label] = dict(ppl=round(ppl, 3), key_recon=round(rm, 4))

    print(f"  Spearman(offline mean, RoPE-averaged bias) = {sp_bias}", flush=True)
    res = dict(
        model=name,
        theta=theta,
        dc_channel_fraction=round(float(dc_mask.float().mean().item()), 3),
        spearman_offline_mu_vs_rope_avg_bias=(
            round(sp_bias, 3) if sp_bias is not None else None
        ),
        rows=rows,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    results = []
    for name in MODELS:
        results.append(analyze(name))
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)
    print("ITEM4B_DONE", flush=True)


if __name__ == "__main__":
    main()
