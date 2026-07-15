# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Are asym-NF4 key offsets RoPE-frequency-structured?

Hypothesis (turboquant-pro review, tying the v1.4.0 DC-offset finding to the
spectral-wavelength story): post-RoPE keys' *slowest* rotary channels (long
wavelength relative to the context window) are near-constant across
positions, so they behave as per-channel DC components -- exactly what
asym-NF4's per-channel mean (`nf4_mean`) spends its zero-point on. If the
per-channel |offset| correlates with rotary wavelength, the zero-point could
be derived deterministically from the model config (no per-channel
calibration metadata), and the Qwen GQA-collapse gets a mechanism
(key-offset statistics, amplified by GQA). If not, the two effects are
cleanly separated. Either way informative.

Protocol: for each model, run W wikitext-2-raw test passages of SEQ tokens,
capture the post-RoPE key cache per layer (transformers stores keys after
rotary embedding for the Llama/Qwen/Mistral families), and accumulate per
(layer, kv_head, channel): mean over tokens (the asym-NF4 offset mu_c) and
mean |k| (scale context). Rotary map (rotate_half convention): channel c has
inverse frequency inv_freq[c mod d/2], inv_freq[i] = theta^(-2i/d),
wavelength 2*pi/inv_freq. Statistics per model: Spearman(|mu_c|, wavelength)
pooled and median per layer; fraction of total |mu| mass in channels whose
wavelength exceeds the window (the "DC-in-window" channels); binned
|mu|-vs-wavelength curve for the plot.

Run on a GPU host:  python benchmarks/rope_offset_frequency.py
Writes item4_result.json and item4_offsets.png to OUT_DIR (default ~/item4).
Results in RESULTS_rope_offsets.md were produced on 2x-GV100 'atlas'
(fp16, one model at a time on a single 32 GB GPU).
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
import torch

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-v0.1",  # skipped gracefully if gated/absent
]
SEQ, N_PASSAGES = 512, 8
OUT_DIR = os.path.expanduser("~/item4")
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 is set by the launcher


def get_layer_keys(cache, layer: int) -> torch.Tensor:
    """Post-RoPE keys (b, n_kv, seq, hd) from a transformers Cache, any version."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer]
    if hasattr(cache, "layers"):
        lay = cache.layers[layer]
        for attr in ("keys", "key_cache", "k_cache"):
            if hasattr(lay, attr):
                return getattr(lay, attr)
    if isinstance(cache, (tuple, list)):
        return cache[layer][0]
    raise TypeError(f"unknown cache type {type(cache)}")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wikitext_passages(tokenizer, n: int, seq: int):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return [ids[i * seq : (i + 1) * seq].unsqueeze(0) for i in range(n)]


def analyze_model(name: str) -> dict | None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map=DEVICE
        )
    except Exception as e:  # gated / missing / OOM: skip, don't fail the run
        print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
        return None
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    def _theta(c) -> float:
        v = getattr(c, "rope_theta", None)
        if v:
            return float(v)
        for attr in ("rope_parameters", "rope_scaling"):
            d = getattr(c, attr, None)
            if isinstance(d, dict) and d.get("rope_theta"):
                return float(d["rope_theta"])
        return 10000.0

    theta = _theta(cfg)

    sum_k = torch.zeros(n_layers, n_kv, hd, dtype=torch.float64)
    sum_abs = torch.zeros_like(sum_k)
    n_tok = 0
    for ids in wikitext_passages(tok, N_PASSAGES, SEQ):
        with torch.no_grad():
            out = model(ids.to(DEVICE), use_cache=True)
        for layer in range(n_layers):
            k = get_layer_keys(out.past_key_values, layer)[0].to(torch.float64)
            sum_k[layer] += k.sum(dim=1).cpu()  # (n_kv, hd)
            sum_abs[layer] += k.abs().sum(dim=1).cpu()
        n_tok += ids.shape[1]
        del out
        torch.cuda.empty_cache()

    mu = (sum_k / n_tok).numpy()  # per-channel offset, (L, n_kv, hd)
    mabs = (sum_abs / n_tok).numpy()

    half = hd // 2
    inv_freq = theta ** (-2.0 * np.arange(half) / hd)
    wavelength = 2.0 * np.pi / inv_freq
    wl_of_channel = np.tile(wavelength, 2)[:hd]  # rotate_half: c mod half

    abs_mu = np.abs(mu)  # (L, n_kv, hd)
    pooled_sp = spearman(abs_mu.reshape(-1, hd).mean(axis=0), wl_of_channel)
    per_layer_sp = [
        spearman(abs_mu[layer].mean(axis=0), wl_of_channel) for layer in range(n_layers)
    ]
    dc_mask = wl_of_channel > SEQ  # wavelength exceeds the window
    mass = abs_mu.mean(axis=(0, 1))
    dc_fraction = float(mass[dc_mask].sum() / mass.sum())

    # binned curve for the plot: mean |mu| per frequency pair
    pair_mu = 0.5 * (mass[:half] + mass[half : 2 * half])
    result = dict(
        model=name,
        n_layers=n_layers,
        n_kv_heads=n_kv,
        head_dim=hd,
        rope_theta=theta,
        seq=SEQ,
        n_tokens=n_tok * 1,
        spearman_absmu_vs_wavelength_pooled=round(pooled_sp, 3),
        spearman_per_layer_median=round(float(np.nanmedian(per_layer_sp)), 3),
        spearman_per_layer_q25=round(float(np.nanpercentile(per_layer_sp, 25)), 3),
        spearman_per_layer_q75=round(float(np.nanpercentile(per_layer_sp, 75)), 3),
        dc_channel_fraction_of_offset_mass=round(dc_fraction, 3),
        dc_channel_fraction_of_channels=round(float(dc_mask.mean()), 3),
        mean_abs_offset=round(float(abs_mu.mean()), 4),
        mean_abs_key=round(float(mabs.mean()), 4),
        wavelength_pairs=[round(float(w), 1) for w in wavelength],
        offset_by_pair=[round(float(x), 4) for x in pair_mu],
    )
    print(
        f"{name}: pooled spearman={pooled_sp:.3f} "
        f"layer-median={result['spearman_per_layer_median']:.3f} "
        f"DC-mass={dc_fraction:.1%} (DC channels: {dc_mask.mean():.1%})",
        flush=True,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def make_figure(results: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.6))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        wl = np.array(r["wavelength_pairs"])
        y = np.array(r["offset_by_pair"])
        ax.plot(
            wl,
            y,
            "o",
            ms=3.5,
            color="#2171b5",
            markeredgecolor="#1f3a5f",
            markeredgewidth=0.3,
        )
        ax.axvline(r["seq"], color="#bbbbbb", lw=1.0, ls="--")
        ax.annotate(
            "window",
            (r["seq"] * 1.2, ax.get_ylim()[1] * 0.05),
            fontsize=7,
            color="#888888",
        )
        ax.set_xscale("log")
        ax.set_xlabel("rotary wavelength (tokens)", fontsize=8)
        ax.set_title(
            f"{r['model'].split('/')[-1]}\n"
            f"spearman={r['spearman_absmu_vs_wavelength_pooled']:.2f}, "
            f"DC mass={r['dc_channel_fraction_of_offset_mass']:.0%}",
            fontsize=8.5,
            color="#333333",
        )
        ax.grid(color="#eeeeee", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#dddddd")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("mean |per-channel key offset|", fontsize=8)
    fig.suptitle(
        "asym-NF4 key offsets vs RoPE wavelength (post-RoPE keys)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "item4_offsets.png"), dpi=170, bbox_inches="tight"
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for name in MODELS:
        r = analyze_model(name)
        if r is not None:
            results.append(r)
            with open(os.path.join(OUT_DIR, "item4_result.json"), "w") as f:
                json.dump(results, f, indent=1)
    make_figure(results)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
