# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""K-row damage vs rotary wavelength (the section-2.3 pre-registered test).

The matched-bit experiment found that Llama-3.2-3B's W^K alone becomes the most
damaging projection at 3-bit (out_kl 0.741 vs V 0.125) with ordinary weight
drift. Hypothesis: W^K rows carrying the low-frequency (long-wavelength)
RoPE channels are the fragile ones -- the weight-space edition of the
RESULTS_rope_offsets.md activation finding.

Design: rows of W^K map to key channels; channel c pairs with rotary frequency
index f = (c mod head_dim) mod (head_dim/2), wavelength lambda_f = 2*pi /
inv_freq[f] (inv_freq read from the model's own rotary embedding buffer, so
rope_scaling is included). Split the head_dim/2 frequency indices into 8
contiguous octiles (bucket 0 = shortest wavelengths ... bucket 7 = longest).
For each bucket, at NB_BITS bits:

  * ONLY   -- quantize only that bucket's rows in every layer's k_proj;
              measure mean KL(p_base || p_q) and top-1 flip rate.
  * SPARED -- quantize ALL k_proj rows except that bucket's; same readout.

Plus full-K quantization (reproduces the matched-bit number) and per-row
weight-level stats (relative Frobenius error, kurtosis, absmax/std) against
wavelength -- Spearman computed scipy-free.

Prediction (pre-registered in docs/notes/projection_sensitivity_deconfounded.md
section 2.3): ONLY-damage rises with wavelength on Llama-3.2-3B (rope-scaling
factor 32) and is comparatively flat on the Mistral-7B-v0.1 control (theta=1e4,
no scaling, no K anomaly at 3-bit).

Usage:
    NB_MODEL=unsloth/Llama-3.2-3B NB_BITS=3 NB_SEQ=256 NB_SAMPLES=32 \
        python k_wavelength_probe.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch


def quantize_per_outchannel(W: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    codes = torch.clamp(torch.round(W / scale), -qmax - 1, qmax)
    return codes * scale


def _load_texts(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    out, buf = [], ""
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        buf = (buf + " " + t).strip()
        if len(buf) > 1200:
            out.append(buf)
            buf = ""
        if len(out) >= n:
            break
    return out


def output_logprobs(model, batch, device) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for ids in batch:
            ids = ids.to(device)
            logits = model(ids.unsqueeze(0)).logits[0]
            outs.append(torch.log_softmax(logits.float(), dim=-1))
    return torch.cat(outs, dim=0)


def kl_top1(base_lp, var_lp):
    with torch.no_grad():
        p = base_lp.exp()
        kl = (p * (base_lp - var_lp)).sum(dim=-1)
        flip = (base_lp.argmax(-1) != var_lp.argmax(-1)).float().mean()
    return float(kl.mean().item()), float(flip.item())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / max(d, 1e-30))


def _kurtosis_rows(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-30
    return ((x - m) / s).astype(np.float64).__pow__(4).mean(axis=1) - 3.0


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "unsloth/Llama-3.2-3B")
    bits = int(os.environ.get("NB_BITS", "3"))
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    n_samples = int(os.environ.get("NB_SAMPLES", "32"))
    n_buckets = int(os.environ.get("NB_BUCKETS", "8"))
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device).eval()
    cfg = model.config
    head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    half = head_dim // 2
    print(
        f"[load] {model_name} in {time.time()-t0:.1f}s bits={bits} head_dim={head_dim}"
    )

    # ground-truth inv_freq from the model's own rotary buffer (rope_scaling included)
    inv_freq = None
    for mod in model.modules():
        if hasattr(mod, "inv_freq"):
            inv_freq = mod.inv_freq.detach().float().cpu().numpy()
            break
    if inv_freq is None or inv_freq.shape[0] != half:
        raise RuntimeError(f"no inv_freq buffer of length {half} found")
    wavelength = 2.0 * math.pi / inv_freq  # ascending with index
    print(
        f"[rope] wavelengths {wavelength[0]:.1f} .. {wavelength[-1]:.3g} "
        f"(window {n_seq}; {int((wavelength > n_seq).sum())}/{half} freqs are DC "
        f"at this window)"
    )

    kprojs = [mod.k_proj for mod in model.modules() if hasattr(mod, "k_proj")]
    print(f"[model] {len(kprojs)} k_proj layers, shape {tuple(kprojs[0].weight.shape)}")

    texts = _load_texts(n_samples)
    batch = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=n_seq).input_ids[
            0
        ]
        if ids.numel() >= 16:
            batch.append(ids)
    print(f"[data] {len(batch)} sequences, ~{sum(x.numel() for x in batch)} tokens")

    base_lp = output_logprobs(model, batch, device)
    print(f"[base] logprobs {tuple(base_lp.shape)}")

    saved = [m.weight.data.clone() for m in kprojs]

    # ── per-row weight-level stats vs wavelength (fine-grained, cheap) ──
    row_relfro, row_kurt, row_peak, row_wave = [], [], [], []
    for m in kprojs:
        W = m.weight.data
        Wq = quantize_per_outchannel(W, bits)
        num = (W - Wq).float().norm(dim=1)
        den = W.float().norm(dim=1) + 1e-30
        a = W.float().cpu().numpy()
        row_relfro.append((num / den).cpu().numpy())
        row_kurt.append(_kurtosis_rows(a))
        row_peak.append(np.abs(a).max(axis=1) / (a.std(axis=1) + 1e-30))
        f = (np.arange(W.shape[0]) % head_dim) % half
        row_wave.append(wavelength[f])
    row_relfro = np.concatenate(row_relfro)
    row_kurt = np.concatenate(row_kurt)
    row_peak = np.concatenate(row_peak)
    row_wave = np.concatenate(row_wave)
    sp_err = spearman(row_relfro, row_wave)
    sp_kurt = spearman(row_kurt, row_wave)
    sp_peak = spearman(row_peak, row_wave)
    print(
        f"[rows] n={row_wave.size}  Spearman vs wavelength: "
        f"relfro={sp_err:+.3f}  kurtosis={sp_kurt:+.3f}  peak/std={sp_peak:+.3f}"
    )

    # ── full-K reference (should reproduce the matched-bit anomaly) ──
    for m in kprojs:
        m.weight.data.copy_(quantize_per_outchannel(m.weight.data, bits))
    kl_full, flip_full = kl_top1(base_lp, output_logprobs(model, batch, device))
    for m, w0 in zip(kprojs, saved):
        m.weight.data.copy_(w0)
    print(f"[fullK] out_kl={kl_full:.4f} top1_flip={flip_full:.4f}")

    # ── bucket probes ──
    per = half // n_buckets
    results = []
    for b in range(n_buckets):
        f_lo, f_hi = b * per, (b + 1) * per if b < n_buckets - 1 else half
        lam_lo, lam_hi = float(wavelength[f_lo]), float(wavelength[f_hi - 1])

        def rows_of(m):
            j = torch.arange(m.weight.shape[0], device=m.weight.device)
            f = (j % head_dim) % half
            return (f >= f_lo) & (f < f_hi)

        # ONLY: quantize just this bucket's rows
        for m, w0 in zip(kprojs, saved):
            sel = rows_of(m)
            Wq = quantize_per_outchannel(w0, bits)
            m.weight.data[sel] = Wq[sel]
        kl_only, flip_only = kl_top1(base_lp, output_logprobs(model, batch, device))
        for m, w0 in zip(kprojs, saved):
            m.weight.data.copy_(w0)

        # SPARED: quantize everything except this bucket's rows
        for m, w0 in zip(kprojs, saved):
            sel = rows_of(m)
            Wq = quantize_per_outchannel(w0, bits)
            m.weight.data.copy_(Wq)
            m.weight.data[sel] = w0[sel]
        kl_sp, flip_sp = kl_top1(base_lp, output_logprobs(model, batch, device))
        for m, w0 in zip(kprojs, saved):
            m.weight.data.copy_(w0)

        rec = {
            "bucket": b,
            "f_lo": f_lo,
            "f_hi": f_hi,
            "wavelength_lo": lam_lo,
            "wavelength_hi": lam_hi,
            "kl_only": kl_only,
            "flip_only": flip_only,
            "kl_spared": kl_sp,
            "flip_spared": flip_sp,
        }
        results.append(rec)
        print(
            f"  bucket {b} (lam {lam_lo:9.1f}..{lam_hi:12.1f}): "
            f"only kl={kl_only:.4f} flip={flip_only:.4f} | "
            f"spared kl={kl_sp:.4f} flip={flip_sp:.4f}"
        )

    kl_only_arr = np.array([r["kl_only"] for r in results])
    sp_bucket = spearman(kl_only_arr, np.arange(n_buckets).astype(float))
    frac_top2 = float(kl_only_arr[-2:].sum() / max(kl_only_arr.sum(), 1e-30))
    print(
        f"[verdict] Spearman(bucket-only kl, wavelength rank)={sp_bucket:+.3f}  "
        f"share of damage in 2 longest-wavelength buckets={frac_top2:.2%}"
    )

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"k_wavelength_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "bits": bits,
                "head_dim": head_dim,
                "n_seq": len(batch),
                "window": n_seq,
                "full_k": {"out_kl": kl_full, "top1_flip": flip_full},
                "row_spearman": {
                    "relfro": sp_err,
                    "kurtosis": sp_kurt,
                    "peak": sp_peak,
                },
                "bucket_spearman_kl_only": sp_bucket,
                "damage_share_top2_buckets": frac_top2,
                "buckets": results,
            },
            f,
            indent=2,
        )
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
