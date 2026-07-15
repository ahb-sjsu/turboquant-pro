# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Behavioral-metric demo: what Correctness Agreement hides, and the noise floor.

Demonstrates :mod:`turboquant_pro.behavioral_agreement` on a real model, using
next-token prediction as the correctness task (pred == true next token). For a
base model and quantized variants we report:

  * the paper's Correctness Agreement (CA),
  * the symmetric flip rate (regressions vs recoveries, McNemar),
  * prediction-level behavioral agreement (same token, right or wrong), and
  * excess disagreement over a NOISE FLOOR built from two near-lossless (8-bit)
    requantizations -- the "free" churn that any quantization inherits.

The point: CA and accuracy can look stable across bit-widths while the flip /
behavioral-agreement / excess-over-floor signals reveal real answer churn -- the
correction to the paper's metric.

Usage (GPU):
    NB_MODEL=Qwen/Qwen2.5-1.5B-Instruct NB_BITS=8,4,3 NB_SEQ=256 NB_SAMPLES=32 \
        python experiments/behavioral_metric_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

# local import of the metric (shipped alongside for the Atlas run)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from turboquant_pro.behavioral_agreement import (
        correctness_agreement,
        evaluate,
        flip_rate,
        noise_floor,
    )
except Exception:  # noqa: BLE001 - allow running from a flat ship dir
    from behavioral_agreement import (  # type: ignore
        correctness_agreement,
        evaluate,
        flip_rate,
        noise_floor,
    )


def quantize_per_outchannel_(W: torch.Tensor, bits: int, dither: float = 0.0,
                             gen: torch.Generator | None = None) -> None:
    """In-place uniform per-output-channel symmetric absmax quant->dequant.

    ``dither`` adds a tiny sub-LSB noise (seeded) so two 'near-lossless' runs
    differ only microscopically -- used to build the behavioral noise floor."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    x = W / scale
    if dither > 0.0:
        x = x + dither * torch.randn(W.shape, generator=gen, device=W.device,
                                     dtype=W.dtype)
    codes = torch.clamp(torch.round(x), -qmax - 1, qmax)
    W.copy_(codes * scale)


def linears(model):
    """Transformer-block nn.Linear weights (skip the LM head / embeddings)."""
    import torch.nn as nn
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ("layers." in name or "block" in name):
            out.append(mod)
    return out


def predict_argmax(model, batch, device):
    """Per-position next-token argmax and the true next token (gold)."""
    preds, gold = [], []
    with torch.no_grad():
        for ids in batch:
            ids = ids.to(device)
            logits = model(ids.unsqueeze(0)).logits[0]  # [seq, vocab]
            p = logits[:-1].argmax(-1).cpu().numpy()     # predict token t+1
            g = ids[1:].cpu().numpy()
            preds.append(p)
            gold.append(g)
    return np.concatenate(preds), np.concatenate(gold)


def _apply_quant(model, saved, bits, dither=0.0, seed=0, device="cpu"):
    # saved weights live on CPU to avoid doubling GPU memory; copy back per apply
    gen = torch.Generator(device=device).manual_seed(seed)
    for mod, w0 in zip(linears(model), saved):
        mod.weight.data.copy_(w0.to(mod.weight.device))
        quantize_per_outchannel_(mod.weight.data, bits, dither=dither, gen=gen)


def _restore(model, saved):
    for mod, w0 in zip(linears(model), saved):
        mod.weight.data.copy_(w0.to(mod.weight.device))


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    bits_list = [int(b) for b in os.environ.get("NB_BITS", "8,4,3").split(",")]
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    n_samples = int(os.environ.get("NB_SAMPLES", "32"))
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    print(f"[load] {model_name} in {time.time()-t0:.1f}s device={device}")

    texts = _load_texts(n_samples)
    batch = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=n_seq).input_ids[0]
        if ids.numel() >= 16:
            batch.append(ids)
    print(f"[data] {len(batch)} seqs, ~{sum(x.numel() for x in batch)} tokens")

    saved = [m.weight.data.clone().cpu() for m in linears(model)]  # CPU snapshots
    base_pred, gold = predict_argmax(model, batch, device)
    acc_base = float(np.mean(base_pred == gold))
    print(f"[base] next-token acc={acc_base:.4f} over {gold.size} positions")

    # noise floor: two 8-bit requantizations that differ only by sub-LSB dither
    _apply_quant(model, saved, 8, dither=0.02, seed=1, device=device)
    ref_a, _ = predict_argmax(model, batch, device)
    _apply_quant(model, saved, 8, dither=0.02, seed=2, device=device)
    ref_b, _ = predict_argmax(model, batch, device)
    _restore(model, saved)
    floor = noise_floor(ref_a, ref_b, gold=gold)
    print(f"[floor] two 8-bit variants: disagreement={floor.floor_disagreement:.4f} "
          f"churn={floor.floor_churn:.4f}")

    rows = []
    for bits in bits_list:
        _apply_quant(model, saved, bits, seed=0, device=device)
        qpred, _ = predict_argmax(model, batch, device)
        _restore(model, saved)
        ca = correctness_agreement(base_pred == gold, qpred == gold)
        fr = flip_rate(base_pred == gold, qpred == gold)
        rep = evaluate(base_pred, qpred, gold, floor=floor)
        rows.append({
            "bits": bits, "acc_quant": fr.acc_quant,
            "correctness_agreement": ca,
            "regressions": fr.regressions, "recoveries": fr.recoveries,
            "churn": fr.churn, "mcnemar_p": fr.mcnemar_p,
            "behavioral_agreement": rep.behavioral_agreement,
            "disagreement": rep.disagreement,
            "excess_over_floor": rep.excess_disagreement,
            "excess_z": rep.excess_z,
        })
        print(f"  b{bits}: acc={fr.acc_quant:.4f} CA={ca:.4f} "
              f"churn={fr.churn:.4f} BA={rep.behavioral_agreement:.4f} "
              f"excess={rep.excess_disagreement:+.4f} z={rep.excess_z:.1f}")

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"behavioral_{tag}.json")
    with open(path, "w") as f:
        json.dump({"model": model_name, "acc_base": acc_base,
                   "n_positions": int(gold.size),
                   "floor": floor.as_dict(), "rows": rows}, f, indent=2)
    print(f"[saved] {path}")
    _summary(tag, acc_base, floor, rows)
    return 0


def _load_texts(n: int) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        out, buf = [], ""
        for row in ds:
            t = row["text"].strip()
            if not t:
                continue
            buf += " " + t
            if len(buf) > 1200:
                out.append(buf.strip()); buf = ""
            if len(out) >= n:
                break
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[data] wikitext unavailable ({type(e).__name__}); embedded sample")
    seed = ("Quantization reduces the precision of model weights to save memory. "
            "Whether the quantized model behaves the same as the original is a "
            "separate question from whether its accuracy is preserved. ")
    return [seed * 4 for _ in range(n)]


def _summary(tag, acc_base, floor, rows):
    print(f"\n=== behavioral metrics vs base: {tag} (acc_base={acc_base:.4f}) ===")
    print(f"noise floor (two 8-bit variants): disagreement={floor.floor_disagreement:.4f}")
    print("bit   acc   CA(paper)  churn   behav_agree  disagree  excess/floor   z")
    for r in rows:
        print(f"{r['bits']:>2}  {r['acc_quant']:.3f}   {r['correctness_agreement']:.3f}"
              f"     {r['churn']:.3f}    {r['behavioral_agreement']:.3f}     "
              f"{r['disagreement']:.3f}     {r['excess_over_floor']:+.3f}    "
              f"{r['excess_z']:.1f}")


if __name__ == "__main__":
    sys.exit(main())
