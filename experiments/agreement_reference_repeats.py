# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Sequence-level intervals and reference-perturbation repeats (experiment E1).

behavioral_metric_demo.py reported decision-level agreement for whole-model
per-output-channel quantization, with a "noise floor" D0 from ONE pair of
8-bit requantizations dithered at 0.02 of a step (seeds 1 and 2). This repeats
that measurement and adds what a reader needs to interpret it:

  * per-sequence outcome counts, so every statistic gets a bootstrap interval
    that resamples whole sequences (positions within a sequence are dependent);
  * the reference disagreement D0 over three dither amplitudes (0.02, 0.05,
    0.1 of a step) and three independent seed pairs each, to show how much D0
    depends on the reference procedure;
  * paired intervals for D - D0 (same resampled sequences for both).

Same data, quantizer and dither as the demo. Usage (GPU):
    NB_MODEL=Qwen/Qwen2.5-1.5B-Instruct NB_SAMPLES=24 \
        python agreement_reference_repeats.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import exp_io
import numpy as np
import torch
import torch.nn as nn


def quantize_per_outchannel_(W, bits, dither=0.0, gen=None) -> None:
    """In-place symmetric absmax per output channel, as in the demo."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    x = W / scale
    if dither > 0.0:
        x = x + dither * torch.randn(
            W.shape, generator=gen, device=W.device, dtype=W.dtype
        )
    codes = torch.clamp(torch.round(x), -qmax - 1, qmax)
    W.copy_(codes * scale)


def linears(model):
    return [
        mod
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and ("layers." in name or "block" in name)
    ]


def predict(model, batch, device) -> list[np.ndarray]:
    out = []
    with torch.no_grad():
        for ids in batch:
            logits = model(ids.to(device).unsqueeze(0)).logits[0]
            out.append(logits[:-1].argmax(-1).cpu().numpy())
            del logits
    return out


def apply_quant(model, saved, bits, dither=0.0, seed=0, device="cpu") -> None:
    gen = torch.Generator(device=device).manual_seed(seed)
    for mod, w0 in zip(linears(model), saved):
        mod.weight.data.copy_(w0.to(mod.weight.device))
        quantize_per_outchannel_(mod.weight.data, bits, dither=dither, gen=gen)


def restore(model, saved) -> None:
    for mod, w0 in zip(linears(model), saved):
        mod.weight.data.copy_(w0.to(mod.weight.device))


def host_copy(t: torch.Tensor) -> torch.Tensor:
    """A CPU copy of a weight, in pinned memory when the device is a GPU (each condition
    copies every weight back, so a pinned copy keeps the GPU busy)."""
    if t.device.type != "cuda":
        return t.clone().cpu()
    try:
        c = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    except RuntimeError:
        c = torch.empty(t.shape, dtype=t.dtype)
    return c.copy_(t)


def per_seq(base, quant, gold) -> np.ndarray:
    """Rows: sequences. Columns: n, base correct, quant correct, regressions,
    recoveries, same prediction."""
    rows = []
    for b, q, g in zip(base, quant, gold):
        bc, qc = b == g, q == g
        rows.append(
            [
                b.size,
                bc.sum(),
                qc.sum(),
                (bc & ~qc).sum(),
                (~bc & qc).sum(),
                (b == q).sum(),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def stats(m: np.ndarray) -> dict:
    n = m[:, 0].sum()
    acc_b, acc_q = m[:, 1].sum() / n, m[:, 2].sum() / n
    r, c, a = m[:, 3].sum() / n, m[:, 4].sum() / n, m[:, 5].sum() / n
    return {
        "acc_base": acc_b,
        "acc_quant": acc_q,
        "acc_drop": acc_b - acc_q,
        "ca": acc_b - r,
        "regressions": r,
        "recoveries": c,
        "agreement": a,
        "disagreement": 1.0 - a,
    }


def disagreement(m: np.ndarray) -> float:
    return 1.0 - m[:, 5].sum() / m[:, 0].sum()


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    n_samples = int(os.environ.get("NB_SAMPLES", "24"))
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    bits_list = [int(b) for b in os.environ.get("NB_BITS", "8,4,3").split(",")]
    amps = [float(a) for a in os.environ.get("NB_AMPS", "0.02,0.05,0.1").split(",")]
    pairs = [(1, 2), (3, 4), (5, 6)]
    n_boot = int(os.environ.get("NB_BOOT", "2000"))
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(0)

    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(target_temp=80.0, max_threads=8, min_threads=2)
        thermal.start()
        torch.set_num_threads(thermal.get_threads())
    except Exception as e:
        n_thr = exp_io.fallback_threads()
        print(f"[thermal] controller unavailable ({e}); capping at {n_thr} threads")
        torch.set_num_threads(n_thr)

    t0 = time.time()
    model, tok = exp_io.model_and_tokenizer(model_name, dtype, device)
    print(f"[load] {model_name} in {time.time() - t0:.1f}s")

    batch = exp_io.token_batch(tok, "test", n_samples, n_seq)
    gold = [ids[1:].numpy() for ids in batch]
    # every condition starts from these: apply_quant copies them back before quantizing
    saved = [host_copy(m.weight.data) for m in linears(model)]
    base = predict(model, batch, device)
    print(f"[data] {len(batch)} seqs, {sum(g.size for g in gold)} positions")

    refs = {}
    for amp in amps:
        for s1, s2 in pairs:
            if thermal is not None:
                torch.set_num_threads(thermal.get_threads())
            apply_quant(model, saved, 8, dither=amp, seed=s1, device=device)
            ra = predict(model, batch, device)
            apply_quant(model, saved, 8, dither=amp, seed=s2, device=device)
            rb = predict(model, batch, device)
            m = per_seq(ra, rb, gold)
            refs[f"amp{amp}_s{s1}{s2}"] = m
            print(f"  ref amp={amp} seeds={s1},{s2} D0={disagreement(m):.4f}")

    quant = {}
    for bits in bits_list:
        apply_quant(model, saved, bits, seed=0, device=device)
        quant[bits] = per_seq(base, predict(model, batch, device), gold)
        s = stats(quant[bits])
        print(f"  {bits}-bit acc_drop={s['acc_drop']:.4f} agree={s['agreement']:.4f}")
    restore(model, saved)

    rng = np.random.default_rng(0)
    n_s = len(batch)
    idx = rng.integers(0, n_s, size=(n_boot, n_s))
    ref0 = refs[f"amp{amps[0]}_s12"]

    def ci(fn, *mats):
        vals = np.array([fn(*(m[i] for m in mats)) for i in idx])
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    out_q = {}
    for bits, m in quant.items():
        s = stats(m)
        row = {k: float(v) for k, v in s.items()}
        for k in s:
            row[f"{k}_ci95"] = ci(lambda x, k=k: stats(x)[k], m)
        row["d_minus_d0_ref"] = float(disagreement(m) - disagreement(ref0))
        row["d_minus_d0_ref_ci95"] = ci(
            lambda x, y: disagreement(x) - disagreement(y), m, ref0
        )
        out_q[str(bits)] = row
    out_r = {}
    for name, m in refs.items():
        out_r[name] = {"d0": float(disagreement(m)), "d0_ci95": ci(disagreement, m)}
    d0s = {str(a): [out_r[f"amp{a}_s{p[0]}{p[1]}"]["d0"] for p in pairs] for a in amps}

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"agreement_repeats_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "n_seqs": n_s,
                "n_positions": int(sum(g.size for g in gold)),
                "window": n_seq,
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "dither_amplitudes": amps,
                "seed_pairs": pairs,
                "bootstrap": {"n": n_boot, "unit": "sequence", "seed": 0},
                "reference_ref0": f"amp{amps[0]}_s12",
                "quantized": out_q,
                "references": out_r,
                "d0_by_amplitude": d0s,
                "per_seq": {
                    "columns": ["n", "base_ok", "quant_ok", "regr", "recov", "same"],
                    "quantized": {str(b): m.tolist() for b, m in quant.items()},
                    "references": {k: m.tolist() for k, m in refs.items()},
                },
            },
            f,
            indent=1,
        )
    print(f"[saved] {path}")
    if thermal is not None:
        try:
            thermal.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
