# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Budget-matched controls for the rotary-row protection of W^K (experiment E2).

k_wavelength_probe.py found that, at 3 bits, sparing the longest-wavelength
eighth of W^K rows removes most of Llama-3.2-3B's key damage. That shows the
rows are ENOUGH. This experiment asks whether they are the RIGHT rows at the
same budget: every condition keeps exactly one eighth of the rows of every
k_proj at NB_PROTECT_BITS (default 8) and the rest at NB_BITS (default 3), so
every condition costs the same 1/8*8 + 7/8*3 = 3.625 bits per key weight
before scale metadata. Only which rows are protected changes.

Conditions (one eighth of the rows of every k_proj protected):
  none            no rows protected (all at NB_BITS), the reference damage
  longest         rotary-frequency octile with the longest wavelengths
  shortest        rotary-frequency octile with the shortest wavelengths
  random_rows_s   a uniform random eighth of the rows of each layer, seed s
  random_freqs_s  a random set of head_dim/16 rotary frequency indices, the same
                  structure as an octile, seed s
  weight_error    rows with the largest relative 3-bit quantization error
  act_error       rows with the largest output error ||dW_row X||^2 on
                  calibration activations X (train split), dW = W - Q_b(W)

Evaluation is on held-out text: the WikiText-2 VALIDATION split (the probe used
the test split; the calibration statistic uses the train split). The probe's
own test-split set is also evaluated for none and longest, for continuity with
the earlier numbers. Readout: mean KL(p_base || p_q) of the next-token
distribution over positions, top-1 flip rate, per-sequence mean KL, and
bootstrap confidence intervals over sequences (whole sequences resampled,
because positions within a sequence are dependent), including paired
differences against "longest".

Quantizer: symmetric absmax per output channel (row), as in the probe.

NB_RESIDENT=1 keeps the base log-probabilities, the weight copies and the
calibration statistic on the GPU instead of in CPU memory. The arithmetic is the
same; it suits a GPU with room for them beside the model and a pod with little host
memory (Mistral-7B's 32k vocabulary on an NRP A10). The default keeps them on the
CPU, which a 128k vocabulary on a shared GPU needs.

Usage (GPU):
    NB_MODEL=unsloth/Llama-3.2-3B NB_OUT=/path python k_row_protection_controls.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import exp_io
import numpy as np
import torch


def quantize_per_outchannel(W: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    codes = torch.clamp(torch.round(W / scale), -qmax - 1, qmax)
    return codes * scale


def seq_logprobs(model, batch, device, store="cpu") -> list[torch.Tensor]:
    """Base log-probabilities, kept on ``store``: CPU memory by default (a 128k
    vocabulary makes the full set several GB, too much to hold on a shared GPU)."""
    outs = []
    with torch.no_grad():
        for ids in batch:
            logits = model(ids.to(device).unsqueeze(0)).logits[0]
            outs.append(torch.log_softmax(logits.float(), dim=-1).to(store))
    return outs


def compare(model, batch, base: list[torch.Tensor], device) -> dict:
    """Run the current model on each sequence and compare with its base, one
    sequence at a time on the GPU."""
    seq_kl, seq_flip, n_pos = [], [], []
    with torch.no_grad():
        for ids, b_cpu in zip(batch, base):
            logits = model(ids.to(device).unsqueeze(0)).logits[0]
            v = torch.log_softmax(logits.float(), dim=-1)
            b = b_cpu.to(device)
            kl = (b.exp() * (b - v)).sum(dim=-1)
            flip = (b.argmax(-1) != v.argmax(-1)).float()
            seq_kl.append(float(kl.mean()))
            seq_flip.append(float(flip.mean()))
            n_pos.append(int(kl.numel()))
            del logits, v, b, kl, flip
    w = np.array(n_pos, dtype=np.float64)
    return {
        "kl": float(np.average(seq_kl, weights=w)),
        "flip": float(np.average(seq_flip, weights=w)),
        "seq_kl": seq_kl,
        "seq_flip": seq_flip,
        "n_pos": n_pos,
    }


def boot_mean(seq_vals, n_pos, rng, n_boot=2000) -> list[float]:
    v = np.asarray(seq_vals, dtype=np.float64)
    w = np.asarray(n_pos, dtype=np.float64)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = (v[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_diff(a, b, n_pos, rng, n_boot=2000) -> list[float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(n_pos, dtype=np.float64)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = ((a[idx] - b[idx]) * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "unsloth/Llama-3.2-3B")
    bits = int(os.environ.get("NB_BITS", "3"))
    pbits = int(os.environ.get("NB_PROTECT_BITS", "8"))
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    n_eval = int(os.environ.get("NB_EVAL", "64"))
    n_calib = int(os.environ.get("NB_CALIB", "16"))
    n_probe = int(os.environ.get("NB_PROBE", "32"))
    seeds = [int(s) for s in os.environ.get("NB_SEEDS", "0,1,2,3,4").split(",")]
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = device if os.environ.get("NB_RESIDENT", "0") == "1" else "cpu"
    torch.manual_seed(0)

    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(target_temp=80.0, max_threads=8, min_threads=2)
        thermal.start()
        torch.set_num_threads(thermal.get_threads())
    except Exception as e:  # the run still needs a thread cap without it
        n_thr = exp_io.fallback_threads()
        print(f"[thermal] controller unavailable ({e}); capping at {n_thr} threads")
        torch.set_num_threads(n_thr)

    t0 = time.time()
    model, tok = exp_io.model_and_tokenizer(model_name, torch.float16, device)
    cfg = model.config
    head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    half = head_dim // 2
    per = half // 8
    print(f"[load] {model_name} in {time.time() - t0:.1f}s bits={bits}/{pbits}")

    inv_freq = None
    for mod in model.modules():
        if hasattr(mod, "inv_freq"):
            inv_freq = mod.inv_freq.detach().float().cpu().numpy()
            break
    if inv_freq is None or inv_freq.shape[0] != half:
        raise RuntimeError(f"no inv_freq buffer of length {half} found")
    wavelength = 2.0 * math.pi / inv_freq

    kprojs = [m.k_proj for m in model.modules() if hasattr(m, "k_proj")]
    n_rows = kprojs[0].weight.shape[0]
    k_prot = n_rows // 8
    print(f"[model] {len(kprojs)} k_proj, {n_rows} rows each, protect {k_prot}")

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {free / 2**30:.1f} GiB free of {total / 2**30:.1f} after load")
        if free < 1.0 * 2**30:
            raise SystemExit(
                "[gpu] under 1 GiB free after load; not running on a shared GPU"
            )

    # weight copies live on `store` (CPU memory by default); one layer at a time goes to
    # the GPU
    saved = [m.weight.data.detach().to(store).clone() for m in kprojs]
    # quantized on the CPU wherever the copies live: the GPU divides by a number as a
    # multiplication by its reciprocal, which moves some steps by one bit
    q_low = [
        quantize_per_outchannel(w.cpu().float(), bits).half().to(store) for w in saved
    ]
    q_high = [
        quantize_per_outchannel(w.cpu().float(), pbits).half().to(store) for w in saved
    ]
    freq_of_row = (torch.arange(n_rows) % head_dim) % half

    # calibration statistic: Sigma_l = X^T X of each k_proj's input, train split
    calib = exp_io.token_batch(tok, "train", n_calib, n_seq)
    sig = [None] * len(kprojs)

    def make_hook(i):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            s = (x.T @ x).to(store)
            sig[i] = s if sig[i] is None else sig[i] + s

        return hook

    hooks = [m.register_forward_pre_hook(make_hook(i)) for i, m in enumerate(kprojs)]
    with torch.no_grad():
        for ids in calib:
            model(ids.to(device).unsqueeze(0))
    for h in hooks:
        h.remove()

    act_score = []
    for i in range(len(kprojs)):
        dw = (saved[i].float() - q_low[i].float()).to(device)
        s = sig[i].to(device)
        act_score.append(((dw @ s) * dw).sum(dim=1).cpu())
        del s, dw
        sig[i] = None
    torch.cuda.empty_cache()
    rel_err = [  # on the CPU wherever the copies live, so the row ranking is the same
        (
            (w.float().cpu() - q.float().cpu()).norm(dim=1)
            / (w.float().cpu().norm(dim=1) + 1e-30)
        )
        for w, q in zip(saved, q_low)
    ]

    def by_freqs(fset):
        mask = torch.zeros(n_rows, dtype=torch.bool)
        for f in fset:
            mask |= freq_of_row == f
        return [mask.clone() for _ in kprojs]

    def top_rows(scores):
        out = []
        for sc in scores:
            mask = torch.zeros(n_rows, dtype=torch.bool)
            mask[torch.topk(sc, k_prot).indices] = True
            out.append(mask)
        return out

    conditions = {
        "none": [torch.zeros(n_rows, dtype=torch.bool) for _ in kprojs],
        "longest": by_freqs(range(half - per, half)),
        "shortest": by_freqs(range(0, per)),
        "weight_error": top_rows(rel_err),
        "act_error": top_rows(act_score),
    }

    def random_rows(g):
        masks = []
        for _ in kprojs:
            mask = torch.zeros(n_rows, dtype=torch.bool)
            mask[torch.randperm(n_rows, generator=g)[:k_prot]] = True
            masks.append(mask)
        return masks

    for s in seeds:
        g = torch.Generator().manual_seed(1000 + s)
        conditions[f"random_rows_{s}"] = random_rows(g)
        fset = torch.randperm(half, generator=g)[:per].tolist()
        conditions[f"random_freqs_{s}"] = by_freqs(fset)
    for name, masks in conditions.items():
        n_sel = sum(int(m.sum()) for m in masks)
        expect = 0 if name == "none" else k_prot * len(kprojs)
        assert n_sel == expect, (name, n_sel, expect)

    def apply(masks):
        for m, lo, hi, sel in zip(kprojs, q_low, q_high, masks):
            w = lo.clone()
            sel = sel.to(lo.device)
            w[sel] = hi[sel]
            m.weight.data.copy_(w.to(m.weight.device))

    def restore():
        for m, w0 in zip(kprojs, saved):
            m.weight.data.copy_(w0.to(m.weight.device))

    def overlap(a, b):
        return sum(int((x & y).sum()) for x, y in zip(a, b)) / max(
            sum(int(x.sum()) for x in a), 1
        )

    evalset = exp_io.token_batch(tok, "validation", n_eval, n_seq)
    probe = exp_io.token_batch(tok, "test", n_probe, n_seq)
    print(f"[data] eval {len(evalset)} validation seqs, probe {len(probe)} test seqs")
    base_eval = seq_logprobs(model, evalset, device, store)
    base_probe = seq_logprobs(model, probe, device, store)

    results = {}
    for name, masks in conditions.items():
        if thermal is not None:
            torch.set_num_threads(thermal.get_threads())
        apply(masks)
        r = compare(model, evalset, base_eval, device)
        if name in ("none", "longest"):
            rp = compare(model, probe, base_probe, device)
            r["probe_set"] = {"kl": rp["kl"], "flip": rp["flip"]}
        restore()
        r["overlap_with_longest"] = overlap(masks, conditions["longest"])
        results[name] = r
        print(
            f"  {name:16s} kl={r['kl']:.4f} flip={r['flip']:.4f}"
            f" overlap_longest={r['overlap_with_longest']:.2f}"
        )

    rng = np.random.default_rng(0)
    n_pos = results["none"]["n_pos"]
    for name, r in results.items():
        r["kl_ci95"] = boot_mean(r["seq_kl"], n_pos, rng)
        r["recovery_vs_none"] = 1.0 - r["kl"] / results["none"]["kl"]
        if name != "longest":
            r["kl_minus_longest_ci95"] = boot_diff(
                r["seq_kl"], results["longest"]["seq_kl"], n_pos, rng
            )
    for name in sorted(results, key=lambda k: results[k]["kl"]):
        r = results[name]
        lo, hi = r["kl_ci95"]
        print(
            f"[ci] {name:16s} kl={r['kl']:.4f} [{lo:.4f},{hi:.4f}]"
            f" recovery={r['recovery_vs_none']:.1%}"
        )

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"k_row_protection_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "bits": bits,
                "protect_bits": pbits,
                "bits_per_key_weight": (pbits + 7 * bits) / 8,
                "rows_per_kproj": n_rows,
                "protected_rows_per_kproj": k_prot,
                "n_kproj": len(kprojs),
                "eval_split": "wikitext-2-raw-v1 validation",
                "calib_split": "wikitext-2-raw-v1 train",
                "probe_split": "wikitext-2-raw-v1 test (the earlier probe's set)",
                "n_eval_seqs": len(evalset),
                "n_calib_seqs": len(calib),
                "window": n_seq,
                "seeds": seeds,
                "longest_wavelengths": [
                    float(wavelength[half - per]),
                    float(wavelength[half - 1]),
                ],
                "torch": torch.__version__,
                "resident": store != "cpu",
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "results": results,
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
