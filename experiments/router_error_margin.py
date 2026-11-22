# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Router quantization error relative to the routing margin (experiment E5).

A top-k set cannot change under a logit perturbation e if the spread of e over
the experts, max_i e_i - min_i e_i, is below the margin m between the k-th and
(k+1)-th logits (the sufficient condition of the TMLR draft, Section 6).
Absolute margins are not comparable across models: scaling a router's weights
scales the margins and, under the per-tensor uniform quantizer used here, the
quantization error by the same factor. The scale-free quantity is

    R = (max_i e_i - min_i e_i) / m,     e = x (Q_b(W_g) - W_g)^T

where x is the router input. R < 1 guarantees the route survives. This
captures x and the logits on WikiText-2 (same chunks as the routing harness),
and reports, per layer and overall, the distribution of R, the fraction with
R < 1, and the exact top-k flip rate, computed from the logits plus e.

Weights are never edited, so CPU offload is safe; gate weights are read from
the cached safetensors (offloaded parameters are meta tensors).

Usage:
    NB_MODEL=allenai/OLMoE-1B-7B-0924 NB_CHUNKS=16 python router_error_margin.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch


def quant_uniform(x: np.ndarray, bits: int) -> np.ndarray:
    """Per-tensor uniform grid over [min, max], as in the routing harness."""
    lo, hi = float(x.min()), float(x.max())
    step = (hi - lo) / max(2**bits - 1, 1)
    return np.round((x - lo) / max(step, 1e-30)) * step + lo


def gate_weights(model_id: str, names: list[str]) -> list[np.ndarray]:
    from safetensors import safe_open

    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    stem = "models--" + model_id.replace("/", "--")
    snap = sorted(glob.glob(os.path.join(home, "hub", stem, "snapshots", "*")))[-1]
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wmap = json.load(f)["weight_map"]
    else:
        wmap = None
    out = []
    for n in names:
        # transformers 5 renamed some MoE modules; try the checkpoint's own names
        cands = [
            f"{n}.weight",
            f"{n.replace('.mlp.gate', '.block_sparse_moe.gate')}.weight",
        ]
        key = next((c for c in cands if wmap is None or c in wmap), None)
        if key is None:
            raise KeyError(f"no checkpoint key for router {n} (tried {cands})")
        fname = wmap[key] if wmap else "model.safetensors"
        with safe_open(os.path.join(snap, fname), framework="pt") as sf:
            out.append(sf.get_tensor(key).float().numpy())
    return out


def analyse(x, logits, W, bits, k) -> dict:
    e = x @ (quant_uniform(W, bits) - W).T
    order = np.argsort(-logits, axis=1)
    top = np.take_along_axis(logits, order[:, :k], 1)
    rest = np.take_along_axis(logits, order[:, k:], 1)
    margin = top[:, -1] - rest[:, 0]
    spread = e.max(axis=1) - e.min(axis=1)
    R = spread / np.maximum(margin, 1e-12)
    new = logits + e
    sel = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(sel, order[:, :k], True, 1)
    min_sel = np.where(sel, new, np.inf).min(axis=1)
    max_rest = np.where(sel, -np.inf, new).max(axis=1)
    flip = min_sel < max_rest
    q = np.percentile(R, [10, 50, 90])
    return {
        "n": int(R.size),
        "R_p10": float(q[0]),
        "R_median": float(q[1]),
        "R_p90": float(q[2]),
        "frac_R_below_1": float((R < 1).mean()),
        "flip_rate": float(flip.mean()),
        "flip_rate_R_below_1": float(flip[R < 1].mean()) if (R < 1).any() else None,
        "flip_rate_R_at_least_1": (
            float(flip[R >= 1].mean()) if (R >= 1).any() else None
        ),
        "median_spread": float(np.median(spread)),
        "median_margin": float(np.median(margin)),
    }


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "allenai/OLMoE-1B-7B-0924")
    n_chunks = int(os.environ.get("NB_CHUNKS", "16"))
    bits_list = [int(b) for b in os.environ.get("NB_BITS", "4,3").split(",")]
    gpu_gib = int(os.environ.get("NB_GPU_GIB", "4"))
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")

    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(target_temp=80.0, max_threads=12, min_threads=2)
        thermal.start()
        torch.set_num_threads(thermal.get_threads())
    except Exception as e:
        print(f"[thermal] controller unavailable ({e}); capping at 6 threads")
        torch.set_num_threads(6)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: f"{gpu_gib}GiB", "cpu": "150GiB"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = model.config
    k = int(cfg.num_experts_per_tok)
    n_exp = int(getattr(cfg, "num_local_experts", None) or cfg.num_experts)
    # routers: modules named *.gate holding an (n_experts, hidden) weight; in
    # transformers 5 these can be custom router modules, not nn.Linear
    gates = [
        (n, m)
        for n, m in model.named_modules()
        if n.endswith(".gate")
        and getattr(m, "weight", None) is not None
        and m.weight.dim() == 2
        and m.weight.shape[0] == n_exp
    ]
    assert gates, "no router modules found"
    dt = time.time() - t0
    print(f"[load] {model_name} top-{k} of {n_exp}, {len(gates)} routers, {dt:.0f}s")

    xs = [[] for _ in gates]
    ls = [[] for _ in gates]

    def mk(i):
        def hook(_mod, inp, out):
            lg = out[0] if isinstance(out, tuple) else out
            xs[i].append(
                inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu().numpy()
            )
            ls[i].append(lg.detach().reshape(-1, lg.shape[-1]).float().cpu().numpy())

        return hook

    handles = [m.register_forward_hook(mk(i)) for i, (_, m) in enumerate(gates)]
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    with torch.no_grad():
        for c in range(n_chunks):
            if thermal is not None:
                torch.set_num_threads(thermal.get_threads())
            chunk = ids[:, c * 1024 : (c + 1) * 1024]
            if chunk.size(1) < 2:
                break
            model(chunk.to(0))
            print(f"  forward {c + 1}/{n_chunks} ({time.time() - t0:.0f}s)")
    for h in handles:
        h.remove()

    weights = gate_weights(model_name, [n for n, _ in gates])
    results = {}
    for bits in bits_list:
        per_layer = []
        allx = []
        for i, W in enumerate(weights):
            x, cap = np.concatenate(xs[i]), np.concatenate(ls[i])
            lg = x @ W.T  # pre-softmax logits, recomputed exactly
            chk = (
                float(np.abs(lg - cap).max() / (np.abs(lg).max() + 1e-12))
                if cap.shape == lg.shape
                else None
            )
            r = analyse(x, lg, W, bits, k)
            r["layer"] = i
            r["logit_recompute_rel_err"] = chk
            per_layer.append(r)
            allx.append((x, lg, W))
        pooled_R = []
        for x, lg, W in allx:
            e = x @ (quant_uniform(W, bits) - W).T
            order = np.argsort(-lg, axis=1)
            top = np.take_along_axis(lg, order[:, :k], 1)
            rest = np.take_along_axis(lg, order[:, k:], 1)
            margin = top[:, -1] - rest[:, 0]
            pooled_R.append((e.max(1) - e.min(1)) / np.maximum(margin, 1e-12))
        R = np.concatenate(pooled_R)
        results[str(bits)] = {
            "R_p10": float(np.percentile(R, 10)),
            "R_median": float(np.median(R)),
            "R_p90": float(np.percentile(R, 90)),
            "frac_R_below_1": float((R < 1).mean()),
            "flip_rate": float(np.mean([p["flip_rate"] for p in per_layer])),
            "per_layer": per_layer,
        }
        res = results[str(bits)]
        print(
            f"[{bits}-bit] R median={res['R_median']:.3g} p10={res['R_p10']:.3g}"
            f" p90={res['R_p90']:.3g} R<1={res['frac_R_below_1']:.1%}"
            f" flip={res['flip_rate']:.3f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"router_error_margin_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "top_k": k,
                "n_experts": n_exp,
                "n_routers": len(gates),
                "router_names": [n for n, _ in gates],
                "n_chunks": n_chunks,
                "tokens_per_chunk": 1024,
                "quantizer": "per-tensor uniform grid over [min, max]",
                "R": "(max_i e_i - min_i e_i) / margin; R < 1 guarantees no flip",
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
