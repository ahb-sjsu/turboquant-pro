# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Does the softmax suppress score error from Q/K weight quantization? (E3)

The matched-bit experiment found that perturbing W^Q or W^K moves the model's
output less than perturbing W^V or W^O. A hypothesis is that the softmax
suppresses part of the score error. The softmax cancels only a constant added
to a whole row of scores, and the error from a weight perturbation,
q_t^T dW_K h_s, varies with the source token, so this has to be measured.

For each attention layer, using the layer's own inputs from an unperturbed run
(so the measurement is local to that layer), we quantize W^Q or W^K at b bits
with the matched-bit quantizer, recompute the post-rotary causal score matrix
S' and compare it with S. For every query row t (support s <= t):

  c_t       = mean_s dS_ts                       the row-constant part (cancelled)
  delta_ts  = dS_ts - c_t                         the centered part (not cancelled)
  const_share     = sum_t n_t c_t^2 / sum_ts dS_ts^2
  centered_rel    = rms(delta) / rms(within-row deviation of S)
  attention_tv    = mean_t 0.5 * || softmax(S'_t) - softmax(S_t) ||_1

const_share near 1 would support "the error is partly constant along each
row"; a small centered_rel and attention_tv would support "small next to the
gaps between scores". The attention layers are the modules with all four
projections, so a vision tower's attention (Gemma 3, whose projections are
q/k/v/out) is left out. NB_DTYPE picks the weight dtype (default float16; Gemma 3
overflows in float16, and its matched-bit run used bfloat16). Usage (GPU):
    NB_MODEL=unsloth/Llama-3.2-3B NB_SAMPLES=8 python score_error_decomposition.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import exp_io
import numpy as np
import torch
import torch.nn.functional as F


def quantize_per_outchannel(W: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    return torch.clamp(torch.round(W / scale), -qmax - 1, qmax) * scale


def seq_intervals(per_layer: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Sequence bootstrap (added 2026-11-22) of const_share, centered_rel,
    attention_tv and the layer median of const_share. Each resample draws whole
    sequences, the same ones for every layer, and recomputes every statistic from
    the resampled sequences' sums, as the reported values are computed."""
    a = {
        k: np.array([x["seq"][k] for x in per_layer], dtype=np.float64)
        for k in per_layer[0]["seq"]
    }  # [layer, sequence]
    n_seq = a["err_den"].shape[1]
    idx = np.random.default_rng(seed).integers(0, n_seq, size=(n_boot, n_seq))
    s = {k: v[:, idx].sum(axis=2) for k, v in a.items()}  # [layer, resample]
    tot = {k: v.sum(axis=0) for k, v in s.items()}  # [resample]
    stats = {
        "const_share": tot["const_num"] / np.maximum(tot["err_den"], 1e-30),
        "centered_rel": np.sqrt(tot["cent_sq"] / np.maximum(tot["spread_sq"], 1e-30)),
        "attention_tv": tot["tv"] / np.maximum(tot["n_rows"], 1),
        "const_share_layer_median": np.median(
            s["const_num"] / np.maximum(s["err_den"], 1e-30), axis=0
        ),
    }
    out = {
        k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in stats.items()
    }
    out |= {"unit": "sequence", "resamples": n_boot, "seed": seed, "n_seqs": n_seq}
    return out


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "unsloth/Llama-3.2-3B")
    n_samples = int(os.environ.get("NB_SAMPLES", "8"))
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    bits_list = [int(b) for b in os.environ.get("NB_BITS", "4,3").split(",")]
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, os.environ.get("NB_DTYPE", "float16"))

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
    attns = [
        m
        for m in model.modules()
        if all(hasattr(m, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
    ]
    rope = sys.modules[type(attns[0]).__module__].apply_rotary_pos_emb
    a0 = attns[0]
    head_dim = a0.head_dim
    n_kv_groups = a0.num_key_value_groups
    scaling = a0.scaling
    dt = time.time() - t0
    print(f"[load] {model_name} {len(attns)} layers head_dim={head_dim} in {dt:.1f}s")

    captured = [[] for _ in attns]

    def make_hook(i):
        def hook(_mod, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            captured[i].append((h.detach(), (pe[0].detach(), pe[1].detach())))

        return hook

    hooks = [
        a.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i, a in enumerate(attns)
    ]
    seqs = exp_io.token_batch(tok, "test", n_samples, n_seq)
    with torch.no_grad():
        for ids in seqs:
            model(ids.to(device).unsqueeze(0))
    for h in hooks:
        h.remove()
    print(f"[data] {len(seqs)} sequences captured")

    def scores(attn, h, cos, sin, wq, wk):
        b, t, _ = h.shape
        q = F.linear(h, wq, attn.q_proj.bias).view(b, t, -1, head_dim).transpose(1, 2)
        k = F.linear(h, wk, attn.k_proj.bias).view(b, t, -1, head_dim).transpose(1, 2)
        if hasattr(attn, "q_norm"):
            q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = rope(q, k, cos, sin)
        k = k.repeat_interleave(n_kv_groups, dim=1)
        return (q.float() @ k.float().transpose(-1, -2)) * scaling

    results = {}
    for proj in ("Q", "K"):
        for bits in bits_list:
            acc = {
                "const_num": 0.0,
                "err_den": 0.0,
                "cent_sq": 0.0,
                "spread_sq": 0.0,
                "tv": 0.0,
                "n_rows": 0,
            }
            per_layer = []
            for li, attn in enumerate(attns):
                if thermal is not None:
                    torch.set_num_threads(thermal.get_threads())
                wq0, wk0 = attn.q_proj.weight.data, attn.k_proj.weight.data
                wq = (
                    quantize_per_outchannel(wq0.float(), bits).to(wq0.dtype)
                    if proj == "Q"
                    else wq0
                )
                wk = (
                    quantize_per_outchannel(wk0.float(), bits).to(wk0.dtype)
                    if proj == "K"
                    else wk0
                )
                lay = {
                    "const_num": 0.0,
                    "err_den": 0.0,
                    "cent_sq": 0.0,
                    "spread_sq": 0.0,
                    "tv": 0.0,
                    "n_rows": 0,
                }
                seq = {k: [] for k in lay}  # the same terms, one per sequence
                with torch.no_grad():
                    for h, (cos, sin) in captured[li]:
                        s0 = scores(attn, h, cos, sin, wq0, wk0)[0]
                        s1 = scores(attn, h, cos, sin, wq, wk)[0]
                        t = s0.shape[-1]
                        mask = torch.ones(
                            t, t, dtype=torch.bool, device=s0.device
                        ).tril()
                        n_t = mask.sum(-1).float()  # support size per row
                        ds = (s1 - s0).masked_fill(~mask, 0.0)
                        c = ds.sum(-1) / n_t
                        cent = (ds - c[..., None]).masked_fill(~mask, 0.0)
                        mu0 = s0.masked_fill(~mask, 0.0).sum(-1) / n_t
                        dev0 = (s0 - mu0[..., None]).masked_fill(~mask, 0.0)
                        p0 = torch.softmax(s0.masked_fill(~mask, float("-inf")), -1)
                        p1 = torch.softmax(s1.masked_fill(~mask, float("-inf")), -1)
                        tv = 0.5 * (p1 - p0).abs().sum(-1)
                        rows = slice(1, None)  # row 0 has one entry, nothing to compare
                        one = {
                            "const_num": float((n_t[rows] * c[:, rows] ** 2).sum()),
                            "err_den": float((ds[:, rows] ** 2).sum()),
                            "cent_sq": float((cent[:, rows] ** 2).sum()),
                            "spread_sq": float((dev0[:, rows] ** 2).sum()),
                            "tv": float(tv[:, rows].sum()),
                            "n_rows": int(tv[:, rows].numel()),
                        }
                        for k in lay:
                            lay[k] += one[k]
                            seq[k].append(one[k])
                for k in acc:
                    acc[k] += lay[k]
                per_layer.append(
                    {
                        "layer": li,
                        "const_share": lay["const_num"] / max(lay["err_den"], 1e-30),
                        "centered_rel": (lay["cent_sq"] / max(lay["spread_sq"], 1e-30))
                        ** 0.5,
                        "attention_tv": lay["tv"] / max(lay["n_rows"], 1),
                        "seq": seq,
                    }
                )
            res = {
                "const_share": acc["const_num"] / max(acc["err_den"], 1e-30),
                "centered_rel": (acc["cent_sq"] / max(acc["spread_sq"], 1e-30)) ** 0.5,
                "attention_tv": acc["tv"] / max(acc["n_rows"], 1),
                "const_share_layer_median": float(
                    np.median([x["const_share"] for x in per_layer])
                ),
                "intervals": seq_intervals(per_layer),
                "per_layer": per_layer,
            }
            results[f"{proj}{bits}"] = res
            print(
                f"  {proj} {bits}-bit const_share={res['const_share']:.3f}"
                f" centered_rel={res['centered_rel']:.4f}"
                f" attention_tv={res['attention_tv']:.4f}"
            )

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"score_error_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "n_seqs": len(seqs),
                "window": n_seq,
                "dtype": str(dtype).replace("torch.", ""),
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "bits": bits_list,
                "local": "each layer perturbed alone, on its own unperturbed inputs",
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
