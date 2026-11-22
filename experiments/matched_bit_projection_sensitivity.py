# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Matched-bit projection sensitivity: de-confounding "Q/K are more sensitive."

Rababah, Akcora & Leung (arXiv:2607.08734, "The Illusion of Equivalency") report
that the query/key projections drift more under quantization than value/output.
But their evidence is llama.cpp K-quantization, whose schemes *assign V and O
more bits than Q/K* (their own appendix: Q2_K quantizes Q,K at 2-bit but V at
4-bit, O at 3-bit). So "Q/K drift more" is confounded with "Q/K got fewer bits."

This experiment removes the confound: quantize W^Q, W^K, W^V, W^O with the
**same** uniform per-output-channel b-bit quantizer, and measure drift two ways:

  * WEIGHT-SPACE (the paper's lens): cosine to the original vec(W), KL of weight
    histograms, |Δkurtosis|, relative Frobenius error.
  * FUNCTIONAL (the mechanism): perturb *only* projection L in every layer at b
    bits, run the model, and measure how much the model's OUTPUT distribution
    moves -- mean KL(p_base || p_quant) over positions, plus top-1 token flip
    rate (the token-level analogue of the paper's answer-flipping).

The hypothesis under test (softmax is an angle-amplifier / the (A2) score path):
at *matched bits*, weight-space drift is roughly comparable across projections,
while the FUNCTIONAL impact of Q/K perturbation exceeds that of V/O -- because
Q/K feed softmax(Q.K^T), which amplifies their error, whereas V/O flow through a
linear, averaging path. That functional asymmetry is invisible to a weight-only
analysis and is the clean, de-confounded version of the paper's finding.

Usage (run on GPU; a small model on CPU also works, just slower):
    NB_MODEL=meta-llama/Llama-3.2-3B NB_BITS=8,4,3,2 NB_SEQ=256 NB_SAMPLES=16 \
        python experiments/matched_bit_projection_sensitivity.py

Per-sequence output (added 2026-11-22). Each record also holds, for every
projection and bit width, the mean KL and the top-1 flip rate of each sequence
(out_kl_seq, out_top1_flip_seq; positions per sequence in seq_positions), and,
for each bit width, a paired sequence bootstrap of the Q/K-over-V/O ratio of
mean out_kl ("intervals"): one set of resampled sequences serves all four
projections, and each projection's KL is position-weighted, as out_kl is.
The perturbed model's KL is now taken one sequence at a time (kl_top1), so
the run fits a 24 GB GPU. out_kl is the same quantity as before, up to the
summation order inside each position's sum over the vocabulary. On an NRP
pod the model and the inputs come from staged copies (exp_io.py, nrp_tmlr/):
NB_STRIPES and NB_MODEL_DIR for the weights, NB_TOKENS for the sequences,
pre-tokenized on Atlas with this file's own chunking (split "matched_bit").

Honest scope: fake-quantization (quantize->dequantize weights, fp inference),
per-output-channel symmetric absmax (a clean matched control, not any vendor
kernel), directional on the model/'sample sizes chosen. The point is the
*relative* Q/K-vs-V/O comparison at matched bits, not absolute degradation.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
SHORT = {"q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O"}


# ── matched quantizer: per-output-channel symmetric absmax, same for all L ──
def quantize_per_outchannel(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform symmetric per-row (output-channel) absmax quantize->dequantize.

    Identical scheme and bit-width for every projection -> the de-confounding
    control the paper lacks. W is [out_features, in_features]."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    codes = torch.clamp(torch.round(W / scale), -qmax - 1, qmax)
    return codes * scale


# ── weight-space drift metrics (the paper's lens) ──────────────────────────
def _hist_kl(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = ha.astype(np.float64) + 1e-9
    pb = hb.astype(np.float64) + 1e-9
    pa /= pa.sum()
    pb /= pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))


def _kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def weight_drift(W: torch.Tensor, Wq: torch.Tensor) -> dict:
    a = W.detach().float().cpu().numpy().ravel()
    b = Wq.detach().float().cpu().numpy().ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    relfro = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    return {
        "cosine": cos,
        "hist_kl": _hist_kl(a, b),
        "abs_dkurtosis": abs(_kurtosis(b) - _kurtosis(a)),
        "rel_fro": relfro,
    }


# ── model helpers ──────────────────────────────────────────────────────────
def iter_attn(model):
    """Yield self-attention modules that expose q/k/v/o_proj (Llama/Qwen/Mistral)."""
    for mod in model.modules():
        if all(hasattr(mod, p) for p in PROJ):
            yield mod


def output_logprobs(model, batch, device) -> torch.Tensor:
    """Stacked log-softmax next-token distributions over all positions.

    Kept **on-device**: pulling [positions x vocab] to CPU and doing the KL there
    leaves the GPU idle for minutes on large-vocab models (152k for Qwen) — which
    both wastes time and trips utilization watchdogs. We hold the base distribution
    on the GPU and do the divergence on-GPU, pulling only scalars."""
    outs = []
    with torch.no_grad():
        for ids in batch:
            ids = ids.to(device)
            logits = model(ids.unsqueeze(0)).logits[0]  # [seq, vocab]
            outs.append(torch.log_softmax(logits.float(), dim=-1))  # stay on GPU
    return torch.cat(outs, dim=0)  # [total_pos, vocab] on device


def kl_top1(
    model, batch, base_lp: torch.Tensor, device
) -> tuple[float, float, list[float], list[float]]:
    """mean KL(base || var) and top-1 argmax flip rate over positions (on-GPU),
    and the same two per sequence.

    The perturbed model's distribution is taken one sequence at a time and
    compared with that sequence's slice of base_lp, so nothing of size
    [all positions x vocab] is held beside base_lp. The whole-batch version
    held five such tensors at once (about 21 GB for Gemma-3's 262k vocabulary
    over 16 x 256 positions), which an A10 cannot hold beside the model. The
    quantity is the same; summation order inside each row's sum over the
    vocabulary may differ in the last bit."""
    kls, flips = [], []
    off = 0
    with torch.no_grad():
        for ids in batch:
            logits = model(ids.to(device).unsqueeze(0)).logits[0]
            var = torch.log_softmax(logits.float(), dim=-1)
            b = base_lp[off : off + var.shape[0]]
            off += var.shape[0]
            kls.append((b.exp() * (b - var)).sum(dim=-1))
            flips.append((b.argmax(-1) != var.argmax(-1)).float())
            del logits, var
        assert off == base_lp.shape[0], (off, base_lp.shape)
        kl, fl = torch.cat(kls), torch.cat(flips)
        seq_kl = [float(x.mean().item()) for x in kls]
        seq_flip = [float(x.mean().item()) for x in flips]
    return float(kl.mean().item()), float(fl.mean().item()), seq_kl, seq_flip


def ratio_intervals(
    results: list[dict], lengths: list[int], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired sequence bootstrap of mean(out_kl Q, K) / mean(out_kl V, O) per bit
    width. One set of resampled sequences serves all four projections, and each
    projection's KL over a resample is position-weighted, as out_kl is."""
    n = np.asarray(lengths, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(n), size=(n_boot, len(n)))
    w = n[idx]

    def kl_of(v, ix=None):
        v = np.asarray(v, dtype=np.float64)
        if ix is None:
            return float((v * n).sum() / n.sum())
        return (v[ix] * w).sum(axis=1) / w.sum(axis=1)

    out = {}
    for b in sorted({r["bits"] for r in results}, reverse=True):
        seq = {r["proj"]: r["out_kl_seq"] for r in results if r["bits"] == b}
        mean = {r["proj"]: r["out_kl"] for r in results if r["bits"] == b}
        boot = (kl_of(seq["Q"], idx) + kl_of(seq["K"], idx)) / (
            kl_of(seq["V"], idx) + kl_of(seq["O"], idx)
        )
        per = {
            p: {
                "out_kl_from_seq": kl_of(seq[p]),
                "ci95": [
                    float(x) for x in np.percentile(kl_of(seq[p], idx), [2.5, 97.5])
                ],
            }
            for p in "QKVO"
        }
        out[str(b)] = {
            "qk_over_vo": (mean["Q"] + mean["K"]) / (mean["V"] + mean["O"]),
            "qk_over_vo_from_seq": (kl_of(seq["Q"]) + kl_of(seq["K"]))
            / (kl_of(seq["V"]) + kl_of(seq["O"])),
            "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "per_projection": per,
        }
    return {
        "unit": "sequence",
        "paired": True,
        "resamples": n_boot,
        "seed": seed,
        "statistic": "mean(out_kl Q, K) / mean(out_kl V, O)",
        "by_bits": out,
    }


def eval_batch(tok, n_samples: int, n_seq: int) -> list[torch.Tensor]:
    """The run's sequences: this file's WikiText-2 chunks through the tokenizer,
    or the same sequences pre-tokenized (NB_TOKENS, split "matched_bit")."""
    path = os.environ.get("NB_TOKENS")
    if path:
        d = json.load(open(path))
        if d["window"] != n_seq:
            raise SystemExit(f"{path}: window {d['window']}, run asks {n_seq}")
        chunks = d["splits"]["matched_bit"]
        if len(chunks) < n_samples:
            raise SystemExit(f"{path}: {len(chunks)} chunks, run asks {n_samples}")
        ids_list = [torch.tensor(c, dtype=torch.long) for c in chunks[:n_samples]]
    else:
        ids_list = [
            tok(t, return_tensors="pt", truncation=True, max_length=n_seq).input_ids[0]
            for t in _load_texts(n_samples)
        ]
    return [ids for ids in ids_list if ids.numel() >= 16]


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    bits_list = [int(b) for b in os.environ.get("NB_BITS", "8,4,3,2").split(",")]
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    n_samples = int(os.environ.get("NB_SAMPLES", "16"))
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Gemma & other bf16-native models overflow in fp16 -> NaN logits; allow override
    _dt = os.environ.get("NB_DTYPE", "float16" if device == "cuda" else "float32")
    dtype = getattr(torch, _dt)
    # Weight-space metrics run in numpy on the CPU for every layer, and do not
    # depend on the sequences. NB_WEIGHT_METRICS=0 skips them (w_* fields null),
    # for a one-CPU GPU pod whose GPU would otherwise sit idle while they run.
    weight_metrics = os.environ.get("NB_WEIGHT_METRICS", "1") != "0"
    torch.manual_seed(0)

    from exp_io import model_and_tokenizer

    t0 = time.time()
    # from_pretrained then .to(device), or the staged stripes on an NRP pod
    model, tok = model_and_tokenizer(model_name, dtype, device)
    cfg = model.config

    def cget(key, default=None):
        # some configs (VLMs like Gemma-3) nest text params under text_config
        if hasattr(cfg, key):
            return getattr(cfg, key)
        tc = getattr(cfg, "text_config", None)
        if tc is not None and hasattr(tc, key):
            return getattr(tc, key)
        return default

    n_layers = cget("num_hidden_layers")
    n_heads = cget("num_attention_heads")
    print(
        f"[load] {model_name} in {time.time()-t0:.1f}s  device={device} dtype={dtype}"
    )
    print(
        f"[cfg] layers={n_layers} hidden={cget('hidden_size')} "
        f"heads={n_heads} kv_heads={cget('num_key_value_heads', n_heads)}"
    )

    # ── eval batch: wikitext-2 sample, else embedded text ──
    batch = eval_batch(tok, n_samples, n_seq)
    lengths = [int(x.numel()) for x in batch]
    print(f"[data] {len(batch)} sequences, ~{sum(x.numel() for x in batch)} tokens")

    attn = list(iter_attn(model))
    print(f"[model] {len(attn)} attention blocks with q/k/v/o_proj")

    base_lp = output_logprobs(model, batch, device)
    print(f"[base] logprobs {tuple(base_lp.shape)}")

    results = []
    for bits in bits_list:
        for proj in PROJ:
            # snapshot originals, quantize this projection in every layer
            saved = []
            wmetrics = []
            for mod in attn:
                W = getattr(mod, proj).weight
                saved.append(W.data.clone())
                Wq = quantize_per_outchannel(W.data, bits)
                if weight_metrics:
                    wmetrics.append(weight_drift(W.data, Wq))
                W.data.copy_(Wq)
            # functional: output-distribution move from perturbing only this proj
            okl, oflip, seq_kl, seq_flip = kl_top1(model, batch, base_lp, device)
            # restore
            for mod, w0 in zip(attn, saved):
                getattr(mod, proj).weight.data.copy_(w0)

            if weight_metrics:
                wm = {k: float(np.mean([m[k] for m in wmetrics])) for k in wmetrics[0]}
            else:
                wm = dict.fromkeys(("cosine", "hist_kl", "abs_dkurtosis", "rel_fro"))
            rec = {
                "bits": bits,
                "proj": SHORT[proj],
                "w_cosine": wm["cosine"],
                "w_hist_kl": wm["hist_kl"],
                "w_abs_dkurtosis": wm["abs_dkurtosis"],
                "w_rel_fro": wm["rel_fro"],
                "out_kl": okl,
                "out_top1_flip": oflip,
                # amplification: functional move per unit weight drift
                "amplification": (
                    okl / (wm["rel_fro"] + 1e-9) if weight_metrics else None
                ),
                "out_kl_seq": seq_kl,
                "out_top1_flip_seq": seq_flip,
            }
            results.append(rec)
            if weight_metrics:
                print(
                    f"  b{bits} {SHORT[proj]}: w_cos={wm['cosine']:.4f} "
                    f"w_relfro={wm['rel_fro']:.4f} | out_kl={okl:.4f} "
                    f"top1_flip={oflip:.4f} amp={rec['amplification']:.3f}"
                )
            else:
                print(
                    f"  b{bits} {SHORT[proj]}: out_kl={okl:.4f} top1_flip={oflip:.4f}"
                )

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"matched_bit_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "config": {
                    "layers": n_layers,
                    "hidden": cget("hidden_size"),
                    "heads": n_heads,
                    "kv_heads": cget("num_key_value_heads", n_heads),
                },
                "n_seq": len(batch),
                "seq_len": n_seq,
                "seq_positions": lengths,
                "dtype": _dt,
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "versions": {
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                },
                "inputs": "NB_TOKENS" if os.environ.get("NB_TOKENS") else "datasets",
                "weights": "NB_STRIPES" if os.environ.get("NB_STRIPES") else "hub",
                "weight_metrics": weight_metrics,
                "results": results,
                "intervals": ratio_intervals(results, lengths),
            },
            f,
            indent=2,
        )
    print(f"[saved] {path}")
    if weight_metrics:
        _print_summary(tag, results)
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
                out.append(buf.strip())
                buf = ""
            if len(out) >= n:
                break
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[data] wikitext unavailable ({type(e).__name__}); embedded sample")
    seed = (
        "The quantization of large language models reduces memory and latency, "
        "but its effect on behavior is subtle. Attention routes information by "
        "comparing queries and keys; the softmax over their inner products decides "
        "what each token attends to. Small perturbations in the score pathway can "
        "reshape those decisions even when aggregate accuracy looks unchanged. "
    )
    return [seed * 3 for _ in range(n)]


def _print_summary(tag: str, results: list[dict]) -> None:
    print(f"\n=== matched-bit projection sensitivity: {tag} ===")
    print("bit  proj   w_cos   w_relfro   out_kl   top1_flip   amp(out_kl/relfro)")
    for r in results:
        print(
            f"{r['bits']:>2}   {r['proj']:>2}   {r['w_cosine']:.4f}   "
            f"{r['w_rel_fro']:.4f}    {r['out_kl']:.4f}    {r['out_top1_flip']:.4f}     "
            f"{r['amplification']:.3f}"
        )
    # QK vs VO functional ratio per bit
    print("\nQ/K vs V/O functional impact (mean out_kl), by bit:")
    bits = sorted({r["bits"] for r in results}, reverse=True)
    for b in bits:
        qk = np.mean(
            [r["out_kl"] for r in results if r["bits"] == b and r["proj"] in "QK"]
        )
        vo = np.mean(
            [r["out_kl"] for r in results if r["bits"] == b and r["proj"] in "VO"]
        )
        wqk = np.mean(
            [r["w_rel_fro"] for r in results if r["bits"] == b and r["proj"] in "QK"]
        )
        wvo = np.mean(
            [r["w_rel_fro"] for r in results if r["bits"] == b and r["proj"] in "VO"]
        )
        ratio = qk / (vo + 1e-12)
        wratio = wqk / (wvo + 1e-12)
        print(
            f"  b{b}: out_kl QK/VO = {ratio:6.2f}x   (weight relfro QK/VO = {wratio:4.2f}x)"
        )


if __name__ == "__main__":
    sys.exit(main())
