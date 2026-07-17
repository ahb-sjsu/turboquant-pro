# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Real Mamba SSM: decay quantization in the linear vs native A_log basis.

Phase-7 SSM/recurrence validation. The operator-sensitivity thesis says a
state-space model's continuous decay ``A = -exp(A_log)`` must be quantized in its
native log-time-constant basis (``A_log``), not on a linear grid on ``A`` — the
SSM analog of NF4-for-keys, because a fixed error in a slow channel (``a -> 1``)
is amplified by ``1/(1-a)`` and compounds over the sequence.

This runs the real test on a real Mamba: for each SSM layer we fake-quantize the
model's own ``A_log`` two ways at the same bit width — a **linear** grid on the
continuous decay ``A`` (the naive baseline) and a grid in the **native A_log /
log-time-constant** basis (what ``operator_sensitivity.quantize_decay(basis=
'log_tau')`` implements) — put the quantized decay back into the model, and
measure WikiText-2 perplexity. The sensitivity summary is computed by the shipped
``state_decay_sensitivity`` on the model's real decays.

    python benchmarks/validate_mamba_decay.py \
        --model state-spaces/mamba-790m-hf --bits 3 --out results_mamba_decay.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("TQP_REPO", os.getcwd()))
from turboquant_pro.operator_sensitivity import state_decay_sensitivity  # noqa: E402


def _quant_uniform(x: np.ndarray, bits: int) -> np.ndarray:
    """Per-tensor uniform fake-quant of ``x`` to ``bits`` levels."""
    lo, hi = float(x.min()), float(x.max())
    step = (hi - lo) / max(2**bits - 1, 1)
    return np.round((x - lo) / max(step, 1e-30)) * step + lo


def _perplexity(model, ids, device, window=2048, max_windows=40) -> float:
    """Corpus perplexity over non-overlapping windows (same chunking each call,
    so relative changes between quantization variants are exact). Returns
    ``inf`` if any window diverges (a collapsed configuration)."""
    nll = torch.zeros((), dtype=torch.float64)
    ntok = 0
    seen = 0
    for i in range(0, ids.size(1) - 1, window):
        chunk = ids[:, i : i + window].to(device)
        if chunk.size(1) < 2:
            break
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss.double().cpu()
        if not torch.isfinite(loss):
            return float("inf")
        toks = chunk.size(1) - 1
        nll += loss * toks
        ntok += toks
        seen += 1
        if seen >= max_windows:
            break
    return float(torch.exp(nll / max(ntok, 1)))


def _layers(model):
    """The list of SSM mixer modules exposing ``A_log`` (HF Mamba / Mamba2)."""
    backbone = getattr(model, "backbone", None) or model.model
    return [layer.mixer for layer in backbone.layers if hasattr(layer.mixer, "A_log")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real Mamba decay-basis validation.")
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=40)
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="float32 avoids fp16 overflow so a collapsed variant reports a finite "
        "(huge) perplexity rather than NaN",
    )
    ap.add_argument("--out", default="results_mamba_decay.json")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} on {device} ({args.dtype}) ...", flush=True)
    model = (
        MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.model)

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    print(f"wikitext-2 test: {ids.size(1)} tokens", flush=True)

    mixers = _layers(model)
    orig = [m.A_log.detach().clone() for m in mixers]
    print(f"{len(mixers)} SSM layers, A_log shape {tuple(orig[0].shape)}", flush=True)

    # Sensitivity summary on the real decays (nominal discrete decay a=exp(A), A<0).
    all_alog = np.concatenate([o.float().cpu().numpy().ravel() for o in orig])
    a_nom = np.exp(-np.exp(all_alog))
    sens = state_decay_sensitivity(a_nom, seq_len=args.window).as_dict()

    base_ppl = _perplexity(model, ids, device, args.window, args.max_windows)
    print(f"baseline ppl {base_ppl:.4f}", flush=True)

    results = {
        "model": args.model,
        "bits": args.bits,
        "dataset": "wikitext-2-raw-v1 (test)",
        "n_layers": len(mixers),
        "baseline_ppl": base_ppl,
        "sensitivity": sens,
    }

    for basis in ("linear", "log_tau"):
        for m, o in zip(mixers, orig):
            alog = o.float().cpu().numpy()
            if basis == "log_tau":
                # Uniform grid on A_log == the native log-time-constant basis.
                alog_q = _quant_uniform(alog, args.bits)
            else:
                # Linear grid on the continuous decay A = -exp(A_log).
                a = -np.exp(alog)
                a_q = _quant_uniform(a, args.bits)
                alog_q = np.log(np.maximum(-a_q, 1e-9))
            m.A_log.data.copy_(
                torch.tensor(alog_q, dtype=m.A_log.dtype, device=m.A_log.device)
            )
        ppl = _perplexity(model, ids, device, args.window, args.max_windows)
        results[f"ppl_{basis}"] = ppl
        print(f"{basis} {args.bits}-bit ppl {ppl:.4f}", flush=True)
        for m, o in zip(mixers, orig):  # restore
            m.A_log.data.copy_(o)

    lin, log = results["ppl_linear"], results["ppl_log_tau"]
    results["linear_collapsed"] = not np.isfinite(lin)
    results["ppl_ratio_linear_over_log"] = (
        None if not np.isfinite(lin) else lin / max(log, 1e-30)
    )
    # JSON-safe: a diverged (collapsed) perplexity serializes as null + a flag.
    for key in ("ppl_linear", "ppl_log_tau", "baseline_ppl"):
        if not np.isfinite(results[key]):
            results[key] = None
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
