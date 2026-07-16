# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""End-to-end validation of the SHIPPED rope_aware_k path (PR #120).

Runs turboquant_pro.model_compress.quantize_weights (the merged module,
imported standalone) on Llama-3.2-3B and checks the shipped default against
the k_wavelength_probe numbers it was derived from.

Pre-registered predictions (default protect_frac=0.125 selects exactly the
probe's bucket 7):
  K-only @3-bit, rope_aware_k=True  -> out_kl ~ 0.099, flip ~ 0.153
  K-only @3-bit, rope_aware_k=False -> out_kl ~ 0.741, flip ~ 0.374
Plus the deployment setting: whole-model (FFN+attn) @4-bit, on vs off.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="[mc] %(message)s")
try:
    from turboquant_pro import model_compress as mc
except ImportError:  # standalone on remote hosts: model_compress.py alongside
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import model_compress as mc


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


def output_logprobs(model, batch, device):
    outs = []
    with torch.no_grad():
        for ids in batch:
            logits = model(ids.to(device).unsqueeze(0)).logits[0]
            outs.append(torch.log_softmax(logits.float(), dim=-1))
    return torch.cat(outs, dim=0)


def kl_top1(base_lp, var_lp):
    with torch.no_grad():
        p = base_lp.exp()
        kl = (p * (base_lp - var_lp)).sum(dim=-1)
        flip = (base_lp.argmax(-1) != var_lp.argmax(-1)).float().mean()
    return float(kl.mean().item()), float(flip.item())


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "unsloth/Llama-3.2-3B")
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    n_samples = int(os.environ.get("NB_SAMPLES", "32"))
    device = "cuda"
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device).eval()
    print(f"[load] {model_name} in {time.time()-t0:.1f}s", flush=True)

    texts = _load_texts(n_samples)
    batch = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=n_seq).input_ids[
            0
        ]
        if ids.numel() >= 16:
            batch.append(ids)
    print(f"[data] {len(batch)} sequences", flush=True)

    base_lp = output_logprobs(model, batch, device)
    print(f"[base] logprobs {tuple(base_lp.shape)}", flush=True)

    results = {}

    def run_variant(tag, bits, k_only, **kw):
        m2 = copy.deepcopy(model)
        comp = mc.ModelCompressor(m2)
        if k_only:
            comp._ffn_layers = []
            comp._attn_layers = [(n, m) for n, m in comp._attn_layers if "k_proj" in n]
        comp.quantize_weights(bits=bits, inplace=True, **kw)
        kl, flip = kl_top1(base_lp, output_logprobs(m2, batch, device))
        del m2, comp
        torch.cuda.empty_cache()
        results[tag] = {"out_kl": kl, "top1_flip": flip}
        print(f"  {tag}: out_kl={kl:.4f} top1_flip={flip:.4f}", flush=True)

    # the measured claim, via the shipped code path
    run_variant("k_only_b3_protected", 3, True, rope_aware_k=True)
    run_variant("k_only_b3_unprotected", 3, True, rope_aware_k=False)
    # mixed precision variant of the fix
    run_variant("k_only_b3_protect8bit", 3, True, rope_aware_k=True, k_protect_bits=8)
    # deployment setting: everything quantized at 4-bit
    run_variant("full_b4_protected", 4, False, rope_aware_k=True)
    run_variant("full_b4_unprotected", 4, False, rope_aware_k=False)

    r = results
    rescue = 1 - r["k_only_b3_protected"]["out_kl"] / max(
        r["k_only_b3_unprotected"]["out_kl"], 1e-9
    )
    print(
        f"[verdict] shipped default recovers {rescue:.1%} of K-only 3-bit "
        f"damage (predicted ~87%)",
        flush=True,
    )
    tag = model_name.split("/")[-1]
    out = os.environ.get(
        "NB_OUT",
        os.path.join(
            "experiments/results_matched_bit",
            f"validate_rope_aware_k_{tag}.json",
        ),
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2)
    print(f"[saved] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
