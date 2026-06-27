#!/usr/bin/env python3
"""WikiText-2 perplexity under KV-cache quantization, using the EXACT same quant machinery
as ``tq_paper_lb_shard.py`` (imported, so the DynamicCache patch / rope hook are installed
from the same env vars). Single sliding-window forward per chunk: each query attends to a
KV cache whose settled region [0 : S-HOT] is quantized (post-RoPE via the cache patch, or
pre-RoPE via the rope hook), exactly as in the LongBench runs.

Env: same as the LB harness (MODEL, CODEBOOK, KEY_BITS, ..., PREROPE, NOQUANT) plus
  SEQLEN (default 2048), MAXCHUNKS (default 0 = all). Prints: ``PPL <tag> <value>``.
"""
import math
import os

# The LB harness reads SHARD_ID/NUM_SHARDS at import; ppl is single-process, so default them.
os.environ.setdefault("SHARD_ID", "0")
os.environ.setdefault("NUM_SHARDS", "1")

import torch

import tq_paper_lb_shard as H  # installs cache patch + reads env (CODEBOOK/KEY_BITS/...)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MAXCHUNKS = int(os.environ.get("MAXCHUNKS", "0"))
TAG = os.environ.get("TAG", "ppl")


def main():
    model = AutoModelForCausalLM.from_pretrained(
        H.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).cuda().eval()
    try:
        tok = AutoTokenizer.from_pretrained(H.MODEL, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(H.MODEL, use_fast=False, trust_remote_code=True)

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids[0]

    nll, ntok = 0.0, 0
    nchunks = 0
    for i in range(0, enc.shape[0] - SEQLEN, SEQLEN):
        chunk = enc[i : i + SEQLEN].unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(chunk, labels=chunk, use_cache=True)
        # out.loss is mean CE over (SEQLEN-1) shifted tokens; accumulate token-weighted.
        nll += out.loss.float().item() * (SEQLEN - 1)
        ntok += SEQLEN - 1
        nchunks += 1
        if MAXCHUNKS and nchunks >= MAXCHUNKS:
            break
    ppl = math.exp(nll / ntok)
    print(f"PPL {TAG} {ppl:.4f}  (chunks={nchunks} seqlen={SEQLEN} "
          f"codebook={H.CODEBOOK} kb={H.KB} noquant={H.NOQUANT} prerope={H.PREROPE})", flush=True)


if __name__ == "__main__":
    main()
