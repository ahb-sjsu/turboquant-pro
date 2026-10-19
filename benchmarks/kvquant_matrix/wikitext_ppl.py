#!/usr/bin/env python3
"""WikiText-2 perplexity under KV-cache quantization, using the EXACT same quant machinery
as ``tq_paper_lb_shard.py`` (imported, so the DynamicCache patch / rope hook are installed
from the same env vars). Single sliding-window forward per chunk: each query attends to a
KV cache whose settled region [0 : S-HOT] is quantized (post-RoPE via the cache patch, or
pre-RoPE via the rope hook), exactly as in the LongBench runs.

Env: same as the LB harness (MODEL, CODEBOOK, KEY_BITS, ..., PREROPE, NOQUANT) plus
  SEQLEN (default 2048), MAXCHUNKS (default 0 = all), CHUNKS_OUT (optional path: one JSON
line per chunk with its summed NLL, for paired per-chunk comparisons between arms).
  KL_REF=1 (Part V): before each chunk's forward, the same chunk runs once more with
quantization off (the fp16 arm's forward, in the same process), and the chunk's record
gains ``kl``, the summed KL(fp16 || arm) of the next-token distributions over the same
SEQLEN - 1 positions as the NLL, and ``fit_s``, the seconds spent fitting the key coding.
Prints: ``PPL <tag> <value>``.
"""
import json
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
KL_REF = int(os.environ.get("KL_REF", "0"))
KL_BLOCK = 256  # positions per block: one block's float32 log-softmax is ~150 MB at 152k vocab


def kl_sum(ref_logits: torch.Tensor, logits: torch.Tensor, n: int) -> float:
    """Sum over positions 0..n-1 of KL(softmax(ref) || softmax(arm)), in float32 blocks.
    Logits are (1, T, V); position t predicts token t+1, as the loss does."""
    total = 0.0
    for a in range(0, n, KL_BLOCK):
        b = min(n, a + KL_BLOCK)
        lp = torch.log_softmax(ref_logits[0, a:b].float(), -1)
        lq = torch.log_softmax(logits[0, a:b].float(), -1)
        total += float((lp.exp() * (lp - lq)).sum())
    return total


def reference_logits(model, chunk) -> torch.Tensor:
    """The chunk's logits with quantization off: the cache patch and the rope hook read
    H.NOQUANT at call time, so this is the fp16 arm's forward. Restored afterwards
    whatever happens."""
    saved = H.NOQUANT
    H.NOQUANT = 1
    try:
        with torch.no_grad():
            return model(chunk, use_cache=True).logits
    finally:
        H.NOQUANT = saved


NL = chr(10)  # a newline, spelled so no shell can mangle it


def _read_chunks(path):
    """Chunk records already written; a torn last line is cut off."""
    good, rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                break
            good.append(line if line.endswith(NL) else line + NL)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(good)
    return rows


def main():
    with H.FR.reads():
        model = AutoModelForCausalLM.from_pretrained(
            H.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa",
            low_cpu_mem_usage=True, trust_remote_code=True, **H.load_device(),
        ).cuda().eval()
    try:
        tok = AutoTokenizer.from_pretrained(H.MODEL, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(H.MODEL, use_fast=False, trust_remote_code=True)

    H.ensure_basis_calibration(model, tok)  # no-op unless BASIS_FIT=calib
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids[0]

    nll, ntok = 0.0, 0
    nchunks = 0
    chunks_out = os.environ.get("CHUNKS_OUT", "")
    # RESUME=1 keeps the chunks already scored (each chunk is an independent forward).
    done = {}
    if chunks_out and H.RESUME and os.path.exists(chunks_out):
        done = {o["start"]: o for o in _read_chunks(chunks_out)}
    fo = open(chunks_out, "a" if done else "w") if chunks_out else None
    for i in range(0, enc.shape[0] - SEQLEN, SEQLEN):
        if i in done:
            nll += done[i]["nll"]
            ntok += done[i]["tokens"]
            nchunks += 1
            if MAXCHUNKS and nchunks >= MAXCHUNKS:
                break
            continue
        chunk = enc[i : i + SEQLEN].unsqueeze(0).cuda()
        ref = reference_logits(model, chunk) if KL_REF and not H.NOQUANT else None
        if H.EA.active() and not H.NOQUANT:  # pass 1: what Expected Attention keeps
            H.EA.set_prompt(SEQLEN, H.EA.ref_key_bits("ppl", i), H.HOT,
                            H.EA.value_bits(H.VB, H.ea_head_dim(model), H.VAL_GROUP))
            H.EA.select(model, chunk)
        H._ACCT.clear()
        H.KC.FIT_S = 0.0
        with torch.no_grad():
            out = model(chunk, labels=chunk, use_cache=True)
        # out.loss is mean CE over (SEQLEN-1) shifted tokens; accumulate token-weighted.
        chunk_nll = out.loss.float().item() * (SEQLEN - 1)
        nll += chunk_nll
        ntok += SEQLEN - 1
        if fo:
            rec = {"chunk": nchunks, "start": i, "nll": chunk_nll,
                   "tokens": SEQLEN - 1, "key_bits": H._bits_summary()}
            if H.EA.active() and not H.NOQUANT:
                rec["kv_bits"] = H.EA.kv_bits((rec["key_bits"] or {}).get("total"))
            if ref is not None:
                rec["kl"] = kl_sum(ref, out.logits, SEQLEN - 1)
                rec["fit_s"] = round(H.KC.FIT_S, 4)
            fo.write(json.dumps(rec) + NL)
            fo.flush()
        if H.EA.active():
            H.EA.done()
        del ref, out
        nchunks += 1
        if MAXCHUNKS and nchunks >= MAXCHUNKS:
            break
    ppl = math.exp(nll / ntok)
    print(f"PPL {TAG} {ppl:.4f}  (chunks={nchunks} seqlen={SEQLEN} "
          f"codebook={H.CODEBOOK} kb={H.KB} noquant={H.NOQUANT} prerope={H.PREROPE} "
          f"key_coding={H.KC.config() if H.KC.active() else '-'})", flush=True)


if __name__ == "__main__":
    main()
