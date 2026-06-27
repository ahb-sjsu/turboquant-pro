#!/usr/bin/env python3
"""Enhanced TurboQuant KV-cache LongBench shard runner (calibration-free).

A torch ``DynamicCache`` that quantizes the KV cache during real ``generate()``,
adding the three ingredients that make KVQuant win -- but **without** Fisher
calibration:

  * group-wise per-channel **keys** (G=32) with a fp16 **residual hot window**,
  * optional **NF4 non-uniform** levels (calibration-free normal-float),
  * optional **1% dense-sparse fp16 outliers** (per-channel top-frac kept fp16) --
    fixes the attention-sink/outlier-channel open item in RESULTS_longbench.md,
  * optional **attention-sink** (first-N tokens kept fp16),
  * per-token uniform **values** with the same residual window.

Config via env so one script ablates many variants. Sharded by ``SHARD_ID`` /
``NUM_SHARDS`` (round-robin over samples) so N GPUs run in parallel; writes
``/root/out_<TAG>/<dataset>.<shard>.jsonl`` for the aggregator.
"""

import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "NousResearch/Llama-2-7b-chat-hf"
LBROOT = "/root/LongBench/LongBench"
DATADIR = "/root/lb_data/data"
MAXLEN = 3500
DATASETS = ["trec", "triviaqa", "qasper"]

KB = int(os.environ.get("KEY_BITS", "4"))
VB = int(os.environ.get("VAL_BITS", "4"))
G = int(os.environ.get("GROUP", "32"))
RESID = int(os.environ.get("RESID", "128"))
SINK = int(os.environ.get("SINK", "0"))
OUT_FRAC = float(os.environ.get("OUTLIER_FRAC", "0.0"))
NUQ = int(os.environ.get("NUQ", "0"))
TAG = os.environ.get("TAG", "v0")
SHARD = int(os.environ["SHARD_ID"])
NSH = int(os.environ["NUM_SHARDS"])
OUT = f"/root/out_{TAG}"
os.makedirs(OUT, exist_ok=True)

# 4-bit NormalFloat levels (QLoRA NF4), calibration-free non-uniform codebook.
NF4 = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)


def _group_pad(x, g):
    n = x.shape[2]
    pad = (g - n % g) % g
    if pad:
        x = torch.cat([x, x[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
    return x, n, pad


def _quant_uniform_group(x, bits, g):
    B, H, n, D = x.shape
    xp, n0, _ = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mn = xg.amin(3, keepdim=True)
    mx = xg.amax(3, keepdim=True)
    qm = 2**bits - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    xq = ((xg - mn) / sc).round().clamp(0, qm) * sc + mn
    return xq.reshape(B, H, Tg * g, D)[:, :, :n0, :]


def _quant_nf4_group(x, g, nf4):
    B, H, n, D = x.shape
    xp, n0, _ = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    amax = xg.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xg / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def qdq_key(x, bits):
    B, H, T, D = x.shape
    if T <= RESID:
        return x
    n = T - RESID
    head = x[:, :, :n, :]
    tail = x[:, :, n:, :]
    if NUQ and bits == 4:
        hq = _quant_nf4_group(head, G, NF4)
    else:
        hq = _quant_uniform_group(head, bits, G)
    keep = torch.zeros(B, H, n, D, dtype=torch.bool, device=x.device)
    if SINK > 0:
        keep[:, :, : min(SINK, n), :] = True
    if OUT_FRAC > 0:
        k = max(1, int(round(n * OUT_FRAC)))
        absn = head.abs()
        thr = absn.kthvalue(n - k + 1, dim=2, keepdim=True).values
        keep |= absn >= thr
    out_head = torch.where(keep, head, hq)
    return torch.cat([out_head, tail], dim=2).to(x.dtype)


def qdq_val(x, bits):
    B, H, T, D = x.shape
    if T <= RESID:
        return x
    n = T - RESID
    head = x[:, :, :n, :]
    tail = x[:, :, n:, :]
    mn = head.amin(3, keepdim=True)
    mx = head.amax(3, keepdim=True)
    qm = 2**bits - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    hq = ((head - mn) / sc).round().clamp(0, qm) * sc + mn
    return torch.cat([hq, tail], dim=2)


NOQUANT = int(os.environ.get("NOQUANT", "0"))

# Inject quantization by monkeypatching DynamicCache.update globally. Passing a
# custom cache via generate(past_key_values=...) is IGNORED in transformers 4.38
# (verified: 2-bit == fp16 exactly), because generate() instantiates its own
# DynamicCache. Patching the class method guarantees every cache update is
# quantized regardless of how generate() builds the cache.
_orig_update = DynamicCache.update


def _patched_update(self, k, v, li, cache_kwargs=None):
    fk, fv = _orig_update(self, k, v, li, cache_kwargs)
    if NOQUANT:
        return fk, fv
    return qdq_key(fk, KB), qdq_val(fv, VB)


DynamicCache.update = _patched_update


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def main():
    print(
        f"[shard {SHARD}/{NSH}] TAG={TAG} KB={KB} VB={VB} G={G} RESID={RESID} "
        f"SINK={SINK} OUT_FRAC={OUT_FRAC} NUQ={NUQ}",
        flush=True,
    )
    config = AutoConfig.from_pretrained(MODEL)
    config.use_cache = True
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            config=config,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .cuda()
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    device = torch.device("cuda:0")
    d2p = json.load(open(f"{LBROOT}/config/dataset2prompt.json"))
    d2m = json.load(open(f"{LBROOT}/config/dataset2maxlen.json"))
    for dataset in DATASETS:
        data = load_jsonl(f"{DATADIR}/{dataset}.jsonl")
        pf = d2p[dataset]
        mg = int(d2m[dataset])
        fo = open(f"{OUT}/{dataset}.{SHARD}.jsonl", "w")
        for gi, o in enumerate(data):
            if gi % NSH != SHARD:
                continue
            prompt = pf.format(**o)
            tp = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tp) > MAXLEN:
                h = MAXLEN // 2
                prompt = tok.decode(tp[:h], skip_special_tokens=True) + tok.decode(
                    tp[-h:], skip_special_tokens=True
                )
            if dataset not in [
                "trec",
                "triviaqa",
                "samsum",
                "lsht",
                "lcc",
                "repobench-p",
            ]:
                prompt = f"[INST]{prompt}[/INST]"
            inp = tok(prompt, truncation=False, return_tensors="pt").to(device)
            cl = inp.input_ids.shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=mg,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            pred = tok.decode(out[cl:], skip_special_tokens=True)
            fo.write(
                json.dumps(
                    {
                        "idx": gi,
                        "pred": pred,
                        "answers": o["answers"],
                        "all_classes": o["all_classes"],
                    }
                )
                + "\n"
            )
            fo.flush()
        fo.close()
        print(f"[shard {SHARD}] {dataset} done", flush=True)
    print(f"SHARD_{SHARD}_DONE", flush=True)


if __name__ == "__main__":
    main()
