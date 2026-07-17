# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""LongBench-qasper 4-bit KEY separation (P3): per-channel asym-NF4 vs
nvfp4 block-16, keys fake-quantized in the live KV cache, values fp16.

Registered expectation from the code-space result: per-channel NF4 >= nvfp4 at
task level (it won ~2.2x on attention KL). Runs three legs on Llama-3.2-3B over
LongBench qasper -- fp16 baseline, per-channel asym-NF4 keys (+2% fp16
outliers), and NVFP4 block-16 keys -- and reports qasper F1 per leg.

Usage (needs torch + transformers + the tqp-trtllm plugin; a CUDA GPU):
    pip install -e . -e plugins/tqp-trtllm transformers accelerate sentencepiece
    NB_N=100 python benchmarks/lb_keys_4bit.py
"""

import io
import json
import os
import re
import string
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "unsloth/Llama-3.2-3B"
N = int(os.environ.get("NB_N", "100"))
MAXCTX = 3500


def get_data():
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip",
        "data.zip",
    )
    with zipfile.ZipFile("data.zip") as z:
        with z.open("data/qasper.jsonl") as f:
            rows = [json.loads(ln) for ln in io.TextIOWrapper(f, "utf-8")]
    return rows[:N]


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gts):
    best = 0.0
    for gt in gts:
        p, g = norm(pred).split(), norm(gt).split()
        common = Counter(p) & Counter(g)
        ns = sum(common.values())
        if ns == 0 or not p or not g:
            best = max(best, 0.0)
            continue
        prec, rec = ns / len(p), ns / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


class KeyQuantCache(DynamicCache):
    quantizer = None  # class attr set per run

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.quantizer is not None and key_states.shape[2] > 0:
            k = key_states.float().cpu().numpy()  # (B, H, S, D)
            kq = np.asarray(self.quantizer.decompress(self.quantizer.compress(k)))
            key_states = torch.as_tensor(
                kq, dtype=key_states.dtype, device=key_states.device
            )
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def run(tag, quantizer, rows, tok, model):
    KeyQuantCache.quantizer = quantizer
    scores = []
    for i, r in enumerate(rows):
        prompt = (
            f"Read the article and answer the question briefly.\n\n"
            f"Article: {r['context']}\n\nQuestion: {r['input']}\nAnswer:"
        )
        ids = tok(
            prompt, return_tensors="pt", truncation=True, max_length=MAXCTX
        ).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=32,
                do_sample=False,
                past_key_values=KeyQuantCache(),
                pad_token_id=tok.eos_token_id,
            )
        pred = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        pred = pred.split("\n")[0]
        scores.append(f1(pred, r["answers"]))
        if (i + 1) % 25 == 0:
            print(
                f"  [{tag}] {i + 1}/{len(rows)} f1={np.mean(scores) * 100:.2f}",
                flush=True,
            )
    s = float(np.mean(scores) * 100)
    print(f"[score] {tag}: qasper F1 = {s:.2f}", flush=True)
    return s


def main():
    from tqp_trtllm.plugin import NVFP4KVQuantizer

    from turboquant_pro.plugins import create

    rows = get_data()
    print(f"[data] qasper {len(rows)} samples", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    kvh = model.config.num_key_value_heads
    hd = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    results = {}
    results["fp16"] = run("fp16", None, rows, tok, model)
    results["per_channel_nf4"] = run(
        "per_channel_nf4",
        create(
            "per_channel", head_dim=hd, n_heads=kvh, nf4_asym=True, outlier_frac=0.02
        ),
        rows,
        tok,
        model,
    )
    results["nvfp4_block16"] = run(
        "nvfp4_block16", NVFP4KVQuantizer(), rows, tok, model
    )
    print("[verdict] " + json.dumps(results), flush=True)
    print("=== JSON ===")
    print(
        json.dumps(
            {
                "model": MODEL,
                "n": len(rows),
                "task": "qasper",
                "keys_only": True,
                "f1": results,
            }
        )
    )


if __name__ == "__main__":
    main()
