"""Run Part III for one model (docs/PREREG_observer_advantage_weights.md). Resumable.

    python -m weight_observer.run --model-key qwen2.5-1.5b --model-path /data/wo/models/qwen2.5-1.5b \\
        --text /data/wo/text --out /data/wo/runs/qwen2.5-1.5b

Phase A (once): statistics on the calibration text, the predictor tables and the atlas.
Phase B: every registered variant, its KL from the full-precision model on the evaluation
text, appended one line per variant to results.jsonl (a restart skips finished variants).
The environment (GPU, package versions) is recorded; a restart elsewhere is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from . import tables as T
from .variants import generate

N_CALIB, N_EVAL, SEQ = 128, 48, 1024
CALIB_SEED = 20261002
GROUP_LAYERS = 2


def environment() -> dict:
    import transformers

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def chunks(tok, text: str, n: int) -> list:
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i : i + SEQ] for i in range(0, (len(ids) // SEQ) * SEQ, SEQ)][:n]


def load(path: str, device: str, dtype=torch.float16):
    from transformers import AutoModelForCausalLM

    m = (
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    m.requires_grad_(False)
    return m


def predictions(tbl: dict, bits: dict) -> dict:
    return {p: sum(tbl[n][str(b)][p] for n, b in bits.items()) for p in T.PREDICTORS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--text", required=True, help="directory with train.txt and test.txt"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    env = environment()
    ep = os.path.join(a.out, "env.json")
    if os.path.exists(ep) and json.load(open(ep)) != env:
        raise SystemExit(f"started on {json.load(open(ep))}, now {env}")
    json.dump(env, open(ep, "w"))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_path)
    ref = load(a.model_path, a.device)
    calib = [
        c[None].to(a.device)
        for c in chunks(
            tok, open(f"{a.text}/train.txt", encoding="utf-8").read(), N_CALIB
        )
    ]
    evalq = [
        c[None].to(a.device)
        for c in chunks(
            tok, open(f"{a.text}/test.txt", encoding="utf-8").read(), N_EVAL
        )
    ]
    print(
        f"[run] {a.model_key} {env} calib {len(calib)}x{SEQ} eval {len(evalq)}x{SEQ}",
        flush=True,
    )

    tp = os.path.join(a.out, "tables.json")
    if not os.path.exists(tp):
        t0 = time.time()
        tbl, atl = T.build(
            ref, calib, GROUP_LAYERS, CALIB_SEED, log=lambda s: print(s, flush=True)
        )
        json.dump(
            {"atlas": atl, "seconds": round(time.time() - t0, 1)},
            open(os.path.join(a.out, "atlas.json"), "w"),
        )
        json.dump(tbl, open(tp + ".tmp", "w"))
        os.replace(tp + ".tmp", tp)
    tbl = json.load(open(tp))

    mods = T.linear_modules(ref)
    names = list(mods)
    variants = generate(names, [mods[n].weight.numel() for n in names])
    vp = os.path.join(a.out, "variants.json")
    if os.path.exists(vp) and json.load(open(vp)) != variants:
        raise SystemExit("variants.json differs from the registered generator")
    json.dump(variants, open(vp, "w"))

    rp = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(rp):
        for line in open(rp, encoding="utf-8"):
            try:
                done.add(json.loads(line)["variant"])
            except ValueError:
                pass
    var = load(a.model_path, a.device)
    with open(rp, "a", encoding="utf-8") as fo:
        for vid, bits in variants.items():
            if vid in done:
                continue
            t0 = time.time()
            from .measure import apply_variant, kl_per_sequence

            apply_variant(ref, var, bits)
            per = kl_per_sequence(ref, var, evalq)
            rec = {
                "variant": vid,
                "seqs": per,
                "pred": predictions(tbl, bits),
                "seconds": round(time.time() - t0, 1),
            }
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            kl = sum(s["kl_sum"] for s in per) / sum(s["tokens"] for s in per)
            print(f"[run] {vid} kl/token {kl:.4f} {rec['seconds']}s", flush=True)
    print("RUN_DONE", a.model_key, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
