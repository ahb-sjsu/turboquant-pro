# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Atlas side of the NRP runs: stripe plans, reference stripes, inputs, and checks.

Everything a pod needs that must match Atlas exactly is made here, from Atlas's cached
checkpoints and WikiText-2 (the cache the E1 to E4 runs read):

    python prep.py plans  OUT     # stripe plans + source-file hashes, per model
    python prep.py stripes OUT    # reference stripes on Atlas and their sha256
    python prep.py tokens OUT     # the experiments' sequences, pre-tokenized
    python prep.py equiv OUT      # loader == from_pretrained, on the CPU

A pod's stripes must hash to the reference stripes, and the reference stripes load into
a model equal to from_pretrained's (every parameter and buffer, and the logits), so a
pod's model is the model the Atlas runs used.
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import stripe  # noqa: E402

HUB = os.environ.get("HF_HUB_CACHE", "/archive/cache/huggingface/hub")
MODELS = {
    "mistral": {
        "repo": "mistralai/Mistral-7B-v0.1",
        "rev": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
        "tag": "Mistral-7B-v0.1",
        "dtype": "F16",  # the experiments load it as float16
        "rename": None,
        "torch_dtype": "float16",
        "tokens": {"test": 48, "validation": 64, "train": 16},
    },
    "gemma": {
        "repo": "unsloth/gemma-3-4b-it",
        "rev": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
        "tag": "gemma-3-4b-it",
        "dtype": None,  # bfloat16, as its matched-bit run
        "rename": "gemma3-text",
        "torch_dtype": "bfloat16",
        "tokens": {"test": 8},
    },
    "qwen": {  # Atlas regression only: its E1 and E3 records exist
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "rev": None,
        "tag": "Qwen2.5-1.5B-Instruct",
        "dtype": "F16",
        "rename": None,
        "torch_dtype": "float16",
        "tokens": {"test": 24, "validation": 8, "train": 4},
    },
}
K = 4


def snapshot(m: dict) -> str:
    d = os.path.join(HUB, "models--" + m["repo"].replace("/", "--"), "snapshots")
    revs = [m["rev"]] if m["rev"] else sorted(os.listdir(d))
    if len(revs) != 1:
        raise SystemExit(f"{m['repo']}: snapshots {revs}")
    return os.path.join(d, revs[0])


def source_hashes(snap: str) -> dict:
    """Each file's hash as the Hub names its blob (LFS: sha256, else git sha1)."""
    return {
        f: os.path.basename(os.path.realpath(os.path.join(snap, f)))
        for f in sorted(os.listdir(snap))
    }


def plans(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for key, m in MODELS.items():
        snap = snapshot(m)
        plan = stripe.make_plan(snap, K, m["dtype"], m["rename"])
        plan.update(
            {
                "key": key,
                "repo": m["repo"],
                "revision": os.path.basename(snap),
                "tag": m["tag"],
                "torch_dtype": m["torch_dtype"],
                "source_hashes": source_hashes(snap),
                "weight_map": {
                    t["dst"]: s["file"] for s in plan["stripes"] for t in s["tensors"]
                },
            }
        )
        if m["rename"] == "gemma3-text":
            from transformers import AutoConfig

            tc = AutoConfig.from_pretrained(snap).text_config
            plan["config"] = json.loads(tc.to_json_string(use_diff=False))
        path = os.path.join(out, f"plan_{key}.json")
        json.dump(plan, open(path, "w"), indent=1)
        print(path, [round(s["size"] / 1e9, 2) for s in plan["stripes"]], "GB")


def stripe_dirs(out: str, key: str) -> list[str]:
    return [os.path.join(out, "stripes", key, f"s{k}") for k in range(K)]


def model_dir(out: str, key: str) -> str:
    return os.path.join(out, "stripes", key, "model")


def write_model_dir(plan: dict, snap: str, dest: str) -> None:
    """The config directory a pod builds the model from (convert_hf_to_gguf too)."""
    os.makedirs(dest, exist_ok=True)
    if "config" in plan:  # Gemma 3: the text model's own config
        json.dump(
            plan["config"], open(os.path.join(dest, "config.json"), "w"), indent=2
        )
        return
    for f in plan["source_hashes"]:
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        with open(os.path.join(snap, f), "rb") as fi:
            open(os.path.join(dest, f), "wb").write(fi.read())
    index = {"metadata": {}, "weight_map": plan["weight_map"]}
    json.dump(index, open(os.path.join(dest, "model.safetensors.index.json"), "w"))


def stripes(out: str) -> None:
    for key, m in MODELS.items():
        path = os.path.join(out, f"plan_{key}.json")
        plan = json.load(open(path))
        snap = snapshot(m)
        dests = stripe_dirs(out, key)
        for f in plan["src_files"]:
            n = stripe.write(plan, os.path.join(snap, f), dests)
            print(f"  {key} {f}: {n / 1e9:.2f} GB")
        plan["sha256"] = stripe.stripe_hashes(plan, dests)
        write_model_dir(plan, snap, model_dir(out, key))
        json.dump(plan, open(path, "w"), indent=1)
        print(key, plan["sha256"])


def tokens(out: str) -> None:
    import exp_io
    from transformers import AutoTokenizer

    for key, m in MODELS.items():
        tok = AutoTokenizer.from_pretrained(snapshot(m))
        splits = {}
        for split, n in m["tokens"].items():
            splits[split] = [
                tok(t, return_tensors="pt", truncation=True, max_length=256)
                .input_ids[0]
                .tolist()
                for t in exp_io.load_texts(split, n)
            ]
        rec = {"model": m["repo"], "window": 256, "splits": splits}
        path = os.path.join(out, f"tokens_{key}.json")
        json.dump(rec, open(path, "w"), separators=(",", ":"))
        print(path, {s: len(v) for s, v in splits.items()}, os.path.getsize(path))


def equiv(out: str) -> None:
    """Every parameter and buffer of the striped load equals from_pretrained's, and so
    do the logits on 32 tokens (CPU). Gemma 3 is compared against the full
    Gemma3ForConditionalGeneration on text input."""
    import torch
    from loader import load
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(int(os.environ.get("NB_THREADS", "8")))
    for key in sys.argv[3:] or MODELS:
        m = MODELS[key]
        dt = getattr(torch, m["torch_dtype"])
        files = sorted(
            glob.glob(os.path.join(out, "stripes", key, "s?", "*.safetensors"))
        )
        a = load(model_dir(out, key), files, "cpu", dt)
        b = AutoModelForCausalLM.from_pretrained(snapshot(m), dtype=dt).eval()
        sa = dict(a.named_parameters(remove_duplicate=False))
        sa |= dict(a.named_buffers(remove_duplicate=False))
        sb = dict(b.named_parameters(remove_duplicate=False))
        sb |= dict(b.named_buffers(remove_duplicate=False))
        if m["rename"] == "gemma3-text":
            pre = "model.language_model."
            sb = {
                "model." + k[len(pre) :]: v for k, v in sb.items() if k.startswith(pre)
            } | {"lm_head.weight": sb["lm_head.weight"]}
        bad = [k for k in sa if k not in sb or not torch.equal(sa[k], sb[k])]
        miss = [k for k in sb if k not in sa]
        ids = torch.tensor([[2, 1014, 6087, 302, 272, 13023, 349, 5276] * 4])
        if m["rename"] == "gemma3-text":
            ids = torch.tensor([[2, 818, 5279, 529, 506, 21023, 563, 3363] * 4])
        with torch.no_grad():
            la, lb = a(ids).logits, b(ids).logits
        same = torch.equal(la, lb)
        diff = float((la - lb).abs().max())
        print(
            f"{key}: {len(sa)} tensors, {len(bad)} differ {bad[:3]},"
            f" {len(miss)} missing {miss[:3]}; logits equal: {same} (max {diff:.3g})"
        )
        del a, b


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    {"plans": plans, "stripes": stripes, "tokens": tokens, "equiv": equiv}[cmd](out)
