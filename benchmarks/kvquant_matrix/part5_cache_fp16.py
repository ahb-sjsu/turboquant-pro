#!/usr/bin/env python3
"""Load a checkpoint once exactly as the harness does, and save that fp16 model to local disk.

    python part5_cache_fp16.py --src /tmp/model/qwen2.5-7b-instruct --dest /tmp/model16/qwen2.5-7b-instruct

Part V's runner starts one process per (arm, task) cell, and each loads the model. From a
linstor set a 7B load took 2 min 41 s (339 tensors, bf16 converted to fp16 on two cores,
7B set-smoke 2026-10-13), so the cells' compute was a small part of the pod's life and its
GPU averaged 19%. This pays that load once: the model is loaded with the harness's own
call (fp16, sdpa, low_cpu_mem_usage, straight to the GPU), saved as fp16 safetensors on
the pod's local disk, and every cell then loads those, with no conversion and no network
volume. The saved checkpoint is reloaded and every tensor compared with the one in memory;
any difference ends the job, so the cells provably see the weights the harness would load.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time

import torch
from transformers import AutoModelForCausalLM

KEEP = (".json", ".txt", ".model", ".jinja", ".tiktoken")
WRITTEN_BY_SAVE = {"config.json", "generation_config.json", "model.safetensors.index.json"}


def load(path: str, device: str = "cuda:0"):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--device", default="cuda:0", help="cpu only for tests")
    a = ap.parse_args(argv)
    t0 = time.time()
    model = load(a.src, a.device)
    t1 = time.time()
    print(f"[cache] loaded {a.src} in {t1 - t0:.0f}s", flush=True)
    os.makedirs(a.dest, exist_ok=True)
    model.save_pretrained(a.dest, safe_serialization=True, max_shard_size="4GB")
    for f in os.listdir(a.src):  # tokenizer and other small files, as the source has them
        if f.endswith(KEEP) and f not in WRITTEN_BY_SAVE:
            shutil.copy(os.path.join(a.src, f), os.path.join(a.dest, f))
    t2 = time.time()
    print(f"[cache] saved fp16 to {a.dest} in {t2 - t1:.0f}s", flush=True)
    ref = digests(model)  # one 7B copy fits an A10, two do not: compare by digest
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    again = digests(load(a.dest, a.device))
    t3 = time.time()
    if ref.keys() != again.keys():
        raise SystemExit(f"[cache] parameter names differ: {sorted(set(ref) ^ set(again))[:5]}")
    bad = [k for k in ref if ref[k] != again[k]]
    if bad:
        raise SystemExit(f"[cache] {len(bad)} tensors differ after the round trip: {bad[:5]}")
    print(f"[cache] verified {len(ref)} tensors bit-identical (sha256 of dtype, shape and "
          f"bytes); reload and check took {t3 - t2:.0f}s", flush=True)
    return 0


def digests(model) -> dict:
    """sha256 of each parameter and buffer's dtype, shape and raw bytes."""
    out = {}
    for name, t in model.state_dict().items():
        h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        out[name] = h.hexdigest()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
