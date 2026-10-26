#!/usr/bin/env python3
"""part5_cache_fp16.py with a short idle start: the same load and save, checked on the GPU.

    python part5_cache_fast.py --src /tmp/model/mistral-7b-instruct --dest /tmp/model16/mistral-7b-instruct

part5_cache_fp16 verifies its copy by hashing every tensor on the CPU, reloading the whole
model from the copy and hashing it again. On a 7B model that is about a minute of a run
pod's ~2 min start with its GPU idle, and NRP removes GPU pods that sit idle at start.
This loads with part5_cache_fp16's own call and saves the same files, then reads each saved
tensor straight to the GPU and compares its bytes with the loaded one there, one tensor at a
time. A parameter left out of the files must share storage with one that was saved (a tied
embedding). On any mismatch it runs part5_cache_fp16 itself, whose full check then decides.

The submitter ships this file into each run pod (part5_nrp.run_script); it is not part of
the registered harness and changes no weight a cell loads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch
from safetensors import safe_open

import part5_cache_fp16 as C


def saved_files(dest: str) -> list:
    """The safetensors files save_pretrained wrote, sharded or not."""
    idx = os.path.join(dest, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    return ["model.safetensors"]


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    a = a.detach().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(a, b.to(a.device).contiguous().reshape(-1).view(torch.uint8))


def check(model, dest: str, device: str):
    """None if every saved tensor equals the loaded one byte for byte and every parameter
    and buffer is saved or shares storage with a saved one; else what is wrong."""
    state = model.state_dict()
    saved = set()
    for fn in saved_files(dest):
        with safe_open(os.path.join(dest, fn), framework="pt", device=device) as f:
            for name in f.keys():
                if name not in state:
                    return f"{fn} holds {name}, which the model does not"
                if not same(state[name], f.get_tensor(name)):
                    return f"{name} differs after the save"
                saved.add(name)
    ptrs = {state[n].data_ptr() for n in saved}
    left = [n for n in state if n not in saved and state[n].data_ptr() not in ptrs]
    return f"not saved and not tied: {left[:5]}" if left else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--device", default="cuda:0", help="cpu only for tests")
    a = ap.parse_args(argv)
    t0 = time.time()
    model = C.load(a.src, a.device)
    t1 = time.time()
    print(f"[cache] loaded {a.src} in {t1 - t0:.0f}s", flush=True)
    os.makedirs(a.dest, exist_ok=True)
    model.save_pretrained(a.dest, safe_serialization=True, max_shard_size="4GB")
    for f in os.listdir(a.src):  # tokenizer and other small files, as part5_cache_fp16
        if f.endswith(C.KEEP) and f not in C.WRITTEN_BY_SAVE:
            shutil.copy(os.path.join(a.src, f), os.path.join(a.dest, f))
    t2 = time.time()
    print(f"[cache] saved fp16 to {a.dest} in {t2 - t1:.0f}s", flush=True)
    bad = check(model, a.dest, a.device)
    n = len(model.state_dict())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if bad:
        print(f"[cache] fast check failed ({bad}); running part5_cache_fp16", flush=True)
        shutil.rmtree(a.dest)
        return C.main(["--src", a.src, "--dest", a.dest, "--device", a.device])
    print(f"[cache] verified {n} tensors bit-identical to the loaded model, compared on "
          f"{a.device}; check took {time.time() - t2:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
