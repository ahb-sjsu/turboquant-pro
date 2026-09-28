"""Gate G0 on the device that will produce the results, run while the job unpacks.

    python -m weight_observer.g0_device --model-path M --out O [--device cuda]

The prereg's G0 (``docs/PREREG_weights_codec_allocation.md``) is that the codecs share one
grid and one rounding rule: GPTQ with H = I and no damping is RTN bit for bit, and so is
AWQ at alpha = 0. The unit tests check it on CPU; this checks it on the job's own GPU, at
every weight shape of the model (read from its ``config.json``) and every width, and that
encoding the same matrix twice gives the same bits (the determinism gate G2 relies on). A
kernel or driver difference then fails the job before any arm is spent. It needs only
torch, which the image carries, so it runs while the environment tar is unpacked; the job
waits for it, and the verdict is written to ``g0_device.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from . import quant as Q

N_CALIB_ROWS = 256  # rows of the synthetic activations that give S and mean |x|


def shapes(cfg: dict) -> dict:
    """(out, in) of each distinct projection shape of a decoder layer."""
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or h // heads
    named = {
        "q_proj": (heads * hd, h),
        "k_proj": (kv * hd, h),
        "o_proj": (h, heads * hd),
        "up_proj": (inter, h),
        "down_proj": (h, inter),
    }
    out = {}
    for n, s in named.items():
        out.setdefault(s, n)
    return {n: s for s, n in out.items()}


def check(rows: int, cols: int, bits: int, device: str, seed: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(rows, cols, generator=g).to(device)
    x = torch.randn(N_CALIB_ROWS, cols, generator=g).to(device)
    S, a = x.T @ x / N_CALIB_ROWS, x.abs().mean(0)
    rtn = Q.rtn(w, bits)
    eye = torch.eye(cols, device=device)
    awq0, alpha = Q.awq(w, S, a, bits, alphas=(0.0,))
    one, two = Q.gptq(w, S, bits), Q.gptq(w, S, bits)
    return {
        "gptq_identity_is_rtn": bool(torch.equal(Q.gptq(w, eye, bits, damp=0.0), rtn)),
        "awq_alpha0_is_rtn": bool(alpha == 0.0 and torch.equal(awq0, rtn)),
        "gptq_repeat_identical": bool(torch.equal(one, two)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = json.load(open(os.path.join(a.model_path, "config.json")))
    t0 = time.time()
    cases = []
    for i, (name, (r, c)) in enumerate(sorted(shapes(cfg).items())):
        for bits in Q.LEVELS:
            res = check(r, c, bits, a.device, seed=1000 * i + bits)
            cases.append({"matrix": name, "shape": [r, c], "bits": bits, **res})
    ok = all(v for cs in cases for k, v in cs.items() if isinstance(v, bool))
    dev = torch.cuda.get_device_name() if a.device.startswith("cuda") else a.device
    rec = {
        "passed": ok,
        "device": dev,
        "torch": torch.__version__,
        "seconds": round(time.time() - t0, 1),
        "cases": cases,
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "g0_device.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[g0] {'PASS' if ok else 'FAIL'} {len(cases)} cases on {dev}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
