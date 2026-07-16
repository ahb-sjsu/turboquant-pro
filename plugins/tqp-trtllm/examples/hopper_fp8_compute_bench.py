# tqp-trtllm: TensorRT-LLM-style KV formats as turboquant-pro plugins
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Hopper fp8-compute micro-bench: upgrade the fp8 KV story from STORAGE to
COMPUTE. Attention-score matmuls q @ K^T three ways on real DC-offset-shaped
keys held in FP8NativeKV containers:

  1. fp16 matmul (baseline)
  2. fp8 storage + upcast matmul (the shipped passthrough path)
  3. fp8 native compute via torch._scaled_mm (Hopper/Ada sm_89+ tensor cores)

Prints latency, effective TFLOP/s, and max |error| vs an fp64 reference.
Run on an H100/H200 (or Ada):  python hopper_fp8_compute_bench.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from tqp_trtllm.native import FP8NativeKV

assert torch.cuda.is_available(), "needs a CUDA GPU (sm_89+ for _scaled_mm)"
DEV = "cuda"
rng = np.random.default_rng(0)


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run(H=32, S=8192, D=128, Q=64):
    off = rng.uniform(-4, 4, size=(1, H, 1, D))
    k = (off + rng.standard_normal((1, H, S, D))).astype(np.float32)
    c = FP8NativeKV().compress(torch.as_tensor(k).to(DEV))
    q = torch.randn(H, Q, D, device=DEV, dtype=torch.float16)

    k16 = torch.as_tensor(k, device=DEV)[0].to(torch.float16)  # (H, S, D)
    ref = torch.einsum("hqd,hsd->hqs", q.double(), k16.double())  # fp64 reference

    # 1. fp16 baseline
    t_fp16 = bench(lambda: torch.einsum("hqd,hsd->hqs", q, k16))
    out16 = torch.einsum("hqd,hsd->hqs", q, k16)

    # 2. shipped passthrough: fp8 storage, upcast, matmul
    def upcast():
        kk = FP8NativeKV().decompress(c)[0].to(torch.float16)
        return torch.einsum("hqd,hsd->hqs", q, kk)

    t_up = bench(upcast)
    outup = upcast()

    # 3. native fp8 compute: per-head _scaled_mm loops (scores in bf16 out)
    q8 = (q.float() / 1.0).to(torch.float8_e4m3fn)  # queries near unit scale
    # mat2 must be column-major: transpose of a contiguous (S, D) is (D, S) col-major
    kT = [c.data[0, h].contiguous() for h in range(H)]
    one = torch.tensor(1.0, device=DEV)

    def scaled():
        outs = []
        for h in range(H):
            outs.append(
                torch._scaled_mm(
                    q8[h],
                    kT[h].t(),
                    scale_a=one,
                    scale_b=c.scale[h].reshape(()),
                    out_dtype=torch.bfloat16,
                )
            )
        return torch.stack(outs)

    t_f8 = bench(scaled)
    out8 = scaled()

    # 4. same per-head loop captured in a CUDA graph: is the 0.65 ms wall
    # launch overhead (graph collapses it) or the GEMMs themselves?
    t_graph = None
    err_graph = None
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                scaled()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = scaled()

        def replay():
            graph.replay()
            return graph_out

        t_graph = bench(replay)
        replay()
        err_graph = float((graph_out.double() - ref).abs().max())
    except Exception as e:  # noqa: BLE001 - graphs can be env-fragile
        print(f"  fp8_scaled_mm_graph   skipped: {type(e).__name__}: {e}", flush=True)

    flops = 2.0 * H * Q * S * D
    res = {
        "shape": {"H": H, "S": S, "D": D, "Q": Q},
        "gpu": torch.cuda.get_device_name(0),
        "fp16": {
            "ms": t_fp16 * 1e3,
            "tflops": flops / t_fp16 / 1e12,
            "max_err": float((out16.double() - ref).abs().max()),
        },
        "fp8_storage_upcast": {
            "ms": t_up * 1e3,
            "tflops": flops / t_up / 1e12,
            "max_err": float((outup.double() - ref).abs().max()),
        },
        "fp8_scaled_mm": {
            "ms": t_f8 * 1e3,
            "tflops": flops / t_f8 / 1e12,
            "max_err": float((out8.double() - ref).abs().max()),
        },
    }
    if t_graph is not None:
        res["fp8_scaled_mm_graph"] = {
            "ms": t_graph * 1e3,
            "tflops": flops / t_graph / 1e12,
            "max_err": err_graph,
        }
    for name in [k for k in res if k not in ("shape", "gpu")]:
        r = res[name]
        print(
            f"  {name:20s} {r['ms']:8.3f} ms  {r['tflops']:7.1f} TF/s  "
            f"maxerr {r['max_err']:.4f}",
            flush=True,
        )
    return res


if __name__ == "__main__":
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    all_res = [run(S=s) for s in (2048, 8192, 32768)]
    print("=== JSON ===")
    print(json.dumps(all_res))
