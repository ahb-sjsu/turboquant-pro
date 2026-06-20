#!/usr/bin/env python3
"""End-to-end edge LLM benchmark: a real generation under PowerSampler.

Produces the Table III numbers -- decode throughput (tok/s), energy-per-token
(J/tok), and peak process memory -- for an actual model generation. Pairs with
``benchmark_edge.py`` (memory budget) and reuses its ``PowerSampler`` (NVML on a
discrete GPU, ``tegrastats`` on Jetson; ``null`` if neither is present).

Backends:
  * ``llama-cpp`` (default; right for Jetson Nano CPU) -- runs a GGUF via
    ``llama-cpp-python``.
  * ``transformers`` -- ``model.generate()`` on CUDA or CPU.

HONESTY: this measures whatever model you point it at. Declare the quantizer
with ``--quantizer`` (e.g. ``turboquant-3bit`` vs ``gguf-q3_k``) so the J/tok
claim is unambiguous -- a GGUF baseline is NOT a TurboQuant result, and the
paper must say which produced the running weights.

Examples:
  # Jetson Nano (CPU, GGUF), energy via tegrastats:
  python -m benchmarks.benchmark_e2e --backend llama-cpp \\
      --model ./models/llama-3.2-1b-q3_k.gguf --quantizer gguf-q3_k \\
      --device "Jetson Nano 4GB" --n-tokens 128 --energy --json out/e2e_nano.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time

try:  # run as `python -m benchmarks.benchmark_e2e`
    from benchmarks.benchmark_edge import PowerSampler, _nullctx
except ImportError:  # run as `python benchmarks/benchmark_e2e.py`
    from benchmark_edge import PowerSampler, _nullctx


def measure_peak_mem_mb() -> float | None:
    """Peak resident set of this process, in MB (Linux/macOS). None on Windows."""
    try:
        import resource
    except ImportError:
        try:
            import psutil

            return psutil.Process().memory_info().rss / 1e6
        except Exception:
            return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    return kb / 1024.0 if platform.system() == "Linux" else kb / 1e6


def _sampler(energy: bool):
    return PowerSampler() if energy else _nullctx()


def run_llama_cpp(
    model: str,
    prompt: str,
    n_tokens: int,
    threads: int | None,
    n_ctx: int,
    energy: bool,
) -> dict:
    from llama_cpp import Llama

    kw = {"model_path": model, "n_ctx": n_ctx, "verbose": False}
    if threads:
        kw["n_threads"] = threads
    llm = Llama(**kw)
    llm.create_completion(
        "Warm up the runtime.", max_tokens=8
    )  # warmup (prefill caches)

    with _sampler(energy) as ps:
        t0 = time.perf_counter()
        out = llm.create_completion(prompt, max_tokens=n_tokens)
        dt = time.perf_counter() - t0
    gen = int(out["usage"]["completion_tokens"])
    return _row(gen, dt, ps, "llama-cpp")


def run_transformers(
    model: str, prompt: str, n_tokens: int, device: str, energy: bool
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(device)
    ids = tok(prompt, return_tensors="pt").to(device)
    m.generate(**ids, max_new_tokens=8, do_sample=False)  # warmup
    if device == "cuda":
        torch.cuda.synchronize()

    with _sampler(energy) as ps:
        t0 = time.perf_counter()
        out = m.generate(**ids, max_new_tokens=n_tokens, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    gen = int(out.shape[1] - ids["input_ids"].shape[1])
    return _row(gen, dt, ps, f"transformers/{device}")


def _row(gen: int, dt: float, ps, backend: str) -> dict:
    e = getattr(ps, "energy_j", None)
    return {
        "backend": backend,
        "power_backend": getattr(ps, "backend", None),
        "gen_tokens": gen,
        "decode_s": round(dt, 3),
        "decode_tok_s": round(gen / dt, 2) if dt > 0 else None,
        "energy_j": round(e, 2) if e else None,
        "avg_power_w": (
            round(ps.avg_power_w, 2) if getattr(ps, "avg_power_w", None) else None
        ),
        "j_per_token": round(e / gen, 4) if (e and gen) else None,
        "peak_mem_mb": (lambda x: round(x, 1) if x else None)(measure_peak_mem_mb()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="End-to-end edge LLM benchmark (Table III)."
    )
    p.add_argument(
        "--model", required=True, help="GGUF path (llama-cpp) or HF id (transformers)"
    )
    p.add_argument(
        "--backend", choices=["llama-cpp", "transformers", "auto"], default="auto"
    )
    p.add_argument(
        "--quantizer",
        required=True,
        help="provenance of the running weights, e.g. turboquant-3bit | gguf-q3_k",
    )
    p.add_argument(
        "--device", default="(unspecified)", help="device label for the Table III row"
    )
    p.add_argument("--prompt", default="Summarize the role of edge AI in IoT systems:")
    p.add_argument("--n-tokens", type=int, default=128)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--hf-device", default="cuda", help="transformers device: cuda|cpu")
    p.add_argument("--energy", action="store_true", help="sample NVML/tegrastats power")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args(argv)

    backend = args.backend
    if backend == "auto":
        backend = "llama-cpp" if args.model.endswith(".gguf") else "transformers"

    try:
        if backend == "llama-cpp":
            r = run_llama_cpp(
                args.model,
                args.prompt,
                args.n_tokens,
                args.threads,
                args.n_ctx,
                args.energy,
            )
        else:
            r = run_transformers(
                args.model, args.prompt, args.n_tokens, args.hf_device, args.energy
            )
    except ImportError as e:
        raise SystemExit(
            f"backend '{backend}' unavailable: {e}. "
            f"Install llama-cpp-python (GGUF) or transformers."
        ) from e

    r.update({"device": args.device, "model": args.model, "quantizer": args.quantizer})

    print(
        f"\nDevice: {r['device']}   Model: {r['model']}   Quantizer: {r['quantizer']}"
    )
    print(f"Backend: {r['backend']}   Power: {r['power_backend'] or 'none'}")
    e = f"{r['energy_j']} J" if r["energy_j"] is not None else "n/a (no power backend)"
    jpt = r["j_per_token"] if r["j_per_token"] is not None else "n/a"
    print(
        f"  decode: {r['decode_tok_s']} tok/s over {r['gen_tokens']} tok ({r['decode_s']} s)"
    )
    print(f"  energy: {e}   ->  {jpt} J/tok   (avg {r['avg_power_w']} W)")
    print(f"  peak memory: {r['peak_mem_mb']} MB")
    print("\nTable III row:")
    print(
        f"  {r['device']} & {r['model']} & {r['quantizer']} & "
        f"{r['decode_tok_s']} & {r['j_per_token']} \\\\"
    )

    if args.json:
        import os

        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
