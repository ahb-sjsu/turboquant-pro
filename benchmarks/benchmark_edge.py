#!/usr/bin/env python3
"""Edge-deployment benchmark for TurboQuant (AIoT memory/throughput/energy).

Measures, per (model, context length, bit-width) on the *current* device:

  * KV-cache memory   -- fp16-equivalent vs TurboQuant-compressed (real, from
    ``TurboQuantKVCache.memory_stats()``), scaled to the full model by n_layers.
  * KV throughput     -- prefill + decode tokens/s for the KV path.
  * Weight footprint  -- params x bits/8 (an *estimate*: fp16 vs compressed).
  * Device budget     -- does (weights + KV) fit the detected device, and a few
    named edge tiers (Jetson Orin, Raspberry Pi 5, a consumer GPU)?
  * Energy            -- average GPU power and energy for the KV workload,
    *measured* via NVML when a CUDA GPU + ``pynvml`` are present; otherwise
    reported as ``null``. We do not estimate energy we cannot measure.

Scope / honesty: this benchmarks the **KV-cache compression and the memory
budget**, not a full transformer forward pass. End-to-end per-token latency and
energy require integrating ``TurboQuantKVCache`` into the model runtime (see
``benchmark_llama.py``); the ``PowerSampler`` below is reusable for that.
Architecture params are approximate -- override with ``--model``.

Usage:
    python benchmarks/benchmark_edge.py
    python benchmarks/benchmark_edge.py --models llama-3.2-3b qwen2.5-7b \
        --contexts 2048 8192 --bits 3 --gen 128 --energy --json out/edge.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from turboquant_pro import TurboQuantKVCache
from turboquant_pro.hardware import get_hardware_profile

# Approximate architectures (n_layers, n_kv_heads [GQA], head_dim, params_billions).
# Override any of these with --model "name:n_layers:n_kv_heads:head_dim:params_b".
MODELS: dict[str, tuple[int, int, int, float]] = {
    "llama-3.2-1b": (16, 8, 64, 1.24),
    "llama-3.2-3b": (28, 8, 128, 3.21),
    "qwen2.5-7b": (28, 4, 128, 7.6),
    "mistral-7b": (32, 8, 128, 7.2),
}

# Named edge memory budgets (GB of usable RAM/VRAM).
EDGE_TIERS: dict[str, float] = {
    "jetson-orin-nano-8gb": 8.0,
    "jetson-orin-nx-16gb": 16.0,
    "rpi5-8gb-cpu": 8.0,
    "consumer-gpu-12gb": 12.0,
}

_GB = 1024.0**3


class PowerSampler:
    """Context manager: integrate NVML GPU power over a workload.

    Yields measured energy (J) and average power (W), or ``None`` for both if
    NVML / a CUDA GPU isn't available. Reusable around any workload (incl. a
    full model generation loop).
    """

    def __init__(self, device_id: int = 0, interval_s: float = 0.01) -> None:
        self.device_id = device_id
        self.interval_s = interval_s
        self.energy_j: float | None = None
        self.avg_power_w: float | None = None
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

    def _loop(self) -> None:
        import pynvml

        while not self._stop.is_set():
            try:
                self._samples.append(
                    pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                )
            except Exception:
                break
            time.sleep(self.interval_s)

    def __enter__(self) -> PowerSampler:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        except Exception:
            self._handle = None
            return self
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is None:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        elapsed = time.perf_counter() - self._t0
        if self._samples:
            self.avg_power_w = sum(self._samples) / len(self._samples)
            self.energy_j = self.avg_power_w * elapsed
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass


@dataclass
class EdgeResult:
    model: str
    context: int
    gen_tokens: int
    bits: int
    n_layers: int
    kv_fp16_gb: float
    kv_tq_gb: float
    measured_store_gb: float
    kv_compression_x: float
    weights_fp16_gb: float
    weights_tq_gb: float
    total_fp16_gb: float
    total_tq_gb: float
    prefill_tok_s: float
    decode_tok_s: float
    energy_j: float | None
    avg_power_w: float | None


def _weight_gb(params_b: float, bits: float) -> float:
    return params_b * 1e9 * bits / 8.0 / _GB


def run_one(
    name: str,
    arch: tuple,
    context: int,
    gen: int,
    bits: int,
    use_gpu: bool,
    energy: bool,
) -> EdgeResult:
    n_layers, n_kv_heads, head_dim, params_b = arch
    rng = np.random.default_rng(0)

    # Measure ONE representative layer (KV memory + time are linear in layers),
    # then scale by n_layers. Avoids allocating every layer on small devices.
    cache = TurboQuantKVCache(
        head_dim=head_dim, n_heads=n_kv_heads, bits=bits, use_gpu=use_gpu
    )
    k_prompt = rng.standard_normal((1, n_kv_heads, context, head_dim)).astype(
        np.float32
    )
    v_prompt = rng.standard_normal((1, n_kv_heads, context, head_dim)).astype(
        np.float32
    )

    t0 = time.perf_counter()
    cache.append(k_prompt, v_prompt)
    prefill_s = time.perf_counter() - t0

    sampler = PowerSampler(device_id=0) if energy else _nullctx()
    with sampler as ps:
        t0 = time.perf_counter()
        for _ in range(gen):
            k = rng.standard_normal((1, n_kv_heads, 1, head_dim)).astype(np.float32)
            v = rng.standard_normal((1, n_kv_heads, 1, head_dim)).astype(np.float32)
            cache.append(k, v)
        decode_s = time.perf_counter() - t0

    # KV memory: analytical budget (unambiguous) -- fp16 (2 B/elem) vs bit-packed
    # at `bits`. This is what the compression claim rests on at scale. The
    # library's measured store (incl. the fp32 hot window + codebooks) is
    # reported separately as `measured_store_gb`; it carries fixed overhead that
    # amortizes only at long context.
    elems = 2 * n_kv_heads * context * head_dim * n_layers  # keys + values, all layers
    kv_fp16_gb = elems * 2 / _GB
    kv_tq_gb = elems * (bits / 8.0) / _GB
    stats = cache.memory_stats()
    measured_store_gb = round(stats["total_bytes"] * n_layers / _GB, 4)
    w_fp16 = _weight_gb(params_b, 16)
    w_tq = _weight_gb(params_b, bits)

    return EdgeResult(
        model=name,
        context=context,
        gen_tokens=gen,
        bits=bits,
        n_layers=n_layers,
        kv_fp16_gb=round(kv_fp16_gb, 4),
        kv_tq_gb=round(kv_tq_gb, 4),
        measured_store_gb=measured_store_gb,
        kv_compression_x=round(kv_fp16_gb / max(kv_tq_gb, 1e-9), 2),
        weights_fp16_gb=round(w_fp16, 3),
        weights_tq_gb=round(w_tq, 3),
        total_fp16_gb=round(w_fp16 + kv_fp16_gb, 3),
        total_tq_gb=round(w_tq + kv_tq_gb, 3),
        prefill_tok_s=round(context / max(prefill_s, 1e-9), 1),
        decode_tok_s=round(gen / max(decode_s, 1e-9), 1),
        energy_j=(round(ps.energy_j, 2) if getattr(ps, "energy_j", None) else None),
        avg_power_w=(
            round(ps.avg_power_w, 1) if getattr(ps, "avg_power_w", None) else None
        ),
    )


class _nullctx:
    energy_j = None
    avg_power_w = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _parse_models(specs: list[str] | None) -> dict[str, tuple]:
    if not specs:
        return MODELS
    out = {}
    for s in specs:
        if ":" in s:
            name, nl, nk, hd, pb = s.split(":")
            out[name] = (int(nl), int(nk), int(hd), float(pb))
        elif s in MODELS:
            out[s] = MODELS[s]
        else:
            raise SystemExit(f"unknown model {s!r}; use name or name:nl:nk:hd:params_b")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TurboQuant edge-deployment benchmark.")
    p.add_argument("--models", nargs="*", help="model names or name:nl:nk:hd:params_b")
    p.add_argument("--contexts", nargs="*", type=int, default=[2048, 8192])
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--cpu", action="store_true", help="force CPU path")
    p.add_argument("--energy", action="store_true", help="sample NVML GPU power")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args(argv)

    prof = get_hardware_profile()
    hw = prof.hardware
    dev_gb = hw.memory_gb if hw.available else 0.0
    use_gpu = hw.available and not args.cpu
    print(
        f"Device: {hw.name} ({hw.arch}, {dev_gb:.1f} GB)  "
        f"backend={'gpu' if use_gpu else 'cpu'}  recommended_bits={prof.recommended_bits}"
    )
    if args.energy and not hw.available:
        print("  (--energy ignored: no CUDA GPU / NVML)")

    models = _parse_models(args.models)
    results: list[EdgeResult] = []
    for name, arch in models.items():
        for ctx in args.contexts:
            results.append(
                run_one(name, arch, ctx, args.gen, args.bits, use_gpu, args.energy)
            )

    # Table
    hdr = (
        f"{'model':<14}{'ctx':>6}{'KV fp16':>9}{'KV TQ':>8}{'KV x':>6}{'store':>8}"
        f"{'tot fp16':>9}{'tot TQ':>8}{'dec tok/s':>10}{'J/run':>8}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        e = f"{r.energy_j:>8.1f}" if r.energy_j is not None else f"{'n/a':>8}"
        print(
            f"{r.model:<14}{r.context:>6}{r.kv_fp16_gb:>9.2f}{r.kv_tq_gb:>8.2f}"
            f"{r.kv_compression_x:>6.1f}{r.measured_store_gb:>8.2f}"
            f"{r.total_fp16_gb:>9.2f}{r.total_tq_gb:>8.2f}{r.decode_tok_s:>10.0f}{e}"
        )
    print(
        "KV fp16/TQ = analytical budget (2 B/elem vs bits/8); store = library's "
        "measured footprint (incl. fp32 hot window). dec tok/s = KV-path only."
    )

    # Device-budget fit table (weights + KV at the largest context tested)
    print("\nFits in device budget (weights + KV @ max context, TQ vs fp16):")
    tiers = dict(EDGE_TIERS)
    if hw.available:
        tiers[f"detected:{hw.name}"] = dev_gb
    worst = {}
    for r in results:
        worst.setdefault(r.model, r)
        if r.total_tq_gb > worst[r.model].total_tq_gb:
            worst[r.model] = r
    for tier, gb in tiers.items():
        fits_tq = [m for m, r in worst.items() if r.total_tq_gb <= gb]
        fits_fp = [m for m, r in worst.items() if r.total_fp16_gb <= gb]
        print(
            f"  {tier:<26} {gb:>5.0f} GB  TQ:{len(fits_tq)}/{len(worst)}  "
            f"fp16:{len(fits_fp)}/{len(worst)}"
        )

    if args.json:
        import os

        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "device": {"name": hw.name, "arch": hw.arch, "memory_gb": dev_gb},
                    "results": [asdict(r) for r in results],
                },
                fh,
                indent=2,
            )
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
