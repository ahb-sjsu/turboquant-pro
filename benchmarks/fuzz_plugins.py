# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Time-budgeted fuzzing sweep over the plugin contracts and backend paths.

Every failure the hardware trials found so far lived in an untested
*combination* (CUDA x torch_xp reference paths; cuBLASLt layout rules), not
in a unit. This fuzzer samples random combinations and checks invariants:

  * conformance (roundtrip / affine==decompress / CSR validity) for every
    registered plugin, at random (H, S, D) incl. GQA shapes, S=1, odd sizes;
  * torch_xp reference paths vs NumPy, on CUDA when available;
  * TurboQuantKVCache.fused_decode vs decompress-then-attend at random
    configs (zero-point modes, outlier fractions, hot-window sizes);
  * invalid shapes must raise cleanly (never crash, never return garbage).

Every case prints its SEED on failure -- rerun with NB_SEED to reproduce.

Usage:
    NB_MINUTES=30 python benchmarks/fuzz_plugins.py          # CPU or GPU
    NB_SEED=12345 python benchmarks/fuzz_plugins.py          # one repro case
"""

from __future__ import annotations

import os
import time
import traceback

import numpy as np

from turboquant_pro.plugin_conformance import run_conformance
from turboquant_pro.plugins import available_plugins, create

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = HAS_CUDA = False

try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

MINUTES = float(os.environ.get("NB_MINUTES", "10"))
ONE_SEED = os.environ.get("NB_SEED")

# formats fuzzable through create(); incubator specs register if importable
for pkg, mod in (
    ("plugins/tqp-bnb", "tqp_bnb"),
    ("plugins/tqp-trtllm", "tqp_trtllm"),
    ("plugins/tqp-gptq-awq", "tqp_gptq_awq"),
):
    import sys

    sys.path.insert(0, pkg)
    try:
        __import__(f"{mod}.plugin")
        from turboquant_pro.plugins import register

        m = sys.modules[f"{mod}.plugin"]
        for attr in dir(m):
            if attr.startswith("SPEC_"):
                try:
                    register(getattr(m, attr))
                except ValueError:
                    pass
    except ImportError:
        pass


def _rand_block(rng, kv=True):
    H = int(rng.choice([1, 2, 4, 8]))
    S = int(rng.choice([1, 2, 7, 33, 96, 257]))
    D = int(rng.choice([16, 32, 64, 128]))
    off = rng.uniform(-6, 6, size=(1, H, 1, D)) if kv else 0.0
    x = (off + rng.standard_normal((1, H, S, D))).astype(np.float32)
    scale = float(rng.choice([0.01, 1.0, 100.0]))
    return x * scale, H, S, D


def _mk(name, rng, H, D):
    if name == "per_channel":
        return create(
            "per_channel",
            head_dim=D,
            n_heads=H,
            nf4_asym=bool(rng.integers(2)),
            nf4=bool(rng.integers(2)),
            outlier_frac=float(rng.choice([0.0, 0.02, 0.1])),
        )
    if name == "polar":
        return None  # polar needs matched tq wiring; covered by unit tests
    if name in ("gptq", "awq"):
        return None  # weight-shaped, fuzzed separately below
    return create(name)


def fuzz_case(seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    fails: list[str] = []
    x, H, S, D = _rand_block(rng)

    for name in sorted(available_plugins(target="kv_key")):
        q = _mk(name, rng, H, D)
        if q is None:
            continue
        if name == "nvfp4_kv" and D % 16:
            try:
                q.compress(x)
                fails.append(f"seed={seed} {name}: bad-D accepted (no raise)")
            except (ValueError, NotImplementedError):
                pass
            continue
        try:
            rep = run_conformance(q, x, rel_err_max=0.95)
            if not rep.passed:
                fails.append(f"seed={seed} {name}: {rep.failures}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"seed={seed} {name}: CRASH {type(e).__name__}: {e}")

    # weight formats on weight-shaped matrices (I divisible by group)
    for name in ("gptq", "awq"):
        if name not in available_plugins():
            continue
        g = int(rng.choice([16, 64, 128]))
        n_out = int(rng.choice([1, 8, 64]))
        n_in = g * int(rng.choice([1, 2, 4]))
        w = (rng.standard_normal((1, 1, n_out, n_in)) * 0.05).astype(np.float32)
        try:
            rep = run_conformance(create(name, group_size=g), w, rel_err_max=0.95)
            if not rep.passed:
                fails.append(f"seed={seed} {name}(g={g}): {rep.failures}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"seed={seed} {name}: CRASH {type(e).__name__}: {e}")

    # torch_xp reference paths vs numpy (CUDA when available)
    if HAS_TORCH:
        from turboquant_pro.backend import to_numpy, torch_xp
        from turboquant_pro.core import TurboQuantKV
        from turboquant_pro.kv_fused import fused_decode_attention

        if D % 8 == 0:
            tq = TurboQuantKV(head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=0)
            v = rng.standard_normal((H, S, D)).astype(np.float32)
            nv = np.linalg.norm(v, axis=-1).astype(np.float32) + 1e-6
            unit = v / nv[..., None]
            rot = np.einsum("hsd,de->hse", unit, np.asarray(tq._Pi_T, dtype=np.float32))
            codes = np.searchsorted(tq.boundaries, rot).astype(np.uint8)
            qv = rng.standard_normal((H, D)).astype(np.float32)
            want = fused_decode_attention(qv, codes, codes, nv, nv, tq, xp=np)
            for dev in ["cpu"] + (["cuda"] if HAS_CUDA else []):
                targs = [
                    torch.as_tensor(a, device=dev) for a in (qv, codes, codes, nv, nv)
                ]
                try:
                    got = fused_decode_attention(*targs, tq, xp=torch_xp)
                    if not np.allclose(to_numpy(got), want, atol=1e-4):
                        fails.append(
                            f"seed={seed} torch_xp[{dev}]: mismatch "
                            f"{float(np.abs(to_numpy(got) - want).max()):.2e}"
                        )
                except Exception as e:  # noqa: BLE001
                    fails.append(
                        f"seed={seed} torch_xp[{dev}]: CRASH "
                        f"{type(e).__name__}: {e}"
                    )

    # cache dispatch exactness at random config
    from turboquant_pro.core import TurboQuantKVCache

    try:
        cache = TurboQuantKVCache(
            head_dim=D,
            n_heads=H,
            bits=4,
            use_gpu=False,
            seed=0,
            per_channel_keys=True,
            key_nf4_asym=bool(rng.integers(2)),
            key_outlier_frac=float(rng.choice([0.0, 0.02])),
            hot_window=int(rng.choice([4, 16, 64])),
        )
        n_tok = int(rng.integers(5, 60))
        for s in range(n_tok):
            cache.append(x[0, :, s % S, :], x[0, :, (s + 1) % S, :])
        qv = rng.standard_normal((H, D)).astype(np.float32)
        got = np.asarray(cache.fused_decode(qv))
        k = np.asarray(cache.get_keys(0, cache.length))[0]
        v = np.asarray(cache.get_values(0, cache.length))[0]
        sc = np.einsum("hd,hsd->hs", qv, k) / np.sqrt(D)
        p = np.exp(sc - sc.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        want = np.einsum("hs,hsd->hd", p, v)
        # scale-aware: fuzzed inputs go up to 100x scale, so outputs reach
        # O(1e3) and pure fp32 reassociation between the fused decomposition
        # and this reference is ~1e-5 relative (measured); absolute-only
        # tolerance would flag exactly that noise.
        tol = 5e-3 + 1e-4 * float(np.abs(want).max())
        if not np.allclose(got, want, atol=tol):
            fails.append(
                f"seed={seed} cache_dispatch: mismatch "
                f"{float(np.abs(got - want).max()):.2e} (tol {tol:.2e})"
            )
    except Exception as e:  # noqa: BLE001
        fails.append(f"seed={seed} cache_dispatch: CRASH {type(e).__name__}: {e}")
        traceback.print_exc()

    # same invariant through the CUDA kernels (M2/M4) when cupy exists --
    # the only randomized coverage the kernel paths get anywhere
    if HAS_CUPY and D % 32 == 0 and D <= 512:
        try:
            import cupy as cp

            cache = TurboQuantKVCache(
                head_dim=D,
                n_heads=H,
                bits=4,
                use_gpu=True,
                seed=0,
                per_channel_keys=True,
                key_nf4_asym=bool(rng.integers(2)),
                key_outlier_frac=float(rng.choice([0.0, 0.02])),
                hot_window=int(rng.choice([4, 16, 64])),
            )
            n_tok = int(rng.integers(5, 60))
            for s in range(n_tok):
                cache.append(x[0, :, s % S, :], x[0, :, (s + 1) % S, :])
            qv = rng.standard_normal((H, D)).astype(np.float32)
            got = cp.asnumpy(cp.asarray(cache.fused_decode(qv)))
            k = np.asarray(cache.get_keys(0, cache.length))[0]
            v = np.asarray(cache.get_values(0, cache.length))[0]
            sc = np.einsum("hd,hsd->hs", qv, k) / np.sqrt(D)
            p = np.exp(sc - sc.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            want = np.einsum("hs,hsd->hd", p, v)
            tol = 5e-3 + 1e-4 * float(np.abs(want).max())
            if not np.allclose(got, want, atol=tol):
                fails.append(
                    f"seed={seed} cache_dispatch_CUDA: mismatch "
                    f"{float(np.abs(got - want).max()):.2e} (tol {tol:.2e})"
                )
        except Exception as e:  # noqa: BLE001
            fails.append(
                f"seed={seed} cache_dispatch_CUDA: CRASH {type(e).__name__}: {e}"
            )

    return fails


def main() -> int:
    if ONE_SEED is not None:
        fails = fuzz_case(int(ONE_SEED))
        print("\n".join(fails) if fails else f"seed {ONE_SEED}: clean")
        return 1 if fails else 0

    t_end = time.time() + MINUTES * 60
    n = 0
    all_fails: list[str] = []
    base = int(np.random.SeedSequence().entropy % (2**31))
    print(
        f"[fuzz] budget {MINUTES} min, base seed {base}, "
        f"torch={HAS_TORCH} cuda={HAS_CUDA} cupy={HAS_CUPY}, "
        f"plugins={sorted(available_plugins())}",
        flush=True,
    )
    while time.time() < t_end:
        seed = base + n
        fails = fuzz_case(seed)
        all_fails.extend(fails)
        for f in fails:
            print(f"  FAIL {f}", flush=True)
        n += 1
        if n % 25 == 0:
            print(f"[fuzz] {n} cases, {len(all_fails)} failures", flush=True)
    print(
        f"[verdict] {n} cases fuzzed, {len(all_fails)} failures"
        + ("" if all_fails else " -- clean sweep"),
        flush=True,
    )
    return 1 if all_fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
