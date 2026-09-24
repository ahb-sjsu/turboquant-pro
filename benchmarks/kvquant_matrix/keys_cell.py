#!/usr/bin/env python3
"""Run ONE LongBench cell of observer-advantage Part II through the harness's own main().

The 2026-08-08 re-validation pattern (``breadth/breadth_arm.py``): everything comes
from env (keys_run.py sets it from keys_grid), the harness is imported unchanged,
and the only additions are Atlas's thermal and Volta accommodations:

* a batch-probe ``ThermalController`` bounds CPU threads (Atlas rule);
* on Volta only: it has no flash kernel and the memory-efficient kernel refuses ``enable_gqa``,
  so transformers' GQA path would fall back to MATH and materialise a heads x T x T
  matrix; forcing the repeat_kv path keeps the memory-efficient kernel eligible.
  It selects a kernel, not an algorithm (the fp16 arm reproduced the recorded
  values under it in August).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tq_paper_lb_shard as H  # noqa: E402

import torch  # noqa: E402

thermal = None
try:
    from batch_probe import ThermalController

    thermal = ThermalController(target_temp=82.0, max_threads=20, min_threads=4,
                                verbose=False)
    thermal.start()
    torch.set_num_threads(thermal.get_threads())
except Exception as e:  # pragma: no cover - Atlas-only
    print(f"[keys] ThermalController unavailable: {e!r}; capping threads at 8", flush=True)
    torch.set_num_threads(8)

from contextlib import nullcontext  # noqa: E402

ctx = nullcontext()
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
    # Volta only (Ampere and later have kernels that take GQA natively). It selects a
    # kernel, not an algorithm, and each model runs every cell on one GPU product.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers.integrations import sdpa_attention as _SDPA

        _SDPA.use_gqa_in_sdpa = lambda *a, **k: False
        ctx = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    except Exception as e:  # older transformers: no GQA switch to turn off
        print(f"[keys] no Volta SDPA accommodation ({e!r})", flush=True)

t0 = time.time()
try:
    with ctx:
        H.main()
finally:
    if thermal is not None:
        thermal.stop()
print(f"[keys] {H.TAG} {H.MODEL_KEY} {H.DATASETS} finished in {time.time() - t0:.0f}s",
      flush=True)
