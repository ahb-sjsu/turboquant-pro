"""Run ONE (arm, task) cell of the Qwen2.5-7B breadth sweep through the harness's
OWN ``main()`` -- the same pattern as the 2026-08-08 re-validation
(``/archive/c12/reval/reval_arm.py``), so the only thing under test is the arm.

Everything comes from env exactly as ``tq_expand.sh`` sets it. Overrides:

* ``HARNESS_DIR``   -- directory holding ``tq_paper_lb_shard.py`` (the sidecar-guard
                      version from the repo at the commit recorded in the log).
* ``BREADTH_OUT``   -- writable output dir (the harness hardcodes ``/root/out_<TAG>``,
                      which user ``claude`` cannot write; ``OUT_DIR`` is honoured by
                      newer harness copies but we set both to be safe).

Thermal discipline (Atlas rule): a batch-probe ``ThermalController`` bounds CPU
threads; the calling shell's watchdog pauses the process on GPU temperature.
"""

import os
import sys
import time

HARNESS_DIR = os.environ["HARNESS_DIR"]
OUT = os.environ["BREADTH_OUT"]
os.environ.setdefault("OUT_DIR", OUT)
sys.path.insert(0, HARNESS_DIR)

# The harness mkdirs its OUT at import; stub across the import, then redirect.
_real_makedirs = os.makedirs
os.makedirs = lambda *a, **k: None
try:
    import tq_paper_lb_shard as H
finally:
    os.makedirs = _real_makedirs

H.OUT = OUT
os.makedirs(H.OUT, exist_ok=True)

# ---- thermal: CPU thread cap via batch-probe (non-negotiable on Atlas) -------
thermal = None
try:
    import torch
    from batch_probe import ThermalController

    thermal = ThermalController(
        target_temp=82.0, max_threads=20, min_threads=4, verbose=False
    )
    thermal.start()
    _n = thermal.get_threads()
    torch.set_num_threads(_n)
    os.environ["OMP_NUM_THREADS"] = str(_n)
    print(f"[breadth] ThermalController on, threads={_n}", flush=True)
except Exception as e:  # pragma: no cover - Atlas-only
    print(f"[breadth] ThermalController unavailable: {e!r}; capping threads at 8", flush=True)
    import torch

    torch.set_num_threads(8)

# ---- Volta accommodation, identical to reval_arm.py --------------------------
# Volta has no flash kernel and the memory-efficient kernel refuses enable_gqa,
# so transformers' GQA path would fall back to MATH and materialise a 28 x T x T
# attention matrix (OOM at T~15k on 32 GiB). Forcing the repeat_kv path keeps the
# memory-efficient kernel eligible. Selects a KERNEL, not an algorithm; the
# fp16 arm reproduced the recorded values under it in August.
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from transformers.integrations import sdpa_attention as _SDPA  # noqa: E402

_SDPA.use_gqa_in_sdpa = lambda *a, **k: False

print(
    f"[breadth] TAG={H.TAG} NOQUANT={H.NOQUANT} CODEBOOK={H.CODEBOOK} KB={H.KB} "
    f"VB={H.VB} G={H.G} HOT={H.HOT} SINK={H.SINK} OUT_FRAC={H.OUT_FRAC} "
    f"PREROPE={H.PREROPE} DATASETS={H.DATASETS}",
    flush=True,
)
print(
    f"[breadth] MAXLEN={H.MAXLEN} MAXGEN_override={H._MAXGEN} "
    f"shard {H.SHARD}/{H.NSH} -> {H.OUT}",
    flush=True,
)

t0 = time.time()
try:
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        H.main()
finally:
    if thermal is not None:
        thermal.stop()
print(f"[breadth] {H.TAG} {H.DATASETS} finished in {time.time() - t0:.0f}s", flush=True)
