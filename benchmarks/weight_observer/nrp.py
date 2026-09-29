"""Part III on NRP Nautilus (namespace ssu-atlas-ai) through nats-bursting. Run on Atlas.

    python -m weight_observer.nrp setup  [--dry-run]    # CPU: pinned venv tar on the volume
    python -m weight_observer.nrp code --commit SHA      # CPU: the pinned harness as one tar
    python -m weight_observer.nrp stage --commit SHA     # CPU: model weights + WikiText text
    python -m weight_observer.nrp run --commit SHA --models qwen2.5-0.5b [--dry-run]
    python -m weight_observer.nrp explore --commit SHA --models qwen2.5-1.5b  # EXPLORATORY
    python -m weight_observer.nrp sens --commit SHA --models qwen2.5-1.5b  # EXPLORATORY
    python -m weight_observer.nrp plans --commit SHA --models qwen2.5-1.5b  # EXPLORATORY
    python -m weight_observer.nrp oracle --commit SHA --models qwen2.5-1.5b  # EXPLORATORY
    python -m weight_observer.nrp flat --commit SHA --models qwen2.5-1.5b  # EXPLORATORY
    python -m weight_observer.nrp ctables --commit SHA --models qwen2.5-0.5b  # Part III-c
    python -m weight_observer.nrp carms --commit SHA --models qwen2.5-0.5b  # Part III-c
    python -m weight_observer.nrp sizecheck --commit SHA --models gemma-2-2b  # before registration
    python -m weight_observer.nrp stage --commit SHA --models gemma-2-2b  # one model (pinned revision)
    python -m weight_observer.nrp fetch --models qwen2.5-1.5b [--what codec --commit SHA]

The GET G3c discipline (experiments/G3c/nrp/submit.py), scored in ``preflight``:
CPU jobs sit in the exempt class (1 CPU, 2 GiB); GPU pods install and download nothing (the
venv, weights, text and code are staged by CPU jobs); requests equal limits; ephemeral
storage is declared; nothing sleeps; a GPU pod's CPU and memory come from the MEASURED usage
of an earlier run of that model or, for a model never run, from the pilot's measurement
scaled by parameter count (the pilot itself is sized by a stated model and runs under the
utilization guard); each model runs on one GPU product. ``watch`` deletes a GPU job whose
utilization stays under 40% for three samples after its warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/home/claude/src/nats-bursting/python")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

NS = "ssu-atlas-ai"
PVC = "tqp-rbq-data"
APP = "tqp-wo"
BATCH = "tqp-weight-observer"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
CPU_ZONE = {"topology.kubernetes.io/zone": "ucsd-nrp"}
GPU_PRODUCT = "NVIDIA-A10"
REPO = "https://github.com/ahb-sjsu/turboquant-pro"
PINS = "transformers==4.56.1 accelerate==1.14.0 safetensors numpy huggingface_hub"
ROOT = "/data/wo"
STATE = "/archive/ahb-sjsu/tqp_weight_observer"
OBSERVATIONS = os.path.join(STATE, "observations.json")
MODELS = {  # key: (hf id, parameters in billions)
    "qwen2.5-0.5b": (
        "Qwen/Qwen2.5-0.5B",
        0.5,
    ),  # pilot: wiring and sizing, never scored
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", 1.5),
    "llama3.2-3b": ("unsloth/Llama-3.2-3B", 3.2),
    # Part III-c's registered models (docs/PREREG_weights_codec_allocation.md, section 1)
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B", 3.09),
    "gemma-2-2b": ("unsloth/gemma-2-2b", 2.61),
    "llama3.1-8b": ("unsloth/Meta-Llama-3.1-8B", 8.03),
}
# The mirror commits the prereg records: staging fetches exactly these.
REVISIONS = {
    "qwen2.5-3b": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "gemma-2-2b": "25319945f7fd83b8b903e12081777b7eef2ba993",
    "llama3.1-8b": "e9a141a2091ea561b96483212645a2a05e6f99fc",
}
REGISTERED = tuple(REVISIONS)
# One GPU product per model (the prereg): two copies of the 8B model need an A6000.
GPU_PRODUCTS = {"llama3.1-8b": "NVIDIA-RTX-A6000"}
nl = chr(10)


def gpu_product(key: str) -> str:
    return GPU_PRODUCTS.get(key, GPU_PRODUCT)


SETUP = f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
if [ -f {ROOT}/env/env.tar ]; then echo "env exists"; exit 0; fi
python -m venv --system-site-packages /tmp/venv
/tmp/venv/bin/pip install -q --no-cache-dir {PINS}
/tmp/venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
/tmp/venv/bin/pip freeze > /tmp/venv/freeze.txt
mkdir -p {ROOT}/env && tar -cf {ROOT}/env/env.tar.tmp -C /tmp venv
mv {ROOT}/env/env.tar.tmp {ROOT}/env/env.tar
df -h /data | tail -1
echo SETUP_DONE
"""


def stage_script(commit: str, what: str) -> str:
    """A CPU job that stages one thing with the pinned code: 'text' or a model key."""
    head = f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PYTHONPATH=/tmp/code
"""
    if what == "text":
        return head + nl.join(
            [
                "/tmp/venv/bin/pip install -q --no-cache-dir datasets==4.3.0",
                f"/tmp/venv/bin/python -m weight_observer.stage_text --dest {ROOT}/text",
                "",
            ]
        )
    hf = MODELS[what][0]
    rev = f" --revision {REVISIONS[what]}" if what in REVISIONS else ""
    return head + (
        "/tmp/venv/bin/python -m weight_observer.stage_models "
        f"--model-id {hf}{rev} --dest {ROOT}/models/{what}" + nl
    )


def code_script(commit: str) -> str:
    tar = f"{ROOT}/code/{commit}.tar"
    return f"""set -euo pipefail
if [ -f {tar} ]; then echo "already staged: {tar}"; exit 0; fi
git clone -q --filter=blob:none --no-checkout {REPO} /tmp/src
git -C /tmp/src sparse-checkout set --no-cone benchmarks/weight_observer
git -C /tmp/src checkout -q {commit}
test "$(git -C /tmp/src rev-parse HEAD)" = "{commit}"
mkdir -p {ROOT}/code
tar -cf {tar}.tmp.$$ -C /tmp/src/benchmarks weight_observer
mv {tar}.tmp.$$ {tar}
echo "staged {tar}"
"""


def run_script(commit: str, key: str, tag: str = "", pilot_env: str = "") -> str:
    """``tag`` and ``pilot_env`` (``K=V;K=V``, values may hold commas) are for pilots only."""
    pairs = [kv.split("=", 1) for kv in pilot_env.split(";") if kv]
    exports = " ".join(f"{k}='{v}'" for k, v in pairs)
    out = f"{key}-{tag}" if tag else key
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{"export " + exports if exports else ""}
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m weight_observer.run --model-key {key} --model-path {ROOT}/models/{key} \\
    --text {ROOT}/text --out {ROOT}/runs/{out}
echo RUN_DONE {key}
"""


def explore_script(commit: str, key: str) -> str:
    """EXPLORATORY (explore.py): the exact-Fisher pass, then its score against the run's KL."""
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m weight_observer.explore --model-path {ROOT}/models/{key} \
    --text {ROOT}/text --out {ROOT}/explore/{key}
python -m weight_observer.explore_score --run {ROOT}/runs/{key} --explore {ROOT}/explore/{key}
echo EXPLORE_SCORED {key}
"""


def sens_script(commit: str, key: str) -> str:
    """EXPLORATORY (sensitivity.py): single-matrix KLs, then the oracle scored beside the rest."""
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m weight_observer.sensitivity --model-path {ROOT}/models/{key} \
    --text {ROOT}/text --out {ROOT}/explore/{key}
python -m weight_observer.explore_score --run {ROOT}/runs/{key} \
    --explore {ROOT}/explore/{key} --sens {ROOT}/explore/{key}/sensitivity.jsonl
echo SENS_SCORED {key}
"""


def plans_script(commit: str, key: str) -> str:
    """EXPLORATORY (plans.py eval): measured KL of the exact plans committed in planned/."""
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m weight_observer.plans eval --model-path {ROOT}/models/{key} \
    --text {ROOT}/text --plans /tmp/code/weight_observer/planned/{key}.json \
    --out {ROOT}/explore/{key}
echo PLANS_EVALUATED {key}
"""


ORACLE_CHECK = "oracle,fisher,exact_block,fisher_tok"


def oracle_script(commit: str, key: str) -> str:
    """EXPLORATORY (plans.py eval --only): the oracle plans measured in one pod beside the
    Fisher plan and its two nearest rivals, into ``oracle_check/``. The additive prediction
    of the oracle's headroom (under 1%) is smaller than additivity's own error (about 10%),
    so only a measurement can say; re-measuring the Fisher plans in the same pod also shows
    whether a plan's KL reproduces across pods. Sixteen plans keep the GPU busy well past
    the model load, which four alone would not."""
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m weight_observer.plans eval --model-path {ROOT}/models/{key} \
    --text {ROOT}/text --plans /tmp/code/weight_observer/planned/{key}.json \
    --only {ORACLE_CHECK} --out {ROOT}/explore/{key}/oracle_check
echo ORACLE_CHECKED {key}
"""


def flat_script(commit: str, key: str) -> str:
    """EXPLORATORY (flatness.py): the Fisher plan and its budget-exact swap perturbations
    (``planned/<key>.flatness.json``, 76 plans) measured in one pod, into ``flatness/``.
    """
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m weight_observer.plans eval --model-path {ROOT}/models/{key} \\
    --text {ROOT}/text --plans /tmp/code/weight_observer/planned/{key}.flatness.json \\
    --out {ROOT}/explore/{key}/flatness
echo FLATNESS_MEASURED {key}
"""


def revision_check(key: str) -> str:
    """For a registered model, a line that ends the job unless the staged weights are the
    mirror commit the prereg records (``stage_models`` writes it to STAGED.json)."""
    if key not in REVISIONS:
        return ""
    m = f"{ROOT}/models/{key}/STAGED.json"
    return (
        f'grep -q \'"revision": "{REVISIONS[key]}"\' {m} '
        f'|| {{ echo "{key}: staged weights are not {REVISIONS[key][:12]}"; exit 1; }}'
        + nl
    )


def _codec_head(commit: str, key: str, out: str) -> str:
    """The small code tar first; then gate G0 on this GPU (weight_observer.g0_device,
    torch from the image) runs while the environment unpacks, and the job waits for its
    verdict: a failed gate ends the job (set -e) before any arm is spent."""
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{revision_check(key)}mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH=/tmp/code python -m weight_observer.g0_device \\
    --model-path {ROOT}/models/{key} --out {out} &
g0=$!
cp -r {ROOT}/models/{key} {LOCAL_MODEL} &
cp_model=$!
tar -xf {ROOT}/env/env.tar -C /tmp
wait $cp_model
wait $g0
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
"""


# The checkpoint is copied to the pod's own disk while G0 keeps the GPU busy: CephFS
# serves one sequential copy at ~370 MB/s but the loaders' scattered reads at ~50-80
# MB/s, which left the GPU idle for 4-5 minutes per load (sizecheck, 2026-09-29).
LOCAL_MODEL = "/tmp/model"
EPHEMERAL = {"llama3.1-8b": "24Gi"}  # the copy (16.1 GB) plus the environment (~2 GB)


def ephemeral(key: str) -> str:
    return EPHEMERAL.get(key, "20Gi")


def ctables_script(commit: str, key: str) -> str:
    """Part III-c (codec_run tables): the sample hashes, then every codec's cost table."""
    return (
        _codec_head(commit, key, codec_dir(key, commit))
        + f"""python -m weight_observer.codec_run tables \\
    --model-key {key} --model-path {LOCAL_MODEL} --text {ROOT}/text \\
    --out {codec_dir(key, commit)}
echo CTABLES_DONE {key}
"""
    )


# Pilot only: the prereg's comparisons re-measured in fp32, the precision check of the
# fp16 harness (sizecheck.PRECISION_INVARIANCE_TOL), before registration.
PRECISION_ARMS = ",".join(
    f"{a}{b}"
    for b in (3, 4)
    for a in ("gptq_f", "gptq_u", "awq_u", "rtn_f", "gptq_frtn")
)


def carms_script(commit: str, key: str, dtype: str = "float16") -> str:
    """Part III-c (codec_run arms): the arms of the plans committed in planned/, encoded
    and measured; the plans travel in the pinned code tar. Another ``dtype`` (pilot
    only) measures ``PRECISION_ARMS`` into its own directory."""
    out = codec_dir(key, commit)
    extra = ""
    if dtype != "float16":
        out = f"{out}/{dtype}"
        extra = f" --dtype {dtype} --only {PRECISION_ARMS}"
    return (
        _codec_head(commit, key, out) + f"""python -m weight_observer.codec_run arms \\
    --model-key {key} --model-path {LOCAL_MODEL} --text {ROOT}/text \\
    --arms-file /tmp/code/weight_observer/planned/{key}.codec_arms.json \\
    --out {out}{extra}
echo CARMS_DONE {key}
"""
    )


def sizecheck_script(commit: str, key: str, reference: str = "") -> str:
    """Before registration (sizecheck.py): the shape probe of what the model's tables and
    arms jobs hold on this GPU, then the reference-precision check on its scored windows;
    with ``reference`` (e.g. float32), that check alone against that reference. G0 runs
    while the environment unpacks and the checkpoint is copied, as for every codec job.
    """
    out = sizecheck_dir(key, commit)
    m = LOCAL_MODEL
    ref = f"python -m weight_observer.sizecheck refcheck --model-path {m} --text {ROOT}/text --out {out}"
    if reference:
        body = f"{ref} --reference {reference}\n"
    else:
        body = f"python -m weight_observer.sizecheck memprobe --model-path {m} --out {out}\n{ref}\n"
    return _codec_head(commit, key, out) + body + f"echo SIZECHECK_DONE {key}\n"


def sizecheck_dir(key: str, commit: str) -> str:
    return f"{ROOT}/sizecheck/{key}/{commit[:12]}"


def codec_dir(key: str, commit: str) -> str:
    """Part III-c output of one model BY THE CODE THAT MADE IT: the jobs resume per
    matrix and per arm, so a directory shared across commits would keep results of old
    code (2026-09-28: the pilot's tables predated the grid fix)."""
    return f"{ROOT}/codec/{key}/{commit[:12]}"


def fetch_script(key: str, what: str = "explore", commit: str = "") -> str:
    """An output directory (explore, or Part III-c's codec or sizecheck at ``commit``) as
    one base64 gzip tar on stdout, read back with ``kubectl logs``."""
    by_commit = {"codec": codec_dir, "sizecheck": sizecheck_dir}
    if what != "explore" and what not in by_commit:
        raise ValueError(f"unknown output {what!r}")
    if what in by_commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{what} output is fetched by the full commit that made it")
    src = by_commit[what](key, commit) if what in by_commit else f"{ROOT}/explore/{key}"
    return f"""set -euo pipefail
cd {src}
echo FETCH_BEGIN
tar -czf - . | base64 -w0
echo
echo FETCH_END
"""


# ----------------------------------------------------------------------------- sizing


def measured(key: str):
    """(cpu cores mean, mem GiB mean, mem GiB peak) of a finished run of this model."""
    try:
        obs = json.load(open(OBSERVATIONS, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows = [o for n, o in obs.items() if n.startswith(f"wo-run-{key.replace('.', '')}")]
    rows = [o for o in rows if o.get("mean_cpu_cores") and o.get("mean_mem_gib")]
    if not rows:
        return None
    return (
        min(o["mean_cpu_cores"] for o in rows),
        min(o["mean_mem_gib"] for o in rows),
        max(o.get("peak_mem_gib") or o["mean_mem_gib"] for o in rows),
    )


def request(key: str):
    """(cpu, mem GiB, why). Measured if this model ran; else the pilot scaled by size; the
    pilot itself by a stated model (host RAM holds two fp16 copies while loading)."""
    m = measured(key)
    if m:
        cpu, mean, peak = m
        mem = max(int(peak * 1.25 + 0.999), 1)
        return (
            max(1, round(cpu / 0.6)),
            mem,
            f"measured: {cpu:.2f} cores, {mean:.1f}/{peak:.1f} GiB",
        )
    pilot = measured("qwen2.5-0.5b")
    if key != "qwen2.5-0.5b":
        if not pilot:
            raise SystemExit(f"{key}: the pilot has not been measured yet")
        cpu, mean, peak = pilot
        s = MODELS[key][1] / MODELS["qwen2.5-0.5b"][1]
        mem = int((peak - 1.5) * s + 1.5 + 0.999) + 1
        return max(1, round(cpu / 0.6)), mem, f"pilot scaled x{s:.1f}"
    return 2, 6, "pilot model: 2 x 1 GB fp16 in host RAM while loading, + 3 GiB runtime"


# Peak host RSS (GiB) of loading the model twice straight to the GPU (run.load), torch and
# the CUDA context included; measured on Atlas GV100 2026-09-27 (1 copy 1.90, the old
# host-first path 2.76). Part III-c jobs are forward-only, so this is their host footprint.
DIRECT_LOAD_PEAK = {"qwen2.5-0.5b": 1.95}
EXEMPT = (1, 2)  # NRP exempt class: requests above 2 GiB are deleted in some windows
DIRECT_LOAD_SINCE = (
    "3a65a01b6960613c6b56b5018b67e5f277299dc1"  # run.load goes to the GPU
)


def has_direct_load(commit: str) -> bool:
    """Whether the pinned code loads straight to the GPU, the path DIRECT_LOAD_PEAK
    measured: an older tar loads to host first and would not fit the exempt class."""
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", DIRECT_LOAD_SINCE, commit],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
    )
    return r.returncode == 0


# Pilot-class models: never scored in Part III-c, so one may run unmeasured to BE the
# measurement (the registered models are sized from these, after the pilot, per the prereg).
MEASURE_PILOTS = ("qwen2.5-0.5b", "qwen2.5-1.5b")


def records_host_mem(commit: str) -> bool:
    """Whether the pinned code writes host_mem.jsonl (weight_observer.hostmem)."""
    r = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:benchmarks/weight_observer/hostmem.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
    )
    return r.returncode == 0


GUARD_HEARTBEAT = os.path.join(STATE, "utilization_guard.heartbeat")
GUARD_MAX_AGE = 300  # seconds; the guard beats every 30


def guard_alive(path: str = GUARD_HEARTBEAT, now: float | None = None) -> bool:
    """Whether the utilization guard (benchmarks/nrp/utilization_guard.py, report-only) is
    recording: a job class whose usage was never measured goes out only while it is."""
    try:
        age = (time.time() if now is None else now) - os.path.getmtime(path)
    except OSError:
        return False
    return age <= GUARD_MAX_AGE


def codec_request(key: str, commit: str, measure: bool = False):
    """(cpu, mem GiB, why) for ctables/carms: the exempt class, only where measured to fit
    and only for code that loads the way it was measured; with ``measure``, a pilot-class
    model may run unmeasured if the pinned code records its own host memory."""
    if not has_direct_load(commit):
        raise SystemExit(
            f"{commit[:12]} predates direct loading ({DIRECT_LOAD_SINCE[:12]}) "
            "or is unknown here: the exempt sizing does not hold for it"
        )
    peak = DIRECT_LOAD_PEAK.get(key)
    if peak is None and measure:
        if key not in MEASURE_PILOTS:
            raise SystemExit(f"{key}: only pilot-class models run to be measured")
        if not records_host_mem(commit):
            raise SystemExit(f"{commit[:12]} does not record host memory")
        return (*EXEMPT, "UNMEASURED pilot: the job records host_mem.jsonl")
    if peak is None:
        raise SystemExit(
            f"{key}: no direct-load host memory measurement (pilot-class: --measure-host)"
        )
    if peak > EXEMPT[1]:
        raise SystemExit(f"{key}: direct-load peak {peak} GiB exceeds the exempt class")
    return (*EXEMPT, f"exempt: direct-load peak {peak:.2f} GiB (two copies)")


def preflight(desc, gpu: bool) -> list:
    bad = []
    r = desc.resources
    if not r.ephemeral_storage:
        bad.append("ephemeral-storage not declared")
    if desc.backoff_limit != 0:
        bad.append("backoff_limit is not 0")
    script = " ".join(desc.command)
    if re.search(r"\bsleep\b", script):
        bad.append("the command contains sleep")
    if gpu and any(
        w in script
        for w in ("pip install", "git clone", "snapshot_download", "apt-get")
    ):
        bad.append("a GPU job installs or downloads")
    if not gpu and (float(r.cpu) > 1 or float(str(r.memory).rstrip("Gi")) > 2):
        bad.append("a CPU job outside the exempt class")
    return bad


def descriptor(
    name, script, cpu, mem_gib, eph, role, gpu=0, image=IMAGE, product=GPU_PRODUCT
):
    from nats_bursting import JobDescriptor, Resources, Volume

    return JobDescriptor(
        name=name,
        image=image,
        command=["/bin/bash", "-lc", script],
        resources=Resources(
            cpu=str(cpu), memory=f"{mem_gib}Gi", gpu=gpu, ephemeral_storage=eph
        ),
        labels={"app": APP, "atlas.io/batch": BATCH, "atlas.io/role": role},
        node_selector=({"nvidia.com/gpu.product": product} if gpu else dict(CPU_ZONE)),
        backoff_limit=0,
        volumes=[Volume(name="data", mount_path="/data", claim_name=PVC)],
    )


def kubectl(*args):
    return subprocess.run(
        ["kubectl", "-n", NS, "--request-timeout=60s", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def submit(desc) -> None:
    from nats_bursting import Client

    kubectl("delete", "job", desc.name, "--ignore-not-found")
    t0 = time.time()
    with Client() as c:
        c.submit(desc)
    for _ in range(60):
        r = kubectl(
            "get", "job", desc.name, "-o", "jsonpath={.metadata.creationTimestamp}"
        )
        if r.returncode == 0 and r.stdout.strip():
            ts = (
                time.mktime(time.strptime(r.stdout.strip(), "%Y-%m-%dT%H:%M:%SZ"))
                - time.timezone
            )
            if ts < t0 - 120:
                raise SystemExit(
                    f"{desc.name}: stale creationTimestamp {r.stdout.strip()}"
                )
            print(desc.name, "created", r.stdout.strip(), flush=True)
            return
        time.sleep(5)
    raise SystemExit(f"{desc.name}: no job appeared")


SCRIPTS = {
    "explore": explore_script,
    "sens": sens_script,
    "plans": plans_script,
    "oracle": oracle_script,
    "flat": flat_script,
    "ctables": ctables_script,
    "carms": carms_script,
    "sizecheck": sizecheck_script,
}


def job_name(
    cmd: str, key: str, dtype: str = "float16", reference: str = "", tag: str = ""
) -> str:
    """A GPU job's name: every variant gets its own, because the breaker keys its queue
    on names (a repeat under new code needs a ``tag``)."""
    name = f"wo-{cmd}-{key.replace('.', '')}"
    if cmd == "carms" and dtype != "float16":
        name += "-fp32"
    if cmd == "sizecheck" and reference:
        name += "-ref-" + {"float32": "fp32", "bfloat16": "bf16"}[reference]
    if cmd == "sizecheck" and tag:
        name += f"-{tag}"
    return name


def sizecheck_request(key: str, commit: str):
    """(cpu, mem GiB, why): a registered model's sizecheck runs in the exempt class,
    unmeasured, because it IS the measurement (it records its own host and GPU peaks);
    like any unmeasured class it goes out only while the utilization guard records."""
    if key not in REGISTERED:
        raise SystemExit(f"{key}: sizecheck is for the registered models {REGISTERED}")
    if not has_direct_load(commit):
        raise SystemExit(f"{commit[:12]} predates direct loading or is unknown here")
    return (*EXEMPT, "UNMEASURED sizecheck: records its own host and GPU peaks")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=(
            "setup",
            "stage",
            "code",
            "run",
            "explore",
            "sens",
            "plans",
            "oracle",
            "flat",
            "ctables",
            "carms",
            "sizecheck",
            "fetch",
        ),
    )
    ap.add_argument("--commit", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--tag", default="", help="pilot runs only: output and job suffix")
    ap.add_argument("--pilot-env", default="", help="pilot only: K=V;K=V overrides")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--what", default="explore", help="fetch: explore, codec or sizecheck"
    )
    ap.add_argument(
        "--measure-host",
        action="store_true",
        help="ctables/carms: let an unmeasured pilot-class model run to be measured",
    )
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "float32"),
        help="carms, pilot only: float32 measures PRECISION_ARMS as the fp16 check",
    )
    ap.add_argument(
        "--reference",
        default="",
        choices=("", "bfloat16", "float32"),
        help="sizecheck: only the reference-precision check, against this reference",
    )
    a = ap.parse_args(argv)
    if a.dtype != "float16" and (a.cmd != "carms" or a.models != "qwen2.5-0.5b"):
        raise SystemExit("--dtype is for the pilot's carms only (qwen2.5-0.5b)")
    if a.reference and a.cmd != "sizecheck":
        raise SystemExit("--reference is for sizecheck only")
    items = []
    if a.cmd == "setup":
        items.append((descriptor("wo-setup", SETUP, 1, 2, "12Gi", "setup"), False))
    elif a.cmd == "code":
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit("--commit must be a full sha")
        items.append(
            (
                descriptor(
                    f"wo-code-{a.commit[:12]}",
                    code_script(a.commit),
                    1,
                    2,
                    "2Gi",
                    "code",
                    image="python:3.12",  # has git; the PyTorch image does not
                ),
                False,
            )
        )
    elif a.cmd == "stage":
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit(
                "--commit must be a full sha (the staging code is pinned too)"
            )
        for what in a.models.split(",") if a.models else ("text", *MODELS):
            if what != "text" and what not in MODELS:
                raise SystemExit(f"unknown model {what!r}")
            name = f"wo-stage-{what.replace('.', '')}"
            items.append(
                (
                    descriptor(
                        name, stage_script(a.commit, what), 1, 2, "12Gi", "stage"
                    ),
                    False,
                )
            )
    elif a.cmd == "run":
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit("--commit must be a full sha")
        for key in a.models.split(","):
            cpu, mem, why = request(key)
            d = descriptor(
                f"wo-run-{key.replace('.', '')}" + (f"-{a.tag}" if a.tag else ""),
                run_script(a.commit, key, a.tag, a.pilot_env),
                cpu,
                mem,
                "20Gi",
                "run",
                gpu=1,
                product=gpu_product(key),
            )
            print(d.name, cpu, f"{mem}Gi", gpu_product(key), "|", why)
            items.append((d, True))
    elif a.cmd in SCRIPTS:
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit("--commit must be a full sha")
        for key in a.models.split(","):
            if a.cmd in ("ctables", "carms", "sizecheck"):
                if a.cmd == "sizecheck":
                    cpu, mem, why = sizecheck_request(key, a.commit)
                else:
                    cpu, mem, why = codec_request(key, a.commit, a.measure_host)
                if not a.dry_run and not guard_alive():
                    raise SystemExit(
                        f"the utilization guard is not recording ({GUARD_HEARTBEAT})"
                    )
            else:
                if not measured(key):
                    raise SystemExit(
                        f"{key}: {a.cmd} is sized from a measured run of it"
                    )
                cpu, mem, why = request(key)
            name, script = job_name(a.cmd, key, a.dtype, a.reference, a.tag), None
            if a.cmd == "carms" and a.dtype != "float16":
                script = carms_script(a.commit, key, a.dtype)
            if a.cmd == "sizecheck" and a.reference:
                script = sizecheck_script(a.commit, key, a.reference)
            d = descriptor(
                name,
                script or SCRIPTS[a.cmd](a.commit, key),
                cpu,
                mem,
                ephemeral(key),
                a.cmd,
                gpu=1,
                product=gpu_product(key),
            )
            print(d.name, cpu, f"{mem}Gi", gpu_product(key), "|", why)
            items.append((d, True))
    elif a.cmd == "fetch":
        for key in a.models.split(","):
            n = f"wo-fetch-{key.replace('.', '')}" + (
                f"-{a.what}-{a.commit[:8]}" if a.what != "explore" else ""
            )
            script = fetch_script(key, a.what, a.commit)
            items.append((descriptor(n, script, 1, 2, "2Gi", "fetch"), False))
    bad = {d.name: preflight(d, g) for d, g in items}
    if any(bad.values()):
        raise SystemExit(
            "PREFLIGHT VETO " + json.dumps({k: v for k, v in bad.items() if v})
        )
    print(f"{a.cmd}: {len(items)} job(s) pass preflight")
    if a.dry_run:
        return 0
    for d, _ in items:
        submit(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
