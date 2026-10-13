#!/usr/bin/env python3
"""Observer-advantage Part V on NRP Nautilus A10 pods, through nats-bursting. Run on Atlas.

    python part5_nrp.py setup [--dry-run]               # CPU: pinned venv tar on the volume
    python part5_nrp.py code  --commit SHA              # CPU: the pinned harness as one tar
    python part5_nrp.py stage --commit SHA --what text  # CPU: WikiText-2 (HF datasets cache)
    python part5_nrp.py stage --commit SHA --what lb    # CPU: LongBench data + config, hash-checked
    python part5_nrp.py stage --commit SHA --what qwen2.5-1.5b-instruct   # CPU: one model
    python part5_nrp.py smoke --commit SHA --model qwen2.5-1.5b-instruct [--dry-run]
    python part5_nrp.py fetch --commit SHA --model qwen2.5-1.5b-instruct --what smoke

docs/PREREG_full_consumer_kv.md, section 6. The discipline is Part III's
(``weight_observer/nrp.py``, whose preflight, submission and kubectl helpers this reuses):
CPU jobs sit in the exempt class (1 CPU, 2 GiB); GPU pods install and download nothing
(the venv, weights, text and LongBench inputs are staged by CPU jobs); requests equal
limits; ephemeral storage is declared; nothing sleeps; every model runs on an A10. A GPU
pod copies its checkpoint to local disk while the environment unpacks, then runs Part II's
runner (``keys_run.py --grid part5``) over the arms. Before submitting a GPU job, start the
Atlas-side watchers for app ``tqp-kv5``: ``part5_watch_gpu.sh`` and
``benchmarks/nrp/utilization_guard.py --selector app=tqp-kv5 --heartbeat <STATE>/guard.hb``.

The staged LongBench inputs must hash to the files Part II scored (recorded below from
Atlas, 2026-10-12); a mismatch fails the staging job.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # benchmarks/, for weight_observer
from weight_observer import nrp as W  # noqa: E402  (preflight, submit, kubectl)

import part5_grid as P  # noqa: E402

NS = W.NS
PVC = W.PVC
APP = "tqp-kv5"
BATCH = "tqp-part5-kv"
IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"  # torch 2.10.0, Part II's
GPU_PRODUCT = "NVIDIA-A10"
REPO = W.REPO
# Part II's environment (Atlas, 2026-10-12), torch from the image.
PINS = (
    "transformers==5.5.0 accelerate==1.13.0 datasets==4.3.0 sentencepiece==0.2.1 "
    "tokenizers==0.22.2 huggingface_hub==1.11.0 safetensors==0.7.0 protobuf"
)
ROOT = "/data/kv5"
STATE = "/archive/ahb-sjsu/tqp_part5"
GUARD_HEARTBEAT = os.path.join(STATE, "guard.hb")
LB_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"  # THUDM/LongBench, Part II's checkout
LB_SHA256 = {
    "data/trec.jsonl": "20a88e14f7442f58a39082cdd7396f1093ea8162da0a4ea7ead7d6757514c8f4",
    "data/triviaqa.jsonl": "ed2529de2e10b12c00f49981870c23c7c51667069cdd8d0740e1423ff337d7fa",
    "data/qasper.jsonl": "29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854",
    "LongBench/config/dataset2prompt.json": "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30",
    "LongBench/config/dataset2maxlen.json": "72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022",
}
# GPU pod shape, the smoke's pilot model stated before it is measured: the harness loads
# the checkpoint to host memory first (low_cpu_mem_usage, then .cuda()), 3.1 GB at 1.5B,
# plus the runtime; one core drives the GPU. Registered models are sized from the smoke's
# measurement scaled by size (the prereg).
SMOKE_SHAPE = {"cpu": 1, "mem_gib": 6, "eph": "12Gi"}
nl = chr(10)


def setup_script() -> str:
    return f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
if [ -f {ROOT}/env/env.tar ]; then echo "env exists"; exit 0; fi
# The 2.10 image's Python is the system 3.12 without ensurepip (kv5-setup failed on a
# plain venv, 2026-10-13): the venv borrows the image's pip and torch from the system
# site-packages, and pip installs into the venv (inside a venv it ignores the system's
# externally-managed marker).
python -m venv --without-pip --system-site-packages /tmp/venv
/tmp/venv/bin/python -m pip --version
/tmp/venv/bin/python -m pip install -q --no-cache-dir {PINS}
/tmp/venv/bin/python -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__, datasets.__version__)"
/tmp/venv/bin/python -m pip freeze > /tmp/venv/freeze.txt
mkdir -p {ROOT}/env && tar -cf {ROOT}/env/env.tar.tmp -C /tmp venv
mv {ROOT}/env/env.tar.tmp {ROOT}/env/env.tar
echo SETUP_DONE
"""


def code_script(commit: str) -> str:
    tar = f"{ROOT}/code/{commit}.tar"
    return f"""set -euo pipefail
if [ -f {tar} ]; then echo "already staged: {tar}"; exit 0; fi
git clone -q --filter=blob:none --no-checkout {REPO} /tmp/src
git -C /tmp/src sparse-checkout set --no-cone benchmarks/kvquant_matrix benchmarks/weight_observer
git -C /tmp/src checkout -q {commit}
test "$(git -C /tmp/src rev-parse HEAD)" = "{commit}"
mkdir -p {ROOT}/code
tar -cf {tar}.tmp.$$ -C /tmp/src/benchmarks kvquant_matrix weight_observer
mv {tar}.tmp.$$ {tar}
echo "staged {tar}"
"""


def _cpu_head(commit: str) -> str:
    return f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 HF_HOME={ROOT}/hf
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PYTHONPATH=/tmp/code
"""


def stage_script(commit: str, what: str) -> str:
    if what == "text":
        return _cpu_head(commit) + (
            '/tmp/venv/bin/python -c "from datasets import load_dataset; '
            "[load_dataset('wikitext', 'wikitext-2-raw-v1', split=s) for s in ('train', 'test')]; "
            "print('TEXT_DONE')\"" + nl
        )
    if what == "lb":
        checks = nl.join(
            f'echo "{h}  {ROOT}/lb/{p}" >> /tmp/lb.sha256' for p, h in LB_SHA256.items()
        )
        return _cpu_head(commit) + f"""mkdir -p /tmp/lb {ROOT}/lb/data {ROOT}/lb/LongBench/config
/tmp/venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('THUDM/LongBench', 'data.zip', repo_type='dataset', local_dir='/tmp/lb'))"
cd /tmp/lb && python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"
for t in trec triviaqa qasper; do cp /tmp/lb/data/$t.jsonl {ROOT}/lb/data/; done
git clone -q --filter=blob:none --no-checkout https://github.com/THUDM/LongBench /tmp/lbsrc
git -C /tmp/lbsrc sparse-checkout set --no-cone LongBench/config
git -C /tmp/lbsrc checkout -q {LB_COMMIT}
cp /tmp/lbsrc/LongBench/config/*.json {ROOT}/lb/LongBench/config/
: > /tmp/lb.sha256
{checks}
sha256sum -c /tmp/lb.sha256
echo LB_DONE
"""
    if what in P.MODELS:
        return _cpu_head(commit) + (
            "/tmp/venv/bin/python -m weight_observer.stage_models "
            f"--model-id {P.MODELS[what]} --dest {ROOT}/models/{what}" + nl
        )
    raise SystemExit(f"unknown stage target {what!r}")


def smoke_script(commit: str, key: str) -> str:
    """The wiring smoke (prereg section 6): every harness arm on documents 0 and 100 of
    each task and 4 perplexity chunks. Nothing is compared from it."""
    out = f"{ROOT}/runs/{commit[:12]}/smoke"
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME={ROOT}/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HARNESS_COMMIT={commit}
t() {{ echo "[t] $(date -u +%T) $*"; }}
t start
mkdir -p /tmp/code /tmp/model && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cp -r {ROOT}/models/{key} /tmp/model/{key} &
cp_model=$!
tar -xf {ROOT}/env/env.tar -C /tmp
t env unpacked
wait $cp_model
t checkpoint copied
export PATH=/tmp/venv/bin:$PATH
cd /tmp/code/kvquant_matrix
python keys_run.py --grid part5 --root {out} --models {key} --arms all --gpu 0 \\
    --docs 2 --ppl-chunks 4 --lbroot {ROOT}/lb/LongBench --datadir {ROOT}/lb/data \\
    --model-dir /tmp/model
t work done
echo SMOKE_DONE {key}
"""


def fetch_script(commit: str, key: str, what: str) -> str:
    """A run directory as one base64 gzip tar on stdout, read back with ``kubectl logs``."""
    if what != "smoke":
        raise SystemExit(f"unknown output {what!r}")
    src = f"{ROOT}/runs/{commit[:12]}/smoke/{key}"
    return f"""set -euo pipefail
cd {src}
echo FETCH_BEGIN
tar -czf - . | base64 -w0
echo
echo FETCH_END
"""


def descriptor(name, script, cpu, mem_gib, eph, role, gpu=0):
    from nats_bursting import JobDescriptor, Resources, Volume

    return JobDescriptor(
        name=name,
        image=IMAGE,
        command=["/bin/bash", "-lc", script],
        resources=Resources(cpu=str(cpu), memory=f"{mem_gib}Gi", gpu=gpu, ephemeral_storage=eph),
        labels={"app": APP, "atlas.io/batch": BATCH, "atlas.io/role": role},
        node_selector=({"nvidia.com/gpu.product": GPU_PRODUCT} if gpu else dict(W.CPU_ZONE)),
        backoff_limit=0,
        volumes=[Volume(name="data", mount_path="/data", claim_name=PVC)],
    )


def short(key: str) -> str:
    return key.replace(".", "").replace("-instruct", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("setup", "code", "stage", "smoke", "fetch"))
    ap.add_argument("--commit", default="")
    ap.add_argument("--what", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd != "setup" and len(a.commit) != 40:
        raise SystemExit("--commit takes the full 40-character SHA")
    gpu = a.cmd == "smoke"
    if a.cmd == "setup":
        desc = descriptor("kv5-setup", setup_script(), 1, 2, "10Gi", "setup")
    elif a.cmd == "code":
        desc = descriptor(f"kv5-code-{a.commit[:8]}", code_script(a.commit), 1, 2, "2Gi", "code")
    elif a.cmd == "stage":
        name = f"kv5-stage-{short(a.what)}"
        desc = descriptor(name, stage_script(a.commit, a.what), 1, 2, "20Gi", "stage")
    elif a.cmd == "smoke":
        if a.model not in P.MODELS:
            raise SystemExit(f"unknown model {a.model!r}")
        if not W.guard_alive(GUARD_HEARTBEAT):
            raise SystemExit(f"the utilization guard is not beating ({GUARD_HEARTBEAT})")
        s = SMOKE_SHAPE
        desc = descriptor(
            f"kv5-smoke-{short(a.model)}",
            smoke_script(a.commit, a.model),
            s["cpu"], s["mem_gib"], s["eph"], "smoke", gpu=1,
        )
    else:
        desc = descriptor(
            f"kv5-fetch-{short(a.model)}", fetch_script(a.commit, a.model, a.what), 1, 2, "2Gi", "fetch"
        )
    bad = W.preflight(desc, gpu)
    if bad:
        raise SystemExit(f"preflight failed: {bad}")
    print(f"{desc.name}: {desc.resources} on {desc.node_selector} (preflight clean)")
    if a.dry_run:
        print(desc.command[-1])
        return 0
    W.submit(desc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
