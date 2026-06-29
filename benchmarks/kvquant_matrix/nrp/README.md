# NRP Nautilus runner — KV-quant LongBench + kernel latency

Ready-to-apply Kubernetes artifacts to run the TurboQuant KV-quant LongBench
harness (Track A) and the fused-decode kernel latency bench (Track B) on
**free-tier** NRP Nautilus GPUs.

| file | what it is |
|---|---|
| `bootstrap.sh` | Track A in-pod entrypoint: deps + code + LongBench data, then the method×shard sweep for one model → PVC. |
| `job_baseline.yaml` | Track A Job **template** (placeholders `__JOB_NAME__ __MODEL__ __MODEL_KEY__ __GPU_PRODUCT__ __NGPU__`), one per model. |
| `latency_bench.py` | Track B: fp16 attn vs M2 warp fused kernel vs dequant — decode latency + peak GPU mem → JSON. |
| `job_latency.yaml` | Track B single-GPU Job (placeholder `__GPU_PRODUCT__`); also runs the existing `benchmark_kv_kernel.py`. |
| `launch.sh` | Manual launcher: sed-substitutes a template and `kubectl apply`s. **You** run it. |
| `validate.py` | Offline structural check (used because server-side dry-run needs interactive OIDC; see below). |

All commands pin the namespace: **`-n ssu-atlas-ai`**.

---

## How code, data, and models get into the pod

Nothing is baked into an image. Each pod, at start:

1. `apt-get install git curl`, then **shallow git-clones the public repo**
   `https://github.com/ahb-sjsu/turboquant-pro` (no auth — repo is public).
2. `pip install` the deps missing from the base image (`transformers==4.44.2`,
   `datasets`, `accelerate`, `sentencepiece`, `einops`, `tiktoken`, and the
   LongBench metric deps `rouge jieba fuzzywuzzy python-Levenshtein`). Torch is
   already in the base image and is **not** reinstalled.
3. Clones **THUDM/LongBench** to `/root/LongBench` (v1 config + `metrics.py`
   live under `/root/LongBench/LongBench/`, matching the runner's `LBROOT` and
   `tq_enh_agg.py`'s hardcoded `sys.path`).
4. Downloads **`data.zip`** from
   `https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip` and
   unzips to `/root/lb_data/data/<task>.jsonl` (matches `DATADIR`).
5. **HF models** download lazily through `transformers` into `HF_HOME=/cache/hf`
   on the **PVC** (`erebus-ego-models`, RWX) → cached & shared across pods, so
   only the first run of each model pays the download.
6. Results (per-task scores + raw predictions + shard logs) are written to the
   **PVC** at `/cache/results/<MODEL_KEY>/<method>/`.

> No HF token is needed: NousResearch Llama-2 mirrors, Mistral-v0.2 and Qwen2.5
> are public/ungated.

### Base image
**`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`** (public Docker Hub,
linux/amd64, ~2.94 GB compressed; tag confirmed to exist). Chosen because it
ships Python 3.11 + Torch 2.4.1 + CUDA 12.1 runtime — enough for the Track A
HF harness with no toolkit. For **Track B**, `cupy-cuda12x` is pip-installed in
the pod; its wheel **bundles nvrtc**, so the RawKernel JIT path works without a
system CUDA toolkit (the `-runtime`, not `-devel`, image is sufficient).

---

## GPUs (free-tier only — premium a100/h100/h200/gh200 are quota-0)

| model | `MODEL_KEY` | ctx | recommended `__GPU_PRODUCT__` | min VRAM |
|---|---|---|---|---|
| NousResearch/Llama-2-7b-chat-hf | `llama2-7b-chat-4k` | 4k | `NVIDIA-GeForce-RTX-3090` (24 GB) | ~16 GB |
| NousResearch/Llama-2-13b-chat-hf | `llama2-13b-chat-4k` | 4k | `NVIDIA-RTX-A6000` / `NVIDIA-L40S` (48 GB) | ~30 GB |
| mistralai/Mistral-7B-Instruct-v0.2 | `mistral-7b-instruct` | 31.5k | `NVIDIA-RTX-A6000` / `NVIDIA-L40S` | ~24–40 GB (long-ctx KV) |
| Qwen/Qwen2.5-7B-Instruct | `qwen2.5-7b-instruct` | 31.5k | `NVIDIA-RTX-A6000` / `NVIDIA-L40S` | ~24–40 GB |

Selected via `nodeAffinity` on `nvidia.com/gpu.product`. The 13B and the two
32k-context 7Bs want 48 GB; only Llama-2-7B@4k is comfortable on a 3090.

---

## Validate (do this first)

> **Server-side dry-run could not be run from the build environment.** This
> kube context (`nautilus`) authenticates via an **interactive OIDC exec plugin**
> (`kubectl oidc-login`); with no cached token and no browser, every API call —
> including `--dry-run=server` and even client-side validation (it downloads the
> server OpenAPI) — fails with
> `getting credentials: exec: executable kubectl failed`. Run the server dry-run
> yourself **after** authenticating:

```bash
# after a normal `kubectl get pods -n ssu-atlas-ai` has refreshed your token:
bash launch.sh validate            # offline structural check + server dry-run of both manifests
bash launch.sh dryrun llama2-13b   # server dry-run a single rendered baseline job
```

The build environment validated the manifests **offline** with `validate.py`
(PyYAML structural + Job/Job-resource sanity). Rendered with concrete values,
both pass:

```
[OK] rendered_baseline.yaml
[OK] rendered_latency.yaml
```

---

## Launch

```bash
# one model (uses the recommended default GPU + 2 shards/GPUs):
bash launch.sh baseline llama2-13b
bash launch.sh baseline qwen NVIDIA-L40S 2     # override GPU / ngpu

# all four baseline jobs:
bash launch.sh all-baselines

# Track B latency (single GPU):
bash launch.sh latency NVIDIA-RTX-A6000
```

`launch.sh` renders the job name DNS-safe (`qwen2.5-...` → `qwen2-5-...`) and
substitutes the placeholders before applying.

### Shrinking the sweep
`bootstrap.sh` defaults to the **full LongBench English subset (16 tasks)** ×
**6 methods**. To run a fast representative slice, set `DATASETS` in
`job_baseline.yaml` (a commented example is in the manifest), e.g.
`DATASETS=trec,triviaqa,qasper`, or trim `METHODS`.

---

## Monitor

```bash
kubectl get jobs,pods -n ssu-atlas-ai -l app=turboquant-kvquant
kubectl logs -f job/tq-lb-llama2-13b-chat-4k -n ssu-atlas-ai     # live progress
# look for: "=== <key>_<method> ... ===", "SHARD_n_DONE", "RESULT ...", "ALL_DONE"
```

## Collect results (from the PVC)

Results persist on `erebus-ego-models` at `/cache/results/`. Read them by
mounting the PVC in a tiny pod:

```bash
kubectl run pvc-peek -n ssu-atlas-ai --rm -it --restart=Never \
  --image=busybox --overrides='
{"spec":{"volumes":[{"name":"c","persistentVolumeClaim":{"claimName":"erebus-ego-models"}}],
"containers":[{"name":"x","image":"busybox","stdin":true,"tty":true,
"command":["sh"],"volumeMounts":[{"name":"c","mountPath":"/cache"}]}]}}'
# then:  cat /cache/results/<MODEL_KEY>/<method>/score.txt
#        cat /cache/results/latency/latency.json
```
Layout: `/cache/results/<MODEL_KEY>/<method>/{score.txt, *.jsonl, *.log}` and
`/cache/results/<MODEL_KEY>/ALL_DONE.txt`; `score.txt` holds the
`RESULT <tag> {json}` line from `tq_enh_agg.py`.

## Tear down

```bash
kubectl delete job tq-lb-llama2-13b-chat-4k -n ssu-atlas-ai   # one
kubectl delete jobs -l app=turboquant-kvquant -n ssu-atlas-ai # all (also auto-cleaned via ttlSecondsAfterFinished=3d)
```

---

## Estimated GPU-hours (full 16-task subset × 6 methods)

Greedy decode, ~200 samples/task (~3,200 prompts/method); quant methods run at
≈fp16 speed (the runner quantizes the prefill **once**). Rough single-GPU rates:

| model | ~s/sample | GPU-h /method | GPU-h × 6 methods |
|---|---:|---:|---:|
| llama2-7b-chat-4k | ~3.5 | ~3.1 | **~19** |
| llama2-13b-chat-4k | ~6 | ~5.3 | **~32** |
| mistral-7b-instruct (32k) | ~9 | ~8.0 | **~48** |
| qwen2.5-7b-instruct (32k) | ~9 | ~8.0 | **~48** |
| **total (4 models)** | | | **≈ 147 GPU-hours** |

> These are order-of-magnitude estimates; long-context prefill dominates the two
> 32k models. **Implication:** the *full* 16-task × 6-method sweep for one model
> does **not** fit in a single 6 h pod (`activeDeadlineSeconds=21600`) at
> `NGPU=2` (e.g. qwen ≈ 24 h wall). Pick one: **(a)** reduce `DATASETS` (the core
> 3 ≈ 600 prompts/method fits easily), **(b)** split `METHODS` across several
> Jobs, or **(c)** raise `NGPU` *and* bump `activeDeadlineSeconds`. The latency
> Job (Track B) is minutes.

---

## NRP-specific risks

- **Auth / dry-run:** the OIDC exec plugin blocks non-interactive API calls
  (above). Refresh your token before `launch.sh`.
- **Admission webhooks:** Nautilus enforces resource requests==limits-ish
  policies and may reject pods with no `activeDeadlineSeconds` or with idle GPUs.
  Both manifests set `activeDeadlineSeconds`, explicit CPU/mem requests+limits,
  and the GPU limit==request. If admission rejects the GPU count, lower `NGPU`.
- **Eviction:** pods are evicted if they under-request **ephemeral-storage**;
  we request 50 Gi (baseline) / 30 Gi (latency) and keep models on the PVC.
- **Image pull:** `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` is ~3 GB —
  first pull on a fresh node adds a few minutes; subsequent pods on that node
  are cached.
- **Scheduling:** free-tier GPUs are contended; a pod may `Pending` a while.
  Prefer `NGPU=1–2`; large multi-GPU requests schedule slower.
- **`data.zip` availability:** if the HF `resolve/main/data.zip` URL ever moves,
  swap the fetch in `bootstrap.sh` for the per-task `datasets.load_dataset(...)`
  path (kept as the documented fallback).
