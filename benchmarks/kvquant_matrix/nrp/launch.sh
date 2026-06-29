#!/usr/bin/env bash
# =============================================================================
# Manual launcher for the NRP Nautilus KV-quant LongBench sweep.
#
# Renders a Job template (sed-substitutes placeholders) and applies it.
# YOU run this -- the build step does not. Always dry-run first.
#
# Usage:
#   ./launch.sh validate                       # offline + server dry-run all manifests
#   ./launch.sh dryrun  <model> [gpu] [ngpu]   # server-side dry-run one baseline job
#   ./launch.sh baseline <model> [gpu] [ngpu]  # apply one baseline job
#   ./launch.sh latency [gpu]                  # apply the Track B latency job
#   ./launch.sh all-baselines                  # apply all 4 baseline jobs (default GPUs)
#
# <model> in: llama2-7b | llama2-13b | mistral | qwen
# [gpu]   in: NVIDIA-RTX-A6000 (default) | NVIDIA-L40S | NVIDIA-GeForce-RTX-3090
# [ngpu]  shard/GPU count (default 2)
# =============================================================================
set -euo pipefail
NS=ssu-atlas-ai
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# model -> "HF_ID|MODEL_KEY|DEFAULT_GPU"
declare -A M
M[llama2-7b]="NousResearch/Llama-2-7b-chat-hf|llama2-7b-chat-4k|NVIDIA-GeForce-RTX-3090"
M[llama2-13b]="NousResearch/Llama-2-13b-chat-hf|llama2-13b-chat-4k|NVIDIA-RTX-A6000"
M[mistral]="mistralai/Mistral-7B-Instruct-v0.2|mistral-7b-instruct|NVIDIA-RTX-A6000"
M[qwen]="Qwen/Qwen2.5-7B-Instruct|qwen2.5-7b-instruct|NVIDIA-RTX-A6000"

render_baseline() {  # args: model gpu ngpu  -> prints rendered yaml on stdout
  local key="$1" gpu="$2" ngpu="$3"
  IFS='|' read -r MODEL MKEY DEFGPU <<<"${M[$key]:?unknown model '$key'}"
  gpu="${gpu:-$DEFGPU}"
  local jobname="tq-lb-$(echo "$MKEY" | tr '.' '-')"   # DNS-safe (qwen2.5 -> qwen2-5)
  sed -e "s#__JOB_NAME__#${jobname}#g" \
      -e "s#__MODEL__#${MODEL}#g" \
      -e "s#__MODEL_KEY__#${MKEY}#g" \
      -e "s#__GPU_PRODUCT__#${gpu}#g" \
      -e "s#__NGPU__#${ngpu}#g" \
      "$HERE/job_baseline.yaml"
}

render_latency() { sed -e "s#__GPU_PRODUCT__#${1:-NVIDIA-RTX-A6000}#g" "$HERE/job_latency.yaml"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  validate)
    echo "## offline structural validation (PyYAML) ##"
    python "$HERE/validate.py" "$HERE/job_baseline.yaml" "$HERE/job_latency.yaml"
    echo "## server-side dry-run (requires a valid kube token) ##"
    render_baseline llama2-13b NVIDIA-RTX-A6000 2 | kubectl apply --dry-run=server -n "$NS" -f -
    render_latency NVIDIA-RTX-A6000              | kubectl apply --dry-run=server -n "$NS" -f -
    ;;
  dryrun)
    render_baseline "${1:?model}" "${2:-}" "${3:-2}" | kubectl apply --dry-run=server -n "$NS" -f -
    ;;
  baseline)
    render_baseline "${1:?model}" "${2:-}" "${3:-2}" | kubectl apply -n "$NS" -f -
    ;;
  latency)
    render_latency "${1:-}" | kubectl apply -n "$NS" -f -
    ;;
  all-baselines)
    for k in llama2-7b llama2-13b mistral qwen; do render_baseline "$k" "" 2 | kubectl apply -n "$NS" -f -; done
    ;;
  *)
    sed -n '5,20p' "$0"; exit 1 ;;
esac
