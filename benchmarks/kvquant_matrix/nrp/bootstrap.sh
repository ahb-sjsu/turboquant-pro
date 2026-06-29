#!/usr/bin/env bash
# =============================================================================
# Track A in-pod entrypoint: KV-quant LongBench sweep for ONE model.
#
# Runs the method x shard matrix for the single MODEL given by env, over the
# methods in $METHODS (default: fp16 nf4 nf4a uniform kvquant kivi), writing
# per-task scores + raw predictions into the PVC at $CACHE/results/$MODEL_KEY/.
#
# It adapts the logic of ../tq_expand.sh / ../tq_nf4a.sh to a single-model,
# single-pod, N-GPU shard layout (one shard per visible GPU) and reuses the
# UNMODIFIED runner (tq_paper_lb_shard.py) and aggregator (tq_enh_agg.py),
# which hardcode /root/LongBench/LongBench and /root/out_$TAG -- so we stage
# code + LongBench at exactly those paths.
#
# Required env : MODEL, MODEL_KEY
# Optional env : METHODS, DATASETS, NUM_SHARDS, TQ_HOME, REPO_URL, REPO_REF,
#                CACHE, HF_HOME, MAXLEN
# =============================================================================
set -uo pipefail
echo "[bootstrap] start $(date -u) host=$(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "[bootstrap] WARN: no nvidia-smi"

# ---- config ----------------------------------------------------------------
: "${MODEL:?set MODEL (HF id)}"
: "${MODEL_KEY:?set MODEL_KEY (LongBench config key)}"
REPO_URL="${REPO_URL:-https://github.com/ahb-sjsu/turboquant-pro}"
REPO_REF="${REPO_REF:-master}"
TQ_HOME="${TQ_HOME:-/workspace/turboquant-pro}"
CACHE="${CACHE:-/cache}"
export HF_HOME="${HF_HOME:-$CACHE/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
METHODS="${METHODS:-fp16 nf4 nf4a uniform kvquant kivi}"
# Full LongBench English subset that tq_enh_agg.py knows how to score.
DATASETS="${DATASETS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
RESULTS="$CACHE/results/$MODEL_KEY"
mkdir -p "$HF_HOME" "$RESULTS"

# one shard per visible GPU unless overridden
if [ -z "${NUM_SHARDS:-}" ]; then
  NUM_SHARDS="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)"
fi
[ "${NUM_SHARDS:-0}" -ge 1 ] 2>/dev/null || NUM_SHARDS=1
echo "[bootstrap] MODEL=$MODEL KEY=$MODEL_KEY METHODS='$METHODS' NUM_SHARDS=$NUM_SHARDS HF_HOME=$HF_HOME"

# ---- OS deps (pytorch runtime image lacks git/curl) ------------------------
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y --no-install-recommends git curl ca-certificates >/dev/null
fi

# ---- python deps NOT in the base image -------------------------------------
# torch is already in pytorch/pytorch:*-runtime; do NOT reinstall it.
# transformers pin: 4.44.x supports Llama-2 / Mistral-v0.2 / Qwen2.5 and the
# DynamicCache.update + apply_rotary_pos_emb monkeypatches the runner relies on.
pip install --no-cache-dir -q \
  "transformers==4.44.2" "tokenizers>=0.19,<0.21" "accelerate>=0.33,<1.1" \
  "datasets>=2.18" sentencepiece protobuf einops tiktoken \
  rouge jieba fuzzywuzzy "python-Levenshtein" || { echo "[bootstrap] pip failed"; exit 1; }

# ---- code ------------------------------------------------------------------
if [ ! -d "$TQ_HOME/.git" ]; then
  echo "[bootstrap] cloning $REPO_URL@$REPO_REF -> $TQ_HOME"
  rm -rf "$TQ_HOME"; mkdir -p "$(dirname "$TQ_HOME")"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$TQ_HOME"
fi
SRC="$TQ_HOME/benchmarks/kvquant_matrix"
# stage runner + aggregator at the hardcoded /root paths they expect
cp "$SRC/tq_paper_lb_shard.py" /root/tq_paper_lb_shard.py
cp "$SRC/tq_enh_agg.py"        /root/tq_enh_agg.py

# ---- LongBench config + metrics.py (v1 files live under LongBench/) ---------
# tq_enh_agg.py does: sys.path.insert(0,"/root/LongBench/LongBench"); from metrics import ...
# tq_paper_lb_shard.py reads $LBROOT/config/dataset2{prompt,maxlen}.json.
if [ ! -f /root/LongBench/LongBench/config/dataset2prompt.json ]; then
  rm -rf /root/LongBench
  git clone --depth 1 https://github.com/THUDM/LongBench /root/LongBench
fi
export LBROOT=/root/LongBench/LongBench
ls "$LBROOT/config/dataset2prompt.json" "$LBROOT/metrics.py" || { echo "[bootstrap] LongBench layout unexpected"; exit 1; }

# ---- LongBench per-task data (data.zip -> data/<task>.jsonl) ----------------
# Canonical distribution (HF datasets dropped script-based loading); unzips to
# /root/lb_data/data/<task>.jsonl, matching DATADIR.
export DATADIR=/root/lb_data/data
if [ ! -d "$DATADIR" ] || [ -z "$(ls -A "$DATADIR" 2>/dev/null)" ]; then
  mkdir -p /root/lb_data
  echo "[bootstrap] fetching LongBench data.zip"
  curl -fSL --retry 4 -o /root/lb_data/data.zip \
    https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
  python - <<'PY'
import zipfile
zipfile.ZipFile("/root/lb_data/data.zip").extractall("/root/lb_data")
print("[bootstrap] extracted LongBench data")
PY
fi
ls "$DATADIR" | head

# ---- method env table ------------------------------------------------------
# fp16 = no quant (reference). The recommended production method is nf4a.
# kvquant / kivi are the new CODEBOOK methods being added in parallel; the
# runner is assumed to accept them. They are passed sensible defaults here.
declare -A MENV
MENV[fp16]="NOQUANT=1"
MENV[nf4]="NOQUANT=0 CODEBOOK=nf4    KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
MENV[nf4a]="NOQUANT=0 CODEBOOK=nf4a  KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
MENV[uniform]="NOQUANT=0 CODEBOOK=uniform KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
MENV[kvquant]="NOQUANT=0 CODEBOOK=kvquant KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
MENV[kivi]="NOQUANT=0 CODEBOOK=kivi   KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=0 OUTLIER_FRAC=0.0  PREROPE=0"

run_method() {
  local meth="$1"
  local extra="${MENV[$meth]:-}"
  if [ -z "$extra" ]; then echo "[bootstrap] unknown method '$meth' -- skipping"; return; fi
  local TAG="${MODEL_KEY}_${meth}"
  echo "=== $TAG $(date -u) :: $extra ==="
  rm -rf "/root/out_$TAG"; mkdir -p "/root/out_$TAG"; rm -f /root/shard_${TAG}_*.log
  local sid=0
  while [ "$sid" -lt "$NUM_SHARDS" ]; do
    env CUDA_VISIBLE_DEVICES="$sid" SHARD_ID="$sid" NUM_SHARDS="$NUM_SHARDS" TAG="$TAG" CHAT=1 \
        MODEL="$MODEL" MODEL_KEY="$MODEL_KEY" DATASETS="$DATASETS" \
        LBROOT="$LBROOT" DATADIR="$DATADIR" ${MAXLEN:+MAXLEN="$MAXLEN"} \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $extra \
        python /root/tq_paper_lb_shard.py > "/root/shard_${TAG}_${sid}.log" 2>&1 &
    sid=$((sid+1)); sleep 8
  done
  # robust barrier: all shards print SHARD_<id>_DONE, or no runner remains (crash)
  while :; do
    d=$(grep -l "SHARD_._DONE" /root/shard_${TAG}_*.log 2>/dev/null | wc -l)
    [ "$d" -ge "$NUM_SHARDS" ] && break
    [ "$(pgrep -fc tq_paper_lb_shard 2>/dev/null || echo 0)" = "0" ] && break
    sleep 15
  done
  # persist raw predictions + shard logs + aggregated score to the PVC
  mkdir -p "$RESULTS/$meth"
  cp /root/out_$TAG/*.jsonl    "$RESULTS/$meth/" 2>/dev/null || true
  cp /root/shard_${TAG}_*.log  "$RESULTS/$meth/" 2>/dev/null || true
  echo "[bootstrap] aggregating $TAG"
  python /root/tq_enh_agg.py "$TAG" 2>&1 | tee "$RESULTS/$meth/score.txt"
}

for meth in $METHODS; do run_method "$meth"; done
echo "ALL_DONE $MODEL_KEY $(date -u)" | tee "$RESULTS/ALL_DONE.txt"
echo "[bootstrap] results in $RESULTS"
