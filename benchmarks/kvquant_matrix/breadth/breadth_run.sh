#!/bin/bash
# Qwen2.5-7B breadth sweep: 7 LongBench tasks x {nf4a, fp16, nf4}, ONE GPU,
# resumable per (arm, task). Replaces the retracted 7-task aggregate in the KV
# paper (CHANGELOG Errata 2026-08-15) with a clean end-to-end run under the
# harness's config-sidecar guard.
#
# Usage on Atlas (inside tmux):
#   SUFFIX=smoke NUM_SHARDS=100 TASKS=2wikimqa ARMS="fp16 nf4a" bash breadth_run.sh
#   SUFFIX=full bash breadth_run.sh
# NUM_SHARDS=1 -> the full 200-doc split; NUM_SHARDS=100 -> docs 0 and 100 (smoke).
#
# Thermal: never start a cell above 75C; a watchdog SIGSTOPs the worker above
# 83C and SIGCONTs below 72C (Atlas Z840 coolers cannot hold a GV100 at 99%).
set -u
BASE=${BASE:-/archive/c12/breadth}
SUFFIX=${SUFFIX:-run}
export HARNESS_DIR=$BASE
export LBROOT=${LBROOT:-$HOME/item4c/LongBench/LongBench}
export DATADIR=${DATADIR:-/archive/longbench/data}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-1}
PY=${PY:-$HOME/env/bin/python}
NSH=${NUM_SHARDS:-1}
TASKS=${TASKS:-"2wikimqa hotpotqa multifieldqa_en narrativeqa samsum multi_news gov_report"}
ARMS=${ARMS:-"nf4a fp16 nf4"}
MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_KEY=qwen2.5-7b-instruct
declare -A ENVS
ENVS[fp16]="NOQUANT=1"
ENVS[nf4]="NOQUANT=0 CODEBOOK=nf4 KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
ENVS[nf4a]="NOQUANT=0 CODEBOOK=nf4a KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
LOG=$BASE/breadth_${SUFFIX}.log

gpu_temp () { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES"; }
cool_wait () {
  while [ "$(gpu_temp)" -gt 75 ]; do
    echo "[breadth] $(date -u +%H:%M) GPU $(gpu_temp)C, waiting to cool" | tee -a "$LOG"
    sleep 30
  done
}
watchdog () {  # pause the worker above 83C, resume below 72C
  local pid=$1 paused=0 t
  while kill -0 "$pid" 2>/dev/null; do
    t=$(gpu_temp)
    if [ "$paused" = 0 ] && [ "$t" -gt 83 ]; then
      kill -STOP "$pid"; paused=1
      echo "[watchdog] $(date -u +%H:%M) GPU ${t}C -> paused" | tee -a "$LOG"
    elif [ "$paused" = 1 ] && [ "$t" -lt 72 ]; then
      kill -CONT "$pid"; paused=0
      echo "[watchdog] $(date -u +%H:%M) GPU ${t}C -> resumed" | tee -a "$LOG"
    fi
    sleep 20
  done
}

echo "=== breadth start $(date -u) GPU=$CUDA_VISIBLE_DEVICES NSH=$NSH arms=[$ARMS] tasks=[$TASKS] harness=$(sha256sum "$BASE/tq_paper_lb_shard.py" | cut -c1-16)" | tee -a "$LOG"
for arm in $ARMS; do
  for task in $TASKS; do
    OUT=$BASE/out_${SUFFIX}_${arm}
    f=$OUT/$task.0.jsonl
    if [ -s "$f" ] && [ -f "$OUT/$task.DONE" ]; then
      echo "skip $arm/$task (done, $(wc -l < "$f") rows)" | tee -a "$LOG"; continue
    fi
    cool_wait
    echo "=== $(date -u +%H:%M) start $arm/$task" | tee -a "$LOG"
    CELL_LOG=$BASE/log_${SUFFIX}_${arm}_${task}.log
    env ${ENVS[$arm]} MODEL=$MODEL MODEL_KEY=$MODEL_KEY TAG=breadth_${arm} DATASETS=$task CHAT=1 \
        SHARD_ID=0 NUM_SHARDS=$NSH BREADTH_OUT=$OUT \
        "$PY" "$BASE/breadth_arm.py" > "$CELL_LOG" 2>&1 &
    pid=$!
    watchdog $pid & wd=$!
    wait $pid; rc=$?
    kill $wd 2>/dev/null
    if [ $rc = 0 ] && grep -q SHARD_0_DONE "$CELL_LOG"; then
      touch "$OUT/$task.DONE"
      echo "=== $(date -u +%H:%M) done $arm/$task ($(wc -l < "$f") rows, $(grep -o 'finished in [0-9]*s' "$CELL_LOG"))" | tee -a "$LOG"
    else
      echo "=== $(date -u +%H:%M) FAILED $arm/$task rc=$rc (see $CELL_LOG)" | tee -a "$LOG"
      tail -5 "$CELL_LOG" | tee -a "$LOG"
    fi
  done
done
"$PY" "$BASE/score_breadth.py" "$SUFFIX" | tee -a "$LOG"
echo "BREADTH_DONE $(date -u)" | tee -a "$LOG"
