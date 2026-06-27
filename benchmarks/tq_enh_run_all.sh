#!/bin/bash
# Drive the enhanced-TurboQuant LongBench ablation across all GPUs, one variant at
# a time. Writes per-variant scores to /root/RESULTS.txt.
#
# NOTE: shards are launched with plain `&` (NOT setsid) so the driver's `wait`
# tracks them; we additionally poll for N SHARD_*_DONE markers before aggregating,
# so a variant never aggregates on partial data.
cd /root
# Pin to the working GPUs (skip faulty GPU5 on this node). Override with GPU_LIST env.
GPU_LIST=${GPU_LIST:-"0 1 2 3 4"}
N=$(echo $GPU_LIST | wc -w)
echo "GPUS=$N list=[$GPU_LIST] $(date)" > /root/driver.log
: > /root/RESULTS.txt

run_variant() {
  local TAG=$1
  echo "=== VARIANT $TAG $(date) NOQUANT=$NOQUANT KB=$KEY_BITS VB=$VAL_BITS SINK=$SINK OUT=$OUTLIER_FRAC NUQ=$NUQ ===" >> /root/driver.log
  rm -rf /root/out_$TAG; mkdir -p /root/out_$TAG
  rm -f /root/shard_${TAG}_*.log
  local sid=0
  for i in $GPU_LIST; do
    CUDA_VISIBLE_DEVICES=$i SHARD_ID=$sid NUM_SHARDS=$N TAG=$TAG \
      NOQUANT=$NOQUANT KEY_BITS=$KEY_BITS VAL_BITS=$VAL_BITS GROUP=$GROUP RESID=$RESID \
      SINK=$SINK OUTLIER_FRAC=$OUTLIER_FRAC NUQ=$NUQ \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python /root/tq_enh_lb_shard.py > /root/shard_${TAG}_$sid.log 2>&1 &
    sid=$((sid+1))
    sleep 15
  done
  # robust barrier: wait until all N shards print their DONE marker
  while true; do
    d=$(grep -h -l "SHARD_._DONE" /root/shard_${TAG}_*.log 2>/dev/null | wc -l)
    if [ "$d" -ge "$N" ]; then break; fi
    # also break if no shard procs remain (crash) to avoid hanging forever
    if [ "$(pgrep -f "TAG=$TAG" | wc -l)" = "0" ] && [ "$(pgrep -f tq_enh_lb_shard | wc -l)" = "0" ]; then break; fi
    sleep 10
  done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS.txt 2>&1
  echo "done $TAG $(date)" >> /root/driver.log
}

# fp16 reference (no quant) -- should reproduce fp16 (~62 / 84.9 / 16.5)
NOQUANT=1 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=0 OUTLIER_FRAC=0.0  NUQ=0 run_variant ref_fp16
# 2-bit uniform sanity (should drop clearly if quant is real)
NOQUANT=0 KEY_BITS=2 VAL_BITS=2 GROUP=32 RESID=128 SINK=0 OUTLIER_FRAC=0.0  NUQ=0 run_variant s_k2
# baseline 4-bit uniform per-channel (== TQ k4v4)
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=0 OUTLIER_FRAC=0.0  NUQ=0 run_variant v0_uniform
# combined: NF4 + 1% outliers + sink (the headline enhanced variant -- run early)
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=4 OUTLIER_FRAC=0.01 NUQ=1 run_variant v4_full
# ablations (which ingredient matters)
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=0 OUTLIER_FRAC=0.01 NUQ=0 run_variant v1_outlier
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=0 OUTLIER_FRAC=0.0  NUQ=1 run_variant v2_nf4
echo "ALL_VARIANTS_DONE $(date)" >> /root/driver.log
