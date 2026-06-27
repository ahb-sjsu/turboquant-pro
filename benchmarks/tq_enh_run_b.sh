#!/bin/bash
# Part (b): single-ingredient ablations + the "can we exceed KVQuant" attempt.
# Reference (from part a, same harness): fp16 qasper 22.06, KVQuant 21.06,
# v0 uniform 14.38, v4 (NF4+1%out+sink) 20.23.
cd /root
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
N=$(echo $GPU_LIST | wc -w)
echo "PARTB GPUS=$N list=[$GPU_LIST] $(date)" >> /root/driver.log
: > /root/RESULTS_B.txt

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
  while true; do
    d=$(grep -h -l "SHARD_._DONE" /root/shard_${TAG}_*.log 2>/dev/null | wc -l)
    if [ "$d" -ge "$N" ]; then break; fi
    if [ "$(pgrep -f tq_enh_lb_shard | wc -l)" = "0" ]; then break; fi
    sleep 10
  done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_B.txt 2>&1
  echo "done $TAG $(date)" >> /root/driver.log
}

# exceed-attempts first (the headline question): NF4 + 2%/3% outliers + sink
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=4 OUTLIER_FRAC=0.02 NUQ=1 run_variant v5_out2pct
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=4 OUTLIER_FRAC=0.03 NUQ=1 run_variant v6_out3pct
# single-ingredient ablations (attribute v4's gain)
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=0 OUTLIER_FRAC=0.01 NUQ=0 run_variant v1_outlier
NOQUANT=0 KEY_BITS=4 VAL_BITS=4 GROUP=32 RESID=128 SINK=0 OUTLIER_FRAC=0.0  NUQ=1 run_variant v2_nf4
echo "ALL_B_DONE $(date)" >> /root/driver.log
