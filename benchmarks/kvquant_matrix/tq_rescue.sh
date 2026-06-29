#!/bin/bash
# Qwen2.5-7B KV-quant rescue / diagnosis sweep. Isolates WHY 4-bit collapses:
#   bit-depth (8/3/2), outlier-channel (10% outliers), long-context fraction (short ctx),
#   finer groups (16). Waits for the generalization matrix to finish first.
cd /root
GPU_LIST="0 1 2 3"; N=4
LOG=/root/rescuedriver.log
: > /root/RESULTS_RESCUE.txt
echo "RESCUE waiting for ALL_GEN_DONE $(date)" > $LOG
while ! grep -q ALL_GEN_DONE /root/gendriver.log 2>/dev/null; do sleep 30; done
echo "RESCUE start $(date)" >> $LOG
QMODEL="MODEL=Qwen/Qwen2.5-7B-Instruct MODEL_KEY=qwen2.5-7b-instruct CHAT=1 DATASETS=trec,triviaqa,qasper"

rv() {
  local TAG=$1; shift
  echo "=== $TAG $(date) ===" >> $LOG
  rm -rf /root/out_$TAG; mkdir -p /root/out_$TAG; rm -f /root/shard_${TAG}_*.log
  local sid=0
  for i in $GPU_LIST; do
    env CUDA_VISIBLE_DEVICES=$i SHARD_ID=$sid NUM_SHARDS=$N TAG=$TAG \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" \
        python /root/tq_paper_lb_shard.py > /root/shard_${TAG}_$sid.log 2>&1 &
    sid=$((sid+1)); sleep 8
  done
  while true; do
    d=$(grep -h -l "SHARD_._DONE" /root/shard_${TAG}_*.log 2>/dev/null | wc -l)
    [ "$d" -ge "$N" ] && break
    [ "$(pgrep -f tq_paper_lb_shard|wc -l)" = "0" ] && break
    sleep 8
  done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_RESCUE.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}

# bit-depth ladder (uniform works at all bit widths)
rv qr_8bit  $QMODEL NOQUANT=0 CODEBOOK=uniform KEY_BITS=8 VAL_BITS=8 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0
rv qr_3bit  $QMODEL NOQUANT=0 CODEBOOK=uniform KEY_BITS=3 VAL_BITS=3 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0
# outlier-channel hypothesis: 10% fp16 outliers at 4-bit nf4
rv qr_4bit_o10 $QMODEL NOQUANT=0 CODEBOOK=nf4 KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.10 PREROPE=0
# long-context-fraction hypothesis: truncate to Llama-length 3500
rv qr_4bit_short $QMODEL NOQUANT=0 CODEBOOK=nf4 KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0 MAXLEN=3500
# 8-bit values only (keys 4-bit nf4): isolates whether VALUES are the killer
rv qr_k4v8  $QMODEL NOQUANT=0 CODEBOOK=nf4 KEY_BITS=4 VAL_BITS=8 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0
echo "ALL_RESCUE_DONE $(date)" >> $LOG
