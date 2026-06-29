#!/bin/bash
cd /root; N=$(nvidia-smi -L|wc -l); GPU_LIST=$(seq 0 $((N-1))); LOG=/root/abl.log
: > /root/RESULTS_ABL.txt; echo "ABL start $(date)" > $LOG
QWEN=Qwen/Qwen2.5-7B-Instruct
NF4A="NOQUANT=0 CODEBOOK=nf4a KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
rv() {
  local TAG=$1; shift
  echo "=== $TAG $(date) ===" >> $LOG
  rm -rf /root/out_$TAG; mkdir -p /root/out_$TAG; rm -f /root/shard_${TAG}_*.log
  local sid=0
  for i in $GPU_LIST; do
    env CUDA_VISIBLE_DEVICES=$i SHARD_ID=$sid NUM_SHARDS=$N TAG=$TAG CHAT=1 DATASETS=gov_report \
        MODEL=$QWEN MODEL_KEY=qwen2.5-7b-instruct \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" \
        python /root/tq_paper_lb_shard.py > /root/shard_${TAG}_$sid.log 2>&1 &
    sid=$((sid+1)); sleep 6
  done
  while [ "$(grep -h -l SHARD_._DONE /root/shard_${TAG}_*.log 2>/dev/null|wc -l)" -lt "$N" ] && [ "$(pgrep -f tq_paper_lb_shard|wc -l)" != "0" ]; do sleep 8; done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_ABL.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}
# max_gen ablation on Qwen gov_report: gap should grow with generation length
for mg in 64 128 256; do
  rv qwen_gov_mg${mg}_fp16 MAXGEN=$mg NOQUANT=1
  rv qwen_gov_mg${mg}_nf4a MAXGEN=$mg $NF4A
done
echo "ALL_ABL_DONE $(date)" >> $LOG
