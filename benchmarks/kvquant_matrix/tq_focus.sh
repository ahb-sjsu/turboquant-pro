#!/bin/bash
cd /root; N=$(nvidia-smi -L|wc -l); GPU_LIST=$(seq 0 $((N-1))); LOG=/root/focus.log
: > /root/RESULTS_FOCUS.txt; echo "FOCUS start $(date) N=$N" > $LOG
LLAMA7=NousResearch/Llama-2-7b-chat-hf; QWEN=Qwen/Qwen2.5-7B-Instruct
NF4A="NOQUANT=0 CODEBOOK=nf4a KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
rv() {
  local TAG=$1; shift
  echo "=== $TAG $(date) ===" >> $LOG
  rm -rf /root/out_$TAG; mkdir -p /root/out_$TAG; rm -f /root/shard_${TAG}_*.log
  local sid=0
  for i in $GPU_LIST; do
    env CUDA_VISIBLE_DEVICES=$i SHARD_ID=$sid NUM_SHARDS=$N TAG=$TAG CHAT=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" \
        python /root/tq_paper_lb_shard.py > /root/shard_${TAG}_$sid.log 2>&1 &
    sid=$((sid+1)); sleep 6
  done
  while [ "$(grep -h -l SHARD_._DONE /root/shard_${TAG}_*.log 2>/dev/null|wc -l)" -lt "$N" ] && [ "$(pgrep -f tq_paper_lb_shard|wc -l)" != "0" ]; do sleep 8; done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_FOCUS.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}
# Exp A: is the long-gen gap Llama-specific? Llama-7B on the 2 sensitive (512-gen) tasks
rv l7_sum_fp16 MODEL=$LLAMA7 MODEL_KEY=llama2-7b-chat-4k DATASETS=gov_report,multi_news NOQUANT=1
rv l7_sum_nf4a MODEL=$LLAMA7 MODEL_KEY=llama2-7b-chat-4k DATASETS=gov_report,multi_news $NF4A
# Exp B: max_gen ablation on Qwen gov_report (does the gap grow with generation length?)
for mg in 64 128 256; do
  rv qwen_gov_mg${mg}_fp16 MODEL=$QWEN MODEL_KEY=qwen2.5-7b-instruct DATASETS=gov_report MAXGEN=$mg NOQUANT=1
  rv qwen_gov_mg${mg}_nf4a MODEL=$QWEN MODEL_KEY=qwen2.5-7b-instruct DATASETS=gov_report MAXGEN=$mg $NF4A
done
echo "ALL_FOCUS_DONE $(date)" >> $LOG
