#!/bin/bash
cd /root; GPU_LIST="0 1 2 3"; N=4; LOG=/root/nf4a.log
: > /root/RESULTS_NF4A.txt; echo "NF4A start $(date)" > $LOG
NF4A="NOQUANT=0 CODEBOOK=nf4a KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0 DATASETS=trec,triviaqa,qasper CHAT=1"
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
  while [ "$(grep -h -l SHARD_._DONE /root/shard_${TAG}_*.log 2>/dev/null|wc -l)" -lt "$N" ] && [ "$(pgrep -f tq_paper_lb_shard|wc -l)" != "0" ]; do sleep 8; done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_NF4A.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}
rv llama7b_nf4a  $NF4A MODEL=NousResearch/Llama-2-7b-chat-hf   MODEL_KEY=llama2-7b-chat-4k
rv qwen_nf4a     $NF4A MODEL=Qwen/Qwen2.5-7B-Instruct          MODEL_KEY=qwen2.5-7b-instruct
rv mistral_nf4a  $NF4A MODEL=mistralai/Mistral-7B-Instruct-v0.2 MODEL_KEY=mistral-7b-instruct
rv llama13b_nf4a $NF4A MODEL=NousResearch/Llama-2-13b-chat-hf  MODEL_KEY=llama2-13b-chat-4k
echo "ALL_NF4A_DONE $(date)" >> $LOG
