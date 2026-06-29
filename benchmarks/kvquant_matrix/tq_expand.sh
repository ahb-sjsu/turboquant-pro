#!/bin/bash
cd /root; GPU_LIST="0 1 2 3"; N=4; LOG=/root/expand.log
: > /root/RESULTS_EXPAND.txt; echo "EXPAND start $(date)" > $LOG
declare -A MID MKEY
MID[llama7b]=NousResearch/Llama-2-7b-chat-hf;    MKEY[llama7b]=llama2-7b-chat-4k
MID[llama13b]=NousResearch/Llama-2-13b-chat-hf;  MKEY[llama13b]=llama2-13b-chat-4k
MID[mistral]=mistralai/Mistral-7B-Instruct-v0.2; MKEY[mistral]=mistral-7b-instruct
MID[qwen]=Qwen/Qwen2.5-7B-Instruct;              MKEY[qwen]=qwen2.5-7b-instruct
m_fp16="NOQUANT=1"
m_nf4="NOQUANT=0 CODEBOOK=nf4  KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
m_nf4a="NOQUANT=0 CODEBOOK=nf4a KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
TASKS="narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,gov_report,multi_news,samsum"
rv() {
  local TAG=$1; shift
  echo "=== $TAG $(date) ===" >> $LOG
  rm -rf /root/out_$TAG; mkdir -p /root/out_$TAG; rm -f /root/shard_${TAG}_*.log
  local sid=0
  for i in $GPU_LIST; do
    env CUDA_VISIBLE_DEVICES=$i SHARD_ID=$sid NUM_SHARDS=$N TAG=$TAG CHAT=1 DATASETS=$TASKS \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" \
        python /root/tq_paper_lb_shard.py > /root/shard_${TAG}_$sid.log 2>&1 &
    sid=$((sid+1)); sleep 8
  done
  while [ "$(grep -h -l SHARD_._DONE /root/shard_${TAG}_*.log 2>/dev/null|wc -l)" -lt "$N" ] && [ "$(pgrep -f tq_paper_lb_shard|wc -l)" != "0" ]; do sleep 10; done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_EXPAND.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}
for mdl in qwen mistral llama7b llama13b; do
  for meth in fp16 nf4 nf4a; do
    v="m_$meth"; ENV="${!v}"
    rv ${mdl}_x_${meth} $ENV MODEL=${MID[$mdl]} MODEL_KEY=${MKEY[$mdl]}
  done
done
echo "ALL_EXPAND_DONE $(date)" >> $LOG
