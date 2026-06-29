#!/bin/bash
# Recovery + expansion driver. Redo all 12 WikiText perplexities (clean, consistent set)
# then the expanded LongBench breadth. Methods: fp16 / NF4 / asym-NF4. Auto-detects GPUs.
cd /root
N=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
GPU_LIST=$(seq 0 $((N-1))); LOG=/root/recover.log
: > /root/RESULTS_PPL.txt; : > /root/RESULTS_EXPAND.txt
echo "RECOVER start $(date)" > $LOG

declare -A MID MKEY
MID[llama7b]=NousResearch/Llama-2-7b-chat-hf;    MKEY[llama7b]=llama2-7b-chat-4k
MID[llama13b]=NousResearch/Llama-2-13b-chat-hf;  MKEY[llama13b]=llama2-13b-chat-4k
MID[mistral]=mistralai/Mistral-7B-Instruct-v0.2; MKEY[mistral]=mistral-7b-instruct
MID[qwen]=Qwen/Qwen2.5-7B-Instruct;              MKEY[qwen]=qwen2.5-7b-instruct
MODELS="qwen mistral llama7b llama13b"   # qwen first: the high-value perplexity
m_fp16="NOQUANT=1"
m_nf4="NOQUANT=0 CODEBOOK=nf4  KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"
m_nf4a="NOQUANT=0 CODEBOOK=nf4a KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0"

# ---- WikiText-2 perplexity: 4 models x 3 methods, 4 jobs at a time ----
echo "== PPL $(date) ==" >> $LOG
j=0
for mdl in $MODELS; do
  for meth in fp16 nf4 nf4a; do
    gpu=$((j % N)); v="m_$meth"; ENV="${!v}"
    env CUDA_VISIBLE_DEVICES=$gpu MODEL=${MID[$mdl]} MODEL_KEY=${MKEY[$mdl]} TAG=${mdl}_${meth} SEQLEN=2048 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENV \
        python /root/wikitext_ppl.py >> /root/RESULTS_PPL.txt 2>/root/ppl_${mdl}_${meth}.err &
    j=$((j+1)); [ $((j % N)) -eq 0 ] && wait
  done
done
wait
echo "PPL_DONE $(date)" >> $LOG

# ---- expanded LongBench tasks (sharded per model,method) ----
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
  while [ "$(grep -h -l SHARD_._DONE /root/shard_${TAG}_*.log 2>/dev/null|wc -l)" -lt "$N" ] && [ "$(pgrep -f tq_paper_lb_shard|wc -l)" != "0" ]; do sleep 8; done
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_EXPAND.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}
for mdl in $MODELS; do
  for meth in fp16 nf4 nf4a; do
    v="m_$meth"; ENV="${!v}"
    rv ${mdl}_x_${meth} $ENV MODEL=${MID[$mdl]} MODEL_KEY=${MKEY[$mdl]}
  done
done
echo "ALL_RECOVER_DONE $(date)" >> $LOG
