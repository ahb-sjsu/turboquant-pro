#!/bin/bash
# Cross-model uniform-4bit (asymmetric) with outliers+sink: completes NF4-vs-uniform comparison.
cd /root; GPU_LIST="0 1 2 3"; N=4; LOG=/root/unidriver.log
: > /root/RESULTS_UNI.txt
echo "UNI waiting for qr_uniform4 CTRL_DONE $(date)" > $LOG
while ! grep -q CTRL_DONE /root/rescuedriver.log 2>/dev/null; do sleep 30; done
echo "UNI start $(date)" >> $LOG
UQ="NOQUANT=0 CODEBOOK=uniform KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0 DATASETS=trec,triviaqa,qasper CHAT=1"
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
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_UNI.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}
rv llama7b_uni4  $UQ MODEL=NousResearch/Llama-2-7b-chat-hf  MODEL_KEY=llama2-7b-chat-4k
rv llama13b_uni4 $UQ MODEL=NousResearch/Llama-2-13b-chat-hf MODEL_KEY=llama2-13b-chat-4k
rv mistral_uni4  $UQ MODEL=mistralai/Mistral-7B-Instruct-v0.2 MODEL_KEY=mistral-7b-instruct
echo "ALL_UNI_DONE $(date)" >> $LOG
