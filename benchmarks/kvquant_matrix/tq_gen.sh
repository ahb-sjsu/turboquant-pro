#!/bin/bash
cd /root
GPU_LIST="0 1 2 3"; N=4
LOG=/root/gendriver.log
echo "GEN start $(date)" > $LOG
: > /root/RESULTS_GEN.txt
COMMON_Q="NOQUANT=0 CODEBOOK=nf4 KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02"

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
  python /root/tq_enh_agg.py $TAG >> /root/RESULTS_GEN.txt 2>&1
  echo "done $TAG $(date)" >> $LOG
}
wait_dl(){ local tag=$1; echo "waiting dl $tag $(date)">>$LOG; while ! grep -q DLDONE /root/dl_$tag.log 2>/dev/null; do sleep 20; done; echo "dl $tag ready $(date)">>$LOG; }

do_model(){
  local pfx=$1 MODEL=$2 MKEY=$3
  local BASE="MODEL=$MODEL MODEL_KEY=$MKEY CHAT=1 DATASETS=trec,triviaqa,qasper"
  rv ${pfx}_fp16    $BASE NOQUANT=1
  rv ${pfx}_nf4post $BASE $COMMON_Q PREROPE=0
  rv ${pfx}_nf4pre  $BASE $COMMON_Q PREROPE=1
}

wait_dl qwen;     do_model qwen     Qwen/Qwen2.5-7B-Instruct        qwen2.5-7b-instruct
wait_dl llama13b; do_model llama13b NousResearch/Llama-2-13b-chat-hf llama2-13b-chat-4k
wait_dl mistral;  do_model mistral  mistralai/Mistral-7B-Instruct-v0.2 mistral-7b-instruct
echo "ALL_GEN_DONE $(date)" >> $LOG
