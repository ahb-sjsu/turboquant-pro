#!/bin/bash
cd /root; TAG=qr_uniform4
rm -rf /root/out_$TAG; mkdir -p /root/out_$TAG; rm -f /root/shard_${TAG}_*.log
sid=0
for i in 1 2 3; do
  env CUDA_VISIBLE_DEVICES=$i SHARD_ID=$sid NUM_SHARDS=3 TAG=$TAG \
    MODEL=Qwen/Qwen2.5-7B-Instruct MODEL_KEY=qwen2.5-7b-instruct CHAT=1 DATASETS=trec,triviaqa,qasper \
    NOQUANT=0 CODEBOOK=uniform KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python /root/tq_paper_lb_shard.py > /root/shard_${TAG}_$sid.log 2>&1 &
  sid=$((sid+1)); sleep 8
done
while [ "$(grep -h -l SHARD_._DONE /root/shard_${TAG}_*.log 2>/dev/null|wc -l)" -lt 3 ] && [ "$(pgrep -f tq_paper_lb_shard|wc -l)" != "0" ]; do sleep 8; done
python /root/tq_enh_agg.py $TAG > /root/RESULTS_CTRL.txt 2>&1
echo "CTRL_DONE $(date)" >> /root/rescuedriver.log
