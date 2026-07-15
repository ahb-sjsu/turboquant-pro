#!/bin/bash
cd ~/item4c
export LBROOT=$HOME/item4c/LongBench/LongBench
export DATADIR=$HOME/item4c/data
export OUTROOT=$HOME/item4c
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cb in nf4a nf4a_bias nf4a_sparse; do
  TAG="zp_${cb}"
  rm -rf $OUTROOT/out_$TAG; mkdir -p $OUTROOT/out_$TAG
  echo "=== $TAG start $(date) ===" >> zp_run.log
  env CUDA_VISIBLE_DEVICES=1 CODEBOOK=$cb TAG=$TAG NOQUANT=0 \
      KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0 \
      DATASETS=qasper CHAT=1 MODEL=Qwen/Qwen2.5-7B-Instruct \
      MODEL_KEY=qwen2.5-7b-instruct SHARD_ID=0 NUM_SHARDS=1 \
      $HOME/env/bin/python tq_zp_lb_shard.py > shard_$TAG.log 2>&1
  $HOME/env/bin/python tq_zp_agg.py $TAG >> zp_run.log 2>&1
  echo "=== $TAG done $(date) ===" >> zp_run.log
done
echo "ALL_ZP_DONE $(date)" >> zp_run.log
