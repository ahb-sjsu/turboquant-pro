#!/bin/bash
# Observer-advantage Part II, the registered execution order on Atlas GPU 1
# (docs/PREREG_observer_advantage_keys.md section 6). Resumable: rerunning skips
# every completed cell. Run inside screen:
#   screen -dmS keys -L -Logfile /archive/ahb-sjsu/keys/screen.log bash keys_campaign.sh
set -u
ROOT=${ROOT:-/archive/ahb-sjsu/keys}
PY=${PY:-$HOME/env/bin/python}
# Amendment 2: each Tier A model runs every cell on one GPU product. Atlas (GV100)
# runs Llama-2 and Tier B; Mistral and Qwen run on Colab A100 (keys_colab.ipynb).
TIER_A=${TIER_A:-llama2-7b-chat-4k}
cd "$(dirname "$0")"
# 1. every arm a verdict reads, on the three Tier A models (G0 and fp16 first)
$PY keys_run.py --root "$ROOT" --models $TIER_A --arms priority --gpu 1
# 2. Tier B, every arm
$PY keys_run.py --root "$ROOT" --models tierB --arms all --gpu 1
# 3. the reported arms on Tier A
$PY keys_run.py --root "$ROOT" --models $TIER_A --arms all --gpu 1
echo "CAMPAIGN_DONE $(date -u)" >> "$ROOT/run.log"
