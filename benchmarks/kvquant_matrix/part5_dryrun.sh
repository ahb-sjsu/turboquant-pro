#!/bin/bash
# Dry-run every Part V submitter command (preflight only, nothing submitted). Run on Atlas.
#   bash part5_dryrun.sh <40-char commit>
set -u
C=${1:?commit}
PY=/home/claude/env/bin/python3
cd "$(dirname "$0")"
for args in "setup" "code --commit $C" "stage --commit $C --what text" \
    "stage --commit $C --what lb" "stage --commit $C --what qwen2.5-1.5b-instruct" \
    "fetch --commit $C --model qwen2.5-1.5b-instruct --what smoke"; do
  $PY part5_nrp.py $args --dry-run 2>&1 | head -1
done
$PY part5_nrp.py smoke --commit "$C" --model qwen2.5-1.5b-instruct --dry-run 2>&1 | head -2
