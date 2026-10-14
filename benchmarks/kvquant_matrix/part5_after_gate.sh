#!/bin/bash
# Part V, one Tier A model: when its gate wave is done, fetch the model's cells, score G1
# (score_part5.py), and on PASS start its verdict chain (g0, verdict, head, baselines) at
# the registered harness; the chain holds until the prereg is merged. A G1 that does not
# pass starts nothing: it is a finding, explained under Part II's Amendment 5 first.
# Runs on Atlas, in a screen session:
#   bash part5_after_gate.sh <model key> <gate commit>
set -uo pipefail
MK=${1:?model key}
GATE_COMMIT=${2:?gate commit}
export KUBECONFIG=/home/claude/.kube/config
cd "$(dirname "$0")"
S=/archive/ahb-sjsu/tqp_part5
ROOT=$S/run_part5
PY=/home/claude/env/bin/python
HARNESS=$($PY -c 'import part5_nrp as N; print(N.HARNESS)')
KEY=$($PY -c "import part5_nrp as N; print(N.short('$MK'))")
LOG=$S/after_gate_$KEY.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
say "waiting for $MK's gate wave at ${GATE_COMMIT:0:12}"
until $PY -c "import json,sys; d=json.load(open('$S/chain_$KEY.json')); sys.exit(0 if 'gate' in d.get('$GATE_COMMIT', []) else 1)" 2>/dev/null; do
  sleep 300
done
say "gate done; fetching"
for try in 1 2 3 4 5; do
  $PY part5_nrp.py fetch-set --commit "$HARNESS" --model "$MK" --what run >> "$LOG" 2>&1 \
    && bash part5_fetch.sh "kv5-fetchset-$KEY" "$ROOT/$MK" >> "$LOG" 2>&1 && break
  say "fetch attempt $try failed"; sleep 300
done
$PY score_part5.py --root "$ROOT" --lbroot /home/claude/item4c/LongBench/LongBench \
  --out "$S/score_gate_$KEY.json" >> "$LOG" 2>&1 || { say "STOP: the scorer failed"; exit 1; }
G1=$($PY -c "import json; print(json.load(open('$S/score_gate_$KEY.json'))['gates']['G1']['$MK']['status'])")
say "G1 $MK: $G1"
if [ "$G1" != "PASS" ]; then
  say "STOP: G1 $G1 on $MK; no verdict wave starts"
  exit 1
fi
exec $PY part5_chain.py --commit "$HARNESS" --model "$MK" --waves tiera >> "$LOG" 2>&1
