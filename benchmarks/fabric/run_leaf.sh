#!/usr/bin/env bash
# The leaf experiment, run on Atlas: record the fabric, start the responder, send
# the client to NRP through nats-bursting (it reaches the hub over the leaf
# link), then check what the recorder saw against what the client counted.
#
#   bash benchmarks/fabric/run_leaf.sh OUT_DIR REHEARSAL_DIR
#
# Preconditions: the rehearsal passed (REHEARSAL_DIR/analysis.json), and the
# preflight in submit_leaf_echo.py passes (it refuses otherwise).
set -euo pipefail
OUT=${1:?out dir}
REH=${2:?rehearsal dir}
PY_NATS=${PY_NATS:-/home/claude/env/bin/python}
PY_TQP=${PY_TQP:-/home/claude/tqp-ip-env/bin/python}
PREFIX=tqp.fabric.exp.leaf.$(date +%s)
mkdir -p "$OUT"
export PYTHONPATH=$PWD

$PY_NATS benchmarks/fabric/submit_leaf_echo.py --prefix "$PREFIX" \
  --footprint "$REH/client.json" > "$OUT/preflight.json"  # dry run: refuses on any rule

$PY_TQP -m turboquant_pro.cli fabric --interval 2 --duration 900 \
  --record "$OUT/record.jsonl" > "$OUT/recorder.log" 2>&1 &
REC=$!
sleep 6  # baseline polls of the idle leaf
$PY_NATS benchmarks/fabric/leaf_echo.py responder --url nats://localhost:4222 \
  --prefix "$PREFIX" --out "$OUT/responder.json" --max-seconds 840 --linger 6 \
  > "$OUT/responder.log" 2>&1 &
RESP=$!
sleep 2
$PY_NATS benchmarks/fabric/submit_leaf_echo.py --prefix "$PREFIX" \
  --footprint "$REH/client.json" --submit > "$OUT/submit.json"
wait $RESP
kill $REC 2>/dev/null || true  # the record so far is complete line by line
wait $REC 2>/dev/null || true
$PY_TQP benchmarks/fabric/analyze.py "$OUT/record.jsonl" "$OUT/responder.json" \
  --path leaf --out "$OUT/analysis.json" > /dev/null
echo "prefix $PREFIX"
