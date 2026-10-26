#!/usr/bin/env bash
# Rehearsal on Atlas, before anything goes to NRP: the same client and responder
# over the local hub, observed by `tqp fabric --record`, then checked by analyze.py.
# It measures the client's footprint (for sizing the NRP pod from a measurement)
# and validates the observer against known traffic on a path we control.
#
#   bash benchmarks/fabric/rehearse_atlas.sh OUT_DIR
set -euo pipefail
OUT=${1:?out dir}
PY_NATS=${PY_NATS:-/home/claude/env/bin/python}      # has nats-py
PY_TQP=${PY_TQP:-/home/claude/tqp-ip-env/bin/python} # runs turboquant_pro
PREFIX=tqp.fabric.exp.rehearsal.$(date +%s)
mkdir -p "$OUT"
export PYTHONPATH=$PWD

$PY_TQP -m turboquant_pro.cli fabric --interval 1 --duration 60 \
  --record "$OUT/record.jsonl" > "$OUT/recorder.log" 2>&1 &
REC=$!
sleep 3  # a few baseline polls before any traffic
$PY_NATS benchmarks/fabric/leaf_echo.py responder --url nats://localhost:4222 \
  --prefix "$PREFIX" --out "$OUT/responder.json" --max-seconds 50 --linger 4 \
  > "$OUT/responder.log" 2>&1 &
RESP=$!
sleep 2  # the responder subscribes, and the recorder sees it idle
/usr/bin/time -v $PY_NATS benchmarks/fabric/leaf_echo.py client \
  --url nats://localhost:4222 --prefix "$PREFIX" \
  > "$OUT/client.json" 2> "$OUT/client.time"
wait $RESP
wait $REC || true
$PY_TQP benchmarks/fabric/analyze.py "$OUT/record.jsonl" "$OUT/responder.json" \
  --path "responder:tqp-fabric-responder $PREFIX" --exact --out "$OUT/analysis.json" > /dev/null
echo "prefix $PREFIX"
grep -E 'Maximum resident|Percent of CPU|Elapsed' "$OUT/client.time"
