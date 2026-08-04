#!/bin/bash
# One-shot transition: bash wave driver -> NATS pool driver.
#
# Waits for the in-flight wave 2 (servers 8-15) to complete under driver1t.sh,
# kills its screen BEFORE it starts wave 3 (so the two drivers never own the
# same servers), then launches driver1t_pool.py for servers 16-499 in screen
# tqp-1t-pool. Run this itself in a screen (see launch_pool.sh).
#
# The post-build phases (qcache/ref/ivf/score) stay with driver1t.sh, re-run
# after POOL_BUILDS_DONE — its build waves fast-skip Complete jobs.
set -u
NS=ssu-atlas-ai

echo "=== $(date -u +%H:%M) waiting for wave 2 (servers 8-15) to complete"
while :; do
  n=0
  for i in 8 9 10 11 12 13 14 15; do
    s=$(kubectl get job tqp-fleet-1t-build-$i -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
    [ "${s:-0}" = "1" ] && n=$((n+1))
  done
  echo "=== $(date -u +%H:%M) wave 2: $n/8 complete"
  [ "$n" -eq 8 ] && break
  sleep 300
done

echo "=== $(date -u +%H:%M) wave 2 done — stopping the bash wave driver"
screen -S tqp-1t -X quit 2>/dev/null || true
sleep 5
screen -ls | grep -q tqp-1t. && echo "=== WARN tqp-1t screen still present"

echo "=== $(date -u +%H:%M) launching pool driver for servers 16-499"
cd /home/claude/tqp_fleet
screen -dmS tqp-1t-pool -L -Logfile /tmp/driver1t_pool.log \
  env TQP_POOL_FROM=16 TQP_POOL_TO=499 \
  /home/claude/agi-hpc/.venv/bin/python3 /home/claude/tqp_fleet/driver1t_pool.py
sleep 5
screen -ls | grep tqp-1t-pool && echo "=== $(date -u +%H:%M) TRANSITION_DONE"
