#!/bin/bash
# Keeps the 1T post-build driver alive: if its screen is gone and the run is not finished, start
# it again (state is checkpointed per phase and in-flight jobs are adopted). Logs one line per
# check. Never touches jobs itself; the driver owns those.
#   screen -dmS tqp-1t-shepherd -L -Logfile /home/claude/tqp_fleet/shepherd1t_post.log bash -c './shepherd1t_post.sh'
set -u
cd /home/claude/tqp_fleet
export KUBECONFIG=/home/claude/.kube/config
DRIVER_LOG=/home/claude/tqp_fleet/driver1t_post.log
while true; do
  if grep -q 'DRIVER1T_POST_DONE\|POST stopped' "$DRIVER_LOG" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) driver finished; shepherd exits"
    exit 0
  fi
  if ! screen -ls 2>/dev/null | grep -q '\.tqp-1t-post\b'; then
    echo "$(date -u +%FT%TZ) driver screen gone; restarting"
    screen -dmS tqp-1t-post -L -Logfile "$DRIVER_LOG" bash -c 'export KUBECONFIG=/home/claude/.kube/config; cd /home/claude/tqp_fleet && TQP_POOL_MAXPAR=20 python3 driver1t_post.py; echo POST_EXIT=$?'
  else
    echo "$(date -u +%FT%TZ) ok: $(grep -c 'DONE aqx-' "$DRIVER_LOG") done, $(grep -c RECYCLE "$DRIVER_LOG") recycles, $(grep -c 'GAVE UP' "$DRIVER_LOG") gave up"
  fi
  sleep 600
done
