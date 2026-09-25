#!/bin/bash
# Atlas-side GPU watcher for Part III run, explore and sens pods (adapted from GET G3c watch_jobs.sh). Every
# two minutes: each running tqp-wo run pod's GPU utilization and memory via nvidia-smi in the
# pod. A job whose GPU sits under the NRP floor of 40% for three samples in a row, after ten
# minutes of age and once the weights are on the card (> 2 GiB), is diagnosed then deleted.
# Exits when no run job is active. Runs on Atlas, never in the cluster.
NS=ssu-atlas-ai
K="kubectl -n $NS --request-timeout=60s"
DIR=/archive/ahb-sjsu/tqp_weight_observer
LOG=$DIR/watch_gpu.log
mkdir -p $DIR
declare -A low
while true; do
  now=$(date -u +%FT%TZ)
  active=0
  for j in $($K get jobs -l "app=tqp-wo,atlas.io/role in (run,explore,sens)" -o name 2>/dev/null); do
    n=${j#job.batch/}
    jst=$($K get $j -o jsonpath='{.status.succeeded}/{.status.failed}/{.status.active}')
    pod=$($K get pods -l job-name=$n -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    phase=$($K get pod $pod -o jsonpath='{.status.phase}' 2>/dev/null)
    util=""; gmem=""; age=""
    if [ "$phase" = "Running" ]; then
      util=$($K exec $pod -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
      gmem=$($K exec $pod -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      st=$($K get pod $pod -o jsonpath='{.status.startTime}' 2>/dev/null)
      age=$(( $(date -u +%s) - $(date -u -d "${st:-now}" +%s 2>/dev/null || date -u +%s) ))
      if [ -n "$util" ] && [ "${gmem:-0}" -gt 2000 ] && [ "$age" -gt 600 ] && [ "$util" -lt 40 ]; then
        low[$n]=$(( ${low[$n]:-0} + 1 ))
      else
        low[$n]=0
      fi
      if [ "${low[$n]:-0}" -ge 3 ]; then
        echo "$now $n GPU under 40 percent three samples running, deleting" >> $LOG
        { echo "== $pod at $now"; $K exec $pod -- bash -c 'ps -eo pid,etimes,pcpu,args | head; nvidia-smi'; $K logs $j --tail=40; } > $DIR/lowutil_$n.txt 2>&1
        $K delete $j >> $LOG 2>&1
      fi
    fi
    echo "$now $n status=$jst phase=$phase gpu=$util gmem=${gmem:-} age=${age:-} low=${low[$n]:-0}" >> $LOG
    case "$jst" in */*/1) active=1 ;; //) active=1 ;; esac
    [ "$phase" = "Pending" ] && active=1
  done
  if [ $active -eq 0 ]; then echo "$now no active run job" >> $LOG; exit 0; fi
  sleep 120  # Atlas-side loop; no cluster job sleeps
done
