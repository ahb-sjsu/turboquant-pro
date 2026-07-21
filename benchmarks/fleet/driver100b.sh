#!/bin/bash
# 100B measurement driver (runs detached on Atlas, in a screen session):
# 50 servers x 400 shards x 5M rows = 100B, in waves of 8 CPU-pegged jobs
# per phase: build -> query cache -> exact full-scan reference -> routed-IVF
# partials -> exact merge/score. No serve window (the 10B run showed it
# measures shard-open latency, not the index). Resumable: completed jobs are
# skipped by name, and every fleet_*.py partial is idempotent on disk.
#
# Sweep-resilient: NRP's utilization enforcement deletes job OBJECTS whose pods
# sit below the ~20% CPU floor during their image-pull/init window. A build that
# gets caught then "vanishes" mid-wave. wait_wave re-applies a vanished OR failed
# job (up to TQP_MAXTRIES) instead of aborting the run, so a transient sweep costs
# a retry, not the whole 100B pass. (A build that reaches its compute loop pegs
# CPU for hours and is never touched; only the init window is vulnerable.)
set -u
NS=ssu-atlas-ai
N=50
WAVE=8
MAXTRIES=${TQP_MAXTRIES:-6}
cd "$(dirname "$0")"

job_done () {  # 0 if job exists and succeeded
  [ "$(kubectl get job "$1" -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)" = "1" ]
}

# wait_wave TMPL PREFIX JOB...  — block until every JOB has succeeded, re-applying
# any that vanish (swept) or fail. TMPL is rendered per job by substituting __I__
# with the job's index (the suffix after PREFIX); for a singleton job pass its full
# name as PREFIX so the index is empty and the concrete yaml applies unchanged.
wait_wave () {
  local tmpl=$1 prefix=$2; shift 2
  declare -A tries
  while :; do
    local ok=1
    for j in "$@"; do
      job_done "$j" && continue
      ok=0
      local bad=0
      kubectl get job "$j" -n $NS >/dev/null 2>&1 || bad=1
      local f
      f=$(kubectl get job "$j" -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
      { [ -n "${f:-}" ] && [ "${f:-0}" -ge 1 ]; } && bad=1
      if [ $bad = 1 ]; then
        tries[$j]=$(( ${tries[$j]:-0} + 1 ))
        if [ "${tries[$j]}" -gt "$MAXTRIES" ]; then
          echo "=== $(date -u +%H:%M) JOB $j gave up after $MAXTRIES retries"
          return 1
        fi
        local I=${j#"$prefix"}
        echo "=== $(date -u +%H:%M) JOB $j vanished/failed -> retry ${tries[$j]}/$MAXTRIES"
        kubectl delete job "$j" -n $NS --ignore-not-found >/dev/null 2>&1
        sed "s/__I__/$I/g" "$tmpl" | kubectl apply -f - >/dev/null
      fi
    done
    [ $ok = 1 ] && return 0
    sleep 120
  done
}

# run_waves NAME_PREFIX TEMPLATE — apply per-server jobs in waves of $WAVE,
# skipping servers whose job already succeeded (resume after interruption).
run_waves () {
  local prefix=$1 tmpl=$2 i=0
  while [ $i -lt $N ]; do
    local wave_jobs=()
    for I in $(seq $i $((i + WAVE - 1))); do
      [ $I -ge $N ] && break
      if job_done "$prefix$I"; then
        echo "$prefix$I already complete, skipping"
        continue
      fi
      kubectl delete job "$prefix$I" -n $NS --ignore-not-found >/dev/null 2>&1
      sed "s/__I__/$I/g" "$tmpl" | kubectl apply -f -
      wave_jobs+=("$prefix$I")
    done
    if [ ${#wave_jobs[@]} -gt 0 ]; then
      echo "=== $(date -u +%H:%M) wave at server $i: ${wave_jobs[*]}"
      wait_wave "$tmpl" "$prefix" "${wave_jobs[@]}" || return 1
    fi
    i=$((i + WAVE))
  done
}

echo "=== $(date -u +%H:%M) 100B PVCs"
for I in $(seq 0 $((N - 1))); do
  sed "s/__I__/$I/g" pvc_100b_tmpl.yaml | kubectl apply -f -
done

echo "=== $(date -u +%H:%M) 100B build (waves of $WAVE)"
run_waves tqp-fleet-100b-build- job_build_100b.yaml || exit 1

echo "=== $(date -u +%H:%M) query cache"
if ! job_done aqx-qcache100; then
  kubectl delete job aqx-qcache100 -n $NS --ignore-not-found >/dev/null 2>&1
  kubectl apply -f job_qcache_100b.yaml
  wait_wave job_qcache_100b.yaml aqx-qcache100 aqx-qcache100 || exit 1
fi

echo "=== $(date -u +%H:%M) exact reference (waves of $WAVE)"
run_waves aqx-ref100- job_ref_100b.yaml || exit 1

echo "=== $(date -u +%H:%M) routed IVF (waves of $WAVE)"
run_waves aqx-ivf100- job_ivf_100b.yaml || exit 1

echo "=== $(date -u +%H:%M) merge + score"
kubectl delete job aqx-score100 -n $NS --ignore-not-found >/dev/null 2>&1
kubectl apply -f job_score100b.yaml
wait_wave job_score100b.yaml aqx-score100 aqx-score100 || exit 1
kubectl logs -n $NS job/aqx-score100 --tail=20 > score_100B.log 2>&1

echo "=== $(date -u +%H:%M) 100B measurement complete"
echo DRIVER100B_DONE
