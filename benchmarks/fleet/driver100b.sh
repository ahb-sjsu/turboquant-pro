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
#
# Auth-resilient (run4 patch): kubelogin's OIDC token refresh hits
# authentik.nrp-nautilus.io, which is INTERMITTENTLY slow/unreachable from
# Atlas. A kubectl call that errors or times out is a CHECK FAILURE, not
# evidence the job vanished: job_state backs off (30s doubling, cap 600s) and
# retries the check without touching the job or its retry budget. Only a
# SUCCESSFUL query (or a definitive server NotFound) yields a verdict.
# The give-up path prints a loud GIVEUP marker and exits non-zero, and an
# auth probe gates driver start, every wave, and every job re-apply.
set -u
NS=ssu-atlas-ai
N=50
WAVE=8
MAXTRIES=${TQP_MAXTRIES:-6}
BACKOFF0=${TQP_BACKOFF0:-30}
BACKOFFCAP=${TQP_BACKOFFCAP:-600}
POLL=${TQP_POLL:-120}
cd "$(dirname "$0")"

# auth_probe -- block until the API server + auth path answer a cheap query.
# Backoff 30s doubling, cap 600s. Never gives up; it waits out the outage so
# no job mutation ever happens against a broken auth path.
auth_probe () {
  local delay=$BACKOFF0 n=0
  while :; do
    timeout 30 kubectl get ns $NS --request-timeout=20s >/dev/null 2>&1 && return 0
    n=$((n + 1))
    echo "=== $(date -u +%H:%M) AUTHPROBE failed (attempt $n) -> backoff ${delay}s"
    sleep $delay
    delay=$((delay * 2)); [ $delay -gt $BACKOFFCAP ] && delay=$BACKOFFCAP
  done
}

# job_state NAME -- print exactly one of: done / failed / running / absent.
# kubectl errors/timeouts are check failures: back off and RE-CHECK, never
# touching the job. Only rc=0 (or an explicit server NotFound) is a verdict.
job_state () {
  local j=$1 delay=$BACKOFF0 out rc s f
  while :; do
    out=$(timeout 60 kubectl get job "$j" -n $NS \
          -o jsonpath='{.status.succeeded}/{.status.failed}' \
          --request-timeout=30s 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
      s=${out%%/*}; f=${out##*/}
      if [ "$s" = "1" ]; then echo done
      elif [ -n "$f" ] && [ "$f" -ge 1 ] 2>/dev/null; then echo failed
      else echo running; fi
      return 0
    fi
    if printf '%s' "$out" | grep -qi 'NotFound\|not found'; then
      echo absent; return 0
    fi
    echo "=== $(date -u +%H:%M) CHECKFAIL job $j (rc=$rc): $(printf '%s' "$out" | head -1) -> backoff ${delay}s, re-checking (retry budget untouched)" >&2
    sleep $delay
    delay=$((delay * 2)); [ $delay -gt $BACKOFFCAP ] && delay=$BACKOFFCAP
  done
}

job_done () {  # 0 if job exists and succeeded (auth-error-safe via job_state)
  [ "$(job_state "$1")" = done ]
}

# wait_wave TMPL PREFIX JOB...  -- block until every JOB has succeeded, re-applying
# any that verifiably vanish (swept) or fail. TMPL is rendered per job by substituting
# __I__ with the job's index (the suffix after PREFIX); for a singleton job pass its full
# name as PREFIX so the index is empty and the concrete yaml applies unchanged.
wait_wave () {
  local tmpl=$1 prefix=$2; shift 2
  declare -A tries
  while :; do
    local ok=1
    for j in "$@"; do
      local st
      st=$(job_state "$j")
      [ "$st" = done ] && continue
      ok=0
      if [ "$st" = failed ] || [ "$st" = absent ]; then
        tries[$j]=$(( ${tries[$j]:-0} + 1 ))
        if [ "${tries[$j]}" -gt "$MAXTRIES" ]; then
          echo "=== $(date -u +%H:%M) GIVEUP GIVEUP GIVEUP: JOB $j still $st after $MAXTRIES retries -- aborting driver, exit 1"
          exit 1
        fi
        local I=${j#"$prefix"}
        echo "=== $(date -u +%H:%M) JOB $j $st -> retry ${tries[$j]}/$MAXTRIES"
        auth_probe
        timeout 120 kubectl delete job "$j" -n $NS --ignore-not-found --request-timeout=60s >/dev/null 2>&1
        sed "s/__I__/$I/g" "$tmpl" | timeout 120 kubectl apply --request-timeout=60s -f - >/dev/null
      fi
    done
    [ $ok = 1 ] && return 0
    sleep $POLL
  done
}

# run_waves NAME_PREFIX TEMPLATE -- apply per-server jobs in waves of $WAVE,
# skipping servers whose job already succeeded (resume after interruption).
run_waves () {
  local prefix=$1 tmpl=$2 i=0
  while [ $i -lt $N ]; do
    auth_probe
    local wave_jobs=()
    for I in $(seq $i $((i + WAVE - 1))); do
      [ $I -ge $N ] && break
      if job_done "$prefix$I"; then
        echo "$prefix$I already complete, skipping"
        continue
      fi
      timeout 120 kubectl delete job "$prefix$I" -n $NS --ignore-not-found --request-timeout=60s >/dev/null 2>&1
      sed "s/__I__/$I/g" "$tmpl" | timeout 120 kubectl apply --request-timeout=60s -f -
      wave_jobs+=("$prefix$I")
    done
    if [ ${#wave_jobs[@]} -gt 0 ]; then
      echo "=== $(date -u +%H:%M) wave at server $i: ${wave_jobs[*]}"
      wait_wave "$tmpl" "$prefix" "${wave_jobs[@]}" || return 1
    fi
    i=$((i + WAVE))
  done
}

# Testability hook: source with TQP_FUNCS_ONLY=1 to load functions without running.
[ "${TQP_FUNCS_ONLY:-0}" = 1 ] && return 0

echo "=== $(date -u +%H:%M) pre-flight auth probe"
auth_probe
echo "=== $(date -u +%H:%M) auth OK"

echo "=== $(date -u +%H:%M) 100B PVCs"
for I in $(seq 0 $((N - 1))); do
  sed "s/__I__/$I/g" pvc_100b_tmpl.yaml | kubectl apply -f -
done

echo "=== $(date -u +%H:%M) 100B build (waves of $WAVE)"
run_waves tqp-fleet-100b-build- job_build_100b.yaml || exit 1

echo "=== $(date -u +%H:%M) query cache"
auth_probe
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
auth_probe
kubectl delete job aqx-score100 -n $NS --ignore-not-found >/dev/null 2>&1
kubectl apply -f job_score100b.yaml
wait_wave job_score100b.yaml aqx-score100 aqx-score100 || exit 1
kubectl logs -n $NS job/aqx-score100 --tail=20 > score_100B.log 2>&1

echo "=== $(date -u +%H:%M) 100B measurement complete"
echo DRIVER100B_DONE
