#!/bin/bash
# 10B straggler-mechanism test (runs detached on Atlas, in a screen session).
#
# Not a new 10B measurement - that exists (results/fleet_run_10B.json). This
# re-runs only the routed-IVF phase over the already-built 10B index to
# exercise straggler re-issue on a real fleet before it is trusted at 1T.
# 8 servers x 250 shards, waves of 4 CPU-pegged jobs
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
N=8
WAVE=4
MAXTRIES=${TQP_MAXTRIES:-6}
# Straggler re-issue. A wave finishes when its SLOWEST job finishes, and on a
# shared cluster node quality varies enough that one bad draw sets the schedule
# (measured at 100B: slowest server 6.9x the median). Re-issuing a straggler is
# cheap here only because builds are resumable: a re-issued pod reloads its
# per-shard sidecars and continues, losing at most the shard in flight, and it
# may land on a faster node. STRAGGLE_X=0 disables.
STRAGGLE_X=${TQP_STRAGGLE_X:-3}      # re-issue at this multiple of the wave median
STRAGGLE_MAX=${TQP_STRAGGLE_MAX:-2}  # per-job cap, separate from failure retries
STRAGGLE_MIN_S=${TQP_STRAGGLE_MIN_S:-1800}  # never touch a job younger than this
cd "$(dirname "$0")"

job_done () {  # 0 if job exists and succeeded
  [ "$(kubectl get job "$1" -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)" = "1" ]
}

# Deleting a Job does NOT synchronously remove its pod, and a terminating pod
# keeps its RWO volume attached. Re-applying immediately makes the new pod fail
# with "Multi-Attach error: volume is already used by pod ...", which this loop
# then reads as a failure and retries, producing a storm of half-dead pods.
# Measured first-hand on the 10B straggler test. Always delete through here.
delete_job_wait () {
  local j=$1 i
  kubectl delete job "$j" -n $NS --ignore-not-found --cascade=foreground     --wait=true --timeout=180s >/dev/null 2>&1
  for i in $(seq 1 60); do
    [ -z "$(kubectl get pods -n $NS -l job-name="$j" --no-headers 2>/dev/null)" ] && return 0
    sleep 5
  done
  echo "=== $(date -u +%H:%M) WARN pods for $j still present after 300s, not re-applying"
  return 1
}

job_elapsed_s () {  # seconds since the job started, or empty if unknown
  local st
  st=$(kubectl get job "$1" -n $NS -o jsonpath='{.status.startTime}' 2>/dev/null)
  [ -z "$st" ] && return 1
  echo $(( $(date -u +%s) - $(date -u -d "$st" +%s) ))
}

# wait_wave TMPL PREFIX JOB...  — block until every JOB has succeeded, re-applying
# any that vanish (swept) or fail. TMPL is rendered per job by substituting __I__
# with the job's index (the suffix after PREFIX); for a singleton job pass its full
# name as PREFIX so the index is empty and the concrete yaml applies unchanged.
wait_wave () {
  local tmpl=$1 prefix=$2; shift 2
  declare -A tries
  declare -A stragg
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
        delete_job_wait "$j" || continue
        sed "s/__I__/$I/g" "$tmpl" | kubectl apply -f - >/dev/null
      fi
    done
    [ $ok = 1 ] && return 0

    # --- straggler re-issue -------------------------------------------------
    # Only meaningful once part of the wave has finished, because the finished
    # jobs are what define "normal" for this wave. Jobs in a wave do identical
    # work (same shard count), so elapsed time is directly comparable.
    if [ "$STRAGGLE_X" -gt 0 ]; then
      local done_s=() e
      for j in "$@"; do
        if job_done "$j"; then e=$(job_elapsed_s "$j") && [ -n "$e" ] && done_s+=("$e"); fi
      done
      if [ ${#done_s[@]} -ge 2 ]; then
        local med
        med=$(printf '%s
' "${done_s[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
        local cutoff=$(( med * STRAGGLE_X ))
        [ "$cutoff" -lt "$STRAGGLE_MIN_S" ] && cutoff=$STRAGGLE_MIN_S
        for j in "$@"; do
          job_done "$j" && continue
          e=$(job_elapsed_s "$j") || continue
          [ -z "$e" ] && continue
          [ "$e" -le "$cutoff" ] && continue
          if [ "${stragg[$j]:-0}" -ge "$STRAGGLE_MAX" ]; then
            echo "=== $(date -u +%H:%M) JOB $j slow (${e}s vs median ${med}s) but at re-issue cap, leaving it"
            continue
          fi
          stragg[$j]=$(( ${stragg[$j]:-0} + 1 ))
          local I=${j#"$prefix"}
          echo "=== $(date -u +%H:%M) JOB $j STRAGGLER ${e}s > ${cutoff}s (median ${med}s) -> re-issue ${stragg[$j]}/$STRAGGLE_MAX"
          delete_job_wait "$j" || continue
          sed "s/__I__/$I/g" "$tmpl" | kubectl apply -f - >/dev/null
        done
      fi
    fi
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
      delete_job_wait "$prefix$I"
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


echo "=== $(date -u +%H:%M) 10B straggler test: routed IVF over the existing index"
echo "=== straggler settings: X=$STRAGGLE_X MAX=$STRAGGLE_MAX MIN_S=$STRAGGLE_MIN_S"
echo "=== writing partials under tag 10bs (the 07-19 results are untouched)"
run_waves aqx-ivf10s- job_ivf_10b_straggle.yaml || exit 1

echo "=== $(date -u +%H:%M) per-job elapsed"
for I in $(seq 0 $((N - 1))); do
  st=$(kubectl get job aqx-ivf10s-$I -n $NS -o jsonpath='{.status.startTime}' 2>/dev/null)
  ct=$(kubectl get job aqx-ivf10s-$I -n $NS -o jsonpath='{.status.completionTime}' 2>/dev/null)
  if [ -n "$st" ] && [ -n "$ct" ]; then
    echo "server $I: $(( $(date -u -d "$ct" +%s) - $(date -u -d "$st" +%s) ))s"
  else
    echo "server $I: incomplete"
  fi
done

echo "=== $(date -u +%H:%M) 10B straggler test complete"
echo DRIVER10BS_DONE
