#!/bin/bash
# 10B measurement driver (runs detached on Atlas): waits for build wave 2,
# runs the scattered reference in two CPU-pegged waves, then a short
# exempt-class serve window for the routed sweep. Serve pods torn down on
# any exit.
set -u
NS=ssu-atlas-ai
cd "$(dirname "$0")"

cleanup () {
  for I in 0 1 2 3 4 5 6 7; do
    kubectl delete deploy aqx-srv10-$I -n $NS --ignore-not-found >/dev/null 2>&1
  done
}
trap cleanup EXIT

wait_jobs () {
  while :; do
    ok=1
    for j in "$@"; do
      if ! kubectl get job "$j" -n $NS >/dev/null 2>&1; then
        echo "JOB $j VANISHED"; return 1
      fi
      s=$(kubectl get job "$j" -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
      f=$(kubectl get job "$j" -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
      if [ -n "${f:-}" ] && [ "${f:-0}" -ge 1 ]; then
        echo "JOB $j FAILED"; return 1
      fi
      [ "${s:-0}" = "1" ] || ok=0
    done
    [ $ok = 1 ] && return 0
    sleep 60
  done
}

echo "=== $(date -u +%H:%M) waiting for 10B build wave 2"
# Wave-2 jobs are created by wave2.sh only after wave 1 completes — wait for
# them to EXIST before treating absence as a sweep.
for i in $(seq 1 480); do
  n=0
  for I in 4 5 6 7; do
    kubectl get job tqp-fleet-10b-build-$I -n $NS >/dev/null 2>&1 && n=$((n+1))
  done
  [ "$n" = "4" ] && break
  sleep 60
done
wait_jobs tqp-fleet-10b-build-4 tqp-fleet-10b-build-5 \
          tqp-fleet-10b-build-6 tqp-fleet-10b-build-7 || exit 1

echo "=== $(date -u +%H:%M) reference wave A (servers 0-3)"
for I in 0 1 2 3; do sed "s/__I__/$I/g" job_ref.yaml | kubectl apply -f -; done
wait_jobs aqx-ref-0 aqx-ref-1 aqx-ref-2 aqx-ref-3 || exit 1
echo "=== $(date -u +%H:%M) reference wave B (servers 4-7)"
for I in 4 5 6 7; do sed "s/__I__/$I/g" job_ref.yaml | kubectl apply -f -; done
wait_jobs aqx-ref-4 aqx-ref-5 aqx-ref-6 aqx-ref-7 || exit 1

echo "=== $(date -u +%H:%M) serve window (8 exempt-class pods)"
for I in 0 1 2 3 4 5 6 7; do
  sed "s/__I__/$I/g" deploy_srv10.yaml | kubectl apply -f -
done
for i in $(seq 1 60); do
  up=$(kubectl get pods -n $NS --no-headers 2>/dev/null | grep aqx-srv10 | grep -c Running)
  [ "$up" = "8" ] && break
  sleep 10
done
sleep 30
kubectl delete job aqx-coord10 -n $NS --ignore-not-found
kubectl apply -f job_coord10.yaml
wait_jobs aqx-coord10 || echo "coord10 FAILED"
kubectl logs -n $NS job/aqx-coord10 --tail=40 > coord_10B.log 2>&1
cleanup

echo "=== $(date -u +%H:%M) 10B measurement complete"
echo DRIVER10B_DONE
