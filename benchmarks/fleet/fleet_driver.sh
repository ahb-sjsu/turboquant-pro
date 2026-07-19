#!/bin/bash
# Fleet driver (runs detached on Atlas): chains the measurement phases once the
# scattered GT jobs finish. Serve pods exist only inside short, busy windows.
set -u
NS=ssu-atlas-ai
cd "$(dirname "$0")"

# Serve pods must never outlive their measurement window — even if this
# driver dies mid-phase (the utilization-floor lesson). Teardown on any exit.
cleanup () {
  for I in 0 1 2 3; do
    kubectl delete deploy tqp-fleet-srv-$I tqp-fleet-topo-$I -n $NS \
      --ignore-not-found >/dev/null 2>&1
  done
}
trap cleanup EXIT

wait_jobs () {
  while :; do
    ok=1
    for j in "$@"; do
      if ! kubectl get job "$j" -n $NS >/dev/null 2>&1; then
        echo "JOB $j VANISHED (swept?)"; return 1
      fi
      s=$(kubectl get job "$j" -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
      f=$(kubectl get job "$j" -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
      if [ -n "${f:-}" ] && [ "${f:-0}" -ge 1 ]; then
        echo "JOB $j FAILED"; return 1
      fi
      [ "${s:-0}" = "1" ] || ok=0
    done
    [ $ok = 1 ] && return 0
    sleep 30
  done
}

echo "=== $(date -u +%H:%M) waiting for GT partials"
wait_jobs aqx-gt-0 aqx-gt-1 aqx-gt-2 aqx-gt-3 || exit 1

echo "=== $(date -u +%H:%M) serve window L=1"
for I in 0 1 2 3; do sed "s/__I__/$I/g" deploy_srv.yaml | kubectl apply -f -; done
for i in $(seq 1 60); do
  up=$(kubectl get pods -n $NS --no-headers 2>/dev/null | grep tqp-fleet-srv | grep -c Running)
  [ "$up" = "4" ] && break
  sleep 10
done
sleep 30
kubectl delete job tqp-fleet-coord -n $NS --ignore-not-found
kubectl apply -f job_coord.yaml
wait_jobs tqp-fleet-coord || echo "coord L=1 FAILED (continuing)"
kubectl logs -n $NS job/tqp-fleet-coord --tail=40 > coord_L1.log 2>&1

echo "=== $(date -u +%H:%M) split for topologies"
for I in 0 1 2 3; do kubectl scale deploy/tqp-fleet-srv-$I -n $NS --replicas=0; done
sleep 45
for I in 0 1 2 3; do sed "s/__I__/$I/g" job_split.yaml | kubectl apply -f -; done
wait_jobs tqp-fleet-split-0 tqp-fleet-split-1 tqp-fleet-split-2 tqp-fleet-split-3 || exit 1
for I in 0 1 2 3; do kubectl delete deploy tqp-fleet-srv-$I -n $NS --ignore-not-found; done

run_topo () {
  L=$1
  echo "=== $(date -u +%H:%M) topo L=$L window"
  for I in 0 1 2 3; do
    sed -e "s/__I__/$I/g" -e "s/__L__/$L/g" deploy_srv_topo.yaml | kubectl apply -f -
  done
  for i in $(seq 1 60); do
    up=$(kubectl get pods -n $NS --no-headers 2>/dev/null | grep tqp-fleet-topo | grep -c Running)
    [ "$up" = "4" ] && break
    sleep 10
  done
  sleep 60
  kubectl delete job tqp-fleet-coord -n $NS --ignore-not-found
  sed "s|- {name: NATS_URL, value: \"nats://atlas-nats:4222\"}|- {name: NATS_URL, value: \"nats://atlas-nats:4222\"}\n        - {name: TQP_LOGICAL_PER_POD, value: \"$L\"}|" \
    job_coord.yaml | kubectl apply -f -
  wait_jobs tqp-fleet-coord || echo "coord L=$L FAILED (continuing)"
  kubectl logs -n $NS job/tqp-fleet-coord --tail=40 > coord_L$L.log 2>&1
  for I in 0 1 2 3; do kubectl delete deploy tqp-fleet-topo-$I -n $NS --ignore-not-found; done
}
run_topo 16
run_topo 64

kubectl delete job tqp-fleet-coord tqp-fleet-ls -n $NS --ignore-not-found
echo "=== $(date -u +%H:%M) all phases complete"
echo DRIVER_DONE
