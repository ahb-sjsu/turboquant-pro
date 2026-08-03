#!/usr/bin/env bash
# Create the 500 Linstor volumes for the 1T run, paced and verified.
#
# Separate from driver1t.sh on purpose. 500 volumes is ten times the object
# count of the 100B run, and asking a shared storage provisioner for all of
# them in a tight loop is the kind of thing that annoys other tenants. This
# creates them in batches, pauses between batches, and reports how many bound,
# so a provisioning problem surfaces before any compute starts.
#
# Idempotent: kubectl apply skips volumes that already exist, so a rerun after
# a partial failure only creates the missing ones.
set -uo pipefail
NS=ssu-atlas-ai
N=${TQP_N:-500}
BATCH=${TQP_BATCH:-25}
PAUSE=${TQP_PAUSE:-10}
export KUBECONFIG=/home/claude/.kube/config
cd "$(dirname "$0")"

echo "=== creating $N PVCs in batches of $BATCH ==="
created=0
for I in $(seq 0 $((N - 1))); do
  if ! kubectl get pvc "tqp-fleet-1t-$I" -n $NS >/dev/null 2>&1; then
    sed "s/__I__/$I/g" pvc_1t_tmpl.yaml | kubectl apply -f - >/dev/null 2>&1 \
      && created=$((created + 1))
  fi
  if [ $(((I + 1) % BATCH)) = 0 ]; then
    b=$(kubectl get pvc -n $NS --no-headers 2>/dev/null | grep -c '^tqp-fleet-1t-')
    echo "  through server $I: $b exist, $created created this run"
    sleep "$PAUSE"
  fi
done

echo
echo "=== waiting for binds ==="
for i in $(seq 1 60); do
  bound=$(kubectl get pvc -n $NS --no-headers 2>/dev/null \
    | awk '/^tqp-fleet-1t-/ && $2=="Bound"' | wc -l)
  total=$(kubectl get pvc -n $NS --no-headers 2>/dev/null | grep -c '^tqp-fleet-1t-')
  echo "  bound $bound / $total"
  [ "$bound" = "$N" ] && break
  sleep 15
done

echo
echo "=== final ==="
kubectl get pvc -n $NS --no-headers 2>/dev/null | awk '/^tqp-fleet-1t-/ {print $2}' | sort | uniq -c
gi=$(kubectl get pvc -n $NS --no-headers 2>/dev/null \
  | awk '/^tqp-fleet-1t-/ {gsub(/Gi/,"",$4); s+=$4} END {print s}')
echo "provisioned: ${gi} Gi ($(echo "scale=1; $gi * 1.0737 / 1000" | bc 2>/dev/null || echo '?') TB)"
