#!/bin/bash
# Unpack a finished Part V fetch job's output (a base64 gzip tar in its log) on Atlas.
#   bash part5_fetch.sh <job name> <destination directory>
set -euo pipefail
JOB=${1:?job}
DEST=${2:?dest}
K="kubectl -n ssu-atlas-ai --request-timeout=120s"
for _ in $(seq 1 40); do
  s=$($K get job "$JOB" -o jsonpath='{.status.succeeded}/{.status.failed}')
  case "$s" in 1/*) break ;; */1) echo "$JOB failed"; $K logs "job/$JOB" --tail=20; exit 1 ;; esac
  sleep 15  # Atlas-side wait; the cluster job itself never sleeps
done
mkdir -p "$DEST"
$K logs "job/$JOB" | sed -n '/^FETCH_BEGIN$/,/^FETCH_END$/p' | sed '1d;$d' | base64 -d | tar -xzf - -C "$DEST"
echo "unpacked $JOB -> $DEST"
