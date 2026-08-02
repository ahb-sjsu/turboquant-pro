#!/usr/bin/env bash
# Test 1: production straggler settings. Expect NO re-issue (8 servers showed
# only 1.4x spread at 10B). Proves the detector does not fire on normal
# variance and does not disturb a healthy wave.
cd /home/claude/tqp_fleet
export KUBECONFIG=/home/claude/.kube/config
bash driver10b_straggle.sh >> driver10bs_prod.log 2>&1
