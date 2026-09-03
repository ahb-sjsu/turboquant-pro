#!/usr/bin/env bash
# Stage 1: 100M smoke with aggressive straggler settings and a short poll.
# Stage 2: 10B straggler test at production settings, ONLY if stage 1 passed.
cd /home/claude/tqp_fleet
export KUBECONFIG=/home/claude/.kube/config

echo "########## STAGE 1: 100M smoke ##########"
TQP_STRAGGLE_X=2 TQP_STRAGGLE_MIN_S=45 TQP_STRAGGLE_MAX=1 TQP_POLL=10 \
  bash driver100m_smoke.sh >> driver100m_smoke.log 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -q DRIVER100M_DONE driver100m_smoke.log; then
  echo "STAGE 1 FAILED (rc=$rc). Not starting 10B." | tee -a driver100m_smoke.log
  exit 1
fi
echo "STAGE 1 PASSED" | tee -a driver100m_smoke.log

echo "########## STAGE 2: 10B straggler test ##########"
bash driver10b_straggle.sh >> driver10bs_run2.log 2>&1
echo "STAGE 2 rc=$?" | tee -a driver10bs_run2.log
