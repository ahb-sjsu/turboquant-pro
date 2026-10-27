#!/bin/bash
# Full-run pilot (resubmit of 1 and 12 with fakecpus.so; adopts the running 300) of the regenerated-index run (tag 1trg): servers 1 (wave 1), 12 (wave 2) and
# 300 (pool), all 400 shards each, exempt class (1 CPU / 2 GiB) on AVX-512 nodes, streamed
# writes, unmatched shards deferred to the Atlas pass, search pins learned per server by
# candidate narrowing (no preset). Same tag and code as the full run, so these three servers'
# partials count toward it.
export KUBECONFIG=/home/claude/.kube/config
cd /home/claude/tqp_fleet
R=/shared/fleet/results
TQP_RUN_TAG=1trg TQP_PHASES=regen TQP_SERVERS=1,12,300 TQP_CODE_MAP=tqp-fleet-code-1trgfull2 TQP_POOL_MAXPAR=3 \
TQP_POST_STATE=/home/claude/tqp_fleet/post_state_1trgfull.json \
TQP_SCORE_LOG=/home/claude/tqp_fleet/score_1trgfull.log \
TQP_WRITES=stream TQP_LIBS=post,pre TQP_DEFER_UNMATCHED=1 \
TQP_RELEASED="1t=$R/queries1t.npy=$R/ref1t_part_{sid}.npz,1tnm=$R/queries1tnm.npy=$R/ref1tnm_part_{sid}.npz" \
python3 driver1t_post.py; echo POST_EXIT=$?
