# End-to-end test of the Atlas-first flow on server 0 shards 15-18 (Atlas only, no cluster jobs).
# Expected: Atlas reproduces 15 and 17 (block), defers 16 (SkylakeX streamed) and 18.
W=/archive/experiments/tqp-fleet-1t/work; H=/archive/experiments/tqp-fleet-1t/scale_transfer/1tnm-all
C=$W/code_1trgtest; R=$W/rg_atlas_1trgtest; L=/home/claude/tqp_fleet/firsttest.log
rm -rf $C $R; mkdir -p $C $R; : > $L
cp /home/claude/tqp_fleet/cm_1trgfull3/* $C/
for f in fleet_regen_score.py fleet_regen_atlas.py fleet_regen_merge.py; do cp /home/claude/tqp_fleet/t_$f $C/$f; done
export KUBECONFIG=/home/claude/.kube/config
cd $C
echo "== 1. Atlas first" >> $L
TQP_ATLAS_CODE=$C TQP_ATLAS_WORKERS=1 TQP_FIRST_SHARDS=15,16,17,18 $W/venv251/bin/python fleet_regen_atlas.py 1trgtest --first 0 >> $L 2>&1
echo "rc=$?" >> $L
grep -E "^shard|^codes|^hash" $R/atlas_0.log >> $L
echo "== 2. driver env for server 0" >> $L
cp /home/claude/tqp_fleet/t_driver1t_post.py /tmp/driver1t_post_test.py
cd /home/claude/tqp_fleet && TQP_RUN_TAG=1trgtest TQP_ATLAS_FIRST=$R/atlas_first_1trgtest.json python3 -c "
import importlib.util, sys
sys.argv=['x']
spec = importlib.util.spec_from_file_location('drv', '/tmp/driver1t_post_test.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
e = m.desc_regen(0).env; print({k: e.get(k) for k in ('TQP_SKIP_SHARDS','TQP_DEFER_UNMATCHED','TQP_BLAS_THREADS','TQP_WRITES')})
e = m.desc_regen(5).env; print('server 5', {k: e.get(k) for k in ('TQP_SKIP_SHARDS','TQP_DEFER_UNMATCHED')})
" >> $L 2>&1
echo "== 3. cluster side, run on Atlas" >> $L
SKIP=$(python3 -c "import json; print(','.join(map(str, json.load(open('$R/atlas_first_1trgtest.json'))['0'])))")
echo "skip=$SKIP" >> $L
cd $C
env PYTHONPATH=$W/repo_build:$C TQP_SERVER_ID=0 TQP_RUN_TAG=1trgtest TQP_PART_KIND=part TQP_SHARD_LIST=15,16,17,18 \
  TQP_SKIP_SHARDS=$SKIP TQP_DEFER_UNMATCHED=0 TQP_SHARDS_PER_SERVER=400 TQP_WRITES=stream TQP_LIBS=post,pre TQP_BLAS_THREADS=1 \
  TQP_LIBDIR=/tmp/ft_l TQP_BOOT=$W/shared/bootstrap TQP_RESULTS=$R TQP_HASH_FILE=$H/hash1tnm_part_0.json \
  TQP_QUERIES_DIR=$W/rg_results TQP_QCACHE_NAME=queries1trg.npy TQP_SCRATCH=/tmp/ft_s TQP_SPILL_DIR=/tmp \
  TQP_RELEASED="1t=$W/shared/results/queries1t.npy=$W/shared/results/ref1t_part_{sid}.npz,1tnm=$H/queries1tnm.npy=$H/ref1tnm_part_{sid}.npz" \
  OPENBLAS_NUM_THREADS=1 taskset -c 12-23 $W/venv251/bin/python $C/fleet_regen_score.py > $R/part_0.log 2>&1
echo "rc=$?" >> $L
grep -E "^shard|^codes|^hash|^DEFERRED" $R/part_0.log >> $L
echo "== 4. merge rule" >> $L
$W/venv251/bin/python - $R >> $L 2>&1 <<'PY'
import sys, numpy as np
R = sys.argv[1]
a = dict(np.load(f"{R}/regen1trgtest_atlas_0.npz")); c = dict(np.load(f"{R}/regen1trgtest_part_0.npz"))
js = a["shards"][a["deferred"][a["shards"]] == 0]
want = np.flatnonzero(c["deferred"] == 1)
print("atlas searched", js.tolist(), "cluster left", want.tolist(), "rule holds", sorted(js.tolist()) == want.tolist())
print("atlas other_ok", {int(j): int(a["other_ok"][j]) for j in a["shards"]}, "atlas kernel", {int(j): str(a["kernel"][j]) for j in a["shards"]})
print("cluster kernel", {int(j): str(c["kernel"][j]) for j in c["shards"]})
PY
kubectl -n ssu-atlas-ai delete configmap tqp-fleet-atlas-1trgtest-0 --ignore-not-found >> $L 2>&1
echo FIRSTTEST_DONE >> $L
