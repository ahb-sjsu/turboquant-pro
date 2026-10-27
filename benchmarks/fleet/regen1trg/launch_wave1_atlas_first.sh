# Atlas-first pass of the 1trg run for wave 1 (servers 0 and 2-7; server 1 is on the cluster).
# Staged on Atlas as /home/claude/tqp_fleet/wave1first.sh, beside the committed scripts as c4_*.py.
# Committed code 61b15fc. Two servers at a time while the Atlas-first test still runs (thermal cap).
W=/archive/experiments/tqp-fleet-1t/work; C=$W/code_1trg4; L=/home/claude/tqp_fleet/wave1first.log
mkdir -p $C
cp /home/claude/tqp_fleet/cm_1trgfull3/* $C/
for f in fleet_regen_score.py fleet_regen_atlas.py fleet_regen_merge.py; do cp /home/claude/tqp_fleet/c4_$f $C/$f; done
md5sum $C/*.py | awk '{print substr($1,1,8), $2}' >> $L
export KUBECONFIG=/home/claude/.kube/config
cd $C
TQP_ATLAS_CODE=$C TQP_ATLAS_WORKERS=${WORKERS:-2} $W/venv251/bin/python fleet_regen_atlas.py 1trg --first 0,2,3,4,5,6,7 >> $L 2>&1
echo "rc=$?" >> $L
echo WAVE1FIRST_EXIT >> $L
