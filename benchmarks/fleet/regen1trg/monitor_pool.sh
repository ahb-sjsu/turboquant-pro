# Monitor for the 1trg cluster pass over the pool servers (run on Atlas): the driver's
# submit failures, recycles, parks and finishes, its exit, and a count of active jobs.
export KUBECONFIG=/home/claude/.kube/config
cd /home/claude/tqp_fleet
grep -E "SUBMIT FAILED|RECYCLE|PARK|LONG|DONE aqx|POST_EXIT|Traceback|Error" driver1trg_pool.log | sed 's/job_id=.*//' | sed 's/^/POOL /'
n=$(grep -c "DONE aqx" driver1trg_pool.log); echo "POOL done $(( (n/10)*10 ))+ of 490"
screen -ls | grep -q rg1trgpool || echo "POOL screen rg1trgpool gone"
