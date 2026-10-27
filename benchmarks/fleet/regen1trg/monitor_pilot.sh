# Monitor for the 1trg cluster pilot (run on Atlas): driver events, released-file checks,
# errors, deferral lines and progress every 25 shards per server, one line each.
export KUBECONFIG=/home/claude/.kube/config
cd /home/claude/tqp_fleet
grep -E "SUBMIT|DONE|RECYCLE|FAIL|PARK|LONG|POST_EXIT|Traceback|Error" driver1trg_pilot.log | grep -v "BACKOFF" | sed 's/job_id=.*//' | sed 's/^/DRV /'
for p in $(kubectl -n ssu-atlas-ai get pods --no-headers 2>/dev/null | grep regen1trg- | awk '{print $1}'); do
  s=$(echo $p | sed -E 's/aqx-regen1trg-([0-9]+)-.*/\1/')
  kubectl -n ssu-atlas-ai logs $p 2>/dev/null > /tmp/m1trgp_$s.log
  grep -E "^released|not found|Traceback|Error|MemoryError|^DEFERRED |REGEN_PART_DONE|^codes" /tmp/m1trgp_$s.log | cut -c1-200 | sed "s/^/S$s /"
  grep -oE "^shard (25|50|75|100|125|150|175|200|225|250|275|300|325|350|375|400)/400 \(g=[0-9]+\) [0-9.]+s" /tmp/m1trgp_$s.log | sed "s/^/S$s /"
done
screen -ls | grep -q rg1trgpilot || echo "DRV screen rg1trgpilot gone"
