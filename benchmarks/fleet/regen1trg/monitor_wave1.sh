# Monitor for the wave-1 Atlas-first pass (run on Atlas): progress every 25 shards per server,
# each server's summary and errors, the pass's own exit lines, and server 1's cluster job.
R=/archive/experiments/tqp-fleet-1t/work/rg_atlas_1trg
for f in $R/atlas_*.log; do
  s=$(basename $f .log | sed 's/atlas_//')
  grep -oE "^shard (25|50|75|100|125|150|175|200|225|250|275|300|325|350|375|400)/400 \(g=[0-9]+\) [0-9.]+s" $f | sed "s/^/W1 S$s /"
  grep -E "^hash:|^codes|^variants|^DEFERRED|Traceback|Error|SystemExit" $f | cut -c1-220 | sed "s/^/W1 S$s /"
done
grep -E "reproduced here|rc=|ATLAS_FIRST_DONE|WAVE1FIRST_EXIT|Traceback" /home/claude/tqp_fleet/wave1first.log | sed 's/^/W1 /'
screen -ls | grep -q wave1first || echo "W1 screen wave1first gone"
bash /home/claude/tqp_fleet/monitor_pilot.sh | grep -E "^S1 |^DRV"
