# When the Atlas-first test ends, restart the wave-1 Atlas-first pass with three workers.
# The pass resumes from each server's state file, so only the shard in progress is rebuilt.
L=/home/claude/tqp_fleet/w1three.log; : > $L
until grep -q FIRSTTEST_DONE /home/claude/tqp_fleet/firsttest.log || ! screen -ls | grep -q firsttest; do sleep 60; done
echo "$(date -u) test ended" >> $L
screen -S wave1first -X quit
sleep 2
pkill -f "code_1trg4/fleet_regen_[a]tlas.py"
pkill -f "code_1trg4/fleet_regen_[s]core.py"
pkill -f "code_1trg4/fleet_regen_[b]uild.py"
pkill -f "code_1trg4/fleet_regen_[c]heck.py"
sleep 5
echo "$(date -u) left: $(pgrep -fc 'code_1trg4/fleet_regen_[a-z]*.py')" >> $L
echo "== restart with 3 workers $(date -u)" >> /home/claude/tqp_fleet/wave1first.log
( screen -dmS wave1first env WORKERS=3 bash /home/claude/tqp_fleet/wave1first.sh ) && echo "$(date -u) relaunched with 3 workers" >> $L
