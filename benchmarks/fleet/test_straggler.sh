#!/bin/bash
# Exercise the straggler decision with synthetic elapsed times.
STRAGGLE_X=3; STRAGGLE_MAX=2; STRAGGLE_MIN_S=1800
decide () {  # decide "<done csv>" <candidate_elapsed> <prior_reissues>
  local done_s=(); IFS=, read -ra done_s <<< "$1"
  local e=$2 prior=$3
  [ ${#done_s[@]} -ge 2 ] || { echo "wait (only ${#done_s[@]} finished)"; return; }
  local med cutoff
  med=$(printf '%s\n' "${done_s[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  cutoff=$(( med * STRAGGLE_X )); [ "$cutoff" -lt "$STRAGGLE_MIN_S" ] && cutoff=$STRAGGLE_MIN_S
  if [ "$e" -le "$cutoff" ]; then echo "leave (${e}s <= cutoff ${cutoff}s, median ${med}s)"
  elif [ "$prior" -ge "$STRAGGLE_MAX" ]; then echo "leave (at cap, ${e}s > ${cutoff}s)"
  else echo "RE-ISSUE (${e}s > cutoff ${cutoff}s, median ${med}s)"; fi
}
echo "1 the measured 100B case, server 11 at 16467s vs ~1956s median:"
echo "   $(decide 1956,1745,2085,1702 16467 0)"
echo "2 normal variation, 1.4x the median (the 10B spread):"
echo "   $(decide 1956,1745,2085,1702 2700 0)"
echo "3 young job, under the 1800s floor, must never be touched:"
echo "   $(decide 300,320,290,310 900 0)"
echo "4 already re-issued twice, must stop thrashing:"
echo "   $(decide 1956,1745,2085,1702 16467 2)"
echo "5 too little evidence, only one job finished:"
echo "   $(decide 1956 16467 0)"
