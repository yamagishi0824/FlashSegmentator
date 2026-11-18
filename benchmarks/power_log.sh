#!/bin/bash
# usage: power_log.sh <out.csv> <cmd...>   samples GPU power every 1 s while <cmd> runs; prints energy (J) and wall (s).
OUT=$1; shift
Q=power.draw,utilization.gpu,memory.used
nvidia-smi --query-gpu=module.power.draw --format=csv,noheader >/dev/null 2>&1 && Q=power.draw,module.power.draw,utilization.gpu,memory.used
echo "t,$Q" > $OUT
( while true; do echo "$(date +%s.%N),$(nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits | head -1)" >> $OUT; sleep 1; done ) & MON=$!
T0=$(date +%s.%N); "$@"; RC=$?; T1=$(date +%s.%N)
kill $MON; wait $MON 2>/dev/null
python3 - "$OUT" "$T0" "$T1" <<'PY'
import sys,csv
rows=list(csv.reader(open(sys.argv[1])))
h=rows[0]; r=[x for x in rows[1:] if len(x)==len(h)]
t=[float(x[0]) for x in r]
for j,name in enumerate(h[1:],1):
    if 'power' not in name: continue
    try: p=[float(x[j]) for x in r]
    except ValueError: continue
    E=sum((t[i+1]-t[i])*(p[i]+p[i+1])/2 for i in range(len(t)-1))
    print(f"ENERGY {name} {E:.0f} J mean {sum(p)/len(p):.1f} W")
print(f"WALL {float(sys.argv[3])-float(sys.argv[2]):.1f} s")
PY
exit $RC
