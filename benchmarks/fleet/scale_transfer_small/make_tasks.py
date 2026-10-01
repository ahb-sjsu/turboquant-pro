"""Write the thermal-pool task files for the scale-transfer calibration (run in its workspace).

usage: python make_tasks.py build|measure QNAME QFILE
"""

import os
import sys

import scale_transfer_small as t

B = os.path.dirname(os.path.abspath(__file__))
PY = "/archive/ahb-sjsu/ovb_tiers/venv/bin/python"
# Package 1 of Atlas's two sockets (CPUs 12-23 and 36-47): the thermal guardian judges a batch
# process by the packages its affinity allows, and package 0 carries the desktop's load.
PKG1 = "12-23,36-47"
ENV = (
    f"cd {B} && PYTHONPATH={B}:{B}/src:{B}/src/benchmarks/fleet "
    f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c {PKG1} nice -n 10 "
)
COMMON = "--boot boot --pool pool --work work --out out"

if sys.argv[1] == "build":
    lines = [
        f"{ENV}{PY} scale_transfer_small.py build {COMMON} --shards {g}"
        for g in t.all_shards()
        if not os.path.exists(os.path.join(B, "pool", f"g{g:06d}", "DONE"))
    ]
    path = "build_tasks.txt"
else:
    qname, qfile = sys.argv[2], sys.argv[3]
    lines = [
        f"{ENV}{PY} scale_transfer_small.py measure {COMMON} --qname {qname} "
        f"--queries {qfile} --perm {i} --size {m}"
        for i in range(t.N_PERM)
        for m in sorted(t.SIZES, reverse=True)
    ]
    path = f"measure_{qname}_tasks.txt"
open(os.path.join(B, path), "w").write("\n".join(lines) + "\n")
print(path, len(lines))
