"""One 1T reference scan, re-run where the placement experiment puts it.

This is the pod side of ``placement_experiment.py``. It repeats exactly what
``fleet_ref.py`` computed for server ``TQP_SERVER_ID`` in the 1T measurement
(same index, same query cache, ``max_open_shards=2``, ``block=65536``, k=10),
with the placement record around it, and writes nothing: the volume and the
shared results are mounted read-only. It compares its top-10 ids with the
partial the 1T run recorded for the same server, so every wall time it reports
is for a computation shown to be the same one.

The Job ships this file with ``fingerprint.py`` prepended (one exec'd script).
"""

import os
import time

import numpy as np

from turboquant_pro import ShardedIndex

SID = int(os.environ["TQP_SERVER_ID"])
RESULTS = "/shared/fleet/results"

fp = Fingerprint(data_path="/idx")  # noqa: F821 - prepended by the driver
q = np.load(f"{RESULTS}/queries1t.npy")
fp.mark("open_start")
sh = ShardedIndex.open("/idx/manifest.json", mmap=True, max_open_shards=2)
fp.mark("scan_start")
t0 = time.time()
ids, sc = sh.search(q, k=10, block=65536)
wall = time.time() - t0
fp.mark("scan_end")

rec = np.load(f"{RESULTS}/ref1t_part_{SID}.npz")
same = bool(np.array_equal(ids, rec["ids"]))
fin = np.isfinite(sc) & np.isfinite(rec["scores"])
fp.emit(
    phase="ref-replica",
    server=SID,
    wall_s=wall,
    rows=int(sh.n_rows),
    nq=int(len(q)),
    same_ids_as_1t=same,
    max_abs_score_diff=float(np.max(np.abs(sc[fin] - rec["scores"][fin]))),
    recorded_1t_wall_s=float(rec["wall_s"]),
)
