"""The systems cell grid (docs/PREREG_systems.md section 2).

DRAFT until the preregistration commit; after it, editing this file is an
amendment and is recorded in the preregistration.

Datasets. The two DBpedia 1M arms of the RaBitQ public campaign (their files
and ground truth are reused as staged; nothing of that campaign's results is
read), and OpenVector Bench's real tiers T6 (10^6 rows) and T7 (10^7 rows),
built from the unsealed rows of the Cohere Wikipedia Embed-V3 dump by
OpenVector Bench's tier builder: a content-addressed permutation, so T6 is a
strict subset of T7 and only N changes between them. Their held-out queries,
L1 ground truth and difficulty strata come with the tier.
"""

from __future__ import annotations

import math

from rabitq_public.datasets import SPECS, Spec

OVB_TIERS = {
    # name: (directory under the data root, parts of 10^6 rows)
    "ovb-wiki1024-t6": ("ovb/wiki1024-t6", 1),
    "ovb-wiki1024-t7": ("ovb/wiki1024-t7", 10),
}
NQ = 1000  # queries scored per arm (the tier ships more; the first NQ are used)
for _name, (_path, _parts) in OVB_TIERS.items():
    SPECS.setdefault(_name, Spec(_name, "npy", _path, NQ, parts=_parts))

DIMS = {
    "dbpedia-ada002-1m": 1536,
    "dbpedia-3large-1536-1m": 1536,
    "ovb-wiki1024-t6": 1024,
    "ovb-wiki1024-t7": 1024,
}
ROWS = {
    "dbpedia-ada002-1m": 990_000,
    "dbpedia-3large-1536-1m": 990_000,
    "ovb-wiki1024-t6": 1_000_000,
    "ovb-wiki1024-t7": 10_000_000,
}
SEED = 0


def nlist(dataset: str) -> int:
    """The RaBitQ campaign's rule, fixed before any data is read."""
    return 2 ** round(math.log2(4 * math.sqrt(ROWS[dataset])))


def configs(dataset: str) -> list[dict]:
    d, L = DIMS[dataset], nlist(dataset)
    probe = max(1, L // 32)
    out = [
        dict(method="tq", out_dim=o, bits=b)
        for o, b in ((d // 4, 4), (d // 2, 3), (d // 2, 4))
    ]
    out.append(dict(method="tq_ivf", out_dim=d // 2, bits=4, nlist=L, nprobe=probe))
    out += [dict(method=m, m=k) for m in ("pq", "opq") for k in (d // 16, d // 8)]
    out += [dict(method="rabitq_ivf", bits=b, nlist=L, nprobe=probe) for b in (1, 4)]
    out.append(
        dict(
            method="scann",
            num_leaves=L,
            leaves_to_search=probe,
            dims_per_block=2,
            aq_threshold=0.2,
            reorder=100,
        )
    )
    return out


def cell_id(dataset: str, cfg: dict) -> str:
    parts = [dataset, cfg["method"]]
    parts += [
        f"{k}{v}"
        for k, v in sorted(cfg.items())
        if k not in ("method", "dataset", "seed")
    ]
    return "-".join(str(p).replace(".", "p") for p in parts)


def cells(datasets=None) -> list[dict]:
    out = []
    for ds in datasets or DIMS:
        for cfg in configs(ds):
            c = dict(cfg, dataset=ds, seed=SEED)
            c["cell_id"] = cell_id(ds, c)
            out.append(c)
    return out


if __name__ == "__main__":
    from collections import Counter

    cs = cells()
    print(len(cs), "cells", Counter(c["method"] for c in cs))
