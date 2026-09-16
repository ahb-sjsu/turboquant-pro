"""Measure how fast a staged corpus reads, sequentially and by random gather.

    python -m rabitq_public.io_probe --dataset wiki1024-10m --data-root /data

Cells spend long phases reading corpora from the CephFS volume, and on 2026-09-15 that sank
eight pods below the cluster's CPU floor (1-4% of 4 requested cores). Before restructuring
those phases this measures them, in the exempt class (1 CPU, 2 GiB), where no floor applies:

- sequential: stream blocks the way ``Dataset.blocks`` does, reporting MiB/s and the cores
  the process used while doing it (a number near zero means the phase is pure waiting);
- random gather: fetch scattered rows the way ``rerank`` does, reporting rows/s, which is
  what decides whether reranking 10k queries x 50 candidates costs seconds or an hour.

It reads at most --seconds of each and never writes anything.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .datasets import SPECS


def _cpu():
    try:
        with open("/proc/self/stat") as f:
            f = f.read().rpartition(") ")[2].split()
        return sum(int(f[i]) for i in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class Timer:
    def __enter__(self):
        self.t, self.c = time.perf_counter(), _cpu()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t
        end = _cpu()
        self.cores = (
            round((end - self.c) / self.wall, 3)
            if end is not None and self.c is not None and self.wall > 0
            else None
        )


def sequential(read_block, n_rows, row_bytes, seconds, block):
    done = 0
    with Timer() as t:
        for s in range(0, n_rows, block):
            read_block(s, min(n_rows, s + block))
            done += min(n_rows, s + block) - s
            if time.perf_counter() - t.t > seconds:
                break
    mib = done * row_bytes / 2**20
    return dict(
        rows=done,
        mib=round(mib, 1),
        s=round(t.wall, 1),
        mib_s=round(mib / t.wall, 1),
        cores=t.cores,
    )


def probe(name, root, seconds, gather_rows):
    sp = SPECS[name]
    rng = np.random.default_rng(0)
    out = dict(dataset=name, kind=sp.kind)
    if sp.kind == "hdf5":
        import h5py

        with h5py.File(os.path.join(root, sp.path), "r") as f:
            d = f["train"]
            n, dim = d.shape
            row_bytes = dim * 4
            out["sequential"] = sequential(
                lambda a, b: np.array(d[a:b], np.float32, copy=True),
                n,
                row_bytes,
                seconds,
                50_000,
            )
            rows = np.sort(rng.choice(n, gather_rows, replace=False))
            with Timer() as t:
                for s in range(0, len(rows), 4096):  # h5py wants monotonic selections
                    d[rows[s : s + 4096].tolist()]
            out["gather"] = dict(
                rows=int(gather_rows),
                s=round(t.wall, 2),
                rows_s=round(gather_rows / t.wall),
                cores=t.cores,
            )
    else:
        d = os.path.join(root, sp.path)
        parts = [
            np.load(os.path.join(d, f"part_{i:03d}.npy"), mmap_mode="r")
            for i in range(sp.parts)
        ]
        n, dim = sum(len(p) for p in parts), parts[0].shape[1]
        row_bytes = dim * 4
        first = parts[0]
        out["sequential"] = sequential(
            # np.asarray on a same-dtype memmap slice returns a view and reads nothing;
            # copy=True forces the pages in, which is what a cell actually pays.
            lambda a, b: np.array(first[a:b], np.float32, copy=True),
            len(first),
            row_bytes,
            seconds,
            50_000,  # 200 MiB per block at d=1024: the probe runs in the exempt 2 GiB class
        )
        rows = np.sort(rng.choice(len(first), gather_rows, replace=False))
        with Timer() as t:
            first[rows]
        out["gather"] = dict(
            rows=int(gather_rows),
            s=round(t.wall, 2),
            rows_s=round(gather_rows / t.wall),
            cores=t.cores,
        )
    out["n"] = int(n)
    out["dim"] = int(dim)
    out["corpus_gib"] = round(n * row_bytes / 2**30, 2)
    return out


def construct(name, root):
    """Time what Dataset.__init__ does, step by step.

    A wiki1024-10m cell sat in its load phase at 0.01 of 4 cores, and the phase covers only
    the constructor, so this takes the constructor apart: which step waits, and on how much.
    """
    from .datasets import SPECS, _npy_layout, normalize

    sp = SPECS[name]
    out = {"dataset": name}
    d = os.path.join(root, sp.path)
    with Timer() as t:
        parts = [
            np.load(os.path.join(d, f"part_{i:03d}.npy"), mmap_mode="r")
            for i in range(sp.parts)
        ]
    offsets = np.cumsum([0] + [len(p) for p in parts])
    pool = int(offsets[-1])
    out["open_memmaps"] = dict(s=round(t.wall, 2), cores=t.cores, parts=sp.parts)
    out["pool_rows"] = pool
    with Timer() as t:
        qrows = np.sort(
            np.random.default_rng(20260914).choice(pool, sp.nq, replace=False)
        )
    out["choose_queries"] = dict(s=round(t.wall, 2), cores=t.cores, nq=sp.nq)
    if sp.holdout_from_pool:
        from .datasets import Dataset

        ds = Dataset.__new__(Dataset)
        ds.spec, ds.root, ds._parts, ds._offsets = sp, root, parts, offsets
        ds.dim, ds._mem, ds._keep = parts[0].shape[1], None, None
        ds._layout = [
            _npy_layout(os.path.join(d, f"part_{i:03d}.npy")) for i in range(sp.parts)
        ]
        with Timer() as t:
            normalize(ds._gather_pool(qrows))
        out["gather_queries"] = dict(s=round(t.wall, 2), cores=t.cores)
        with Timer() as t:
            mask = np.ones(pool, bool)
            mask[qrows] = False
            np.flatnonzero(mask)
        out["keep_mask"] = dict(s=round(t.wall, 2), cores=t.cores)
    gt_path = os.path.join(root, "gt", f"{name}.npy")
    if os.path.exists(gt_path):
        out["gt_file_mib"] = round(os.path.getsize(gt_path) / 2**20, 1)
        with Timer() as t:
            gt = np.load(gt_path)
            shape = gt.shape
        out["load_gt"] = dict(s=round(t.wall, 2), cores=t.cores, shape=list(shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, nargs="+")
    ap.add_argument(
        "--construct",
        action="store_true",
        help="time Dataset.__init__ step by step instead of reading the corpus",
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--gather-rows", type=int, default=50_000)
    a = ap.parse_args()
    for name in a.dataset:
        if a.construct:
            print("CONSTRUCT " + json.dumps(construct(name, a.data_root)), flush=True)
        else:
            r = probe(name, a.data_root, a.seconds, a.gather_rows)
            print("PROBE " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
