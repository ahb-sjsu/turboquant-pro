"""Run one registered cell and write ``<out>/<cell_id>.json`` (+ ``.ids.npz``).

    python -m rabitq_public.cell --cell-id <id> --data-root /data --out /data/results

Every method returns the top-50 corpus positions per query from its compressed search
alone; recall is then scored identically for all methods (docs/PREREG_rabitq_public.md
section 3): single-pass recall@10 from the first 10, and +rerank recall@10 after exact
cosine re-scoring of the first 20 (x2) and first 50 (x5) candidates. A finished cell is
never recomputed; a partial result is never visible (write to a temp name, then rename).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import tempfile
import threading
import time

import numpy as np

from .datasets import Dataset
from .grid import cells, supplementary_cells

K = 50
QB = 0  # faiss RaBitQ query bits; 0 = unquantized queries, the most accurate setting


def _kernel_source_sha():
    import hashlib

    from turboquant_pro import _adc

    path = os.path.join(os.path.dirname(_adc.__file__), "adc_scan.cpp")
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def m_tq(ds, c, threads):
    from turboquant_pro import ADCIndex, PCAMatryoshka

    t = time.perf_counter()
    pca = PCAMatryoshka(input_dim=ds.dim, output_dim=c["out_dim"])
    pca.fit(ds.train_sample(c["seed"], 100_000))
    pipe = pca.with_quantizer(bits=c["bits"], seed=c["seed"])
    index = ADCIndex(pipe)
    # Fill preallocated arrays block by block; ADCIndex.add on the whole stream would
    # re-concatenate its code array once per block.
    codes = np.empty((ds.n, c["out_dim"]), np.uint8)
    cnorm = np.empty(ds.n, np.float32)
    vrnorm = np.empty(ds.n, np.float32)
    for s, blk in ds.blocks():
        part = ADCIndex(pipe).add(blk)
        e = s + len(blk)
        codes[s:e], cnorm[s:e], vrnorm[s:e] = part._codes, part._cnorm, part._vrnorm
    index._codes, index._cnorm, index._vrnorm = codes, cnorm, vrnorm
    build = time.perf_counter() - t
    t = time.perf_counter()
    ids, _ = index.search(ds.queries, k=K)
    search = time.perf_counter() - t
    stored = -(-c["out_dim"] * c["bits"] // 8) + 4
    extra = dict(
        kernel=bool(index.uses_kernel),
        in_memory_bytes_per_vec=c["out_dim"] + 8,
        kernel_source_sha256=_kernel_source_sha(),
    )
    return np.asarray(ids), stored, build, search, extra


def _faiss(threads):
    import faiss

    faiss.omp_set_num_threads(threads)
    return faiss


def _rabitq_spec(bits):
    return "RaBitQ" if bits == 1 else f"RaBitQ{bits}"


def m_rabitq_flat(ds, c, threads):
    faiss = _faiss(threads)
    t = time.perf_counter()
    index = faiss.index_factory(
        ds.dim, _rabitq_spec(c["bits"]), faiss.METRIC_INNER_PRODUCT
    )
    faiss.downcast_index(index).qb = QB
    index.train(ds.train_sample(c["seed"], 200_000))
    for _, blk in ds.blocks():
        index.add(blk)
    build = time.perf_counter() - t
    t = time.perf_counter()
    _, ids = index.search(ds.queries, K)
    search = time.perf_counter() - t
    rq = faiss.downcast_index(index)
    return ids, int(rq.code_size), build, search, dict(qb=int(rq.qb), metric="ip")


FAR = 4.0


def far_zero_rows(x: np.ndarray) -> np.ndarray:
    """Move all-zero rows to a point outside the unit sphere, for the L2-metric methods.

    L2 ranks like cosine only for unit vectors. A normalized zero row stays at the origin,
    distance 1 from every query, and outranks every unit vector with cosine below 0.5: on
    NYTimes (239 zero rows) an exact IVFFlat scan fell to recall 0.43. At (4, 0, ..., 0) the
    row is at distance >= 3 from every query, farther than any unit vector (<= 2), so it is
    never retrieved ahead of a real neighbour. Cosine methods and rerank use the original
    rows. Registered as Amendment 1 of docs/PREREG_rabitq_public.md.
    """
    zero = ~np.any(x, axis=1)
    if not zero.any():
        return x
    x = np.array(x, dtype=np.float32, copy=True)
    x[zero] = 0.0
    x[zero, 0] = FAR
    return x


def _ivf_rabitq(faiss, x_train, blocks, queries, dim, c):
    index = faiss.index_factory(
        dim, f"IVF{c['nlist']},{_rabitq_spec(c['bits'])}", faiss.METRIC_L2
    )
    ivf = faiss.extract_index_ivf(index)
    ivf.cp.seed = c["seed"]
    faiss.downcast_index(ivf).qb = QB
    index.train(x_train)
    for blk in blocks:
        index.add(blk)
    ivf.nprobe = c[
        "nlist"
    ]  # exhaustive: every list is scanned, so recall reflects the estimator
    t = time.perf_counter()
    _, ids = index.search(queries, K)
    search = time.perf_counter() - t
    extra = dict(
        nprobe=int(ivf.nprobe), metric="l2", qb=int(faiss.downcast_index(ivf).qb)
    )
    return ids, int(ivf.code_size), search, extra


def m_rabitq_ivf(ds, c, threads):
    faiss = _faiss(threads)
    t = time.perf_counter()
    train = far_zero_rows(ds.train_sample(c["seed"], 40 * c["nlist"]))
    ids, stored, search, extra = _ivf_rabitq(
        faiss, train, (far_zero_rows(b) for _, b in ds.blocks()), ds.queries, ds.dim, c
    )
    build = time.perf_counter() - t - search
    return ids, stored, build, search, extra


def m_pca_rabitq_ivf(ds, c, threads):
    from turboquant_pro import PCAMatryoshka

    faiss = _faiss(threads)
    t = time.perf_counter()
    pca = PCAMatryoshka(input_dim=ds.dim, output_dim=c["out_dim"])
    pca.fit(ds.train_sample(c["seed"], 100_000))

    def proj(x):
        return np.ascontiguousarray(pca.transform(x), dtype=np.float32)

    train = proj(far_zero_rows(ds.train_sample(c["seed"], 40 * c["nlist"])))
    ids, stored, search, extra = _ivf_rabitq(
        faiss,
        train,
        (proj(far_zero_rows(b)) for _, b in ds.blocks()),
        proj(ds.queries),
        c["out_dim"],
        c,
    )
    build = time.perf_counter() - t - search
    return ids, stored, build, search, extra


STALE_SCRATCH_S = 6 * 3600


def _sweep_stale_scratch(scratch, older_than=STALE_SCRATCH_S):
    """Delete corpus copies left behind by pods that were killed before their cleanup ran.

    Seven of these, 3.6 GiB each, held 26 GiB of a 120 GiB volume after the cells that wrote
    them were stopped mid-run on 2026-09-15. A pod cannot clean up after SIGKILL, so the next
    one does it.
    """
    try:
        now = time.time()
        for name in os.listdir(scratch):
            if not name.endswith(".corpus.npy"):
                continue
            path = os.path.join(scratch, name)
            if now - os.path.getmtime(path) > older_than:
                os.unlink(path)
                print(f"removed stale scratch {name}", flush=True)
    except OSError:
        pass


def m_rabitqlib_ivf(ds, c, threads):
    import rabitqlib

    faiss = _faiss(threads)
    t = time.perf_counter()
    km = faiss.Kmeans(ds.dim, c["nlist"], niter=20, seed=c["seed"])
    km.train(far_zero_rows(ds.train_sample(c["seed"], 40 * c["nlist"])))
    quant = faiss.IndexFlatL2(ds.dim)
    quant.add(km.centroids)
    # rabitqlib.build takes the whole float32 corpus. By default that copy is a memory map on
    # scratch storage (reclaimable page cache, not anonymous memory). TQP_RBQ_SCRATCH=ram
    # keeps it in RAM instead: on the 10M x 1024 arm the build reads the map in random order,
    # and over CephFS that left the pod in disk sleep at 3% CPU.
    scratch = os.environ.get("TQP_RBQ_SCRATCH") or tempfile.gettempdir()
    _sweep_stale_scratch(scratch)
    spath = None
    if scratch == "ram":
        data = np.empty((ds.n, ds.dim), np.float32)
    else:
        os.makedirs(scratch, exist_ok=True)
        spath = os.path.join(scratch, f"{c['cell_id']}.corpus.npy")
        data = np.lib.format.open_memmap(
            spath, mode="w+", dtype=np.float32, shape=(ds.n, ds.dim)
        )
    cid = np.empty(ds.n, np.uint32)
    try:
        for s, blk in ds.blocks():
            blk = far_zero_rows(blk)
            data[s : s + len(blk)] = blk
            cid[s : s + len(blk)] = quant.search(blk, 1)[1][:, 0]
        if spath:
            data.flush()
        index = rabitqlib.IvfIndex(ds.dim, ds.n, c["nlist"], c["bits"], "l2")
        index.build(
            data,
            np.ascontiguousarray(km.centroids, dtype=np.float32),
            cid,
            threads,
            False,
        )
    finally:
        del data
        if spath:
            os.unlink(spath)
    build = time.perf_counter() - t
    t = time.perf_counter()
    res = index.search(ds.queries, K, c["nlist"], True, threads)
    search = time.perf_counter() - t
    ids = np.asarray(res[0], dtype=np.int64)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ix.bin")
        index.save(p)
        file_bytes = os.path.getsize(p)
    per_vec = (file_bytes - c["nlist"] * ds.dim * 4) / ds.n
    return (
        ids,
        round(per_vec, 2),
        build,
        search,
        dict(nprobe=c["nlist"], file_bytes=file_bytes),
    )


def m_pq(ds, c, threads, opq=False):
    faiss = _faiss(threads)
    t = time.perf_counter()
    spec = f"OPQ{c['m']},PQ{c['m']}x8" if opq else f"PQ{c['m']}x8"
    index = faiss.index_factory(ds.dim, spec, faiss.METRIC_INNER_PRODUCT)
    base = faiss.downcast_index(index.index if opq else index)
    base.pq.cp.seed = c["seed"]
    train = ds.train_sample(c["seed"], 200_000)
    note("pq: training rows gathered")
    index.train(train)
    del train
    note("pq: trained")
    for i, (_, blk) in enumerate(ds.blocks()):
        index.add(blk)
        if i % 10 == 0:
            note(f"pq: added block {i}")
    note("pq: added")
    build = time.perf_counter() - t
    t = time.perf_counter()
    _, ids = index.search(ds.queries, K)
    search = time.perf_counter() - t
    return ids, int(c["m"]), build, search, dict(metric="ip")


METHODS = dict(
    tq=m_tq,
    tqfix=m_tq,  # same pipeline; the pod compiles the v2 kernel (Amendment 2)
    rabitq_flat=m_rabitq_flat,
    rabitq_ivf=m_rabitq_ivf,
    pca_rabitq_ivf=m_pca_rabitq_ivf,
    rabitqlib_ivf=m_rabitqlib_ivf,
    pq=m_pq,
    opq=lambda ds, c, th: m_pq(ds, c, th, opq=True),
)


def hits_at_10(gt: np.ndarray, top: np.ndarray) -> np.ndarray:
    g = gt[:, :10]
    return np.array(
        [len(np.intersect1d(g[i], top[i][top[i] >= 0])) for i in range(len(g))],
        np.uint8,
    )


def rerank(ds: Dataset, ids: np.ndarray, depth: int) -> np.ndarray:
    cand = ids[:, :depth]
    uniq = np.unique(cand[cand >= 0])
    rows = ds.take(uniq)
    out = np.full((len(cand), 10), -1, np.int64)
    for i in range(len(cand)):
        c = cand[i][cand[i] >= 0]
        s = rows[np.searchsorted(uniq, c)] @ ds.queries[i]
        top = c[np.argsort(-s, kind="stable")[:10]]
        out[i, : len(top)] = top
    return out


def _peak_rss_gib():
    if sys.platform == "win32":
        return None
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def note(label):
    """One line of memory at a named step, so an OOM kill says which step did it.

    A wiki PQ cell went from 3.3 GiB to past its 12 GiB limit inside one reporting interval,
    which is a single large allocation rather than the page cache the earlier fix addressed.
    The periodic line brackets such a jump; these name it.
    """
    cur = _cgroup_mem_bytes()
    anon = 0
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("RssAnon:"):
                    anon = int(ln.split()[1]) << 10
                    break
    except OSError:
        pass
    total = (cur or anon) / 2**30
    print(f"STEP {label}: mem={total:.1f} anon={anon / 2**30:.1f} GiB", flush=True)


def _cpu_seconds():
    """CPU seconds this process and its reaped children have burned, or None off Linux.

    Read from /proc/self/stat rather than the cgroup's cpu.stat: inside a pod the two agree,
    but on a plain host the cgroup root reports the whole machine.
    """
    try:
        with open("/proc/self/stat") as f:
            f = f.read().rpartition(") ")[2].split()
        return sum(int(f[i]) for i in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _cgroup_mem_bytes():
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read())
    except OSError:
        return None


class AnonPeak:
    """Time-averaged CPU and memory for the pod, a peak, and a per-phase breakdown.

    ru_maxrss counts file pages mapped from the memory-mapped corpus, which the kernel can
    reclaim; anonymous memory is what a pod's memory request has to cover. But the cluster
    judges a pod on *averages* over its life, not peaks: on 2026-09-15 eight cells were
    flagged at 1-4% of 4 requested CPUs while a snapshot showed several near 3.7 cores,
    because their CephFS-bound phases sink the average. So this also samples the cgroup's
    memory and CPU counters, and ``mark`` splits the run into phases, which is what tells us
    *which* phase is idle. benchmarks/nrp/sizing.py turns the result into a request.
    """

    def __init__(self, report_s=None):
        # A pod that is OOM-killed never writes its record, so the peak it reached has to be
        # in the log before it dies: the line below carries the running peak, not just the
        # current value, and the interval is short enough to bracket a sudden allocation.
        self.report_s = report_s or float(os.environ.get("TQP_USAGE_REPORT_S", "60"))
        self.current = None
        self.peak_kib = 0
        self.mem_sum = self.mem_n = self.mem_peak = 0
        self.anon_sum = self.anon_n = 0
        self.t0 = time.perf_counter()
        self.cpu0 = _cpu_seconds()
        self.phases = {}
        self._said = self.t0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                with open("/proc/self/status") as f:
                    for ln in f:
                        if ln.startswith("RssAnon:"):
                            kib = int(ln.split()[1])
                            self.peak_kib = max(self.peak_kib, kib)
                            self.anon_sum += kib << 10
                            self.anon_n += 1
                            break
            except OSError:
                return
            cur = _cgroup_mem_bytes()
            if cur is not None:
                self.mem_sum += cur
                self.mem_n += 1
                self.mem_peak = max(self.mem_peak, cur)
            now = time.perf_counter()
            if now - self._said >= self.report_s:
                self._said = now
                cpu, wall = _cpu_seconds(), now - self.t0
                cores = (
                    (cpu - self.cpu0) / wall
                    if cpu is not None and self.cpu0 is not None and wall > 0
                    else float("nan")
                )
                now_gib = (cur or self.peak_kib << 10) / 2**30
                peak_gib = max(self.mem_peak, self.peak_kib << 10) / 2**30
                # anon is what the process allocated; the rest of memory.current is page
                # cache from reading the corpus, which the cgroup charges to the pod too.
                print(
                    f"USAGE {wall / 60:5.1f} min phase={self.current} "
                    f"mean_cores={cores:.2f} mem={now_gib:.1f} peak={peak_gib:.1f} "
                    f"anon={(self.peak_kib << 10) / 2**30:.1f} GiB",
                    flush=True,
                )
            self._stop.wait(0.5)

    @contextlib.contextmanager
    def phase(self, name):
        """Time one phase and record the CPU cores it averaged."""
        self.current = name
        t, cpu = time.perf_counter(), _cpu_seconds()
        try:
            yield
        finally:
            wall = time.perf_counter() - t
            now = _cpu_seconds()
            self.phases[name] = dict(
                s=round(wall, 1),
                cpu_cores=(
                    round((now - cpu) / wall, 3)
                    if now is not None and cpu is not None and wall > 0
                    else None
                ),
            )
            self.current = None

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def gib(self):
        return round(self.peak_kib / 2**20, 3) if self.peak_kib else None

    def _mean_mem(self):
        """Mean charged memory, or mean anonymous memory where the cgroup file is absent
        (as on a plain host): anonymous memory omits the page cache the cluster counts, so
        the fallback understates the average and can only make a request smaller."""
        if self.mem_n:
            return round(self.mem_sum / self.mem_n / 2**30, 3)
        return round(self.anon_sum / self.anon_n / 2**30, 3) if self.anon_n else None

    def usage(self):
        """What the cluster would average over this pod: cores, mean and peak GiB."""
        wall = time.perf_counter() - self.t0
        cpu = _cpu_seconds()
        return dict(
            mean_cpu_cores=(
                round((cpu - self.cpu0) / wall, 3)
                if cpu is not None and self.cpu0 is not None and wall > 0
                else None
            ),
            mean_mem_gib=self._mean_mem(),
            peak_mem_gib=round(max(self.mem_peak, self.peak_kib << 10) / 2**30, 3)
            or None,
            peak_anon_gib=self.gib,
            wall_s=round(wall, 1),
            phases=self.phases,
        )


def environment() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    import faiss

    import turboquant_pro

    try:
        rbq = version("rabitqlib")
    except PackageNotFoundError:
        rbq = None
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(
                ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    return dict(
        host=platform.node(),
        cpu=cpu,
        python=platform.python_version(),
        numpy=np.__version__,
        faiss=faiss.__version__,
        turboquant_pro=turboquant_pro.__version__,
        rabitqlib=rbq,
        commit=os.environ.get("TQP_COMMIT", "unknown"),
    )


def run(cell: dict, data_root: str, out_dir: str, threads: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cell['cell_id']}.json")
    if os.path.exists(path):
        return path
    t0 = time.time()
    with AnonPeak() as anon:
        with anon.phase("load"):
            ds = Dataset(cell["dataset"], data_root)
        if ds.gt is None:
            raise SystemExit(f"no ground truth for {cell['dataset']}; run gt.py first")
        with anon.phase("index"):
            ids, stored, build_s, search_s, extra = METHODS[cell["method"]](
                ds, cell, threads
            )
        ids = np.asarray(ids, dtype=np.int64)
        with anon.phase("rerank"):
            hits = dict(
                hits_single=hits_at_10(ds.gt, ids[:, :10]).tolist(),
                hits_rr2=hits_at_10(ds.gt, rerank(ds, ids, 20)).tolist(),
                hits_rr5=hits_at_10(ds.gt, rerank(ds, ids, 50)).tolist(),
            )
    rec = dict(
        cell=cell,
        n=ds.n,
        dim=ds.dim,
        nq=len(ds.queries),
        stored_bytes_per_vec=stored,
        **hits,
        build_s=round(build_s, 2),
        search_s=round(search_s, 3),
        threads=threads,
        peak_rss_gib=_peak_rss_gib(),
        peak_anon_gib=anon.gib,
        usage=anon.usage(),
        wall_s=round(time.time() - t0, 1),
        extra=extra,
        env=environment(),
        finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    for k in ("hits_single", "hits_rr2", "hits_rr5"):
        rec[k.replace("hits", "recall10")] = round(float(np.mean(rec[k])) / 10, 5)
    np.savez_compressed(
        os.path.join(out_dir, f"{cell['cell_id']}.ids.npz"), ids=ids.astype(np.int32)
    )
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell-id")
    g.add_argument("--cell-json", help="an unregistered cell, for smoke tests only")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--threads", type=int, default=int(os.environ.get("CELL_THREADS", "4"))
    )
    a = ap.parse_args()
    if a.cell_json:
        cell = json.loads(a.cell_json)
    else:
        cell = next(
            c for c in cells() + supplementary_cells() if c["cell_id"] == a.cell_id
        )
    print(run(cell, a.data_root, a.out, a.threads), flush=True)


if __name__ == "__main__":
    main()
