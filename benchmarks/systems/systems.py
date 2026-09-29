"""The systems under test, behind one interface.

Every builder returns a :class:`Built`: a ``search(queries, k)`` callable per
named variant, the bytes each row costs, and provenance. The harness
(``cell.py``) times and scores every system the same way.

**Bytes per row** are measured, never computed from a formula: what the index
holds that grows with N, divided by N. For faiss that is the serialized size
after ``add`` minus the serialized size of the same trained index empty (so
codebooks, rotations and centroids, which do not grow with N, are excluded, as
the campaign excludes them, and IVF's stored ids are included). For TurboQuant
Pro it is the nbytes of the arrays the index keeps per row: packed codes plus
the per-row norms the kernel reads. For ScaNN it is the serialized arrays whose
leading dimension is N. The fp32 originals a rerank reads are reported apart,
as the rerank tier, for every system.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Built:
    system: str
    variants: dict  # name -> search(queries, k) -> (ids, scores)
    bytes_per_row: float
    build_s: float
    provenance: dict = field(default_factory=dict)
    # variant whose ranking a routed variant is measured against (routing stage)
    routing_reference: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# TurboQuant Pro                                                              #
# --------------------------------------------------------------------------- #


def _tq_row_bytes(index) -> float:
    per_row = 0
    for c in index._chunks:
        per_row += int(np.asarray(c.codes.blocked).nbytes)
    for a in (index._cnorm, index._vrnorm, index._segw):
        if a is not None:
            per_row += int(np.asarray(a).nbytes)
    return per_row / max(index.size, 1)


def build_tq(ds, c, threads) -> Built:
    """PCA-Matryoshka + scalar codes in an ADCIndex, searched under each scorer:
    ``exact`` (float reference) and ``fast`` (the compiled kernel)."""
    from turboquant_pro import ADCIndex, PCAMatryoshka
    from turboquant_pro import scorer as S

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    t = time.perf_counter()
    pca = PCAMatryoshka(input_dim=ds.dim, output_dim=c["out_dim"])
    pca.fit(ds.train_sample(c["seed"], 100_000))
    index = ADCIndex(pca.with_quantizer(bits=c["bits"], seed=c["seed"]))
    for _, blk in ds.blocks():
        index.add(blk)
    build = time.perf_counter() - t

    def run(mode):
        def search(q, k):
            ids, sc = index.search(q, k=k, mode=mode)
            return np.asarray(ids), np.asarray(sc)

        return search

    return Built(
        system="tq",
        variants={"exact": run("exact"), "fast": run("fast")},
        bytes_per_row=_tq_row_bytes(index),
        build_s=build,
        provenance={
            "scorers": S.describe(),
            "fast_scorer": index.kernel_scorer if index.uses_kernel else "exact-float",
            "fallback_reason": index._fallback_reason(),
        },
    )


def build_tq_ivf(ds, c, threads) -> Built:
    """TurboQuant Pro's residual IVF: ``fast`` at the registered nprobe, and the
    same index at nprobe = nlist, the routing stage's reference."""
    from turboquant_pro import IVFIndex, PCAMatryoshka
    from turboquant_pro import scorer as S

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    t = time.perf_counter()
    pca = PCAMatryoshka(input_dim=ds.dim, output_dim=c["out_dim"])
    pca.fit(ds.train_sample(c["seed"], 100_000))
    ivf = IVFIndex.from_blocks(
        pca,
        lambda: (b for _, b in ds.blocks()),
        n=ds.n,
        train=ds.train_sample(c["seed"], 40 * c["nlist"]),
        bits=c["bits"],
        nlist=c["nlist"],
        seed=c["seed"],
        residual=True,
    )
    build = time.perf_counter() - t

    def run(nprobe):
        def search(q, k):
            ids, sc = ivf.search(q, k=k, nprobe=nprobe, mode="fast")
            return np.asarray(ids), np.asarray(sc)

        return search

    st = ivf.stats()
    return Built(
        system="tq_ivf",
        variants={
            f"nprobe{c['nprobe']}": run(c["nprobe"]),
            "nprobe_all": run(c["nlist"]),
        },
        bytes_per_row=float(st["index_bytes_per_row"]),
        build_s=build,
        provenance={"scorers": S.describe()},
        routing_reference={f"nprobe{c['nprobe']}": "nprobe_all"},
    )


# --------------------------------------------------------------------------- #
# faiss: PQ, OPQ, IVF-RaBitQ                                                  #
# --------------------------------------------------------------------------- #


def _faiss(threads):
    import faiss

    faiss.omp_set_num_threads(threads)
    return faiss


def _faiss_row_bytes(faiss, index, empty_bytes: int, n: int) -> float:
    return (len(faiss.serialize_index(index)) - empty_bytes) / max(n, 1)


def build_pq(ds, c, threads, opq=False) -> Built:
    faiss = _faiss(threads)
    t = time.perf_counter()
    spec = f"OPQ{c['m']},PQ{c['m']}x8" if opq else f"PQ{c['m']}x8"
    index = faiss.index_factory(ds.dim, spec, faiss.METRIC_INNER_PRODUCT)
    base = faiss.downcast_index(index.index if opq else index)
    base.pq.cp.seed = c["seed"]
    base.do_polysemous_training = False  # permutes code ids only (#210)
    index.train(ds.train_sample(c["seed"], 200_000))
    empty = len(faiss.serialize_index(index))
    for _, blk in ds.blocks():
        index.add(blk)
    build = time.perf_counter() - t

    def search(q, k):
        s, i = index.search(np.ascontiguousarray(q, np.float32), k)
        return i, s

    return Built(
        system="opq" if opq else "pq",
        variants={"single": search},
        bytes_per_row=_faiss_row_bytes(faiss, index, empty, ds.n),
        build_s=build,
        provenance={"faiss": faiss.__version__, "spec": spec},
    )


def build_rabitq_ivf(ds, c, threads) -> Built:
    """faiss IVF + RaBitQ exactly as the RaBitQ public campaign runs it (L2
    metric on unit rows, zero rows moved outside the sphere, unquantized queries
    ``qb = 0``), at the registered nprobe and at nprobe = nlist (the routing
    stage's reference)."""
    from rabitq_public.cell import QB, _rabitq_spec, far_zero_rows

    faiss = _faiss(threads)
    t = time.perf_counter()
    spec = f"IVF{c['nlist']},{_rabitq_spec(c['bits'])}"
    index = faiss.index_factory(ds.dim, spec, faiss.METRIC_L2)
    ivf = faiss.extract_index_ivf(index)
    ivf.cp.seed = c["seed"]
    faiss.downcast_index(ivf).qb = QB
    index.train(far_zero_rows(ds.train_sample(c["seed"], 40 * c["nlist"])))
    empty = len(faiss.serialize_index(index))
    for _, blk in ds.blocks():
        index.add(far_zero_rows(blk))
    build = time.perf_counter() - t

    def run(nprobe):
        def search(q, k):
            ivf.nprobe = nprobe
            s, i = index.search(np.ascontiguousarray(q, np.float32), k)
            return i, -s  # L2 distances on unit rows: smaller is better

        return search

    return Built(
        system="rabitq_ivf",
        variants={
            f"nprobe{c['nprobe']}": run(c["nprobe"]),
            "nprobe_all": run(c["nlist"]),
        },
        bytes_per_row=_faiss_row_bytes(faiss, index, empty, ds.n),
        build_s=build,
        provenance={"faiss": faiss.__version__, "spec": spec, "qb": QB, "metric": "l2"},
        routing_reference={f"nprobe{c['nprobe']}": "nprobe_all"},
    )


# --------------------------------------------------------------------------- #
# ScaNN                                                                       #
# --------------------------------------------------------------------------- #


def _scann_row_bytes(searcher, n: int) -> tuple[float, dict]:
    """Serialized arrays with leading dimension n, per row, split into the
    hot index and the float dataset ScaNN keeps for its own reorder."""
    with tempfile.TemporaryDirectory() as d:
        searcher.serialize(d)
        per_file = {}
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npy"):
                continue
            a = np.load(os.path.join(d, f), mmap_mode="r")
            if a.ndim and a.shape[0] == n:
                per_file[f] = a.nbytes / n
    reorder = {f: b for f, b in per_file.items() if f in ("dataset.npy",)}
    hot = sum(b for f, b in per_file.items() if f not in reorder)
    return hot, {
        "per_row_files": per_file,
        "reorder_bytes_per_row": sum(reorder.values()),
    }


def build_scann(ds, c, threads) -> Built:
    """ScaNN: a partitioning tree, anisotropic hashing (AH) scoring, and its own
    exact reorder. ``ah`` is the tree + AH ranking alone (the compressed stage);
    ``reorder`` adds ScaNN's reorder of ``reorder`` candidates against the float
    dataset it holds; ``all_leaves`` searches every leaf (routing reference)."""
    import scann

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    t = time.perf_counter()
    x = np.empty((ds.n, ds.dim), np.float32)
    for s, blk in ds.blocks():
        x[s : s + len(blk)] = blk
    searcher = (
        scann.scann_ops_pybind.builder(x, 10, "dot_product")
        .tree(
            num_leaves=c["num_leaves"],
            num_leaves_to_search=c["leaves_to_search"],
            training_sample_size=min(ds.n, 250_000),
        )
        .score_ah(
            c["dims_per_block"], anisotropic_quantization_threshold=c["aq_threshold"]
        )
        .reorder(c["reorder"])
        .set_n_training_threads(threads)
        .build()
    )
    del x
    build = time.perf_counter() - t

    def run(leaves, reorder):
        def search(q, k):
            q = np.ascontiguousarray(q, np.float32)
            ids, sc = searcher.search_batched_parallel(
                q,
                final_num_neighbors=k,
                pre_reorder_num_neighbors=reorder if reorder else k,
                leaves_to_search=leaves,
            )
            return np.asarray(ids, np.int64), np.asarray(sc)

        return search

    hot, detail = _scann_row_bytes(searcher, ds.n)
    return Built(
        system="scann",
        variants={
            "ah": run(c["leaves_to_search"], 0),
            "reorder": run(c["leaves_to_search"], c["reorder"]),
            "all_leaves": run(c["num_leaves"], 0),
        },
        bytes_per_row=hot,
        build_s=build,
        provenance={"scann": getattr(scann, "__version__", None), **detail},
        routing_reference={"ah": "all_leaves"},
    )


BUILDERS = dict(
    tq=build_tq,
    tq_ivf=build_tq_ivf,
    pq=build_pq,
    opq=lambda ds, c, th: build_pq(ds, c, th, opq=True),
    rabitq_ivf=build_rabitq_ivf,
    scann=build_scann,
)
