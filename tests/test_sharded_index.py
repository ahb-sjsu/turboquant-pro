"""Sharded index: fan-out search over single-basis shards, merged to a global top-k.

Validates that a single shard reduces exactly to a `TQEIndex`, and that a
multi-shard index recovers the true neighbours (shared basis -> comparable scores
-> correct merge), all over memory-mapped shards.
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import ShardedIndex, TQEIndex


def _corpus(n=2000, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    coeffs = rng.standard_normal((n, rank)) * np.linspace(1.0, 0.3, rank)
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def _exact_topk(queries, base, k):
    bn = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-30)
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-30)
    return np.argsort(-(qn @ bn.T), axis=1)[:, :k]


def test_single_shard_equals_tqeindex(tmp_path):
    corpus = _corpus(800)
    single = TQEIndex.create(corpus, output_dim=32, bits=4, seed=42, ids=np.arange(800))
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s1"), shard_size=10_000, output_dim=32, bits=4, seed=42
    )
    assert sh.n_shards == 1 and sh.n_rows == 800
    q = corpus[:40]
    a, _ = single.search(q, k=10, rerank=10)
    b, _ = sh.search(q, k=10, rerank=10)
    np.testing.assert_array_equal(a, b)


def test_multishard_recall_and_global_ids(tmp_path):
    corpus = _corpus(2000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=500, output_dim=32, bits=4
    )
    assert sh.n_shards == 4 and sh.n_rows == 2000
    assert sh.stats()["shard_rows"] == [500, 500, 500, 500]

    q = corpus[:80]
    gt = _exact_topk(q, corpus, 10)  # ids == positions for a fresh contiguous build
    ids, _ = sh.search(q, k=10, rerank=10)
    recall = float(np.mean([len(set(a) & set(g)) / 10 for a, g in zip(ids, gt)]))
    assert recall > 0.9  # exact rerank across shards recovers the neighbours
    # global ids span all shards, and each query finds itself
    assert ids.max() >= 1500  # results come from the last shard too
    assert all(i in ids[i] for i in range(len(q)))


def test_streaming_ingest_equals_in_memory(tmp_path):
    # Building from an iterator of row-blocks (never holding the full corpus)
    # must produce a byte-for-byte equivalent index to the in-memory create().
    corpus = _corpus(2000)
    shard_size = 500
    ref = ShardedIndex.create(
        corpus, str(tmp_path / "mem"), shard_size=shard_size, output_dim=32, bits=4
    )

    def blocks():  # yields the corpus one shard-sized block at a time
        for s in range(0, len(corpus), shard_size):
            yield corpus[s : s + shard_size]

    streamed = ShardedIndex.create_streaming(
        blocks(), str(tmp_path / "stream"), output_dim=32, bits=4, shard_size=shard_size
    )
    assert streamed.n_shards == ref.n_shards == 4
    assert streamed.n_rows == ref.n_rows == 2000
    q = corpus[:60]
    a, sa = ref.search(q, k=10, rerank=10)
    b, sb = streamed.search(q, k=10, rerank=10)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(sa, sb, rtol=0, atol=0)


def test_streaming_uneven_blocks_and_empty(tmp_path):
    # Blocks may be uneven; ids stay global/contiguous in iteration order.
    corpus = _corpus(1000)
    sizes = [400, 300, 300]
    bounds = np.cumsum([0, *sizes])

    def blocks():
        for a, b in zip(bounds[:-1], bounds[1:]):
            yield corpus[a:b]

    sh = ShardedIndex.create_streaming(
        blocks(), str(tmp_path / "s"), output_dim=32, bits=4
    )
    assert sh.n_shards == 3 and sh.n_rows == 1000
    assert sh.stats()["shard_rows"] == sizes
    ids, _ = sh.search(corpus[:20], k=5, rerank=10)
    assert all(i in ids[i] for i in range(20))

    import pytest

    with pytest.raises(ValueError, match="at least one non-empty block"):
        ShardedIndex.create_streaming(iter([]), str(tmp_path / "empty"))


def test_bounded_open_shards_matches_full_open(tmp_path):
    # Many shards with a tiny open budget must (a) never hold more than the budget
    # open at once and (b) return exactly the same results as opening everything.
    corpus = _corpus(1200)
    ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=100, output_dim=32, bits=4
    )  # 12 shards
    q = corpus[:50]
    manifest = str(tmp_path / "s" / "manifest.json")
    full = ShardedIndex.open(manifest, max_open_shards=64)
    bounded = ShardedIndex.open(manifest, max_open_shards=3)
    assert bounded.n_shards == 12
    a, _ = full.search(q, k=10, rerank=10)
    b, _ = bounded.search(q, k=10, rerank=10)
    np.testing.assert_array_equal(a, b)
    assert len(bounded._open) <= 3  # fd budget respected across the fan-out
    bounded.close()
    assert len(bounded._open) == 0


def _recall(got, ref, k):
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(got[:, :k], ref)]))


def test_sharded_ivf_matches_fullscan_and_is_selective(tmp_path):
    corpus = _corpus(3000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=750, output_dim=32, bits=4
    )  # 4 shards
    sh.build_ivf(nlist=64)
    assert sh.has_ivf
    q = corpus[:60]

    # Reference: the full-scan sharded ADC ranking (what IVF must reproduce).
    full, _ = sh.search(q, k=10)
    # Probing every cell scores every row -> identical to the full scan.
    allcells, _ = sh.search(q, k=10, nprobe=64)
    assert _recall(allcells, full, 10) > 0.99
    # A few cells: still high recall, at a fraction of the rows; each finds itself.
    few, _ = sh.search(q, k=10, nprobe=8)
    assert _recall(few, full, 10) > 0.75
    assert all(i in few[i] for i in range(len(q)))


def test_sharded_hierarchical_ivf_matches_and_is_local(tmp_path):
    from turboquant_pro.adc_index import _normalize
    from turboquant_pro.ivf import probed_leaves_hier

    corpus = _corpus(4000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=800, output_dim=32, bits=4
    )  # 5 shards
    sh.build_ivf(hierarchical=True, top_nlist=8, sub_nlist=8)  # 64 leaves, 2 levels
    assert sh.has_ivf and sh._ivf_hier
    assert sh._ivf_meta["top_nlist"] == 8 and sh._ivf_meta["sub_nlist"] == 8
    q = corpus[:80]

    full, _ = sh.search(q, k=10)  # full-scan sharded ADC ranking (the reference)
    # Probing every top and every leaf == a full scan -> reproduces the ranking.
    allcells, _ = sh.search(q, k=10, nprobe=64, top_probe=8)
    assert _recall(allcells, full, 10) > 0.99
    # A few leaves under a few tops: still strong recall at a fraction of the rows.
    few, _ = sh.search(q, k=10, nprobe=16, top_probe=4)
    assert _recall(few, full, 10) > 0.7

    # Locality: the probed leaves for each query cluster into <= top_probe top cells,
    # which is what lets a router touch only a few servers (leaf // sub_nlist == top).
    cent, radius = sh._load_ivf()
    q_rot, _ = sh._get_shard(0)._adc._query_terms(q)
    probed, top_sel = probed_leaves_hier(
        _normalize(q_rot), sh._ivf_top, cent, radius, sh._ivf_sub_nlist, 16, 4
    )
    tops_touched = probed // sh._ivf_sub_nlist
    for i in range(len(q)):
        assert len(np.unique(tops_touched[i])) <= 4  # <= top_probe tops per query
        assert set(tops_touched[i].tolist()) <= set(top_sel[i].tolist())
    assert tops_touched.max() < 8  # never routes outside the top_nlist top cells


def test_hierarchical_ivf_persists_across_reopen(tmp_path):
    corpus = _corpus(3000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=1000, output_dim=32, bits=4
    ).build_ivf(hierarchical=True, top_nlist=6, sub_nlist=6)
    q = corpus[:40]
    before, _ = sh.search(q, k=10, nprobe=12, top_probe=3)
    reopened = ShardedIndex.open(str(tmp_path / "s" / "manifest.json"))
    assert reopened._ivf_hier and reopened._ivf_sub_nlist == 6
    after, _ = reopened.search(q, k=10, nprobe=12, top_probe=3)
    np.testing.assert_array_equal(before, after)  # hierarchy reloads identically


def test_tiered_rerank_beats_adc_ceiling(tmp_path):
    from turboquant_pro import NpyOriginalStore

    corpus = _corpus(4000)
    q = corpus[:80]
    # True neighbours: exact fp32 cosine top-10 (what ADC only approximates).
    cn = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    true = np.argsort(-(qn @ cn.T), axis=1)[:, :10]

    # Hot tier: codes only (no originals), aggressive 2-bit so ADC is lossy; IVF.
    sh = ShardedIndex.create(
        corpus,
        str(tmp_path / "s"),
        shard_size=1000,
        output_dim=20,
        bits=2,
        keep_originals=False,
    ).build_ivf(nlist=64)
    # Cold tier: originals as an id-indexed npy store.
    store = NpyOriginalStore.write(str(tmp_path / "orig.npy"), corpus)

    plain, _ = sh.search(q, k=10, nprobe=32)  # ADC-only shortlist ranking
    tiered, _ = sh.search(q, k=10, nprobe=32, rerank=10, rerank_store=store)

    plain_recall = _recall(plain, true, 10)
    tiered_recall = _recall(tiered, true, 10)
    # Exact rescoring over a superset shortlist can only help, and here clearly does.
    assert tiered_recall >= plain_recall
    assert tiered_recall > plain_recall + 0.03  # breaks the ADC ceiling
    assert tiered_recall > 0.9  # near-exact once the shortlist is rescored


def test_rerank_candidates_exact_over_shortlist(tmp_path):
    from turboquant_pro import NpyOriginalStore, rerank_candidates

    rng = np.random.default_rng(1)
    corpus = rng.standard_normal((200, 16)).astype(np.float32)
    store = NpyOriginalStore.write(str(tmp_path / "o.npy"), corpus)
    q = corpus[:5]
    # Shortlist = a superset of the true top-3 plus distractors; rerank must recover it.
    cn = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    true = np.argsort(-(qn @ cn.T), axis=1)[:, :3]
    cand = np.array([list(true[r]) + [50, 51, 52, 60, 61] for r in range(len(q))])
    ids, sc = rerank_candidates(q, cand, 3, store, metric="cosine")
    for r in range(len(q)):
        assert set(ids[r]) == set(true[r])  # exact top-3 recovered from the shortlist
    assert np.all(np.diff(sc, axis=1) <= 1e-6)  # scores descending


def test_sharded_ivf_parallel_equals_sequential(tmp_path):
    corpus = _corpus(3000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=500, output_dim=32, bits=4
    ).build_ivf(
        nlist=64
    )  # 6 shards
    q = corpus[:60]
    seq, ss = sh.search(q, k=10, nprobe=16, workers=1)
    par, sp = sh.search(q, k=10, nprobe=16, workers=4)  # thread pool over shards
    np.testing.assert_array_equal(seq, par)  # parallel fan-out is exact
    np.testing.assert_allclose(ss, sp, rtol=0, atol=0)


def test_parallel_build_equals_streaming(tmp_path):
    # Shard 0 fits the basis; the remaining shards build concurrently (each reads
    # shard 0's basis read-only) and must yield a byte-equivalent index to the
    # sequential create() — the ingest side of distributed scale.
    from concurrent.futures import ThreadPoolExecutor

    corpus = _corpus(2000)
    ss = 500  # 4 shards
    ref = ShardedIndex.create(
        corpus, str(tmp_path / "seq"), shard_size=ss, output_dim=32, bits=4
    )
    out = str(tmp_path / "par")
    blocks = [corpus[s : s + ss] for s in range(0, len(corpus), ss)]
    metas = [ShardedIndex.write_shard(out, blocks[0], 0, 0, output_dim=32, bits=4)]
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [
            ex.submit(
                ShardedIndex.write_shard,
                out,
                blocks[i],
                i,
                i * ss,
                output_dim=32,
                bits=4,
            )
            for i in range(1, len(blocks))
        ]
        metas += [f.result() for f in futs]
    par = ShardedIndex.finalize_manifest(out, metas, metric="cosine", shard_size=ss)
    assert par.n_shards == ref.n_shards == 4 and par.n_rows == ref.n_rows == 2000
    q = corpus[:60]
    a, sa = ref.search(q, k=10, rerank=10)
    b, sb = par.search(q, k=10, rerank=10)
    np.testing.assert_array_equal(a, b)  # parallel build == sequential, exactly
    np.testing.assert_allclose(sa, sb, rtol=0, atol=0)


def test_sharded_ivf_gpu_build_matches_recall(tmp_path):
    import pytest

    pytest.importorskip("cupy")  # GPU path; skipped where CuPy is absent (CI)
    corpus = _corpus(3000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=750, output_dim=32, bits=4
    )
    sh.build_ivf(nlist=64, device="gpu")  # k-means + assignment on the GPU
    assert sh.has_ivf
    q = corpus[:60]
    full, _ = sh.search(q, k=10)
    few, _ = sh.search(q, k=10, nprobe=16)
    assert _recall(few, full, 10) > 0.7  # GPU-built partition recovers the neighbours


def test_sharded_ivf_persists_across_reopen(tmp_path):
    corpus = _corpus(2000)
    ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=500, output_dim=32, bits=4
    ).build_ivf(nlist=40)
    reopened = ShardedIndex.open(str(tmp_path / "s" / "manifest.json"))
    assert reopened.has_ivf
    ids, _ = reopened.search(corpus[:20], k=5, nprobe=8)
    assert all(i in ids[i] for i in range(20))  # each query finds itself via IVF


def test_reopen_from_manifest(tmp_path):
    corpus = _corpus(1200)
    ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=400, output_dim=32, bits=4
    )
    reopened = ShardedIndex.open(str(tmp_path / "s" / "manifest.json"))
    assert reopened.n_shards == 3 and reopened.n_rows == 1200
    ids, _ = reopened.search(corpus[:20], k=5, rerank=10)
    assert all(i in ids[i] for i in range(20))  # each query finds itself
    reopened.close()


def test_ivf_sidecars_are_uint32_and_packed_scan_matches(tmp_path):
    """Format-v3 economy at the sharded layer: uint32 posting lists + packed
    code gathers in the IVF scan, scoring identically to the full scan."""
    import glob
    import os

    corpus = _corpus(3000)
    sh = ShardedIndex.create(
        corpus,
        str(tmp_path / "p"),
        shard_size=1000,
        output_dim=32,
        bits=4,
        keep_originals=False,
    )
    sh.build_ivf(nlist=48)
    membs = sorted(glob.glob(os.path.join(str(tmp_path / "p"), "*.ivf.memb.npy")))
    assert membs and all(np.load(m).dtype == np.uint32 for m in membs)
    # Shards are v3 on disk (packed codes, elided arange ids)...
    shard0 = TQEIndex.open(os.path.join(str(tmp_path / "p"), "shard_00000.tqe"))
    assert shard0.stats()["format_version"] == 3
    # ...and the IVF path (uint32 rows -> PackedCodes gather) agrees with the
    # exhaustive fan-out at a full-coverage nprobe.
    q = corpus[:30]
    full, _ = sh.search(q, k=10)
    ivf, _ = sh.search(q, k=10, nprobe=48)
    np.testing.assert_array_equal(full, ivf)


def test_ivf_source_routing_exact_and_skips_shards(tmp_path):
    """Source routing: the cell->shard occupancy table may change WHICH shards
    are opened, never WHAT is returned. Cluster-per-shard layout so routes are
    actually sparse; equality is checked against the same search with the table
    disabled — the code path an index built before the table existed takes."""
    rng = np.random.default_rng(7)
    dim, per = 64, 600
    centers = np.zeros((4, dim), dtype=np.float32)
    for c in range(4):
        centers[c, c * 8] = 4.0
    corpus = np.concatenate(
        [centers[c] + 0.05 * rng.standard_normal((per, dim)) for c in range(4)]
    ).astype(np.float32)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=per, output_dim=32, bits=4
    )  # shard i holds exactly cluster i
    sh.build_ivf(nlist=16)

    # Simulate a pre-table index: manifest without "occupancy" -> fallback.
    reopened = ShardedIndex.open(str(tmp_path / "s" / "manifest.json"))
    meta = dict(reopened._ivf_meta)
    meta.pop("occupancy", None)
    reopened._ivf_meta, reopened._ivf_occupancy = meta, None
    assert reopened._shard_routes() is None

    # Single-cluster batches (sparse routes) and one spanning all four.
    batches = [corpus[:20], corpus[per * 2 : per * 2 + 20], corpus[:: per // 5][:20]]
    for q in batches:
        for npb, workers in [(2, 1), (4, 4), (16, 1)]:
            ids, sc = sh.search(q, k=5, nprobe=npb, workers=workers)
            ref_ids, ref_sc = reopened.search(q, k=5, nprobe=npb, workers=workers)
            np.testing.assert_array_equal(ids, ref_ids)
            np.testing.assert_array_equal(sc, ref_sc)

    # The routing must actually skip: a one-cluster batch at small nprobe
    # cannot need all four shards, while the fallback still scans everything.
    sh.search(corpus[:20], k=5, nprobe=2)
    assert sh._last_shards_scanned < sh.n_shards
    reopened.search(corpus[:20], k=5, nprobe=2)
    assert reopened._last_shards_scanned == reopened.n_shards
