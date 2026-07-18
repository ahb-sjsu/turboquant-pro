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
