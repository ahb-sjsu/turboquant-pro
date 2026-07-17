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
