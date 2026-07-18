"""Distributed scatter-gather search: partition a sharded index across in-process
shard-servers and verify the coordinator's merged top-k equals single-node search
(the transport here is a plain in-process call; in production it is nats-bursting).
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import (
    ShardedIndex,
    ShardServer,
    partition_manifest,
    scatter_gather,
)


def _corpus(n=3000, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    coeffs = rng.standard_normal((n, rank)) * np.linspace(1.0, 0.3, rank)
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def _recall(got, ref, k):
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(got[:, :k], ref)]))


def test_scatter_gather_ivf_equals_single_node(tmp_path):
    corpus = _corpus(3000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=300, output_dim=32, bits=4
    ).build_ivf(
        nlist=64
    )  # 10 shards, global coarse layer
    q = corpus[:60]
    ref, _ = sh.search(q, k=10, nprobe=16)  # single-node IVF over all shards

    # Partition the shards across 3 servers; the global coarse layer stays shared, so
    # every server probes the same cells (IVF-as-router). Transport is in-process.
    subs = partition_manifest(str(tmp_path / "s" / "manifest.json"), 3)
    servers = {p: ShardServer(p) for p in subs}
    ids, _ = scatter_gather(
        q, 10, subs, lambda ep, b: servers[ep].handle(b), nprobe=16, max_parallel=3
    )
    assert _recall(ids, ref, 10) > 0.999  # merged partials == single-node, exactly


def test_scatter_gather_full_scan_equals_single_node(tmp_path):
    corpus = _corpus(2000)
    sh = ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=250, output_dim=32, bits=4
    )  # 8 shards, no IVF
    q = corpus[:40]
    ref, _ = sh.search(q, k=10)  # single-node full-scan ADC
    subs = partition_manifest(str(tmp_path / "s" / "manifest.json"), 4)
    servers = {p: ShardServer(p) for p in subs}
    ids, _ = scatter_gather(q, 10, subs, lambda ep, b: servers[ep].handle(b))
    assert _recall(ids, ref, 10) > 0.999


def test_partition_manifest_covers_all_shards(tmp_path):
    corpus = _corpus(1200)
    ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=100, output_dim=32, bits=4
    )  # 12 shards
    subs = partition_manifest(str(tmp_path / "s" / "manifest.json"), 5)
    total = sum(ShardServer(p).index.n_shards for p in subs)
    assert total == 12  # disjoint round-robin partition covers every shard exactly once
