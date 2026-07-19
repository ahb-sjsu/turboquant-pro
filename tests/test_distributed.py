"""Distributed scatter-gather search: partition a sharded index across in-process
shard-servers and verify the coordinator's merged top-k equals single-node search
(the transport here is a plain in-process call; in production it is nats-bursting).
"""

from __future__ import annotations

import asyncio
import os

import numpy as np

from turboquant_pro import (
    NatsShardServer,
    Router,
    ShardedIndex,
    ShardServer,
    build_cell_placement,
    nats_transport,
    partition_manifest,
    scatter_gather,
    scatter_gather_routed,
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


def test_ivf_router_matches_and_reduces_fanout(tmp_path):
    # Cell-aligned placement: each server owns a cell-range. The router sends a query
    # only to the servers holding its probed cells -> exact result, fewer servers hit.
    corpus = _corpus(4000)
    nlist, n_servers = 32, 4
    cpp = nlist // n_servers  # cells per server

    # 1) reference index (one shard) + IVF -> global centroids + per-row cell
    orig = ShardedIndex.create(
        corpus, str(tmp_path / "orig"), shard_size=len(corpus), output_dim=32, bits=4
    )
    orig.build_ivf(nlist=nlist)
    odir = str(tmp_path / "orig")
    off = np.load(os.path.join(odir, "shard_00000.ivf.off.npy"))
    memb = np.load(os.path.join(odir, "shard_00000.ivf.memb.npy"))
    row_cell = np.empty(len(corpus), dtype=np.int64)
    for c in range(nlist):
        row_cell[memb[off[c] : off[c + 1]]] = c

    # 2) re-shard cell-aligned, reusing orig's basis + centroids so cells stay stable
    out = str(tmp_path / "aligned")
    basis = os.path.join(odir, "shard_00000.tqe")
    metas = []
    for s in range(n_servers):
        rows = np.where((row_cell >= s * cpp) & (row_cell < (s + 1) * cpp))[0]
        metas.append(
            ShardedIndex.write_shard(
                out, corpus[rows], s, ids=rows, basis_from=basis, bits=4
            )
        )
    aligned = ShardedIndex.finalize_manifest(out, metas)
    centroids = np.load(os.path.join(odir, "coarse_centroids.npy"))
    aligned.build_ivf(centroids=centroids)  # same basis + centroids -> cell-aligned
    ref, _ = aligned.search(corpus[:60], k=10, nprobe=4)  # single-node over all shards

    # 3) one server per shard; placement is sparse (each cell on exactly one server)
    subs = partition_manifest(os.path.join(out, "manifest.json"), n_servers)
    servers = {p: ShardServer(p) for p in subs}
    placement = build_cell_placement(subs, subs)
    assert all(len(v) == 1 for v in placement.values())  # cell-aligned -> sparse map
    router = Router(
        out, placement, pipeline_manifest=os.path.join(out, "manifest.json")
    )

    q = corpus[:60]
    ids, _ = scatter_gather_routed(
        q, 10, router, lambda ep, b: servers[ep].handle(b), nprobe=4
    )
    assert _recall(ids, ref, 10) > 0.999  # routed == single-node, exactly
    # each query reaches fewer than all servers (its probed cells cluster on a subset)
    fan = np.mean(
        [len(router.servers_for(q[i : i + 1], nprobe=2)[0]) for i in range(len(q))]
    )
    assert fan < n_servers


class _FakeBus:
    """In-process stand-in for a NATS connection: ``subscribe`` records (subject -> cb)
    with queue-group semantics; ``request`` routes to a subscriber's cb and returns its
    reply. Exercises the adapter's subject routing + request/reply + sync bridge without
    a real nats-server (the fabric's 'Transport seam for deterministic testing')."""

    def __init__(self):
        self._subs: dict[str, list] = {}

    async def subscribe(self, subject, queue=None, cb=None):
        self._subs.setdefault(subject, []).append(cb)
        return object()

    async def request(self, subject, data, timeout=None):
        cbs = self._subs.get(subject)
        if not cbs:
            raise RuntimeError(f"no subscriber for {subject}")
        holder = {}

        class _Msg:
            def __init__(self, data):
                self.data = data

            async def respond(self, reply):
                holder["reply"] = reply

        await cbs[0](_Msg(data))  # queue group -> one subscriber handles it
        return _Msg(holder["reply"])

    async def drain(self):
        pass


def _build_cell_aligned(tmp_path, nlist=32, n_servers=4, n=4000):
    """A cell-aligned sharded IVF index (one shard/server per cell-range) + its per-cell
    server placement, reused by the router and NATS tests."""
    corpus = _corpus(n)
    cpp = nlist // n_servers
    orig = ShardedIndex.create(
        corpus, str(tmp_path / "orig"), shard_size=len(corpus), output_dim=32, bits=4
    )
    orig.build_ivf(nlist=nlist)
    odir = str(tmp_path / "orig")
    off = np.load(os.path.join(odir, "shard_00000.ivf.off.npy"))
    memb = np.load(os.path.join(odir, "shard_00000.ivf.memb.npy"))
    row_cell = np.empty(len(corpus), dtype=np.int64)
    for c in range(nlist):
        row_cell[memb[off[c] : off[c + 1]]] = c
    out = str(tmp_path / "aligned")
    basis = os.path.join(odir, "shard_00000.tqe")
    metas = []
    for s in range(n_servers):
        rows = np.where((row_cell >= s * cpp) & (row_cell < (s + 1) * cpp))[0]
        metas.append(
            ShardedIndex.write_shard(
                out, corpus[rows], s, ids=rows, basis_from=basis, bits=4
            )
        )
    aligned = ShardedIndex.finalize_manifest(out, metas)
    centroids = np.load(os.path.join(odir, "coarse_centroids.npy"))
    aligned.build_ivf(centroids=centroids)
    subs = partition_manifest(os.path.join(out, "manifest.json"), n_servers)
    placement = build_cell_placement(subs, subs)
    router = Router(
        out, placement, pipeline_manifest=os.path.join(out, "manifest.json")
    )
    return corpus, out, subs, placement, router, aligned


def test_nats_transport_routed_equals_single_node(tmp_path):
    # The full production path (ShardServer <-> NATS request/reply <-> routed
    # coordinator) over a fake bus: same top-k as single-node search.
    corpus, out, subs, placement, router, aligned = _build_cell_aligned(tmp_path)
    ref, _ = aligned.search(corpus[:60], k=10, nprobe=4)

    # server_id == the sub-manifest path (so placement keys == transport endpoints).
    bus = _FakeBus()

    async def _connect(_url):
        return bus

    transport = nats_transport("nats://fake", connect=_connect)
    try:
        # Register each shard-server on the same bus/loop the transport uses.
        for path in subs:
            srv = NatsShardServer(ShardServer(path), path)
            asyncio.run_coroutine_threadsafe(
                srv.subscribe(bus), transport._loop
            ).result()
        q = corpus[:60]
        ids, _ = scatter_gather_routed(q, 10, router, transport, nprobe=4)
        assert _recall(ids, ref, 10) > 0.999  # over-the-wire == single-node, exactly
    finally:
        transport.close()


def test_partition_manifest_covers_all_shards(tmp_path):
    corpus = _corpus(1200)
    ShardedIndex.create(
        corpus, str(tmp_path / "s"), shard_size=100, output_dim=32, bits=4
    )  # 12 shards
    subs = partition_manifest(str(tmp_path / "s" / "manifest.json"), 5)
    total = sum(ShardServer(p).index.n_shards for p in subs)
    assert total == 12  # disjoint round-robin partition covers every shard exactly once
