# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Distributed scatter-gather search — the cross-node layer (experimental).

`ShardedIndex` is the *per-node* engine. Beyond one box, the shards are partitioned
across many **shard-servers** (each holding a disjoint shard-range on its own local
NVMe / block volume); a **coordinator** scatters a query to the relevant servers,
gathers their partial top-k, and merges. The merge is exact — a global top-k row is
in its server's shard-range top-k — via ``ShardedIndex._merge_partials``.

This module is deliberately **transport-agnostic**: the coordinator takes a
``transport(endpoint, request_bytes) -> response_bytes`` callable, so the wire can be
anything. In production that wire is **nats-bursting**: shard-servers are a
persistent NATS queue-group pool, the coordinator uses request/reply, and turboquant-pro
compresses the payloads — but nothing here depends on it, and it is fully testable
in-process. Keep the core engine SQLite-clean; this is the optional coordinator on top.
"""

from __future__ import annotations

import io
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .sharded_index import ShardedIndex

# --------------------------------------------------------------------------- #
# Wire format — a JSON header (params) + a raw .npy query block; .npz response #
# --------------------------------------------------------------------------- #


def encode_request(
    queries: np.ndarray,
    k: int,
    *,
    nprobe: int | None = None,
    radius_scale: float = 0.5,
    bound: str = "weighted",
    workers: int = 1,
    rerank: int = 0,
) -> bytes:
    """Serialize a search request. Payloads are small (queries + params); the wire
    (e.g. nats-bursting) handles any further compression."""
    header = json.dumps(
        {
            "k": k,
            "nprobe": nprobe,
            "radius_scale": radius_scale,
            "bound": bound,
            "workers": workers,
            "rerank": rerank,
        }
    ).encode()
    qbuf = io.BytesIO()
    np.save(qbuf, np.asarray(queries, dtype=np.float32))
    return struct.pack("<I", len(header)) + header + qbuf.getvalue()


def decode_request(blob: bytes):
    hlen = struct.unpack("<I", blob[:4])[0]
    header = json.loads(blob[4 : 4 + hlen])
    queries = np.load(io.BytesIO(blob[4 + hlen :]))
    return header, queries


def encode_response(ids: np.ndarray, scores: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, ids=ids, scores=scores)
    return buf.getvalue()


def decode_response(blob: bytes):
    z = np.load(io.BytesIO(blob))
    return z["ids"], z["scores"]


def serve(index: ShardedIndex, request_bytes: bytes) -> bytes:
    """Answer one search request against a (per-node) index (shard-server handler).
    Returns this server's partial top-k over its shard-range."""
    h, queries = decode_request(request_bytes)
    ids, scores = index.search(
        queries,
        k=h["k"],
        rerank=h.get("rerank", 0),
        nprobe=h.get("nprobe"),
        radius_scale=h.get("radius_scale", 0.5),
        bound=h.get("bound", "weighted"),
        workers=h.get("workers", 1),
    )
    return encode_response(ids, scores)


class ShardServer:
    """A shard-server: holds a `ShardedIndex` over its shard-range and answers request
    bytes with partial-top-k bytes. Wrap it in a nats-bursting worker (subscribe to the
    search subject, reply with ``handle(msg)``) to put it on the wire."""

    def __init__(
        self, manifest_path: str, *, mmap: bool = True, max_open_shards: int = 128
    ):
        self.index = ShardedIndex.open(
            manifest_path, mmap=mmap, max_open_shards=max_open_shards
        )

    def handle(self, request_bytes: bytes) -> bytes:
        return serve(self.index, request_bytes)


def scatter_gather(
    queries: np.ndarray,
    k: int,
    endpoints,
    transport,
    *,
    nprobe: int | None = None,
    radius_scale: float = 0.5,
    bound: str = "weighted",
    workers: int = 1,
    rerank: int = 0,
    max_parallel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinator: scatter a request to ``endpoints`` over ``transport``, merge the
    partial top-ks into the global top-k (exact).

    ``transport(endpoint, request_bytes) -> response_bytes`` is any RPC (in-process for
    tests, NATS request/reply in production). ``max_parallel`` fans out across a thread
    pool (calls are I/O-bound, so the GIL is released)."""
    q = np.asarray(queries, dtype=np.float32)
    if q.ndim == 1:
        q = q[None]
    nq = len(q)
    req = encode_request(
        q,
        k,
        nprobe=nprobe,
        radius_scale=radius_scale,
        bound=bound,
        workers=workers,
        rerank=rerank,
    )
    endpoints = list(endpoints)

    def call(ep):
        return decode_response(transport(ep, req))

    if max_parallel and max_parallel > 1 and len(endpoints) > 1:
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(endpoints))) as ex:
            partials = list(ex.map(call, endpoints))
    else:
        partials = [call(ep) for ep in endpoints]
    return ShardedIndex._merge_partials(partials, nq, k)


def partition_manifest(manifest_path: str, n_servers: int, out_dir: str | None = None):
    """Assign the index's shards to ``n_servers`` disjoint sub-manifests (round-robin),
    so each server owns a shard-range. The global IVF coarse layer + per-shard sidecars
    stay shared, so every server probes the *same* cells (IVF-as-router). Returns the
    sub-manifest paths."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    shards = manifest["shards"]
    out_dir = out_dir or os.path.dirname(os.path.abspath(manifest_path))
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n_servers):
        assigned = shards[i::n_servers]
        sub = dict(manifest)
        sub["shards"] = assigned
        sub["n_shards"] = len(assigned)
        sub["n_rows"] = sum(s["n_rows"] for s in assigned)
        p = os.path.join(out_dir, f"server_{i:03d}.manifest.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sub, f, indent=2)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# IVF-as-router — scatter only to the servers holding a query's probed cells   #
# --------------------------------------------------------------------------- #


def _probed_cells(centroids, radius, q_rot, nprobe, radius_scale, bound):
    """The best ``nprobe`` cells per query (same weighted-A* order the servers use)."""
    qdir = q_rot / np.maximum(np.linalg.norm(q_rot, axis=1, keepdims=True), 1e-30)
    theta = np.arccos(np.clip(qdir @ centroids.T, -1.0, 1.0))
    beta = 1.0 if bound == "admissible" else float(radius_scale)
    ub = np.cos(np.maximum(0.0, theta - beta * radius[None, :]))
    return np.argsort(-ub, axis=1)[:, :nprobe]


def build_cell_placement(server_manifests, endpoints, out_dir=None):
    """Map ``cell -> [endpoints]``: which servers hold non-empty posting lists for each
    cell, read from the servers' IVF sidecars. Sparse when the index is cell-aligned
    (each server owns a cell-range) — that sparsity is what lets routing skip servers.
    """
    cell_servers: dict[int, list] = {}
    for man, ep in zip(server_manifests, endpoints):
        base_dir = out_dir or os.path.dirname(os.path.abspath(man))
        with open(man, encoding="utf-8") as f:
            shards = json.load(f)["shards"]
        here = set()
        for s in shards:
            off = np.load(
                os.path.join(base_dir, os.path.splitext(s["path"])[0] + ".ivf.off.npy")
            )
            here.update(np.nonzero(np.diff(off))[0].tolist())
        for c in here:
            cell_servers.setdefault(int(c), []).append(ep)
    return cell_servers


class Router:
    """The coordinator's routing table: the global coarse quantizer (centroids/radii),
    the shared query pipeline, and a ``cell -> servers`` placement map. Given a query it
    computes the probed cells and returns the (few) servers that hold them, so the
    scatter touches ``nprobe/nlist`` of the fleet rather than all of it."""

    def __init__(self, coarse_dir: str, cell_servers: dict, *, pipeline_manifest=None):
        self.centroids = np.load(os.path.join(coarse_dir, "coarse_centroids.npy"))
        self.radius = np.load(os.path.join(coarse_dir, "coarse_radius.npy"))
        self.cell_servers = {int(c): list(v) for c, v in cell_servers.items()}
        man = pipeline_manifest or os.path.join(coarse_dir, "manifest.json")
        self._adc = ShardedIndex.open(man)._get_shard(0)._adc  # shared basis/pipeline

    def probed(self, queries, nprobe, radius_scale=0.5, bound="weighted"):
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        q_rot, _ = self._adc._query_terms(q)
        return _probed_cells(
            self.centroids, self.radius, q_rot, nprobe, radius_scale, bound
        )

    def servers_for(self, queries, nprobe, radius_scale=0.5, bound="weighted"):
        """The endpoints to scatter this query batch to (union over the batch's cells),
        plus the per-query probed cells."""
        cells = self.probed(queries, nprobe, radius_scale, bound)
        eps, seen = [], set()
        for c in np.unique(cells).tolist():
            for ep in self.cell_servers.get(int(c), []):
                if ep not in seen:
                    seen.add(ep)
                    eps.append(ep)
        return eps, cells


def scatter_gather_routed(
    queries,
    k,
    router: Router,
    transport,
    *,
    nprobe: int,
    radius_scale: float = 0.5,
    bound: str = "weighted",
    workers: int = 1,
    max_parallel: int | None = None,
):
    """Routed scatter-gather: consult ``router`` for the servers that hold the query's
    cells and scatter only to them (vs :func:`scatter_gather`, which hits every server).
    Same exact top-k — the skipped servers have no rows in the probed cells."""
    endpoints, _ = router.servers_for(queries, nprobe, radius_scale, bound)
    return scatter_gather(
        queries,
        k,
        endpoints,
        transport,
        nprobe=nprobe,
        radius_scale=radius_scale,
        bound=bound,
        workers=workers,
        max_parallel=max_parallel,
    )
