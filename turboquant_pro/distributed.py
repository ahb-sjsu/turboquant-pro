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
