# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Corpus-parallel ingest: every build worker fetches only its own rows.

The 10B run proved zero-movement ingest for a *seeded* corpus — each worker
regenerated its own range, so no corpus bytes crossed the network. This module
extends the same principle to **real** corpora: a worker pulls exactly the byte
range (or file subset) backing the shard it is about to index, indexes it, and
discards the raw vectors. Consequences:

- no central staging volume (a 1B SIFT corpus is 128 GB; a 1T corpus is 128 TB
  and could never be staged at all),
- ingest bandwidth scales with worker count rather than a single stream,
- only the ~24 B/row index persists, and
- a corpus can be *defined by its source + row range*, which is what makes a
  procedural reference corpus rebuildable rather than distributable.

Sources
-------
``bigann``      HTTP range GET into ``base.1B.u8bin`` (fixed 128-byte records
                after an 8-byte header; the CDN serves ``accept-ranges: bytes``).
``wiki1024``    Hugging Face parquet shards, assigned to workers by file — the
                natural unit, and deterministic. Reads ``HF_TOKEN`` (env) or
                ``~/.secrets/hf_token`` so ingest is authenticated rather than
                subject to anonymous rate limits.
``synthetic``   The seeded generator (``fleet_common.gen_block``).

Every fetch is retried with backoff and length-checked, because a partial read
that silently succeeds would corrupt an index shard.
"""

from __future__ import annotations

import os
import time
import urllib.request

import numpy as np

BIGANN_URL = (
    "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
)
BIGANN_DIM = 128
BIGANN_HEADER = 8
WIKI_REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"


def hf_token() -> str | None:
    """Token from the environment, else the on-disk secret. Never logged."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    path = os.path.expanduser("~/.secrets/hf_token")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def _get_range(url: str, start: int, length: int, retries: int = 8) -> bytes:
    """One ranged GET, retried; returns exactly ``length`` bytes or raises."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{start + length - 1}"}
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                buf = r.read()
            if len(buf) == length:
                return buf
            last = f"short read {len(buf)}/{length}"
        except Exception as e:  # network flakiness is expected at this scale
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"range fetch failed after {retries} tries: {last}")


def fetch_bigann(start_row: int, n_rows: int, block: int = 2_000_000) -> np.ndarray:
    """Rows ``[start_row, start_row+n_rows)`` of BIGANN as ``(n, 128) uint8``.

    Split into blocks so a transient failure retries a chunk, not the shard.
    """
    out = np.empty((n_rows, BIGANN_DIM), dtype=np.uint8)
    done = 0
    while done < n_rows:
        take = min(block, n_rows - done)
        off = BIGANN_HEADER + (start_row + done) * BIGANN_DIM
        buf = _get_range(BIGANN_URL, off, take * BIGANN_DIM)
        out[done : done + take] = np.frombuffer(buf, dtype=np.uint8).reshape(
            take, BIGANN_DIM
        )
        done += take
    return out


def wiki_files(config: str = "en") -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(
        f
        for f in list_repo_files(WIKI_REPO, repo_type="dataset", token=hf_token())
        if f.startswith(f"{config}/") and f.endswith(".parquet")
    )


def fetch_wiki_files(files: list[str], keep_local: bool = False) -> np.ndarray:
    """Embeddings from a list of parquet shards — the worker's assigned files.

    Files are removed after decode unless ``keep_local``; at scale the point is
    that raw vectors never accumulate.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    blocks = []
    for f in files:
        local = hf_hub_download(WIKI_REPO, f, repo_type="dataset", token=hf_token())
        col = pq.read_table(local, columns=["emb"])["emb"]
        blocks.append(np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float32))
        if not keep_local:
            os.unlink(local)
    return np.concatenate(blocks)


def fetch(source: str, start_row: int, n_rows: int) -> np.ndarray:
    """Uniform entry point: ``fetch("bigann", 0, 5_000_000)``.

    ``synthetic`` maps a row range onto the seeded generator's global shards.
    """
    if source == "bigann":
        return fetch_bigann(start_row, n_rows)
    if source == "synthetic":
        from fleet_common import SHARD_ROWS, gen_block

        assert start_row % SHARD_ROWS == 0 and n_rows % SHARD_ROWS == 0
        g0 = start_row // SHARD_ROWS
        return np.concatenate([gen_block(g0 + i) for i in range(n_rows // SHARD_ROWS)])
    raise ValueError(f"unknown source {source!r} (wiki1024 shards by file)")
