# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Pod entrypoint: serve one shard-range over NATS (a warm shard-server pool worker).

Run one of these per ``server_id`` (one shard-range) in a nats-bursting persistent
pool::

    python -m turboquant_pro.nats_worker \\
        --manifest /idx/server_003.manifest.json --server-id 3 \\
        --nats-url nats://atlas-nats:4222

It opens the shard-range memory-mapped once and blocks on its NATS subscription
(``{prefix}.{server_id}``), answering search requests with partial top-k bytes — no
per-request cold start, no sleep. Every flag also reads from an env var (the pool
Deployment sets these), so the same image serves any shard-range by configuration.
See :func:`turboquant_pro.nats_transport.shard_pool_manifest` for the Deployment.
"""

from __future__ import annotations

import argparse
import os

from .distributed import ShardServer
from .nats_transport import DEFAULT_PREFIX, NatsShardServer


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Serve one shard-range over NATS.")
    p.add_argument(
        "--manifest", default=_env("TQP_MANIFEST"), help="shard-range manifest.json"
    )
    p.add_argument("--server-id", default=_env("TQP_SERVER_ID"), help="this range's id")
    p.add_argument("--nats-url", default=_env("NATS_URL", "nats://atlas-nats:4222"))
    p.add_argument(
        "--subject-prefix", default=_env("TQP_SUBJECT_PREFIX", DEFAULT_PREFIX)
    )
    p.add_argument(
        "--max-open-shards", type=int, default=int(_env("TQP_MAX_OPEN_SHARDS", "128"))
    )
    p.add_argument("--max-workers", type=int, default=int(_env("TQP_MAX_WORKERS", "4")))
    a = p.parse_args(argv)
    if not a.manifest or a.server_id in (None, ""):
        p.error(
            "--manifest and --server-id are required (or TQP_MANIFEST/TQP_SERVER_ID)"
        )

    server = ShardServer(a.manifest, max_open_shards=a.max_open_shards)
    worker = NatsShardServer(
        server,
        a.server_id,
        subject_prefix=a.subject_prefix,
        max_workers=a.max_workers,
    )
    print(
        f"tqp shard-server id={a.server_id} subject={worker.subject} "
        f"shards={server.index.n_shards} -> {a.nats_url}",
        flush=True,
    )
    worker.run(a.nats_url)


if __name__ == "__main__":
    main()
