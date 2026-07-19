# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""NATS deployment adapter for the distributed coordinator (experimental, optional).

:mod:`distributed` is transport-agnostic — its coordinator takes a
``transport(endpoint, request_bytes) -> response_bytes`` callable. This module is the
production wire for that seam: **nats-bursting**. It turns a :class:`ShardServer` into
a warm, always-on NATS subscriber (a persistent queue-group pool worker) and hands the
coordinator a ``transport`` that reaches those servers by NATS request/reply.

Design (matched to the nats-bursting fabric):

* **Request/reply, not work-queue.** A search is a *matched* RPC (a query returns *this*
  server's partial top-k). That is core-NATS request/reply — a sibling to the fabric's
  fire-and-forget ``Worker`` (which publishes to ``results.>``). Core NATS also crosses
  leaf-node boundaries that JetStream work-queues do not.
* **Subject per shard-range, queue group per subject.** Each shard-server owns a subject
  ``{prefix}.{server_id}`` so the *router* can address the specific server holding a
  cell (not round-robin across the fleet); replicas of the *same* shard-range share that
  subject as a queue group, so they load-balance and one can fail without dropping the
  range.
* **Warm pool.** The server blocks on its subscription (no sleep — NRP "no sleeping
  Jobs"), index memory-mapped once, so requests skip the container cold start.
* **No hard dependency.** ``nats`` (nats-py) is imported lazily, only when a real
  connection is opened; a ``connect`` seam allows a fake bus for deterministic tests.

Deploy the servers as a nats-bursting **persistent pool** (``pool_manifest`` /
``PoolDescriptor``, ``cpu``/``mem`` in the NRP ignored range), one server_id per
shard-range; drive them from the workstation with :func:`nats_transport` plugged into
:func:`~turboquant_pro.distributed.scatter_gather_routed`.
"""

from __future__ import annotations

import asyncio
import threading

DEFAULT_PREFIX = "tqp.shard"


async def _default_connect(nats_url: str):
    """Open a real NATS connection (lazy import so ``nats`` stays optional)."""
    import nats  # noqa: PLC0415 — optional dep, imported only when actually connecting

    return await nats.connect(
        nats_url, max_reconnect_attempts=-1, reconnect_time_wait=2
    )


def _subject(prefix: str, server_id) -> str:
    return f"{prefix}.{server_id}"


class NatsShardServer:
    """Serve one :class:`~turboquant_pro.distributed.ShardServer` over NATS.

    Subscribes to ``{subject_prefix}.{server_id}`` in a queue group (``queue`` defaults
    to the subject, so replicas of this shard-range share the load) and answers each
    request with the partial-top-k bytes ``ShardServer.handle`` returns. The search runs
    in a thread (it is CPU-bound and releases the GIL), so the event loop stays free to
    accept more requests.
    """

    def __init__(
        self,
        shard_server,
        server_id,
        *,
        subject_prefix: str = DEFAULT_PREFIX,
        queue: str | None = None,
        max_workers: int = 4,
    ):
        self._server = shard_server
        self.server_id = server_id
        self.subject = _subject(subject_prefix, server_id)
        self.queue = queue or self.subject  # replicas of this range share the queue
        self._max_workers = max_workers
        self._stop = None  # asyncio.Event, created on the serving loop

    async def subscribe(self, nc):
        """Register the request/reply handler on ``nc`` and return the subscription
        (does not block — usable both from :meth:`serve` and from a test loop)."""
        sem = asyncio.Semaphore(self._max_workers)

        async def _cb(msg):
            async with sem:  # bound concurrent in-flight searches on this pod
                reply = await asyncio.to_thread(self._server.handle, msg.data)
            await msg.respond(reply)

        return await nc.subscribe(self.subject, queue=self.queue, cb=_cb)

    async def serve(self, nats_url: str, *, connect=_default_connect):
        """Connect, subscribe, and block forever serving requests (the pool-worker
        entrypoint). Blocks on the subscription, never sleeps."""
        nc = await connect(nats_url)
        await self.subscribe(nc)
        self._stop = asyncio.Event()
        try:
            await self._stop.wait()
        finally:
            await nc.drain()

    def run(self, nats_url: str, *, connect=_default_connect) -> None:
        """Blocking pod entrypoint: ``NatsShardServer(ShardServer(m), id).run(url)``."""
        asyncio.run(self.serve(nats_url, connect=connect))


class NatsTransport:
    """A synchronous ``transport(endpoint, request_bytes) -> response_bytes`` over NATS
    request/reply, for the (synchronous) coordinator. ``endpoint`` is a ``server_id``;
    the call is a NATS request to ``{subject_prefix}.{server_id}``.

    Owns one background event loop (like the fabric's client), so the coordinator's
    thread-pool fan-out (:func:`scatter_gather` ``max_parallel``) can call it from many
    threads concurrently — each request gets its own NATS inbox, so replies never cross.
    """

    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        *,
        subject_prefix: str = DEFAULT_PREFIX,
        timeout: float = 30.0,
        connect=_default_connect,
    ):
        self._url = nats_url
        self._prefix = subject_prefix
        self._timeout = timeout
        self._connect = connect
        self._loop = asyncio.new_event_loop()
        self._nc = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._call(self._open())

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro):
        """Run ``coro`` on the loop from a foreign thread; return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _open(self):
        self._nc = await self._connect(self._url)

    async def _request(self, endpoint, request_bytes):
        msg = await self._nc.request(
            _subject(self._prefix, endpoint), request_bytes, timeout=self._timeout
        )
        return msg.data

    def __call__(self, endpoint, request_bytes: bytes) -> bytes:
        return self._call(self._request(endpoint, request_bytes))

    def close(self) -> None:
        if self._nc is not None:
            self._call(self._nc.drain())
            self._nc = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def nats_transport(
    nats_url: str = "nats://localhost:4222",
    *,
    subject_prefix: str = DEFAULT_PREFIX,
    timeout: float = 30.0,
    connect=_default_connect,
) -> NatsTransport:
    """Convenience: a started :class:`NatsTransport`. Pass the returned object as the
    ``transport`` to :func:`~turboquant_pro.distributed.scatter_gather` /
    :func:`~turboquant_pro.distributed.scatter_gather_routed`; call ``.close()`` when
    done."""
    return NatsTransport(
        nats_url, subject_prefix=subject_prefix, timeout=timeout, connect=connect
    )
