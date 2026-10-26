"""Exercise a NATS path with known traffic, so an observer can be checked against it.

The question this answers: when ``tqp fabric`` reports what crossed a NATS link
(messages, bytes, rates, round-trip time), is it right? The only way to know is to
send traffic whose every message is counted at the source, then compare. Two roles:

``responder`` (runs next to the hub, on Atlas)
    Replies to every request on ``<prefix>.echo`` with its payload, counts what
    arrives on ``<prefix>.sink``, and records the client's result from
    ``<prefix>.result``. It stops by itself when that result arrives or after
    ``--max-seconds``, whichever is first, and writes everything it counted.

``client`` (runs where the path starts: on Atlas for the rehearsal, in an NRP pod
through the leaf for the real run)
    Three bounded phases, then it exits. No phase waits on a timer: each blocks
    only on the network (a request's reply, or the flush of what it published).

    1. ``rtt``   N sequential request/replies of 64 bytes: the round-trip time
                 distribution, measured at the client.
    2. ``sizes`` M request/replies at each payload size: latency and goodput
                 against size.
    3. ``burst`` K fire-and-forget publishes of B bytes to the sink, then one
                 flush: the rate the path absorbs.

    Every message and payload byte it sends and receives is counted, and the
    totals go to stdout (a JSON line) and to ``<prefix>.result``.

Nothing here reads message content back except the echo's length check. Payloads
are random bytes (compression cannot flatter them).

    python benchmarks/fabric/leaf_echo.py responder --url nats://localhost:4222 \\
        --prefix tqp.fabric.exp.r1 --out responder.json --max-seconds 900
    python benchmarks/fabric/leaf_echo.py client --url nats://atlas-nats:4222 \\
        --prefix tqp.fabric.exp.r1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import socket
import sys
import time

DEFAULTS = {
    "rtt_n": 500,
    "sizes": [1024, 16384, 262144],
    "sizes_n": 20,
    "burst_n": 20000,
    "burst_bytes": 1024,
}
RTT_BYTES = 64


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))
    return s[i]


def _usage() -> dict:
    """This process's CPU seconds and peak RSS, for sizing the pod from measurement."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_kib = ru.ru_maxrss if sys.platform != "darwin" else ru.ru_maxrss / 1024
    return {"cpu_s": ru.ru_utime + ru.ru_stime, "peak_rss_mib": peak_kib / 1024}


async def client(args) -> dict:
    import nats

    t_start = time.time()
    nc = await nats.connect(args.url, name=f"tqp-fabric-client {args.prefix}")
    sent = {"msgs": 0, "payload_bytes": 0}
    recv = {"msgs": 0, "payload_bytes": 0}
    phases = {}

    async def request(n_bytes):
        body = os.urandom(n_bytes)
        t0 = time.perf_counter()
        msg = await nc.request(f"{args.prefix}.echo", body, timeout=args.timeout)
        dt = time.perf_counter() - t0
        if len(msg.data) != n_bytes:
            raise RuntimeError(f"echo returned {len(msg.data)} bytes, sent {n_bytes}")
        sent["msgs"] += 1
        sent["payload_bytes"] += n_bytes
        recv["msgs"] += 1
        recv["payload_bytes"] += n_bytes
        return dt

    # 1. round-trip time
    t0 = time.time()
    lat = [await request(RTT_BYTES) for _ in range(args.rtt_n)]
    ms = [x * 1e3 for x in lat]
    phases["rtt"] = {
        "t0": t0,
        "t1": time.time(),
        "n": len(ms),
        "bytes": RTT_BYTES,
        "ms": {q: _pct(ms, p) for q, p in (("p50", 50), ("p90", 90), ("p99", 99))},
        "ms_min": min(ms),
        "ms_max": max(ms),
    }

    # 2. payload sizes
    rows = []
    for size in args.sizes:
        t0 = time.time()
        lat = [await request(size) for _ in range(args.sizes_n)]
        wall = sum(lat)
        rows.append(
            {
                "bytes": size,
                "n": len(lat),
                "t0": t0,
                "t1": time.time(),
                "ms_p50": _pct([x * 1e3 for x in lat], 50),
                "goodput_mb_s": 2 * size * len(lat) / wall / 1e6,  # both directions
            }
        )
    phases["sizes"] = rows

    # 3. burst
    body = os.urandom(args.burst_bytes)
    t0 = time.time()
    p0 = time.perf_counter()
    for _ in range(args.burst_n):
        await nc.publish(f"{args.prefix}.sink", body)
    await nc.flush(timeout=args.timeout * 10)
    wall = time.perf_counter() - p0
    sent["msgs"] += args.burst_n
    sent["payload_bytes"] += args.burst_n * args.burst_bytes
    phases["burst"] = {
        "t0": t0,
        "t1": time.time(),
        "n": args.burst_n,
        "bytes": args.burst_bytes,
        "wall_s": wall,
        "msgs_per_s": args.burst_n / wall,
        "mb_per_s": args.burst_n * args.burst_bytes / wall / 1e6,
    }

    result = {
        "role": "client",
        "prefix": args.prefix,
        "url": args.url,
        "host": socket.gethostname(),
        "t_start": t_start,
        "t_end": time.time(),
        "sent": sent,  # the result message itself is not counted here
        "received": recv,
        "phases": phases,
        "usage": _usage(),
    }
    await nc.publish(f"{args.prefix}.result", json.dumps(result).encode())
    await nc.flush(timeout=args.timeout)
    await nc.drain()
    return result


async def responder(args) -> dict:
    import nats

    nc = await nats.connect(args.url, name=f"tqp-fabric-responder {args.prefix}")
    counts = {"echo_msgs": 0, "echo_bytes": 0, "sink_msgs": 0, "sink_bytes": 0}
    done = asyncio.Event()
    result: dict = {}

    async def on_echo(msg):
        counts["echo_msgs"] += 1
        counts["echo_bytes"] += len(msg.data)
        await msg.respond(msg.data)

    async def on_sink(msg):
        counts["sink_msgs"] += 1
        counts["sink_bytes"] += len(msg.data)

    async def on_result(msg):
        result.update(json.loads(msg.data))
        done.set()

    await nc.subscribe(f"{args.prefix}.echo", cb=on_echo)
    await nc.subscribe(f"{args.prefix}.sink", cb=on_sink)
    await nc.subscribe(f"{args.prefix}.result", cb=on_result)
    await nc.flush()
    t_start = time.time()
    print(json.dumps({"role": "responder", "ready": t_start}), flush=True)
    try:
        await asyncio.wait_for(done.wait(), timeout=args.max_seconds)
        why = "client result received"
        # Stay connected a little so an observer polling the server sees this
        # connection's final counters before it closes (on Atlas; never in a pod).
        await asyncio.sleep(args.linger)
    except asyncio.TimeoutError:
        why = f"no client result within {args.max_seconds} s"
    # The sink is fire-and-forget: its last messages may still be in flight when
    # the result (sent after a flush) arrives. Let delivery settle before counting.
    await nc.flush()
    await nc.drain()
    out = {
        "role": "responder",
        "prefix": args.prefix,
        "url": args.url,
        "host": socket.gethostname(),
        "t_start": t_start,
        "t_end": time.time(),
        "stopped": why,
        "counts": counts,
        "client": result or None,
        "usage": _usage(),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("role", choices=["client", "responder"])
    p.add_argument("--url", default=os.environ.get("NATS_URL", "nats://localhost:4222"))
    p.add_argument("--prefix", default=os.environ.get("PREFIX", "tqp.fabric.exp.local"))
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--rtt-n", type=int, default=DEFAULTS["rtt_n"])
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULTS["sizes"])
    p.add_argument("--sizes-n", type=int, default=DEFAULTS["sizes_n"])
    p.add_argument("--burst-n", type=int, default=DEFAULTS["burst_n"])
    p.add_argument("--burst-bytes", type=int, default=DEFAULTS["burst_bytes"])
    p.add_argument("--max-seconds", type=float, default=900.0)
    p.add_argument("--out", help="responder: write its counts here")
    p.add_argument(
        "--linger",
        type=float,
        default=5.0,
        help="responder: seconds to stay connected after the result (observers)",
    )
    args = p.parse_args(argv)
    run = client if args.role == "client" else responder
    out = asyncio.run(run(args))
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
