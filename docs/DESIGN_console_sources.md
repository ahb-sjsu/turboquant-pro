# Design: new console sources (host, BitTorrent DHT) and a layout that shows only what is attached

**Status: DESIGN, for the owner's review (2026-09-28). No code, nothing installed.**

## Why this belongs next to TurboQuant Pro at all

The BitTorrent DHT is Kademlia: a routed nearest-neighbour search over 160-bit node ids,
where distance is XOR. Every lookup walks hop by hop toward a target, each hop returning
nodes closer to it, the same shape as the routed search the console already shows (probe
cells, converge or not). A DHT source lets the console watch a routed search that runs on
the open internet, at a scale nobody controls, with the same rules as every other panel:
each number is measured, sampled, derived or estimated, and a panel with no data says why.

It is a demonstration of the console's source model, not a compression feature, so it
lives as a plugin (`plugins/tqp-dht`), not in the core.

## Architecture: a daemon that owns the torrent session, a console that only reads

Copied from the NATS source (`console/fabric.py`), which reads a server's monitoring port
and never touches the fabric:

- **`tqp-dht` daemon** (plugin): one libtorrent session that seeds a fixed list of official
  open-source torrents. It serves its own state as JSON on `127.0.0.1` only, read-only, like
  a NATS monitoring port. It is the only process that talks to peers.
- **`tqp console --dht http://127.0.0.1:PORT`**: polls that endpoint. It opens no torrent
  session and sends nothing to the DHT, so watching cannot change what it watches.

This keeps libtorrent (a C++ extension, from apt on Ubuntu) out of the console's install.

## What libtorrent exposes (checked against the RC_2_0 source and Python bindings)

| Data | libtorrent call | In Python | Kind |
|---|---|---|---|
| Routing table: nodes and replacements per bucket | `post_dht_stats()` -> `dht_stats_alert.routing_table` | yes (`num_nodes`, `num_replacements`) | measured |
| Active lookups: type, outstanding, responses, timeouts, nodes left | `dht_stats_alert.active_requests` | yes, **without** `target` | measured |
| DHT counters: nodes, messages and bytes in/out, per query type in/out, invalid queries | `post_session_stats()`, the `dht.*` metrics | yes, as a name -> value dict | measured; rates derived |
| Every DHT packet, bencoded | `dht_pkt_alert.pkt_buf` (`alert_category::dht_log`) | yes, **without** direction or peer address | measured bytes |
| Swarms: peers, seeds, rates, availability | `torrent_handle.status()` | yes | measured |

**The hop trace is reconstructed, not read.** No structured API gives the distance at each
hop. The packets do. Our outgoing `find_node` / `get_peers` queries carry our node id (the
id on most outgoing queries; the bindings do not expose it directly), the target and a
transaction id. Each reply carries the same transaction id and a list of nodes. Pairing
them gives, per lookup and per round, the shared-prefix length between the target and the
closest node returned: 0 bits far away, rising toward 160 as the lookup converges.
Labelled **derived** (from measured packets). Packet alerts are high volume, so the daemon
samples lookups rather than keeping every one.

## Panels, from shapes the Go client already draws

| Panel | Shape (existing) | Shows |
|---|---|---|
| DHT node | metric rows, as the NATS panel | nodes known, node cache, messages in/out per s, bytes in/out, queries by type, invalid queries |
| Routing table | labelled bars, as the pipeline panel | one bar per bucket, nodes / capacity, replacements |
| Lookups | scrolling table, as the query stream | per lookup: type, rounds, responses, timeouts, final shared-prefix bits |
| Convergence | the scope, one trace per recent lookup | shared-prefix bits against round |
| Swarms | metric rows | per torrent: peers, seeds, up/down rate, availability, ratio |

One console change is needed and is worth making anyway: **the grid lays out only the
panels whose source is attached.** Today a session started with `--nats` alone draws eight
empty index panels around the one with data (the owner hit exactly this). With the change,
`--dht` alone draws the DHT panels, `--demo --dht` draws both.

## A second source, cheaper and first: the host

The same layout change makes room for `tqp console --host`: the machine the console runs
on, read from `/proc` and `/sys` with the standard library, no daemon and nothing sent
anywhere. Today panel 1 shows only the console's own process (CPU and RSS, when psutil is
installed); nothing shows the machine.

| Panel | Shape | Read from | Shows |
|---|---|---|---|
| CPU | labelled bars + scope trace | `/proc/stat` | busy % per core and per package, iowait, steal |
| Thermal | metric rows | `/sys/class/hwmon/*` (coretemp, the GPUs' sensors) | package and core temperatures against their own critical limits |
| Memory | metric rows | `/proc/meminfo`, `/proc/vmstat` | used, cache, available, swap, major faults per s |
| Disk I/O | scrolling table | `/proc/diskstats` | per device: read/write MB/s, IOPS, mean wait, utilisation |
| Network | metric rows | `/proc/net/dev` | per interface: bytes and packets in/out per s, errors, drops |
| GPU | metric rows | NVML when present, else "unavailable: no NVML" | utilisation, memory, temperature, power |

Kernel counters are **measured**; every per-second rate is **derived** from two polls, and is
null on the first poll or after a counter resets. Temperatures are **sampled**. On Atlas this
puts the thermal picture (packages near their limits under load) next to the load that
causes it.

## Operating the DHT daemon on Atlas

- **What it seeds:** a short, fixed list of torrents the projects publish themselves (for
  example Ubuntu, Debian and Arch images), each checked against the project's published
  SHA-256 after download. Storage under `/archive`.
- **Caps, set in the session:** an upload rate limit (proposed 2 MB/s), a connection limit,
  UPnP and NAT-PMP **off** (no router port mapping), no port forward. The DHT and seeding
  work outbound-only; inbound peers and a fuller routing table would need a forwarded port,
  a later decision.
- **Process:** a named screen session (or a user systemd unit), niced. libtorrent's CPU use
  is small; no GPU.
- **Exposure:** Atlas's public address becomes visible to the swarms and the DHT, as for any
  BitTorrent client.

## Build order

1. The attached-sources layout in the grid and the Go client (fixes the `--nats`-only
   screen on its own).
2. The host source (`console/host.py`): no install, nothing exposed, testable on any Linux
   machine against `/proc` fixtures.
3. The DHT daemon with the JSON endpoint and the caps; a test with a local libtorrent session.
4. `console/dht.py` (the reader, like `fabric.py`) and the view model rows.
5. The hop-trace reconstruction, with a test on recorded packets.

## Decisions for the owner

1. Install `python3-libtorrent` on Atlas (apt, 2.0.10; needs sudo).
2. Which torrents, and how much of `/archive` (a few images: roughly 2 to 10 GB).
3. The upload cap (proposed 2 MB/s).
4. Plugin inside turboquant-pro (`plugins/tqp-dht`) or a separate repository.
5. The order above: layout, then host, then the DHT (steps 1 and 2 need no decision on
   1 to 3).
