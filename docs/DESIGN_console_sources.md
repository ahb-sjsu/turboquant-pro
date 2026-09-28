# Design: new console sources (the machine, BitTorrent DHT) and a layout that shows only what is attached

**Status (2026-09-28):** the layout and the machine source are **built** (below). The DHT
source is a **design** awaiting the owner's decisions; nothing for it is installed.

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

## Built first: the machine source and the pages (as implemented)

Steps 1 and 2 below are built (`tqp console --machine`; `--host` was already the web
view's bind address). What was built, and what measuring it on Atlas changed:

- **Pages.** The engine's `hello` names the pages the sources give (the index grid when an
  index is attached, the machine page with `--machine`, NATS on its own page when there is
  no index grid to hold it as panel 9). `<` and `>` move between them; each page keeps its
  focus; Tab and the digits move only among the current page's panels. The index grid
  draws panel 9 only with `--nats`. The client and engine check a protocol number (2) and
  refuse to pair across releases.
- **`console/machine.py`**, from `/proc` and `/sys` with the standard library, and NVML
  through ctypes when it is installed. Counters **measured**; rates and busy shares
  **derived** from two polls (null on the first and after any counter went backwards);
  temperatures and GPU readings **sampled**.
- **`console/machine_view.py`**: six panels, one shape (summary, calibrated rows with a
  sparkline, per-CPU strips, a table), drawn by one renderer in the Go client
  (`pages.go`). Colour follows each sensor's own limits and nothing else.

| Panel | Read from | Shows |
|---|---|---|
| 1 CPU | `/proc/stat`, `/proc/loadavg`, CPU topology | busy and iowait for the machine and per package (summed time, not a mean of fractions), one cell per logical CPU |
| 2 thermal | `/sys/class/hwmon`, NVML | each package, its hottest core, other sensors, GPUs, against their own limits; the sensor closest to its limit |
| 3 memory | `/proc/meminfo`, `/proc/vmstat` | used (total minus MemAvailable), cache, dirty, swap, major faults and swap traffic per s |
| 4 disks | `/proc/diskstats`, `/sys/block` | per device: bytes and IOPS each way, mean wait per I/O, share of time busy, queue |
| 5 network | `/proc/net/dev`, `/sys/class/net` | physical NICs and tunnels one row each, everything virtual summed; link share from the NIC's speed |
| 6 GPU | NVML | utilisation, memory, temperature against the slowdown limit, power against the enforced limit |

**Cross-checked on Atlas against independent tools over the same 5 s window:** CPU busy
against vmstat (21.1 % against 21 %), memory available against free (within 22 MiB, read a
moment apart), every active disk against iostat (throughput within about 1 %, utilisation
equal, wait equal to iostat's for write-only devices), GPU temperature (within 1 °C),
utilisation and memory against nvidia-smi (equal). What the cross-check corrected:

1. **GPU memory** read about 280 MiB high on each GPU: NVML's v1 call counts memory the
   driver reserves as used. The source uses the v2 call, as nvidia-smi does.
2. **VM taps** (`vnet*`) were called tunnels: a TAP has `tun_flags` too. A tunnel is now a
   TUN device (`IFF_TUN`) or an interface with no link layer (type 65534, as WireGuard).
3. **md arrays** were assumed to keep no busy time; on this kernel md1 and md5 do, and
   iostat reports their utilisation. The rule is now the device's own `queue/iostats` flag:
   a device with 0 (md127 here) keeps no statistics, and all its rates are unknown ("-"),
   never 0.
4. **GPU power** is an instantaneous reading: back-to-back reads of NVML and nvidia-smi
   agree, and both catch bursts (180 to 207 W between readings near 58 W on a busy GPU).
   The panel says so.

A limit a sensor reports outside 0 to 150 degrees C (Atlas's NVMe reports 65261.85 on two
sensors) is not shown as a limit.

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
   screen on its own). **Built.**
2. The machine source (`console/machine.py`, `--machine`): no install, nothing exposed,
   tested against `/proc` and `/sys` fixture trees. **Built.**
3. The DHT daemon with the JSON endpoint and the caps; a test with a local libtorrent session.
4. `console/dht.py` (the reader, like `fabric.py`) and the view model rows.
5. The hop-trace reconstruction, with a test on recorded packets.

## Decisions for the owner

1. Install `python3-libtorrent` on Atlas (apt, 2.0.10; needs sudo).
2. Which torrents, and how much of `/archive` (a few images: roughly 2 to 10 GB).
3. The upload cap (proposed 2 MB/s).
4. Plugin inside turboquant-pro (`plugins/tqp-dht`) or a separate repository.
5. Steps 1 and 2 are done; steps 3 to 5 wait on decisions 1 to 4.
