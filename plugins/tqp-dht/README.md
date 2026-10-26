# tqp-dht

A BitTorrent DHT source for the TurboQuant Pro console.

The BitTorrent DHT is Kademlia: a routed nearest-neighbour search over 160-bit
node ids, where the distance between two ids is their XOR. Every lookup walks
hop by hop toward its target, each response returning nodes closer to it. That
is the same shape as the routed search TurboQuant Pro's indexes do, run on the
open internet, and `tqp console --dht` watches it.

```bash
tqp-dht serve --data /archive/tqp-dht            # the daemon (screen or a user unit)
tqp console --dht http://127.0.0.1:8290          # the console page, read-only
```

## What the daemon does, and does not do

- **Seeds only what the projects publish.** The images are pinned in
  `tqp_dht/daemon.py` (`TORRENTS`): the `.torrent` each project publishes,
  fetched over HTTPS from an allowed host, and the SHA-256 each project lists.
  A finished image is checked against it; a mismatch pauses that torrent.
- **Caps**: upload 2 MB/s, download 20 MB/s, 50 connections.
- **Opens nothing on the router.** libtorrent turns UPnP, NAT-PMP and local
  peer discovery on by default; the daemon turns all three off. With no port
  forward, the DHT and seeding work outbound only.
- **Serves its state on 127.0.0.1 only**: `GET /snapshot`, nothing else, no
  secrets.

## What the console shows

- **node**: nodes known, messages and bytes in and out per second, queries by
  type in and out, invalid queries.
- **routing table**: the nodes and replacements in each bucket.
- **lookups**: each lookup our node made, rebuilt from its own packets
  (`tqp_dht.krpc`): queries, responses, the best shared prefix with the target.
- **convergence**: for each recent lookup, the best shared prefix after each
  response; the walk toward the target, bit by bit.
- **swarms**: each image: state, peers and seeds, rates, ratio, and whether its
  SHA-256 matched the published one.

libtorrent reports each lookup's progress but not the distance at each step,
so the convergence is **derived** from the packets: our query and its response
are paired by transaction id, and a response that cannot be paired is counted,
not guessed at.
