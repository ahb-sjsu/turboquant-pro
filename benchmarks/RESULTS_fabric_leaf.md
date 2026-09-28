# Does `tqp fabric` see what crosses the NATS leaf link?

`tqp fabric` reports what crosses a NATS link: messages, bytes, rates and round-trip time.
This experiment tests that claim. It sends traffic whose every message is counted at the
source, records the fabric while the traffic runs, and compares the two.

Code: `benchmarks/fabric/`. `leaf_echo.py` holds the client and responder,
`analyze.py` does the comparison, `rehearse_atlas.sh` runs the rehearsal,
`submit_leaf_echo.py` and `run_leaf.sh` run the NRP experiment.

## Design

The **responder** runs on Atlas next to the hub. It echoes every request, counts what
arrives on the sink, and stops on its own when the client reports.

The **client** runs three bounded phases, then exits:

1. `rtt`: 500 request/replies of 64 bytes each.
2. `sizes`: 20 request/replies at each of 1 KiB, 16 KiB and 256 KiB.
3. `burst`: 20,000 fire-and-forget publishes of 1 KiB, then one flush.

No phase waits on a timer. Each blocks only on the network. The client counts every
message and payload byte it sends and receives, and publishes the totals as its result.
Payloads are random bytes, so compression cannot make the byte counts look better.

**The observer** is `tqp fabric --record --interval 2`, polling the hub's monitoring port.

**The paths:**
- **Rehearsal:** the client and responder both run on Atlas, and the observed path is the
  responder's own connection on the hub. That connection carries nothing else, so the
  counts must match exactly.
- **NRP run:** the client runs in an NRP pod connected to the in-cluster leaf node
  (`nats://atlas-nats:4222`). All of its traffic therefore crosses the leaf link, and the
  observed path is that link. Other traffic may share the link, so the counts must be at
  least the injected traffic, and any excess is reported as unrelated.

## Predictions, fixed before the NRP run

| check | prediction |
|---|---|
| messages toward the responder | = client messages sent + 1 (the result message), exact on the rehearsal and at least that on the leaf |
| messages back | = client replies received, exact on the rehearsal and at least that on the leaf |
| bytes | the server counts payload bytes: ratio 1.00 on the rehearsal. On the leaf, reported either way: the link is s2-compressed, and whether the counters count before or after compression is unknown |
| round-trip time | the server's figure (hub to the peer only) is at most the client's median (client to responder and back) |
| burst rate | the peak rate derived over one poll is at most the client's own burst rate, and reaches about burst_n / interval when the burst fits in one interval |
| delivery | responder echo count = client requests, and sink count = burst_n (core NATS delivers at most once, so a loss would show here) |

## Rehearsal on Atlas, 2026-09-27: all checks pass

Source: `benchmarks/fabric/results/rehearsal_atlas_20260927.json`.

| check | observed | expected |
|---|---|---|
| messages toward the responder | 20,561 | 20,561 (exact) |
| messages back | 560 | 560 (exact) |
| bytes toward / back | ratio 1.00 / 1.00 | 1.00 |
| round-trip time | server 0.38 to 0.59 ms, client p50 0.69 ms | server at most client |
| burst | 20,000 messages in 0.085 s (235k msgs/s at the client); the observer's peak over its 1 s poll was 20,544 msgs/s | about 20,000 per 1 s interval |
| delivery | echo 560 of 560, sink 20,000 of 20,000 | lossless |

The client's footprint on Atlas was 44 MiB peak RSS and 0.71 CPU-seconds over 1.0 s of wall
time (`/usr/bin/time -v`). The NRP pod is sized from this measurement: the exempt class,
1 CPU and 512 MiB, covering 1.25 x (44 MiB + 150 MiB for pip).

The rehearsal found one thing: **`/connz` counts from the server's side.** A connection's
`in_msgs` is what the server received from that client, not what the client received. The
analysis therefore orients every path along the experiment, from the client toward the
responder. In `tqp fabric` the leaf counters read the same way: `in` is what the hub
received over the link.

## NRP run over the leaf, 2026-09-27: all checks pass

Source: `benchmarks/fabric/results/leaf_nrp_20260927.json`.

**The run.** Job `tqp-fabric-leaf-1790472048` was submitted through nats-bursting.
- `burst.status` returned `submitted`, and the Job's creationTimestamp was fresh.
- The pod ran on NRP node `igrok-la.cenic.net`, completed in 1 min 51 s, and was then deleted.
- The client's own footprint in the pod was 31 MiB peak RSS and 0.45 CPU-seconds.
- The recorder polled every 2 s and took 62 snapshots of the leaf link during the window.

| check | observed | expected |
|---|---|---|
| messages over the leaf toward Atlas | 20,561 | at least 20,561. Exactly 20,561 arrived: nothing else crossed the link in the window |
| messages back over the leaf | 560 | at least 560. Exactly 560 |
| bytes toward / back | ratio 1.00 / 1.00 | reported. The link is s2-compressed, so these counters are payload bytes before compression, not wire bytes |
| round-trip time | server 87.2 ms (one value across the whole run); client p50 136 ms, p90 254 ms, p99 499 ms, min 117 ms | server at most client |
| burst | 20,000 × 1 KiB published in 0.375 s at the client (53k msgs/s, buffered). All 20,001 messages landed in a single 2 s poll: peak 10,000 msgs/s | about (burst_n + 1) / interval = 10,000 |
| delivery | echo 560 of 560, sink 20,000 of 20,000 | lossless |
| request/reply by size (p50, two-way goodput) | 1 KiB: 145 ms, 0.014 MB/s. 16 KiB: 204 ms, 0.14 MB/s. 256 KiB: 534 ms, 0.79 MB/s | reported |

**What the instrument saw during the run.**
- The request/reply phases showed as a steady 3.5 to 8 msgs/s each way, with bytes/s rising through the size sweep.
- The burst showed as one 2 s interval at 10,000 msgs/s and 10.2 MB/s.
- The per-connection history shows the method working: counts are exact on a shared link that happened to be quiet.

**Findings**
1. **`tqp fabric`'s leaf counters are exact.** Message counts matched the client's own tally to the message, in both directions, over the real NRP link.
2. **Byte counters mean payload, not wire bytes.** The ratio was exactly 1.00 on a compressed link, so the server counts bytes before s2 compression. The instrument now says so on the leaf line ("bytes are payload, before compression") and in `docs/CLI.md`.
3. **The server's leaf RTT is a coarse figure.**
   - It read 87.228367 ms unchanged for the whole run.
   - Earlier today it read 78.441859 ms unchanged for over 25 minutes.
   - It does change over time, but rarely, so it is at best an occasional sample of the hub-to-leaf hop. The instrument's "unchanged for" flag is the right treatment.
4. **The path from an NRP pod to Atlas is longer than the leaf RTT suggests.**
   - The client's fastest round trip was 117 ms, 30 ms more than the server's leaf RTT.
   - The leaf pod runs on `55m-ps.sox.net` and the client pod ran on `igrok-la.cenic.net`, so the in-cluster hop from client to leaf node may itself cross the country.
   - That explanation is consistent with the numbers but not measured. The hop was not timed separately.
5. **Payload throughput is bounded by the WAN.** A 256 KiB request/reply took about 0.53 s, about 0.8 MB/s two-way. Fire-and-forget bursts are absorbed by client buffering and delivered within seconds.

**Policy record.**
- One exempt-class pod: 1 CPU, 512 Mi (sized from the rehearsal's measured 44 MiB peak), 1 Gi ephemeral storage, no GPU.
- The client code has no sleep, and the pod terminated by itself.
- It was submitted via nats-bursting, and the completed Job was deleted afterwards.
- The leaf Deployment was not touched.

## Next

The same method could test storage paths: S3, LINSTOR and CephFS on NRP. That means known
reads and writes counted at the source, compared with what the storage system and the
console report. It would need a storage source for the console and is not built yet.
