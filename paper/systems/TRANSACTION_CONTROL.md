# Transaction control: congestion concepts applied to the storage path

*Design note for the systems paper (§5.4/§6). Borrows the formulation from the
nats-bursting INFOCOM draft — "politeness as a control problem", AIMD with a
probe-driven estimator and an explicit detection delay — and applies it one
layer down, to storage transactions rather than cluster admission.*

---

## 1. The isomorphism

| network (INFOCOM draft) | storage path (this paper) |
|---|---|
| RTT | λ₀ — uncongested per-transaction latency (3.5 s/shard open on network block; ~0.5 s/location on the shared FS) |
| bottleneck bandwidth | B — substrate transaction rate (ops/s), **per inode** on CephFS |
| congestion window | w — in-flight transactions (`workers`, `max_parallel`, fetch fan-out) |
| loss / ECN | latency inflation above λ₀ — storage queues rather than drops, so the signal is delay-based (Vegas/BBR-like), not loss-based |
| detection delay D | time for queueing to become visible in measured latency |
| policy violation | degrading a *shared* substrate other tenants are using |

Little's Law gives throughput `X = w / R`. With `R = λ₀` while uncongested and
`R = w / B` once saturated, the operating point is Kleinrock's optimum:

> **w\* = λ₀ · B**

Below w\* the substrate is idle waiting on us; above it, added window is pure
queueing delay and buys nothing.

## 2. What this explains that we could not before

At 250 shards/server we raised the per-request fan-out from `workers=1` to
`workers=16` and measured **no improvement** — servers stayed at ~2.5% CPU and
a single 1-query probe still exceeded 900 s. Read as congestion control, this
is not a failed fix: **we were already at or beyond w\***, so additional window
added queueing rather than throughput. The model retrodicts the null result,
which is exactly the kind of check that distinguishes a framework from a
metaphor.

It also explains the *shape* of the earlier 1B result: single-node speedup
decaying 4.7× → 2.5× as `nprobe` grew. More probed cells = more transactions
against a fixed B.

## 3. The optimization: shard size has an interior optimum at S ≈ nprobe

Let a server hold `S` shards, and let a query probe `nprobe` cells with
cell-aligned placement. Then per query:

- **transactions** `T ≈ min(S, nprobe)` — one open per shard actually touched;
- **bytes** `≈ N · (nprobe/nlist) · bytes_per_row` — set by the probe, *not* by S;
- **latency** `≈ T·λ₀ / min(w, T) + bytes / BW`.

Two regimes:

- **S < nprobe:** transactions grow linearly with S, and each added shard adds
  λ₀/w of latency for no selectivity benefit that the probe can exploit.
- **S > nprobe:** transaction count saturates at `nprobe`, while placement
  granularity keeps improving — but every *additional* shard beyond that is
  another file, another open, and (crucially) another inode competing for the
  same per-inode-serialized substrate.

So the knee is

> **S\* ≈ nprobe** — enough shards for placement granularity, no more than the
> probe count can use.

**Our two measured points bracket it.** At `nprobe=128`: 50 shards/server is
*under* the knee (leaving placement granularity unused); 250 is *over* it
(2× the transactions the probe can exploit, which is where the 900 s probe
came from). ⬜ Direct test: sweep S ∈ {32, 64, 128, 256} at fixed `nprobe=128`
and fixed corpus, and show the latency minimum near S = 128.

**Caveat that matters.** This assumes cell-aligned placement, which we do not
yet have (round-robin placement puts every cell on every server, so
`servers_touched` was 4/4). Under round-robin, `T ≈ S` always and *fewer
shards is monotonically better* — which is what we observed. The interior
optimum is a **prediction contingent on the placement work**, and must be
labelled as such until measured.

## 4. The controller: replace four hand-tuned constants

`workers`, `max_parallel`, `max_open_shards`, and the cold-tier fetch fan-out
are all window parameters, and every one of them is currently a guess. Worse,
the right value is not a constant: we measured **>10× node-to-node variance**
in client bandwidth on this cluster, so any static window is wrong for most
nodes by construction.

**Probe-driven AIMD, delay-based:**

1. **Estimate λ₀** per (node, tier) with a low-concurrency probe — the
   INFOCOM estimator, reused verbatim in role.
2. **Additive increase** `w += α` while measured `R ≈ λ₀` (no queueing).
3. **Multiplicative decrease** `w ← βw` when `R > λ₀(1+δ)` — delay-based,
   because storage queues instead of dropping.
4. Converges toward `w* = λ₀·B` **without ever needing to know B**, which is
   the property that makes it deployable across a heterogeneous fleet.

**Per-inode allocation (a real departure from TCP).** On CephFS, B is
per-file: we measured that row-parallel fetch helps only *across* distinct
shard files, and serializes within one. So the window is not a single scalar
but an allocation `w = Σ_f w_f` with `w_f ≤ λ₀ · B_f`. Concurrency spent
inside one inode is wasted — a constraint with no TCP analogue, and the reason
naive "more threads" failed.

**Two tiers**, mirroring the INFOCOM macro/micro split:
- *macro* — coordinator-side in-flight transactions across the fleet;
- *micro* — per-server queue depth over its own shards.

## 5. Politeness, one layer down

CephFS is genuinely multi-tenant: over-driving it with concurrent random reads
degrades other NRP users, and unlike CPU there is no utilization *floor*
protecting us — only a ceiling we can breach. So the INFOCOM two-sided
objective transfers intact: **maximize our goodput subject to not degrading
the shared substrate**, with the delay signal serving as the politeness
detector. This is the same thesis as the bursting paper, applied to I/O.

## 6. Why this belongs in the paper

It converts §1.0's thesis from a diagnosis into a solution:

> transaction-bound (thesis) → w\* = λ₀·B and S\* ≈ nprobe (design rules) →
> AIMD controller (mechanism) → $/QPS at fixed recall (§6, the price)

⬜ **Experiments required:** (a) measure λ₀ and B per tier; (b) the shard-size
sweep of §3; (c) controller vs best static window across heterogeneous nodes
— the controller only needs to *match* the best hand-tuned constant while
finding it automatically, and should *beat* it on a heterogeneous fleet;
(d) show the null result of §2 is reproduced and explained.

**Honest positioning (revised after a prior-art search, 2026-07-20).**
Applied AIMD is not new theory, and the INFOCOM draft says so about its own
use. Two claims I made earlier are now **withdrawn**:

* *"The transaction-bound framing is new for ANN."* It is not. DiskANN's own
  accounting is per-transaction (384 B used of a 4 KB page); SPANN reports
  cutting I/O accesses ~10×; FlashANNS overlaps I/O with compute; and
  "Leveraging I/O Stalls for Efficient Scheduling in ANNS" (arXiv 2605.19335,
  May 2026) works the same ground. That paper must be read before any framing
  claim is drafted.
* *"The per-inode window constraint may be novel."* Shared-file lock
  contention and the metadata wall are long-established in HPC parallel I/O —
  "opening a file on thousands of processes" is the textbook example. Ours is
  the read side rather than the write side, which needs a targeted check, not
  an assumption of novelty.

What remains defensible: the **constants** (10³–10⁵× above the local-SSD
regime the ANN literature assumes), the **controlled isolation** (placement
changes which files open while holding bytes scanned exactly constant →
8.5–9.3×), the **retrodiction** of a null result we had already recorded and
misfiled, and the **bridge** — running a billion-scale vector index on a
distributed parallel filesystem, where the metadata wall rather than the data
read sets latency, appears not to have been done. That is a narrower claim
than the one this note originally made, and it is the one the data supports.
