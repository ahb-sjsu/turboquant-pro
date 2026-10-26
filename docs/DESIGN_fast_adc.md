# Design Doc — Fast Compressed ADC

**Status:** built (M1, M3, M3-final measured below; 1M scaling open) ·
**Goal:** a compressed search that is fast *and* compressed *and* high-recall at
the same operating point, with the speed optimization held to a stated contract.

## The contract, in one place

Three things score a query in this library. Every statement in this document,
the README, the claims and the tests refers to exactly one of them.

| name | what it is | promise |
|---|---|---|
| **exact ADC** (`exact-float`) | numpy, full float precision, over the stored codes | the **reference semantics**. Reproducible bit for bit across RAM, memory-mapped and blocked storage. `mode="exact"` |
| **SIMD ADC** (`kernel-uint8-lut`) | the compiled kernel; its per-dim table is quantized to 255 levels of one query-global scale | **approximate by construction.** Its score sits within the table's resolution of the exact one, and it only reorders neighbours whose exact scores lie inside that resolution (section 3b). The kernel's unpruned scan built without AVX2 runs a float table instead (`kernel-float-lut`): equal to the reference up to float32 summation order, still not bit for bit. `mode="fast"` |
| **SIMD ADC + exact rerank** | the SIMD scan returns `k * rerank` candidates; they are rescored exactly (against stored originals) | the two-stage path. The rerank **reconciles the boundary region** the first stage cannot resolve; it does not make the first stage exact |

`TQEIndex.search(mode=...)` chooses (`"fast"` is the default, `"exact"` the
reference; `exact=True` is the older spelling of `mode="exact"`), and every
search records which scorer ran in `last_scorer` and in its trace, including
why `"fast"` fell back to the reference when it did (no kernel built, a
memory-mapped index, a `block`, a metric or code width the kernel does not
scan). `ADCIndex`, `IVFIndex` and `ShardedIndex` take the same `mode`. Results
are comparable bit for bit only when their provenance names the same
first-stage scorer.

The design goal follows from the contract: the fast kernel does not have to be
exact. It has to be good enough at candidate generation that the exact second
stage recovers the ranking, and both halves of that are measured.

## 1. Problem & success criteria

turboquant-pro wins recall and compression (beats RaBitQ, ties OPQ at ~30×) but
its compressed search is a slow scan:

| path | qps @199k | why |
|---|---:|---|
| CPU flat-reconstruct | 162–254 | decompress to fp32, then exact scan |
| `gpu_adc_search` (current) | **2.8** | **per-query** CuPy launch — overhead dominates |
| faiss PQ ADC (target class) | ~900 | **batched** C++ SIMD ADC, linear scan |
| ScaNN (system bar) | ~3400 | AH + IVF tree + reorder |

The bottleneck is **not** the math — it's the absence of a *batched* ADC kernel.
faiss does the same linear scan at ~900 qps because its ADC is vectorized.

**Success criteria.** A batched ADC over tq-pro codes that:
1. **≥ 900 qps @ 199k** on CPU (match faiss PQ), **≥ 5000 qps** on a GV100;
2. **recall after exact rerank equal to the exact-ADC path's** within
   measurement noise, with the first-stage (SIMD, no rerank) disagreement
   bounded and measured rather than assumed to be zero;
3. drops in behind the existing `compress_batch` API (no new training).

*As first written, criterion 2 read "identical recall, it only speeds the
scan". That assumed the kernel was an exact scorer. It is not (section 3b), so
the criterion is restated as the two-stage contract above.*

## 2. Background: ADC for tq-pro codes

tq-pro encodes a vector as: PCA-Matryoshka project to `d'` dims → per-dim scalar
quantize to `b` bits (a shared codebook of `2^b` centroids `c[0..2^b-1]`) →
bit-pack, keeping the L2 norm aside.

**Asymmetric Distance Computation (ADC):** keep the query in full precision;
quantize only the database. For inner product, precompute per query a lookup
table

```
LUT[j, s] = q'[j] * c[s]      for dim j in [0,d'), centroid s in [0,2^b)
```

Then for a database vector with codes `code[0..d')`:

```
score(q, x) = sum_{j} LUT[j, code[j]]          # d' table lookups + adds
```

This is exactly faiss PQ's hot loop. Batched over `Q` queries × `N` db vectors it
is embarrassingly parallel and table-lookup-bound — ideal for SIMD/CUDA.

## 3. Two-track plan

### Track A — *days*: faiss flat-PQ backend over PCA-reduced space (no custom kernel)
The cheapest experiment, worth doing first. Path-1 used IVF (a coarse quantizer)
which **added approximation and hurt recall**. A **flat** `IndexPQ` (exhaustive
ADC, no IVF) over the PCA-Matryoshka-reduced space gives faiss's fast SIMD ADC
*without* the IVF recall hit:

```python
pca = PCAMatryoshka(768, 256); pca.fit(sample)
idx = faiss.IndexPQ(256, M, 8)          # M sub-quantizers, fast SIMD ADC
idx.train(pca.transform(C)); idx.add(pca.transform(C))
```

- **If** recall ≈ tq-pro's scalar codes at faiss-PQ speed (~900+ qps), Track A is
  a low-effort partial A+ and may suffice. Measure recall vs `benchmark_vectordb`
  PCA-256 numbers; PQ groups dims where tq-pro quantizes per-dim, so recall will
  differ — this experiment decides by how much.
- **Risk:** PQ's joint sub-vector codebooks ≠ tq-pro's per-dim scalar codebooks,
  so codes/recall won't match exactly. Track A validates whether "PCA + fast PQ"
  is good enough; Track B makes tq-pro's *own* codes fast regardless.

### Track B — *weeks*: native batched ADC kernel for tq-pro scalar codes
A real kernel so tq-pro's existing codes search fast, no recall change.

**Data layout.** Repack `N×d'` `b`-bit codes into a kernel-friendly layout:
- For `b=4`: 2 codes/byte — directly usable with SIMD `pshufb` (16-entry LUT),
  the faiss PQ4 fast-scan trick. **Recommend a 4-bit "kernel mode"** as the first
  target (byte-aligned, fastest).
- For `b=3`: either (i) a one-time unpack to a query-time 4-bit/byte layout
  (storage stays 3-bit on disk, expand in RAM), or (ii) bit-twiddling unpack in
  the kernel. Start with (i).

**CUDA design (CuPy `RawKernel` or pybind11 + .cu):**
1. *LUT kernel:* given projected queries `Qp (Q×d')` and centroids, compute
   `LUT (Q×d'×2^b)` on device.
2. *Scan kernel:* grid over (query-tile × db-tile). Each block loads its query's
   LUT into **shared memory** (tiny: `d'×2^b`, e.g. 256×8 floats = 8 KB), streams
   db codes with **coalesced** reads, accumulates `d'` lookups per db vector, and
   maintains a per-query **top-k** (warp-level bitonic or a small heap in
   registers/shared). 1-bit/3-bit unpack via precomputed shift/mask.
3. *Reduce:* merge per-tile top-k → final top-k per query.

**CPU SIMD fallback (AVX2/AVX-512):** faiss-style 4-bit PQ fast-scan — `pshufb`
gathers 16-entry LUT, `vpsadbw`/accumulate, blocked over 32 db vectors. Gives the
~900 qps baseline on machines without a GPU.

**Integration:**
```python
# new module turboquant_pro/_adc (pybind11 .so + CuPy kernel)
from turboquant_pro import batched_adc_search
idx, scores = batched_adc_search(queries, packed_codes, tq, top_k=10, device="cuda")
```
Replaces the per-query `gpu_adc_search` loop; `TurboQuantPGVector` gains a
`build_adc_index()` that does the one-time repack.

## 3b. The accuracy contract (issue #171)

The kernel is an **approximate scorer, by construction**. `build_lut` quantizes
the per-dim table to 255 levels of a single query-global scale; that is what
lets the inner loop run on `uint8` SIMD lanes, and it is what makes the score
inexact. The numpy path in `score_block` is the reference: it scores the same
formula at full float precision.

Both run in the same library, and `TQEIndex.search` chooses between them:

```python
if self._mmap or block is not None or exact:   # numpy, exact float ADC
else:                                          # compiled kernel, uint8 LUT
```

For a long time four tests asserted that those two return identical neighbour
ids. They pass with no kernel built and fail with one, and the failure was read
as a tie-ordering problem. It is not. With the kernel suppressed the paths
agree bit for bit; with it present, 33 of 200 returned ids differ and **none of
the disagreeing scores are equal**. The two sides are different scorers, not
two traversal orders of one.

### What is promised

Measured on Atlas (Xeon E5-2690 v3, AVX2) over six shapes, n 800 to 4000, dim
48 to 128, output 16 to 128, bits 2 to 4, 32 queries each:

| quantity | observed |
|---|---|
| max abs deviation of a kernel score from the exact score | 2.9e-3 to 6.3e-3 |
| that deviation as a fraction of the top-k score spread | 0.5% to 2.3% |
| recall@10 of the kernel's top-k against the exact top-k | 0.972 to 0.988 mean, 0.90 worst query |
| how far a disagreed-on id sits from the k-th exact score | always at or below the deviation |

The last row is the one worth stating as a promise: **the kernel only reorders
neighbours whose exact scores lie inside its own resolution.** A disagreement
further out than that would mean it preferred a neighbour it could tell was
worse, and `tests/test_kernel_contract.py` fails if that ever happens.

So:

* **Within one scorer, a ranking is reproducible bit for bit**, whatever the
  storage layout. `mode="exact"` is reproducible across RAM, memory-map and any
  block size, and that is tested.
* **Across the two scorers, it is not**, and no amount of tie-breaking would
  fix it, because the scores genuinely differ.
* **`mode="exact"` is how a caller demands the reference scorer** (`exact=True`
  is the older spelling). Use it when a run has to be comparable to another run
  whose layout you do not control: certificate anchors, claim replay, a
  recorded plan re-run. Before this was a parameter, callers got the exact path
  only as a side effect of passing `block` or memory-mapping, which is not
  something to rely on.
* **Reranking reconciles the boundary region.** An exact rescoring of a wider
  candidate set lands on nearly the same neighbours from either scorer:
  `test_rerank_removes_the_difference` requires at least 0.99 agreement at
  `rerank=8`, and on 100k LaBSE the two-stage recall@10 is 0.9995 through the
  kernel against 0.9997 through the reference (M3-final below). This is the
  two-stage path the library recommends anyway.
* **The scorer is named, never inferred.** `mode="exact"` or `mode="fast"`
  chooses it, `last_scorer` and the trace record it, and `tqp certify
  --environment` stamps the installation's scorers into the certificate. The
  rank certificate is computed on the float reconstruction, so it certifies the
  reference scorer's ranking.

### Why CI did not see it

`.github/workflows/ci.yml` had two jobs that between them covered everything
except the interesting configuration: `adc-kernel` built the kernel and ran
only `tests/test_adc_kernel.py`, and the whole-suite job never built the
kernel, so every kernel-gated path skipped. The kernel was exercised by its own
arithmetic replay and by nothing that consumes it. The `adc-kernel` job now
also runs the index, IVF and contract suites.

## 4. Validation

*As first written, this section asked for an exact set match between the kernel
and the reference (recall-vs-reference = 1.0). That holds for the kernel's
float-table scalar path, not for the SIMD path, and the tests now pin what is
true (`tests/test_kernel_contract.py`):*

- **Reference reproducibility:** `mode="exact"` returns the same ids and
  scores bit for bit in RAM, memory-mapped and blocked; with no kernel built,
  every path agrees bit for bit.
- **SIMD resolution:** every kernel score lies within 5% of the top-k score
  spread of the exact score (worst observed 2.3%).
- **SIMD boundary:** every id the two scorers disagree on sits within the
  kernel's own deviation of the k-th exact score.
- **SIMD recall against the reference:** mean at least 0.95, worst query at
  least 0.7 (observed 0.972 to 0.988 mean).
- **Two-stage agreement:** at `rerank=8` the two scorers agree on at least 0.99
  of the returned neighbours.
- **Recall:** reported per scorer: SIMD single-stage, SIMD + exact rerank and
  exact ADC + exact rerank, never one column standing for all three.
- **Performance:** qps on 199k and 1M vs the targets in §1; report on CPU (SIMD)
  and GV100 (CUDA). Add a `benchmark_adc_kernel.py`.
- **Operating-point plot:** recall@10 against qps at stated bytes per vector,
  with separate points for SIMD single-stage, SIMD + exact rerank and exact ADC +
  exact rerank, beside PQ, OPQ, RaBitQ and ScaNN measured on the same machine
  and protocol.

## 5. Risks & mitigations
- **3-bit unpacking is fiddly** → ship the **4-bit kernel mode first** (byte
  aligned, faiss-proven fast); add 3-bit via RAM-side unpack later.
- **top-k on GPU is the perf-sensitive part** → start with `k=10` warp-level
  selection; reuse a vetted CUDA top-k (e.g. raft/cub) rather than rolling one.
- **Maintenance burden of a native kernel** → keep the pure-Python path as the
  reference/fallback; the kernel is an accelerator, not a rewrite.

## 6. Effort & milestones
1. **Track A experiment** (1–2 days): PCA + flat `IndexPQ`; decide if it suffices.
2. **CPU SIMD 4-bit fast-scan** (3–5 days): hits the ~900 qps bar, no GPU needed.
3. **CUDA batched kernel + top-k** (1–2 weeks): the ≥5000 qps GV100 result.
4. **3-bit support, integration, tests, benchmark, paper update** (3–5 days).

**Net:** ~2–4 weeks for the full A+; Track A may give a publishable partial in
days. This is the *only* route the evidence supports for a clean A+ — and it's
honest engineering, not a reframing.

---

## Track A result (measured): NEGATIVE — confirms Track B is the only route

PCA-256 + faiss **flat** IndexPQ (no IVF), 100k LaBSE:

| config | bytes/vec | comp | qps | recall@10 (+rerank) |
|---|---:|---:|---:|---:|
| flat PQ (m=32) | 32 | 96× | 2743 | 0.41 |
| flat PQ (m=64) | 64 | 48× | 218 | 0.61 |
| *tq-pro scalar (reference)* | *96* | *32×* | *254* | ***0.999*** |

Even exhaustive flat-PQ ADC over the PCA space does **not** reach tq-pro's recall:
faiss product quantization is not as accurate (for this rerank) as tq-pro's
**per-dim scalar** codes. The recall advantage lives in tq-pro's specific
quantizer — so a fast *faiss-PQ backend* cannot deliver it. **Conclusion: only
Track B (a native batched ADC kernel over tq-pro's own scalar codes) reaches
fast + compressed + 0.999-recall.** Both quick alternatives (Path-1 IVF, Track-A
flat-PQ) are now measured and ruled out. Track B is the evidence-confirmed,
sole legitimate route to a clean A+.

---

## M1 investigation (measured): faiss fast-scan reuse ruled out — custom kernel necessary

Before writing kernel code, M1 tested whether faiss's existing PQ4 **fast-scan**
(the SIMD ADC kernel) could be reused for tq-pro-equivalent codes. 100k LaBSE,
recall@10 (+rerank):

| route | recall (+rerank) | qps | build | note |
|---|---:|---:|---:|---|
| per-dim PQ4 fast-scan (M=d') | 0.764 | 8763 | 3 s | no rotation |
| PCA + random-rot + PQ4 fast-scan | 0.768 | 8495 | 4 s | random rotation doesn't help |
| OPQ-96 + PQ4 fast-scan | **0.925** | 13447 | 346 s | learned rotation helps, but 4-bit caps recall + slow build |
| OPQ-64 + PQ4 fast-scan | 0.764 | 18807 | 303 s | too few subquantizers |
| *tq-pro own codes (reference)* | ***0.999*** | *254* | *24 s* | the recall target; slow scan |

**Findings:** (1) the **4-bit fast-scan format caps recall at ~0.925** — below
tq-pro's 0.999 (which uses its own 3-bit codebooks + exact reconstruct); (2) a
**random** rotation does not recover recall (only OPQ's *learned* rotation helps,
and even then only to 0.925 at 4-bit); (3) the one fast route that reaches 0.925
needs OPQ's **346 s** build, surrendering tq-pro's build-cost edge.

**Conclusion:** no faiss-reuse path delivers tq-pro's 0.999 at fast-scan speed.
The custom batched-ADC kernel over tq-pro's *own* codes (which already give 0.999
via exact reconstruct — the kernel only speeds the scan, math unchanged) is
**confirmed necessary** and remains the sole route to a clean A+. M1's next step
is the CPU SIMD kernel itself (the de-risking above is done).

---

## M1 kernel BUILT (measured): correct + 3.6x faster, hits the speed target

Implemented `turboquant_pro/_adc/adc_scan.cpp` — a pybind11 C++ extension with an AVX2
`pshufb` fast-scan (uint8-LUT, 16-entry table lookup, 32 db vectors/step, uint16
accumulation) and a scalar reference. Built on Atlas (g++ -O3 -march=native, AVX2),
100k corpus, 1000 queries, per-dim 4-bit codes:

| metric | result |
|---|---|
| agreement: scalar float-table path vs exact ADC | **0.9998** (validates the math) |
| agreement: AVX2 SIMD (uint8 table) vs exact ADC | 0.98 (the table's resolution) |
| **qps: AVX2 SIMD** | **3789** |
| qps: scalar reference | 1099 |
| qps: numpy reconstruct (baseline) | 1065 |
| **speedup (SIMD vs baseline)** | **3.6x** |

**M1 status: SUCCESS on speed, with the SIMD path an approximate scorer.** The
SIMD path clears the ~900 qps
target (3789 qps) and the scalar path (~numpy) confirms the pshufb SIMD is what
delivers the win. The kernel **supports tq-pro's 3-bit codes too** (S=8 ≤ 16, the
pshufb LUT just uses the first 8 entries) — so it can run tq-pro's *actual* codes
unchanged.

**Next (M3 integration):** feed the kernel tq-pro's real per-dim codes + shared
codebook + rotated queries (from `TurboQuantPGVector`) and confirm it reproduces
tq-pro's 0.999 recall at ~3789 qps. The kernel's *float-table* path reproduces the
exact ADC (0.9998); the SIMD path agrees at 0.98 and relies on the exact rerank
for the rest, as measured in M3-final. The standalone benchmark used a naive quantile
codebook (low recall) only to exercise the kernel — recall comes from the codes,
not the scan. Build: `python -m turboquant_pro._adc`; bench: `benchmarks/benchmark_adc_kernel.py`.

---

## M3 integration (measured): the float-table kernel reproduces tq-pro ADC; SIMD 2.5x faster

Wired tq-pro's REAL codes (PCA-256 + QR rotation + shared 3-bit codebook, recomputed
to match `TurboQuantPGVector.compress_embedding` exactly) through the M1 kernel.
100k LaBSE:

| metric | result |
|---|---|
| kernel (scalar) vs tq-pro faiss flat-reconstruct | **1.0000** (exact top-10 match) |
| kernel (AVX2 SIMD) vs faiss flat-reconstruct | 0.9867 (uint8-LUT quant) |
| recall@10 (256-d operating point): tq-pro ref | 0.405 single / 0.767 +rerank |
| recall@10 (256-d operating point): SIMD kernel | 0.405 single / 0.765 +rerank |
| **qps: SIMD kernel vs flat-reconstruct** | **3538 vs 1412 = 2.5x** |

**Measured:** the kernel's float-table scalar path reproduces tq-pro's compressed
ADC top-10 exactly on this sample (1.0000); the SIMD path agrees at 0.9867 and runs
**2.5x faster**. The integration mechanism works; the SIMD path is the approximate
scorer of the contract above, not a faster copy of the exact one.

**Honest gap — the PCA-mean term.** The paper's headline tq-pro recall (0.784/0.999)
comes from the pipeline reconstructing to **768-d via `pca.inverse_transform`
(mean re-added, uncentered query)**, not the centered 256-d search the kernel did
(0.40/0.77). To reach 0.999 at kernel speed, the cosine must be over 768-d recon:

```
cos(Q, recon768) = (Q.mean + norm * sum_j rotate(qt)[j]*cent[code[j]]) / ||recon768||
  qt = Q @ components^T  (uncentered PCA proj)
  ||recon768||^2 = ||mean||^2 + 2*norm*m_n + norm^2 * ||cent[codes]||^2
  m_n = sum_j rotate(mean_pca)[j]*cent[code[j]]   (one mean-ADC scan, precomputed)
```

So M3-final needs the kernel extended to: (a) a per-query bias `Q.mean`, (b) a
precomputed per-vector `m_n` (mean-ADC) and code-norm `||cent[codes]||`, (c) cosine
+ top-k computed inside the kernel from those. All terms are derived and
precomputable; it is a bounded kernel extension, the one remaining step to
demonstrate **0.999 at ~3500 qps**. Script: `benchmarks/benchmark_m3_integration.py`.

---

## M3-FINAL (measured): 32x storage, 0.9995 reranked recall@10 and 3802 qps at one operating point

Extended the kernel with the derived PCA-mean term (per-query `Q.mean` bias,
precomputed per-vector mean-ADC `m_n` and code-norm, cosine over 768-d recon).
100k LaBSE, 32x compression (96 bytes):

| | recall@10 single | recall@10 +rerank | qps | bytes |
|---|---:|---:|---:|---:|
| tq-pro flat-reconstruct (768-d ref) | 0.7902 | 0.9997 | 481 | 96 |
| **tq-pro + M1 SIMD kernel** | **0.7886** | **0.9995** | **3802** | 96 |

Top-10 agreement with the exact path before rerank: float-table scalar **0.9999**,
SIMD **0.9775**. The exact rerank reconciles the SIMD boundary (+rerank 0.9995
against the reference's 0.9997). **Speedup 7.9x** over tq-pro's own
flat-reconstruct.

**Result.** At this tested operating point (100k LaBSE, one CPU, 96-byte codes)
turboquant-pro simultaneously reaches about **32x compressed storage**, **0.9995
recall@10 after exact rerank** and **about 3.8k QPS** (3802; ScaNN 3441 and OPQ 915
in the same benchmark; OPQ's reranked recall 0.999, RaBitQ's 0.962), with a
training-free build.

What this result does not show, stated beside it:

* **It is a linear scan.** The speedup is a constant factor over reconstruct;
  query cost still grows with N. Sub-linear scaling needs the IVF path.
* **The strongest recall figure is after reranking.** The SIMD single stage is
  0.7886 recall@10 (the reference's is 0.7902); the 0.9995 is the two-stage
  path.
* **The 96 bytes are the codes.** The per-vector auxiliary terms the kernel
  reads (the mean-ADC term `m_n` and the code norm) and any originals kept for
  the rerank are not in that figure.
* **One corpus, one size, one machine.** 1M and larger, with total bytes, the
  latency distribution, the rerank width and its cost, all systems on identical
  hardware and protocol, is the open measurement.
