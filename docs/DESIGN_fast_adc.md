# Design Doc — Fast Compressed ADC (Path 2 to A+)

**Status:** proposed · **Owner:** TBD · **Goal:** break the speed/recall/
compression trilemma so turboquant-pro is fast *and* compressed *and* high-recall.

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
2. **identical recall** to today's tq-pro codes (it only speeds the scan);
3. drops in behind the existing `compress_batch` API (no new training, no recall loss).

Meeting (1)+(2) puts tq-pro in the **fast + compressed + high-recall** corner —
the A+ result.

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

## 4. Validation

- **Correctness:** top-k from the kernel must match the reference per-query ADC
  (exact set match on a 10k sample; assert recall-vs-reference = 1.0).
- **Recall:** unchanged from today's codes — re-run `benchmark_vectordb.py` and
  confirm the tq-pro recall column is identical (it must be; same codes).
- **Performance:** qps on 199k and 1M vs the targets in §1; report on CPU (SIMD)
  and GV100 (CUDA). Add a `benchmark_adc_kernel.py`.
- **A+ check:** a single plot — recall@10 vs qps — with tq-pro's new point landing
  in the **upper-right with the smallest bytes/vec**, dominating PQ/OPQ/RaBitQ and
  competitive with ScaNN at far higher recall.

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
