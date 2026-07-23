# 2.0 positioning — the four moats, stated honestly

> **"The fix was declaration."** — external review of the anatomy
> provenance work, on a case where the implementation was already correct
> (Spearman, as shipped) and the defect was that the artifact didn't SAY so.
> An undeclared-correct implementation is indistinguishable, to the
> artifact's reader, from a wrong one. Correctness that isn't legible isn't
> verifiable, and unverifiable correctness earns no trust. That is the
> claims discipline of this entire page in four words.

Distilled from external review #2 ([`ROADMAP_2.0-review2.txt`](ROADMAP_2.0-review2.txt)),
which frames 2.0's differentiation correctly: **the moat is the contracts, not
the algorithms.** This page is the quotable version — every claim below is
annotated with its shipped/target status, per the project's positioning rule
(measured targets, never promises). Marketing copy quotes THIS page, not the
raw review.

## Moat 1 — Zero-trust state governance 🟢 shipped (in-process scope)

Most KV offload schemes match prefixes by prompt-string or raw-token hashes;
change TP rank, RoPE scaling, the attention backend, or a LoRA adapter and
they silently return corrupted attention state or crash. tqp's
`tqp-kv-identity/1` profile binds model repo/revision, weight + tokenizer
fingerprints, adapters, RoPE, attention backend + KV layout, TP/PP, dtype,
head geometry, block size, quantization discipline, and encoder version into
one content address, with the invariant enforced structurally:

> **Uncertain compatibility ⇒ cache miss and recomputation.** An incomplete
> profile matches nothing — including itself. Prefix keys cannot be minted
> under uncertain identity.

*Honest scope:* shipped and tested at the store/profile layer; persistence
and cross-node reuse (where the moat pays off) land at beta behind these
gates. "Only safe option for multi-tenant clusters" is the *goal*; the
security/tenancy milestone (P1-M3, namespaces + threat model) is what earns
the sentence.

## Moat 2 — TQE1 as the Parquet/GGUF of compressed tensors 🟡

Four-dimension versioning (container / spec revision / record profile /
codec) ✅ ratified in the RFC draft. Golden corpus enforced in CI ✅. A
dependency-free single-file Python reader ✅ (vendor one file, read TQE1
forever). Canonical encoding (IEEE-754 binary32 headers, zero pad bits,
NaN/Inf rejection) ✅ ratified.

*Honest scope:* the Rust reader, fuzz-in-CI, cross-language interop matrix,
and the `kv_block` profile are rc gates — until then the accurate sentence is
"a frozen-core format with a public conformance suite," not "an open
standard." The word *standard* waits for an external adopter (RFC §10).

## Moat 3 — Break-even admission ⚪ designed (P1-M4)

`T_lookup + T_read + T_transfer + T_dequant < T_recompute`, evaluated
per-block against prefix length, tier bandwidth, queue depth, and load.

*Honest scope:* this is a design milestone. The claim to make when it ships
is "**designed to avoid net regressions, with the break-even region measured
and reported**" — NOT "mathematically guaranteed to never degrade
performance": the inequality is an *estimate* under measured costs, and the
GA gate is that the region is *reported*, not hidden. Guarantee-language is
banned by the roadmap's positioning rule.

## Moat 4 — The cross-engine control plane 🟡

One storage layer under every engine rather than a competitor to any:
vLLM V1 connector 🟡 (protocol-conformance lane pinned in CI; engine
execution = beta) · SGLang HiCache backend ⚪ (2.1, public backend surface) ·
DuckDB compressed-domain Arrow scans 🟢 · PostgreSQL Track A bridge 🟢 →
Track B access method ⚪ (2.2, full database semantics).

## The stratified-hubness lead (added 2026-07-23, lit-check partial)

Stated honestly, component by component: SQL-over-vectors is commodity;
partitioned indexes are commodity; hubness measurement is scikit-hubness
plus sixteen years of literature; N_k is a GROUP BY once the kNN graph is
an edge table. And the niche is heating from the adversarial side —
arXiv:2604.05480 (Black-Hole Attack) validates centrality-mechanism
hubness as an attack surface; arXiv:2602.22427 ships an open-source
hubness-poisoning scanner. Worse news for uniqueness, better news for
relevance.

What nobody visible ships, and tqp now does 🟡: **the relational surface
over *certified* stratified hubness** — hubness as queryable relations
that carry contracts (area-scoped, identity-versioned, ABSTAIN-aware,
gateable in the query), where joins across mismatched `area_map_digest`
values are refused by the relation itself. The 2.2 fusion is already on
the roadmap: `WITH (RECALL >= 0.9 PER AREA)` — the recall contract and
STRATA §5 in one syntax.

This is a **lead**, not a portal: roughly two quarters of formalization
advantage plus shipped components. Leads are spent by shipping, citing
both papers (done — primer), and publishing while the timestamp is ours.
Candidate external adopter for `hubdiff`: the open-source detector;
candidate stress batteries for openvector-bench: their adversarial
benchmarks.

## The verdict line (approved phrasing)

> 1.x proved the algorithms; 2.0 proves the contracts. Identity-gated cache
> reuse, a frozen conformance-tested format, and fail-safe fallbacks — chosen
> not because it's fast, but because it's engineered to be safe.

("Unbreakable" is not approved phrasing; contracts are *tested*, and the test
suites are public.)
