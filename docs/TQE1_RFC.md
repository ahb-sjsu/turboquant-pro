# TQE1 Specification — RFC draft

**Status: DRAFT (2.0.0-alpha).** This document is the normative successor to
[`FORMAT_SPEC.md`](FORMAT_SPEC.md); until the 2.0.0-rc freeze, `FORMAT_SPEC.md`
remains the implemented truth and this RFC tracks it plus the ratified design
decisions below. The words MUST/SHOULD/MAY are used per RFC 2119.

## 1. Versioning model (four dimensions, never conflated)

| dimension | value(s) today | governs |
|---|---|---|
| **Container format** | `TQE1` | magic, framing, header layouts |
| **Specification revision** | `1.0-draft` | the normative text itself |
| **Record profile** | `embedding` (implemented) · `kv_block` (reserved) | what a record's payload *means* |
| **Codec ID** | `tq-lloydmax-qr`, `tq-lloydmax-hadamard` (implemented) · `key-channel/*`, `polar/*` (reserved for `kv_block`) | how payload bytes decode |

Bit-packing generations of the *persisted index* (TQIX "v3" etc.) are index
concerns and MUST NOT be referred to as versions of TQE1. The bare word "v3"
never appears in this specification.

## 2. Core record (normative, as implemented)

The v1 (20-byte) and v2 (21-byte) header layouts, LSB-first bit-packing,
Lloyd-Max codebooks, and rotation families are exactly as specified in
[`FORMAT_SPEC.md`](FORMAT_SPEC.md) §"Record layout" through §"Decode
algorithm"; those sections are incorporated here by reference and are already
frozen in practice (golden corpus, below). In this RFC's vocabulary: header
`version=1` ⇒ profile `embedding`, codec `tq-lloydmax-qr`; `version=2` ⇒
profile `embedding`, codec selected by the `rotation` byte.

**Conformance target:** the golden corpus
([`tests/golden/tqe1/`](../tests/golden/tqe1/)) with its sha256 manifest. An
implementation conforms iff it reproduces `expected.npz` from the `.tqe`
bytes alone and rejects the malformed cases. The dependency-free reference
reader is [`contrib/tqe1_reader.py`](../contrib/tqe1_reader.py).

## 3. Canonical encoding (ratified for 1.0)

Two conforming writers given identical input and parameters MUST produce
byte-identical records:

- Header fields are fixed-layout (no ordering freedom by construction).
- Floating-point header fields (`norm`) are IEEE-754 binary32, round-to-
  nearest-even from the writer's float64 intermediate. NaN and ±Inf are
  invalid in any header field; writers MUST reject them, readers MUST treat
  them as corruption.
- Bit-packing pad bits MUST be zero. Readers MUST ignore pad-bit values on
  decode but validators SHOULD flag nonzero padding (non-canonical).
- Future profile metadata (kv_block) uses the canonical JSON of the identity
  module (`sorted keys, ",":"" separators, ensure_ascii, allow_nan=False`).
- Hash coverage: a record's content hash covers header + codes exactly
  (trailing extension bytes excluded; they carry their own hashes).

## 4. Integrity vs identity (three fields, three problems)

| field | purpose | algorithm |
|---|---|---|
| corruption checksum | fast bit-rot detection | CRC32C (per record or per block) |
| content hash | dedup / content addressing | SHA-256 over §3 coverage |
| semantic identity | "may I reuse this?" | the KV identity-profile digest (`tqp-kv-identity/1`) carried as metadata — equality REQUIRED for reuse; uncertain ⇒ miss |

A reader MUST NOT treat a correct checksum as permission to reuse (identity
is separate), and MUST NOT treat identity match as proof of integrity.

## 5. Extension behavior (ratified for 1.0)

- Unknown **optional field/trailer**: skip (length-delimited, per
  FORMAT_SPEC's trailer rule).
- Unknown **mandatory feature bit**: reject the record.
- Unknown **record profile**: enumerate (size, profile id) but do not decode.
- Unknown **codec**: reject that record — not the file.
- A feature-bit word and reserved id ranges (profiles `0x00–0x0F` core,
  `0x10–0x7F` registered, `0x80–0xFF` private-use) ship with the `kv_block`
  profile revision of the header.

## 6. Parser limits (security; ratified for 1.0)

Readers MUST enforce: maximum record size (default 1 GiB, configurable
down), maximum `dim` (65535 by field width; implementations MAY cap lower),
checked arithmetic on all size computations, allocation bounded by declared
sizes before reading payloads, and no recursion in parsing. A malformed-file
fuzz corpus is a release requirement (rc gate); `contrib/tqe1_reader.py` is
the first fuzz target.

## 7. Random access & recovery (design, targeted at `kv_block`)

Batch files today are sequential concatenations (sufficient for embedding
batches). The `kv_block` profile adds: an optional footer index (record
offsets + content hashes), an atomic-commit marker (a record is visible only
after its index entry lands), truncation recovery (scan-forward from the last
valid marker), tombstones + compaction, and single-writer/multi-reader
append rules. Writers MUST make partial records unobservable (write-ahead
then index).

## 8. `kv_block` record profile (reserved; design)

Payload: one layer's (K, V) block pair under a declared codec pair; metadata
(canonical JSON): identity-profile digest, layer index, head geometry, token
range, prefix block hash (the chained content address from
`prefix_block_hashes`), codec IDs + parameters, dtype. Governing rule
inherited: a reader MUST refuse to decode a `kv_block` whose identity digest
it cannot match — enumerate, don't decode.

## 9. Compatibility promise (P2-M4 wording, ratified)

- TQE1 readers MUST always understand all **finalized TQE1 core records**
  (the §2 embedding profile as frozen by the golden corpus).
- Deprecated codecs remain readable for a documented minimum of one major
  release after deprecation.
- `tqp format migrate` upgrades older experimental records.
- Pre-freeze files (anything written before the 1.0 spec freeze other than
  the golden-corpus-covered core) are **legacy**: readable via migrate, not
  part of the permanent contract.

## 10. Interoperability requirements for calling this a standard (rc/GA gates)

Python writer → Rust reader; Rust writer → Python reader; old reader → new
writer (optional extensions only); big-endian environments explicitly
rejected or simulated in CI; fuzz suite green; golden files reproduced by an
independently-written implementation; at least one reader maintained outside
this package. The word "standard" is not used prominently in project
positioning before an external adopter exists.
