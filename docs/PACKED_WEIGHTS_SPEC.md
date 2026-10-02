# TQPW: packed weights (`tqp.packed_weights/1`)

The stored form of a planned weight encoding. `tqp plan encode-weights` writes it, and
`tqp plan decode-weights` turns it back into a runnable model. A TQPW file holds a
codec's integer codes for every planned matrix, each at its own width, plus the grid
that decodes them. Its payload is exactly the plan's `stored_bits`.

## Container

The TQIX container ([FORMATS.md §2](FORMATS.md#2-tqix--the-persisted-index-container)),
little-endian, with magic `TQPW`:

```
header (12 bytes)                         directory: n_sections x 56 bytes
 off sz field                              off sz field
 0   4  magic      b"TQPW"                 0   32 name   utf-8, NUL-padded
 4   2  version    uint16 (== 1)           32  8  offset uint64 (from file start)
 6   2  n_sections uint16                  40  8  length uint64
 8   4  reserved   uint32 (0)              48  4  crc32  uint32 (of the section)
                                           52  4  flags  uint32 (0)
then the section payloads, back to back, in directory order
```

Sections, in order: `meta`, then for matrix `i` = 0, 1, ... the codes `c<i>` and the
grid `g<i>`. A reader must check every section's CRC32 and refuse a mismatch, a
section that runs past the end of the file, a magic other than `TQPW`, or a version
it does not implement.

## `meta`

UTF-8 JSON with sorted keys and no insignificant whitespace. Required fields:

| field | value |
|---|---|
| `format` | `"tqp.packed_weights/1"` |
| `group` | `128`, the input columns per grid group |
| `matrices` | list, in section order, of `{"name", "shape": [out, in], "bits"}` |
| `grid`, `decode` | human-readable restatements of this spec |

Writers may add fields. `tqp plan encode-weights` adds `codec`, `model_path`,
`plan_sha256` and `cost_table_hash`. `in` is a multiple of 128 and `bits` is in 1..8.

## Codes: `c<i>`

The `out * in` codes of the matrix in row-major order, each an unsigned integer
below `2^bits`. They are packed as one LSB-first stream: code `j` occupies stream
bits `[j*bits, (j+1)*bits)`, and stream bit `p` is bit `p mod 8` of byte `p div 8`.
This is the packing of [FORMAT_SPEC.md](FORMAT_SPEC.md#bit-packing), extended to every
width from 1 to 8. Since `out * in` is a multiple of 128, the stream fills whole
bytes. The section is `out * in * bits / 8` bytes long.

## Grid: `g<i>`

`out x (in / 128) x 2` IEEE half-precision values, little-endian, row-major. Entry
`[o, g]` is `(lo, step)` for row `o`, input columns `128g .. 128g + 127`. The section
is `out * (in / 128) * 4` bytes, i.e. 32 bits per group.

## Decoding

For code `r` in row `o`, column `c`, with `(lo, step)` the grid of `(o, c div 128)`:

```
w[o, c] = float32(r) * float32(step) + float32(lo)      (float32 arithmetic)
```

The product is exact in float32, since `r` has at most 8 significant bits and `step`
at most 11. So `w` is the single float32 rounding of the exact `r * step + lo`,
whether or not the sum is fused. Every conforming reader produces the same bits. A
runtime then casts `w` to its own dtype.

## Size

Each matrix stores `out*in*bits + (out*in/128)*32` bits. That is
`turboquant_pro.weight_plan.stored_bits`, the quantity a `tqp.weight_plan/1` budgets.
The file adds the 12-byte header, 56 bytes per section and the `meta` JSON.

## Relation to the codec

The GPTQ and RTN codecs (`turboquant_pro.weight_codec`) compute each group's grid in
float32 and write `codes * step + lo` in float32. The stored codes are those exact
codes. The stored grid is that float32 grid rounded to float16, which is what the
32 bits per group the plan counts can hold. So a decoded weight differs from the
codec's output by at most about `2^-11 * (|lo| + r * step)`.
`tqp plan encode-weights` measures the largest such difference, in units of the
group's step, and records it in `weight_encoding.json` (`grid_rounding`). The
Part III-c results were measured on the codec's float32 output. They were not
measured on the decoded weights.

## Versioning

The `magic` + `version` prefix is permanent. Within version 1 the layout and the
decode rule are frozen. A change to either is a new version, which readers that
predate it refuse.

## Conformance

`tests/golden/tqpw/` holds `golden.tqpw` (sha256 pinned in `manifest.json`), with a
matrix at widths 2, 3, 4, 5, 6 and 8, and `expected.npz`, the float32 tensors it
decodes to. A reader conforms if it reproduces `expected.npz` from the file bit for
bit and refuses corrupt, truncated, foreign and unknown-version files.
`contrib/tqpw_reader.py` is a dependency-free reader written from this page alone.
`tests/test_tqpw_golden.py` and `tests/test_packed_weights.py` check all of this.
