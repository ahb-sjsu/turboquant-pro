# Writing a quantizer plugin

turboquant-pro accepts out-of-tree quantization formats through a small,
executable contract (design: [`DESIGN_hardware_and_plugins.md`](DESIGN_hardware_and_plugins.md),
section 2). A plugin gets, for free: the certification instruments
(rank certificates, the (A2) probe, `behavioral_agreement`), the comparison
harness, and — if it exposes the affine capability — the fused
compute-on-codes decode.

## The minimum

Provide an object with `compress(x, **kw) -> container` and
`decompress(container) -> ndarray`, and register a named factory:

```python
# your package: tqp_myformat/plugin.py
from turboquant_pro.plugins import PluginSpec, TARGET_KV_KEY

def make(**config):
    return MyQuantizer(**config)

SPEC = PluginSpec(
    name="myformat",
    factory=make,
    targets=frozenset({TARGET_KV_KEY}),
    tier="experimental",           # entry tier; see docs/api-stability.md
    description="one line: what it is, what it's for",
)
```

```toml
# your package: pyproject.toml
[project.entry-points."turboquant_pro.plugins"]
myformat = "tqp_myformat.plugin:SPEC"
```

That's it — `turboquant_pro.available_plugins()` now lists it and
`turboquant_pro.create_quantizer("myformat", **cfg)` builds it. Targets:
`"kv_key"`, `"kv_value"`, `"weight"`, `"embedding"` (keys and values are
deliberately separate targets — they have opposite quantization disciplines;
see `docs/KV_KEYS_FINDING.md`).

## The affine capability (unlocks fused decode)

If your KEY format dequantizes as **per-channel affine over a fixed grid**

```
dequant[b, h, s, j] = mu[h, j] + weight[h, j] * grid[code[b, h, s, j]]
                      (+ optional sparse fp16 overlay)
```

expose it, and the M4 fused kernel serves your format with no kernel work:

```python
class MyQuantizer:
    ...
    def grid_params(self, c):   # (mu (H,D), weight (H,D), grid (L,)) or None
    def codes(self, c):         # unpacked (B, H, S, D) uint8
    def outlier_csr(self, c):   # (row_ptr, cols, deltas) token-major, or None
```

`weight` may be per-channel `(H, D)` or token-block-granular `(H, S, D)`
(the design doc §6 extension — fold per-block scales by expanding them per
element; `tqp_bnb.BnbNF4Quantizer.grid_params` is the reference). Return
`None` from `grid_params` for containers with no affine form (e.g.
learned per-channel tables) — that is the documented graceful degrade to
decompress-then-attend, not an error. blockwise-scaled formats (bitsandbytes
NF4, block-16 FP4): fold the block scale into `weight`; GPTQ/AWQ-style
scale/zero: `mu = zero * scale`, `weight = scale`, `grid = arange(2**bits)`.

## The conformance kit (run it in your CI)

```python
from turboquant_pro.plugin_conformance import assert_conformance

def test_conformance():
    q = make(head_dim=128, n_heads=8, ...)
    x = load_representative_block()      # (1, H, S, D) float32
    assert_conformance(q, x)
```

Checks: round-trip envelope, packed/unpacked equivalence, **affine
reconstruction == decompress** (the gate that makes fused decode safe to
inherit), CSR structural validity, and byte-serialization round-trip — each
reported as pass / skip-with-reason / FAIL-with-detail. Correctness must not
require your target hardware: emulate exotic dtypes (e.g. `ml_dtypes`) so the
suite runs on CPU CI.

## Reference implementation

`turboquant_pro.plugins.PerChannelKVQuantizer` is the in-tree
`PerChannelKV` registered through this exact interface (name
`"per_channel"`), including the affine capability consumed by
`kv_fused_pck.PreparedPCKBlock`. `"polar"` (PolarQuant values) demonstrates
the non-affine case. `tests/test_plugins.py` shows both passing the same
suite an external plugin would run.
