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

## The capability declaration (how the planner enumerates you)

The control plane (`tqp plan run`) builds its candidate list from this registry.
It will try your codec at a default grid of bit widths if you say nothing, which
works but wastes calls on widths you do not support. Declare them instead:

```python
class MyQuantizer:
    ...
    def capabilities(self):
        return {
            "bit_widths": (2, 4, 8),      # widths the factory accepts as bits=
            "default_bits": 4,            # used when the factory takes no bits=
            "requires_calibration": False,
            "hardware": None,             # e.g. "sm_90" when silicon-specific
        }
```

Everything is optional and unknown keys are carried into the plan record rather
than dropped — the planner is not the authority on what a future codec has to say
about itself. A factory that accepts a parameter named `head_dim`, `n_heads`,
`dim` or `input_dim` is handed the artifact's measured value, so a codec whose
geometry must match the data does not have to be configured by hand for every
artifact; the hint is offered, and a factory that rejects it is built again
without it.

A codec that cannot be built at all is recorded in the plan as `unsupported`
with the error, never silently dropped: a candidate that vanishes without a
record is how a planner ends up recommending from a list of one.

## The consumer side of the same contract

`turboquant_pro.plugins` plugs in the codec. `turboquant_pro.read_operators`
plugs in the operator `P_C` it is judged against. `turboquant_pro.consumers`
plugs in the **measurement** that stands in for the consumer, item by item, which
is what the planner ranks codecs on. Same registry shape, entry-point group
`turboquant_pro.consumers`:

```python
from turboquant_pro.consumers import ConsumerSpec, register_consumer

class MyConsumer:
    name = "my_metric"
    higher_is_better = True

    def per_item(self, original, reconstructed, **context):
        ...  # one score per item the consumer reads

    def nominal_per_item(self, original, reconstructed, **context):
        ...  # optional: the cheap metric on those same items

register_consumer(
    ConsumerSpec(
        name="my_consumer",
        factory=lambda **cfg: MyConsumer(**cfg),
        targets=frozenset({"embedding"}),
        exact=True,              # is this the consumer's own computation?
        evidence_kind="statistical",
    )
)
```

`nominal_per_item` is what makes the false-clear diagnostic meaningful: the cheap
metric has to be scored on the same items as the consumer, or the rate compares
two different populations. Return nothing and the planner says the false clear is
unavailable rather than computing a misaligned one.

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
