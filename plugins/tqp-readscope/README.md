# tqp-readscope

readscope-backed **read-operator providers** for turboquant-pro.

turboquant-pro has two extension points. `turboquant_pro.plugins` plugs in
**quantizers**, the things being certified. `turboquant_pro.read_operators`
plugs in the **read operator** `P_C` they are certified against, so that

```
D = tr(P_C @ Sigma_delta)
```

is a number you compute rather than a claim you make. This package supplies
`P_C` for consumers that have no closed form, by measuring it.

```python
from turboquant_pro.read_operators import (
    create_read_operator, consumer_distortion, error_covariance,
)

P = create_read_operator("readscope_jacobian", consumer=my_head).operator(keys)
sigma = error_covariance(keys, codec.decompress(codec.compress(keys)))
consumer_distortion(P, sigma)      # what the consumer actually loses
```

## Two providers

| name | consumer shape | recovers |
|---|---|---|
| `readscope_blind` | vector → scalar margin | `E[g gᵀ]` |
| `readscope_jacobian` | vector → vector | `E[Jᵀ J]` |

Prefer `readscope_jacobian` where the consumer emits a vector: each probe
direction then returns `m` numbers instead of one, at the same call cost.

## The budget is not a tuning knob

readscope's calibration found that recovery against the direction budget
`k/d` is a **cliff at `k = d`**, and that the cliff is **rank independent** —
asking for one direction costs the same as asking for sixteen, because below
full dimension the estimate is a projection onto a random subspace and a
projected operator's leading eigenvector is not the operator's.

So both providers default to `k = d` and **warn** when given less, rather
than quietly returning a one-direction reading that looks like a
measurement. Full specification, including what is still unmeasured:
[readscope SPEC.md](https://github.com/ahb-sjsu/readscope/blob/master/SPEC.md).

## Neither package depends on the other

turboquant-pro declares the protocol. readscope measures the operator. This
adapter is the only code that knows about both, and it is about a hundred
lines. readscope stays numpy-only and usable by people with no interest in
compression, which is worth more than the convenience of merging them.

## Install

```bash
pip install -e plugins/tqp-readscope
```

Discovered automatically through the `turboquant_pro.read_operators` entry
point. MIT.
