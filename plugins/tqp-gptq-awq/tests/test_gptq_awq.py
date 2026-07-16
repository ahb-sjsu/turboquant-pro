"""Conformance + semantics for the GPTQ/AWQ weight-format adapters."""

from __future__ import annotations

import numpy as np
from tqp_gptq_awq.plugin import AWQQuantizer, GPTQQuantizer

from turboquant_pro.plugin_conformance import assert_conformance

RNG = np.random.default_rng(17)


def _weight():
    return RNG.standard_normal((1, 1, 64, 256)).astype(np.float32) * 0.05


def test_gptq_conformance_affine_pass():
    report = assert_conformance(GPTQQuantizer(), _weight(), rel_err_max=0.30)
    assert report.results["affine"].startswith("pass"), report


def test_gptq_exact_on_grid():
    q = GPTQQuantizer(group_size=16)
    x = (np.arange(16, dtype=np.float32) * 0.1 - 0.8).reshape(1, 1, 1, 16)
    got = q.decompress(q.compress(x))
    assert np.allclose(got, x, atol=1e-5)


def test_awq_scale_fold_roundtrip():
    s = RNG.uniform(0.5, 2.0, 256).astype(np.float32)
    q = AWQQuantizer(group_size=128, act_scale=s)
    x = _weight()
    got = q.decompress(q.compress(x))
    rel = np.linalg.norm(got - x) / np.linalg.norm(x)
    assert rel < 0.25  # random (non-smoothing) act-scales widen 4-bit groups
    assert q.grid_params(q.compress(x)) is None  # honest degrade when scaled


def test_awq_unit_scale_is_gptq():
    x = _weight()
    a = AWQQuantizer().decompress(AWQQuantizer().compress(x))
    g = GPTQQuantizer().decompress(GPTQQuantizer().compress(x))
    assert np.allclose(a, g, atol=1e-6)


def test_registry_recommendation_end_to_end():
    from tqp_gptq_awq.plugin import SPEC_AWQ, SPEC_GPTQ

    from turboquant_pro.plugins import TARGET_WEIGHT, available_plugins, register

    for spec in (SPEC_GPTQ, SPEC_AWQ):
        try:
            register(spec)
        except ValueError:
            pass
    names = set(available_plugins(target=TARGET_WEIGHT))
    assert {"gptq", "awq"} <= names
