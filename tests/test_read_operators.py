# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Read-operator provider protocol, registry, and conformance kit.

The exit criterion mirrors the quantizer plugins': the in-tree providers must
register and pass their own conformance suite through exactly the interface
an out-of-tree package would use.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.read_operator_conformance import (
    assert_read_operator_conformance,
    run_read_operator_conformance,
)
from turboquant_pro.read_operators import (
    ReadOperator,
    ReadOperatorSpec,
    available_read_operators,
    consumer_distortion,
    create_read_operator,
    error_covariance,
    get_read_operator,
    register_read_operator,
)

RNG = np.random.default_rng(11)
N, D = 128, 16


def _acts():
    return RNG.standard_normal((N, D))


def _queries(n=24):
    return RNG.standard_normal((n, D))


# ------------------------------------------------------------- the registry


def test_in_tree_providers_are_registered():
    names = set(available_read_operators())
    assert {"identity", "declared", "attention_analytic"} <= names


def test_unknown_provider_names_the_alternatives():
    with pytest.raises(KeyError, match="available"):
        get_read_operator("no_such_provider")


def test_providers_satisfy_the_protocol():
    for name, cfg in (
        ("identity", {}),
        ("declared", {"matrix": np.eye(D)}),
        ("attention_analytic", {}),
    ):
        assert isinstance(create_read_operator(name, **cfg), ReadOperator)


def test_registration_refuses_silent_overwrite():
    spec = ReadOperatorSpec(name="identity", factory=lambda **_: None)
    with pytest.raises(ValueError, match="already registered"):
        register_read_operator(spec)


def test_declared_requires_missing_imports_up_front():
    register_read_operator(
        ReadOperatorSpec(
            name="_needs_absent",
            factory=lambda **_: None,
            requires=("a_module_that_does_not_exist",),
        ),
        overwrite=True,
    )
    with pytest.raises(ImportError, match="requires"):
        create_read_operator("_needs_absent")


# ------------------------------------------------------------- conformance


@pytest.mark.parametrize(
    "name,cfg,ctx",
    [
        ("identity", {}, {}),
        ("declared", {"matrix": np.diag(np.linspace(1.0, 4.0, D))}, {}),
        ("attention_analytic", {"n_probe_keys": 8}, {"queries": None}),
    ],
)
def test_in_tree_providers_pass_conformance(name, cfg, ctx):
    provider = create_read_operator(name, **cfg)
    context = dict(ctx)
    if "queries" in context:
        context["queries"] = _queries()
    assert_read_operator_conformance(provider, _acts(), name=name, **context)


def test_conformance_catches_a_degenerate_provider():
    """An all-zero operator is algebraically valid and certifies everything."""

    class Zero:
        def operator(self, activations, **_):
            d = np.asarray(activations).shape[-1]
            return np.zeros((d, d))

    rep = run_read_operator_conformance(Zero(), _acts(), name="zero")
    assert not rep.passed
    assert "non_degenerate" in rep.failures


def test_conformance_catches_a_non_psd_provider():
    class Indefinite:
        def operator(self, activations, **_):
            d = np.asarray(activations).shape[-1]
            return np.diag(np.linspace(-1.0, 1.0, d))

    rep = run_read_operator_conformance(Indefinite(), _acts(), name="indef")
    assert not rep.passed
    assert "psd" in rep.failures


def test_conformance_catches_a_nondeterministic_provider():
    class Jittery:
        def operator(self, activations, **_):
            d = np.asarray(activations).shape[-1]
            return np.eye(d) * (1.0 + np.random.default_rng().random())

    rep = run_read_operator_conformance(Jittery(), _acts(), name="jitter")
    assert not rep.passed
    assert "deterministic" in rep.failures


def test_conformance_catches_a_raising_provider():
    class Broken:
        def operator(self, activations, **_):
            raise RuntimeError("nope")

    rep = run_read_operator_conformance(Broken(), _acts(), name="broken")
    assert not rep.passed
    assert rep.failures == ["callable"]


# ------------------------------------------------------- the distortion side


def test_distortion_ignores_error_the_consumer_cannot_read():
    P = np.diag([1.0, 0.0])
    assert consumer_distortion(P, np.diag([0.0, 1e6])) == pytest.approx(0.0)


def test_distortion_is_linear_in_the_error():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((D, D))
    P, S = A @ A.T, np.eye(D)
    assert consumer_distortion(P, 3.0 * S) == pytest.approx(
        3.0 * consumer_distortion(P, S)
    )


def test_error_covariance_removes_a_shared_offset():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((64, D))
    shifted = a + 5.0
    assert np.allclose(error_covariance(a, shifted), 0.0, atol=1e-9)


def test_identity_provider_reproduces_reconstruction_error():
    """With no consumer knowledge, the metric degenerates to the trace."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((64, D))
    b = a + 0.1 * rng.standard_normal((64, D))
    sigma = error_covariance(a, b)
    P = create_read_operator("identity").operator(a)
    assert consumer_distortion(P, sigma) == pytest.approx(np.trace(sigma))


def test_attention_operator_is_spanned_by_its_queries():
    """The structural claim the closed form rests on."""
    rng = np.random.default_rng(6)
    keys = rng.standard_normal((64, D))
    q = rng.standard_normal((5, D))
    P = create_read_operator("attention_analytic", n_probe_keys=8).operator(
        keys, queries=q
    )
    assert np.linalg.matrix_rank(P, tol=1e-10) <= 5
    # and the operator's range lies inside the query span
    qb = np.linalg.qr(q.T)[0]
    resid = P - qb @ (qb.T @ P)
    assert np.abs(resid).max() < 1e-8 * max(1.0, np.abs(P).max())


def test_attention_over_a_single_key_reads_nothing():
    """A softmax over one element is constant, so it has no gradient and the
    operator is exactly zero. The conformance kit must call that degenerate
    rather than certify it."""
    rng = np.random.default_rng(7)
    one_key = rng.standard_normal((1, D))
    P = create_read_operator("attention_analytic", n_probe_keys=1).operator(
        one_key, queries=_queries()
    )
    assert np.abs(P).max() == pytest.approx(0.0, abs=1e-12)

    rep = run_read_operator_conformance(
        create_read_operator("attention_analytic", n_probe_keys=1),
        one_key,
        name="attention_analytic_single_key",
        queries=_queries(),
    )
    assert not rep.passed
    assert "non_degenerate" in rep.failures


def test_attention_provider_says_what_it_needs():
    provider = create_read_operator("attention_analytic")
    with pytest.raises(ValueError, match="queries"):
        provider.operator(_acts())


def test_providers_ignore_context_they_do_not_understand():
    """Callers must be able to pass a superset without breaking a provider."""
    P = create_read_operator("identity").operator(
        _acts(), queries=_queries(), consumer=lambda x: 0.0, nonsense=7
    )
    assert P.shape == (D, D)
