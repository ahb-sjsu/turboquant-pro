# tqp-readscope: readscope-backed read operators for turboquant-pro
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""The adapter, exercised through the interface turboquant-pro publishes.

Two things are being checked. That the providers satisfy the read-operator
contract, using turboquant-pro's own conformance kit rather than a private
one. And that a blindly recovered operator agrees with the closed form where
a closed form exists, which is the only way to know the adapter is wired up
correctly rather than merely returning something PSD.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("readscope")
pytest.importorskip("turboquant_pro")

from turboquant_pro.read_operator_conformance import (  # noqa: E402
    assert_read_operator_conformance,
)
from turboquant_pro.read_operators import (  # noqa: E402
    create_read_operator,
    register_read_operator,
)

from tqp_readscope.plugin import SPEC_BLIND, SPEC_JACOBIAN  # noqa: E402

RNG = np.random.default_rng(3)
N, D = 96, 12


@pytest.fixture(autouse=True)
def _registered():
    """Register exactly as the entry point would, so the test path is real."""
    register_read_operator(SPEC_BLIND, overwrite=True)
    register_read_operator(SPEC_JACOBIAN, overwrite=True)


def _linear_scalar(w):
    return lambda x: float(w @ x)


def _linear_vector(A):
    return lambda x: A @ x


def test_blind_provider_passes_conformance():
    w = RNG.standard_normal(D)
    provider = create_read_operator("readscope_blind", consumer=_linear_scalar(w))
    assert_read_operator_conformance(
        provider, RNG.standard_normal((N, D)), name="readscope_blind"
    )


def test_jacobian_provider_passes_conformance():
    A = RNG.standard_normal((5, D))
    provider = create_read_operator("readscope_jacobian", consumer=_linear_vector(A))
    assert_read_operator_conformance(
        provider, RNG.standard_normal((N, D)), name="readscope_jacobian"
    )


def test_blind_recovery_matches_the_closed_form():
    """A linear margin has a constant gradient, so P_C is exactly w w^T."""
    w = RNG.standard_normal(D)
    provider = create_read_operator("readscope_blind", consumer=_linear_scalar(w))
    P = provider.operator(RNG.standard_normal((32, D)))
    assert np.allclose(P, np.outer(w, w), atol=1e-5)


def test_jacobian_recovery_matches_the_closed_form():
    """For C(x) = A x the Jacobian is A, so the Gram is exactly A^T A."""
    A = RNG.standard_normal((4, D))
    provider = create_read_operator("readscope_jacobian", consumer=_linear_vector(A))
    P = provider.operator(RNG.standard_normal((16, D)))
    assert np.allclose(P, A.T @ A, atol=1e-5)


def test_consumer_may_arrive_in_the_call_context():
    """Providers are reusable; the consumer need not be fixed at creation."""
    w = RNG.standard_normal(D)
    provider = create_read_operator("readscope_blind")
    P = provider.operator(RNG.standard_normal((16, D)), consumer=_linear_scalar(w))
    assert np.allclose(P, np.outer(w, w), atol=1e-5)


def test_missing_consumer_is_a_clear_error():
    provider = create_read_operator("readscope_blind")
    with pytest.raises(ValueError, match="needs the consumer"):
        provider.operator(RNG.standard_normal((8, D)))


def test_subdimensional_budget_warns_rather_than_quietly_degrading():
    """readscope's budget law is a cliff at k = d, so a smaller budget is a
    specification violation and must not pass silently."""
    w = RNG.standard_normal(D)
    provider = create_read_operator(
        "readscope_blind", consumer=_linear_scalar(w), n_directions=D // 3
    )
    with pytest.warns(RuntimeWarning, match="cliff at k = d"):
        provider.operator(RNG.standard_normal((16, D)))


def test_full_budget_does_not_warn():
    import warnings

    w = RNG.standard_normal(D)
    provider = create_read_operator(
        "readscope_blind", consumer=_linear_scalar(w), n_directions=D
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provider.operator(RNG.standard_normal((8, D)))


def test_blind_agrees_with_the_in_tree_attention_closed_form():
    """The cross-check that matters: two independent providers, one truth.

    Both sides must see the same key set and accumulate over the same probed
    keys. An earlier version of this test handed the analytic provider a
    single key, whose softmax is constant at one and therefore reads nothing,
    so it compared a real operator against an exact zero.
    """
    n_keys = 8
    keys = RNG.standard_normal((n_keys, D))
    q = RNG.standard_normal((6, D))
    d = D

    analytic = create_read_operator("attention_analytic", n_probe_keys=n_keys, seed=0)
    exact = analytic.operator(keys, queries=q)

    blind = create_read_operator("readscope_jacobian")
    got = np.zeros((d, d))
    for s in range(n_keys):

        def attention(k_s, _s=s):
            kp = keys.copy()
            kp[_s] = k_s
            z = (q @ kp.T) / np.sqrt(d)
            z = z - z.max(axis=-1, keepdims=True)
            p = np.exp(z)
            return (p / p.sum(axis=-1, keepdims=True)).ravel()

        got += blind.operator(keys[s][None, :], consumer=attention)

    scale = max(np.abs(exact).max(), 1e-12)
    assert scale > 1e-6, "the analytic operator must not be degenerate here"
    assert np.abs(got - exact).max() < 1e-3 * scale
