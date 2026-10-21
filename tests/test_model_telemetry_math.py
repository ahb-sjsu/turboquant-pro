"""The numerics of model_telemetry, with numpy only (no torch, no transformers)."""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.model_telemetry import (
    ReadPlan,
    Window,
    calibration_floor,
    compare_head,
    plan_regret,
    read_subspace,
    softmax_gram_weights,
    subspace_kappa,
)
from turboquant_pro.read_allocation import allocate_bits
from turboquant_pro.read_operators import AttentionAnalyticReadOperator
from turboquant_pro.validity import INCONCLUSIVE, STALE, UNCHECKED, VALID


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _brute(p, g=None):
    """sum_s || dp/dz_s ||^2 g_s^2, with dp_u/dz_s = p_u (delta_us - p_s)."""
    n = p.shape[-1]
    g = np.ones_like(p) if g is None else g
    out = np.zeros(p.shape[:-1])
    for s in range(n):
        e = np.zeros(n)
        e[s] = 1.0
        w = p * (e - p[..., s : s + 1])
        out += (w**2).sum(-1) * g[..., s] ** 2
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_form_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    p = _softmax(rng.normal(size=(5, 17)) * 3)
    np.testing.assert_allclose(softmax_gram_weights(p), _brute(p), rtol=1e-12)
    g = rng.uniform(0.1, 1.0, size=p.shape)
    np.testing.assert_allclose(softmax_gram_weights(p, g), _brute(p, g), rtol=1e-12)


def test_masked_keys_carry_nothing():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(4, 9))
    z[:, 6:] = -np.inf
    p = _softmax(z)
    np.testing.assert_allclose(softmax_gram_weights(p), softmax_gram_weights(p[:, :6]))


def test_operator_equals_analytic_provider_with_every_key_probed():
    rng = np.random.default_rng(4)
    d, n_keys, n_q = 8, 12, 10
    keys = rng.normal(size=(n_keys, d))
    q = rng.normal(size=(n_q, d)) * 2
    ref = AttentionAnalyticReadOperator(n_probe_keys=n_keys).operator(keys, queries=q)
    p = _softmax(q @ keys.T / np.sqrt(d))
    a = softmax_gram_weights(p)
    mine = (q * a[:, None]).T @ q / d
    np.testing.assert_allclose(mine, ref, rtol=1e-10, atol=1e-14)


def _psd(rng, d, rank=None, scale=1.0):
    A = rng.normal(size=(d, rank or d))
    return scale * A @ A.T


def test_kappa_is_one_under_a_pure_change_of_scale():
    rng = np.random.default_rng(5)
    P = _psd(rng, 6)
    U = np.linalg.eigh(P)[1][:, -3:]
    assert subspace_kappa(P, 7.0 * P, U) == pytest.approx(1.0, rel=1e-10)


def test_kappa_reads_a_known_change():
    U = np.eye(4)[:, :2]
    P = np.diag([4.0, 2.0, 1.0, 1.0])
    Q = np.diag([4.0, 8.0, 1.0, 1.0])  # direction 2 read 4x harder
    assert subspace_kappa(P, Q, U) == pytest.approx(4.0)


def test_regret_is_exactly_one_against_itself_and_above_one_after_a_rotation():
    rng = np.random.default_rng(6)
    d = 8
    P, C = _psd(rng, d, 3), _psd(rng, d)
    r, dp, dr = plan_regret(P, C, P, C, budget_bits=4.0 * d)
    assert r == pytest.approx(1.0, rel=1e-9) and dp == pytest.approx(dr, rel=1e-9)
    R = np.linalg.qr(rng.normal(size=(d, d)))[0]
    r2, _, _ = plan_regret(P, C, R @ P @ R.T, C, budget_bits=4.0 * d)
    assert r2 > 1.0


def test_allocate_bits_covariance_equals_activations():
    rng = np.random.default_rng(7)
    d = 6
    P = _psd(rng, d)
    x = rng.normal(size=(500, d)) @ rng.normal(size=(d, d))
    C = np.cov(x, rowvar=False, bias=True)
    a = allocate_bits(P, budget_bits=12.0, activations=x)
    b = allocate_bits(P, budget_bits=12.0, covariance=C)
    np.testing.assert_allclose(a.bits, b.bits, atol=1e-9)
    assert b.meta["whitened"] is False
    with pytest.raises(ValueError):
        allocate_bits(P, budget_bits=12.0, activations=x, covariance=C)


def _entry(P, C=None):
    return {"P": P, "C": C, "sketch": read_subspace(P)}


def _lowrank(rng, d=16, r=2):
    B = rng.normal(size=(d, r))
    return B @ B.T + 1e-4 * np.eye(d)


def test_status_valid_on_its_own_traffic_and_stale_after_a_rotation():
    rng = np.random.default_rng(8)
    P = _lowrank(rng)
    e = _entry(P, np.eye(16))
    same = compare_head(0, 0, e, P, (P, P), np.eye(16), budget_bits=64.0)
    assert same.status == VALID and same.overlap == pytest.approx(1.0)
    assert same.kappa == pytest.approx(
        1.0, rel=1e-6
    ) and same.scale_log2 == pytest.approx(0)
    assert same.regret == pytest.approx(1.0, rel=1e-9)
    R = np.linalg.qr(rng.normal(size=(16, 16)))[0]
    Q = R @ P @ R.T
    moved = compare_head(0, 0, e, Q, (Q, Q), np.eye(16), budget_bits=64.0)
    assert moved.status == STALE and moved.overlap < 0.5


def test_status_abstains_when_the_halves_disagree():
    rng = np.random.default_rng(9)
    P = _lowrank(rng)
    R = np.linalg.qr(rng.normal(size=(16, 16)))[0]
    Q = R @ P @ R.T
    c = compare_head(0, 0, _entry(P), 0.5 * (P + Q), (P, Q), None)
    assert c.status == INCONCLUSIVE


def test_status_unchecked_for_a_flat_operator():
    c = compare_head(0, 0, _entry(np.eye(16)), np.eye(16), (np.eye(16),) * 2, None)
    assert c.status == UNCHECKED and "too flat" in c.reason


def _window(rng, layers=(0, 1), H=2, d=8):
    w = Window(0, 100, {})
    for li in layers:
        P = np.stack([np.stack([_lowrank(rng, d) for _ in range(H)]) for _ in range(2)])
        k = rng.normal(size=(2, 50, H, d))
        w.layers[li] = {
            "P": {"softmax": P * 10, "logit": P * 30},
            "rows": np.full((2, H), 10.0),
            "ksum": k.sum(1),
            "kk": np.einsum("snhd,snhe->shde", k, k),
            "krows": np.array([50, 50]),
            "ent": np.ones(H),
            "ent_n": np.ones(H),
            "err": np.zeros((H, d, d)),
            "err_n": 0,
        }
    return w


def test_window_operator_is_the_row_weighted_mean_of_its_halves():
    w = _window(np.random.default_rng(10))
    L = w.layers[0]
    expect = L["P"]["softmax"].sum(0) / L["rows"].sum(0)[:, None, None]
    np.testing.assert_allclose(w.operator(0), expect)
    m = w.merged(w)
    np.testing.assert_allclose(m.operator(0), w.operator(0))
    assert m.tokens == 200


def test_plan_round_trips_and_validates_its_own_window(tmp_path):
    w = _window(np.random.default_rng(11))
    plan = ReadPlan.from_window(w, model="tiny", bits_per_channel=4)
    path = tmp_path / "plan.npz"
    plan.save(str(path))
    back = ReadPlan.load(str(path))
    assert back.meta == plan.meta and set(back.entries) == set(plan.entries)
    for key, e in plan.entries.items():
        np.testing.assert_array_equal(back.entries[key]["P"], e["P"])
        np.testing.assert_array_equal(back.entries[key]["C"], e["C"])
    for c in back.compare(w):
        assert c.overlap == pytest.approx(1.0) and c.regret == pytest.approx(1.0)


def test_read_subspace_is_sized_by_trace():
    P = np.diag([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.125])  # trace 16
    sk = read_subspace(P, 0.9)
    assert sk["rank"] == 4 and sk["trace_fraction"] == pytest.approx(15 / 16)
    np.testing.assert_allclose(np.abs(sk["basis"]), np.eye(8)[:4])
    assert read_subspace(P, 0.5)["rank"] == 1  # 8/16 reaches one half exactly


def test_too_few_rows_abstain_and_unsupported_numbers_say_why():
    rng = np.random.default_rng(12)
    P = _lowrank(rng)
    e = _entry(P, np.eye(16))
    c = compare_head(0, 0, e, P, (P, P), np.eye(16), rows=(1, 0), key_rows=2,
                     budget_bits=64.0)  # fmt: skip
    assert c.status == INCONCLUSIVE and "too few" in c.reason
    assert c.regret is None and "rank-deficient" in c.unavailable["model.plan.regret"]
    assert c.kappa is None and "directions" in c.unavailable["model.plan.kappa"]
    ok = compare_head(0, 0, e, P, (P, P), np.eye(16), rows=(40, 40), key_rows=40,
                      budget_bits=64.0)  # fmt: skip
    assert ok.status == VALID and ok.kappa is not None and ok.regret is not None
    assert ok.unavailable == {}


def test_a_plan_whose_own_halves_disagree_abstains_instead_of_going_stale():
    rng = np.random.default_rng(13)
    P = _lowrank(rng)
    R = np.linalg.qr(rng.normal(size=(16, 16)))[0]
    Q = R @ P @ R.T  # the calibration's two halves read different subspaces
    fl = calibration_floor(P, Q)
    assert fl["overlap"] < 0.5
    e = dict(_entry(0.5 * (P + Q)), floor=fl)
    c = compare_head(0, 0, e, Q, (Q, Q), None, rows=(40, 40))
    assert c.status == INCONCLUSIVE and "calibrate on more traffic" in c.reason
    same = calibration_floor(P, 3.0 * P)
    assert same["overlap"] == pytest.approx(1.0) and same["kappa"] == pytest.approx(1.0)


def test_the_regret_floor_is_above_one_on_unchanged_traffic():
    rng = np.random.default_rng(14)
    d = 8
    B = rng.normal(size=(d, 3))
    P = B @ B.T + 0.01 * np.eye(d)
    mix = rng.normal(size=(d, d))
    xa, xb = (rng.normal(size=(400, d)) @ mix for _ in range(2))
    Ca, Cb = np.cov(xa, rowvar=False), np.cov(xb, rowvar=False)
    fl = calibration_floor(P, P, Ca=Ca, Cb=Cb, budget_bits=4.0 * d)
    assert fl["regret"] > 1.0  # two samples of one source: a refit still looks better
    same = calibration_floor(P, P, Ca=Ca, Cb=Ca, budget_bits=4.0 * d)
    assert same["regret"] == pytest.approx(1.0)


def test_a_window_keeps_key_moments_per_half():
    w = _window(np.random.default_rng(15))
    assert w.key_rows(0) == 100 and w.key_rows(0, 1) == 50
    full = w.key_covariance(0)
    halves = [w.key_covariance(0, k) for k in (0, 1)]
    assert full.shape == halves[0].shape == (2, 8, 8)
    assert not np.allclose(halves[0], halves[1])
