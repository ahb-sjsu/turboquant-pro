# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""The learning price of a sampled read operator (T1 note, geometric-observation
paper/t1-learn-then-serve v4.1): the note's own numbers, the band's coverage
with a self-test that a band too narrow is caught, the refutation of a wrong
declared constant, invariances, the lower-rank scope, the moving-kernel
diagnostic of the addendum, finite differences, the calibration message, and
the CLI."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from turboquant_pro.cli import main
from turboquant_pro.learning_price import (
    CERTIFIED,
    NO_CERTIFICATE,
    REFUTED,
    calibration_message,
    combine_bands,
    confidence_band,
    finite_difference_error,
    kernel_geometry,
    learn_weight,
    learning_price,
    max_step_for_price,
    probes_for_price,
    quantize_weight,
)
from turboquant_pro.observer import ObserverContract
from turboquant_pro.read_operators import AttentionAnalyticReadOperator

# ---------------------------------------------------------------------------
# The note's numbers
# ---------------------------------------------------------------------------


def _sin_arm_constants(beta):
    """The registration draft's consumer x + beta sin x, coordinate-wise, with
    Sigma = diag(2, 0.5) and G = diag(4, 1), as in the note's verify_t1_v4.py."""
    s2 = np.array([2.0, 0.5])
    q = 1 + 2 * beta * np.exp(-s2 / 2) + (beta**2 / 2) * (1 + np.exp(-2 * s2))
    nu_minus = 1 - min((1 - beta) ** 2 / q)
    nu_plus = max((1 + beta) ** 2 / q) - 1
    return nu_plus, nu_minus


def test_band_reproduces_the_notes_worked_constants():
    # "On the registration draft's consumer x+0.2 sin x, m=32, delta=0.1, the
    # constants are eps+ = 0.186 and eps- = 0.208 ... the exact roots t+-,
    # 0.177 and 0.189, may be used in their place." (after Lemma 1)
    nup, num = _sin_arm_constants(0.2)
    assert (round(nup, 3), round(num, 3)) == (0.233, 0.522)
    ep, em = confidence_band(32, 2, 0.1, nup, num)
    assert (round(ep, 3), round(em, 3)) == (0.186, 0.208)
    tp, tm = confidence_band(32, 2, 0.1, nup, num, exact_roots=True)
    assert (round(tp, 3), round(tm, 3)) == (0.177, 0.189)


def test_exact_roots_never_exceed_the_stated_band():
    rng = np.random.default_rng(1)
    for _ in range(500):
        m, p = int(rng.integers(1, 5000)), int(rng.integers(1, 300))
        nup, num, delta = rng.uniform(0, 5), rng.uniform(0, 1), rng.uniform(1e-6, 0.5)
        e = confidence_band(m, p, delta, nup, num)
        t = confidence_band(m, p, delta, nup, num, exact_roots=True)
        assert t[0] <= e[0] + 1e-15 and t[1] <= e[1] + 1e-15


def test_price_formula_and_its_edges():
    assert learning_price(0.0, 0.0, 7) == 0.0
    assert learning_price(0.1, 0.1, 4) == pytest.approx(2 * math.log2(1.1 / 0.9))
    assert learning_price(0.1, 1.0, 4) == math.inf
    with pytest.raises(ValueError):
        learning_price(-0.1, 0.1, 4)


@pytest.mark.parametrize(
    "p,nup,num,eta", [(1, 0.2, 1.0, 0.1), (8, 1.0, 1.0, 1.0), (64, 3.0, 0.3, 4.0)]
)
def test_probe_count_meets_the_target_and_minimal_is_least(p, nup, num, eta):
    def price(m):
        return learning_price(*confidence_band(m, p, 0.05, nup, num), p)

    out = probes_for_price(eta, p, 0.05, nup, num)
    assert price(out["corollary"]) <= eta
    assert out["minimal"] <= out["corollary"]
    assert price(out["minimal"]) <= eta
    if out["minimal"] > 1:
        assert price(out["minimal"] - 1) > eta


def test_probe_count_rejects_a_target_outside_the_corollary():
    with pytest.raises(ValueError):
        probes_for_price(5.0, 4, 0.05, 1.0)


# ---------------------------------------------------------------------------
# Coverage, with the self-test that a narrow band fails it
# ---------------------------------------------------------------------------


def _two_point_coverage(scale, nup, num, m, p=2, delta=0.1, reps=3000, seed=0):
    """The law at which the variance bound is attained (Bhatia-Davis): each
    coordinate of X is nu+ with probability nu-/(nu+ + nu-) and -nu- otherwise,
    P = Q^1/2 (I + X) Q^1/2 with X diagonal, as in the note's check C2."""
    rng = np.random.default_rng(seed)
    ep, em = confidence_band(m, p, delta, nup, num)
    ep, em = scale * ep, scale * em
    pr = num / (nup + num)
    Z = np.where(rng.random((reps, m, p)) < pr, nup, -num)
    mean = Z.mean(axis=1)
    return float(np.mean((mean.max(axis=1) <= ep) & (mean.min(axis=1) >= -em)))


@pytest.mark.parametrize("nup,num", [(0.25, 0.5), (0.9, 0.9), (2.0, 0.3)])
@pytest.mark.parametrize("m", [8, 32, 128])
def test_band_covers_at_least_one_minus_delta(nup, num, m):
    assert _two_point_coverage(1.0, nup, num, m) >= 0.9


def test_self_test_a_band_a_quarter_as_wide_is_caught():
    # I0: the coverage check must be able to fail. A band at 25% of Lemma 1's
    # width covers far less than 1 - delta at the symmetric two-point law.
    assert _two_point_coverage(0.25, 0.9, 0.9, 32) < 0.9


# ---------------------------------------------------------------------------
# learn_weight: certificates, refutation, invariance
# ---------------------------------------------------------------------------


def _bounded_metrics(m, p, nup, num, seed):
    """Samples Q^1/2 U(I + X)U^T Q^1/2 with X in [-num, nup]: the declared
    constants are true by construction."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    Q = A @ A.T + p * np.eye(p)
    w, V = np.linalg.eigh(Q)
    R = (V * np.sqrt(w)) @ V.T
    U, _ = np.linalg.qr(rng.standard_normal((p, p)))
    pr = num / (nup + num)
    X = np.where(rng.random((m, p)) < pr, nup, -num)
    return np.stack([R @ (U * (1 + x)) @ U.T @ R for x in X]), Q


def test_true_constants_are_rarely_refuted():
    refuted = 0
    for seed in range(200):
        P, _ = _bounded_metrics(64, 3, 0.8, 0.6, seed)
        rec = learn_weight(metrics=P, delta=0.1, nu_plus=0.8, nu_minus=0.6)
        refuted += rec.status == REFUTED
    # the test has level delta = 0.1; it is conservative, so far fewer
    assert refuted / 200 <= 0.1


def test_a_declared_constant_the_sample_contradicts_is_refuted():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((400, 6)) * np.array([5, 3, 1, 1, 0.5, 0.2])
    # q^T Q^-1 q averages p = 6 for a query sample, so nu_plus = 0.5 is false
    rec = learn_weight(q, kind="linear", delta=0.05, nu_plus=0.5)
    assert rec.status == REFUTED
    assert not math.isfinite(rec.price_bits)
    assert "wrong at level delta" in " ".join(rec.reasons)


def test_a_constant_sample_prices_at_the_bernstein_floor():
    P = np.stack([np.diag([3.0, 1.0, 0.5])] * 50)
    rec = learn_weight(metrics=P, nu_plus=0.0, nu_minus=0.0)
    assert rec.status == CERTIFIED
    assert rec.eps_plus == 0.0 and rec.eps_minus == 0.0
    assert rec.price_bits == 0.0
    assert np.allclose(rec.Q_hat, np.diag([3.0, 1.0, 0.5]))


def test_too_few_samples_give_no_certificate():
    rng = np.random.default_rng(4)
    q = rng.standard_normal((40, 16))  # full rank, but far too few for p = 16
    rec = learn_weight(q, kind="linear", nu_plus=20.0)
    assert rec.rank == 16
    assert rec.status == NO_CERTIFICATE
    assert rec.eps_minus >= 1.0
    assert "bound the weight from above only" in " ".join(rec.reasons)


def test_the_hint_does_not_promise_samples_a_large_difference_error_cannot_reach():
    q = np.random.default_rng(5).standard_normal((40, 4))
    rec = learn_weight(q, nu_plus=1.0, fd_error=1.2)
    assert rec.status == NO_CERTIFICATE
    assert "take a smaller step" in " ".join(rec.reasons)


def test_rotation_and_scale_change_nothing_that_is_relative():
    P, _ = _bounded_metrics(80, 4, 1.0, 0.7, seed=11)
    base = learn_weight(metrics=P, nu_plus=1.0, nu_minus=0.7)
    U, _ = np.linalg.qr(np.random.default_rng(12).standard_normal((4, 4)))
    rot = learn_weight(metrics=U @ P @ U.T * 37.0, nu_plus=1.0, nu_minus=0.7)
    assert rot.status == base.status
    for f in ("eps_plus", "eps_minus", "price_bits", "observed_plus", "observed_minus"):
        assert getattr(rot, f) == pytest.approx(getattr(base, f), rel=1e-9, abs=1e-12)
    assert np.allclose(rot.Q_hat, U @ base.Q_hat @ U.T * 37.0)


def test_factor_and_metric_forms_agree():
    rng = np.random.default_rng(13)
    A = rng.standard_normal((60, 3, 5))
    f = learn_weight(A, nu_plus=40.0)
    g = learn_weight(metrics=np.einsum("mki,mkj->mij", A, A), nu_plus=40.0)
    assert f.status == g.status and f.rank == g.rank
    assert np.allclose(f.Q_hat, g.Q_hat)
    assert f.observed_plus == pytest.approx(g.observed_plus, rel=1e-9)
    assert f.geometry == g.geometry


def test_a_lower_rank_weight_is_not_priced_even_when_its_kernel_looks_fixed():
    # Remark 14 extends only Theorem 11, under (A1) declared in advance; a kernel
    # that looks fixed in a sample does not supply it (an earlier draft priced it).
    P, _ = _bounded_metrics(200, 3, 0.5, 0.5, seed=21)
    E = np.zeros((200, 6, 6))
    E[:, :3, :3] = P  # read only the first three of six directions
    rec = learn_weight(metrics=E, nu_plus=0.5, nu_minus=0.5)
    assert rec.rank == 3 and rec.dim == 6 and not rec.geometry.kernel_moves
    assert rec.status == NO_CERTIFICATE
    assert not math.isfinite(rec.price_bits)
    assert "Remark 14" in " ".join(rec.reasons)


def test_a_lower_rank_weight_with_a_moving_kernel_is_not_priced():
    rng = np.random.default_rng(22)
    q = np.zeros((3000, 4))
    q[:, :2] = rng.standard_normal((3000, 2))  # rank-one samples in a plane of R^4
    rec = learn_weight(q, nu_plus=30.0)
    assert rec.rank == 2 and rec.geometry.kernel_moves
    assert rec.status == NO_CERTIFICATE


def test_an_ill_conditioned_weight_cut_by_the_rank_is_not_priced():
    # a full-rank weight whose smallest direction falls under rank_rtol: the
    # refutation test cannot see that direction, so there is no certificate
    rng = np.random.default_rng(24)
    q = rng.standard_normal((5000, 3)) * np.array([1.0, 1.0, 1e-6])
    rec = learn_weight(q, kind="linear", nu_plus=30.0)
    assert rec.rank == 2
    assert rec.status == NO_CERTIFICATE


def test_query_samples_price_a_linear_consumer_as_a_regret_bound():
    rng = np.random.default_rng(23)
    q = rng.standard_normal((4000, 3))
    rec = learn_weight(q, kind="linear", nu_plus=30.0)
    assert rec.status == CERTIFIED and math.isfinite(rec.price_bits)
    assert rec.geometry is None  # per-query ranks say nothing about a kernel
    assert "regret bound" in rec.price_scope and "Proposition 12" in rec.price_scope
    with pytest.raises(ValueError, match="kind='points'"):
        rec.gap(0.1, lipschitz=1.0, hessian_bound=0.0, third_bound=0.0)


def test_point_metrics_at_full_rank_price_a_learning_part():
    # Corollary 8 needs Q > 0, not a fixed kernel: x1 + a sin x2 reads one
    # turning direction per point, and the mean has full rank
    x2 = np.random.default_rng(25).standard_normal(4000)
    J = np.stack([np.ones_like(x2), 0.8 * np.cos(x2)], axis=1)[:, None, :]
    rec = learn_weight(J, kind="points", nu_plus=30.0)
    assert rec.rank == 2 and rec.geometry.kernel_moves
    assert rec.status == CERTIFIED
    assert "learning part" in rec.price_scope


def test_corollary8_gap_formula_and_refusal():
    J = np.random.default_rng(26).standard_normal((3000, 1, 2))
    rec = learn_weight(J, nu_plus=30.0)
    assert rec.status == CERTIFIED
    lam = float(np.linalg.eigvalsh(rec.Q_hat).min())
    M1, M2, M3, D = 2.0, 0.3, 0.2, 0.01
    g = rec.gap(D, lipschitz=M1, hessian_bound=M2, third_bound=M3)
    s = D / lam
    rho = (
        (1.75 * M2**2 + 2 * M1 * M3) * s**2
        + math.sqrt(5) / 2 * M2 * M3 * s**2.5
        + 5 / 12 * M3**2 * s**3
    )
    ep, em = confidence_band(rec.m, 2, rec.delta, 30.0, 1.0)
    assert g["rho_bar"] == pytest.approx(rho)
    assert g["design_distortion"] == pytest.approx((1 - em) * (D - rho))
    assert g["tau_bar_bits"] == pytest.approx(
        math.log2((1 + ep) / ((1 - em) * (1 - rho / D)))
    )
    assert g["tau_bar_bits"] > g["learning_part_bits"]
    # a linear consumer (M2 = M3 = 0) pays only the learning part
    lin = rec.gap(D, lipschitz=M1, hessian_bound=0.0, third_bound=0.0)
    assert lin["tau_bar_bits"] == pytest.approx(lin["learning_part_bits"])
    with pytest.raises(ValueError, match="certifies nothing"):
        rec.gap(5.0, lipschitz=M1, hessian_bound=M2, third_bound=M3)


def test_combined_band_is_the_largest_error_of_each_side():
    assert combine_bands([(0.1, 0.3), (0.2, 0.05), (0.0, 0.0)]) == (0.2, 0.3)


# ---------------------------------------------------------------------------
# The moving-kernel addendum: x1 + a sin x2
# ---------------------------------------------------------------------------


def test_kernel_geometry_of_the_addendums_example():
    # g(x) = x1 + a sin x2 reads one direction per point, (1, a cos x2), which
    # turns with x2: q = rank Q = 2 > r = 1, and with k = 1 output the fixed
    # weight pays (q - k)/2 = 1/2 bit per halving of the loss (Corollary 9).
    rng = np.random.default_rng(31)
    x2 = rng.standard_normal(500)
    J = np.stack([np.ones_like(x2), 0.8 * np.cos(x2)], axis=1)[:, None, :]
    g = kernel_geometry(J, output_dim=1)
    assert (g.q, g.point_rank_min, g.point_rank_max) == (2, 1, 1)
    assert g.kernel_moves and g.penalty_coefficient == 0.5


def test_kernel_geometry_of_a_linear_consumer_is_still():
    J = np.tile(np.array([[1.0, 0.8]]), (50, 1, 1))
    g = kernel_geometry(J, output_dim=1)
    assert (g.q, g.kernel_moves, g.penalty_coefficient) == (1, False, 0.0)


def test_no_coefficient_when_the_output_cannot_have_a_density():
    J = np.random.default_rng(32).standard_normal((40, 1, 3))
    assert kernel_geometry(J, output_dim=2).penalty_coefficient is None


# ---------------------------------------------------------------------------
# Proposition 2 and Proposition 4
# ---------------------------------------------------------------------------


def test_finite_differences_obey_proposition_2():
    a = 0.7

    def g(x):
        return np.array([x[0] + a * np.sin(x[1])])

    def jac(x):
        return np.array([[1.0, a * np.cos(x[1])]])

    M1, M2, M3 = math.sqrt(1 + a * a), a, a
    rng = np.random.default_rng(41)
    for h in (1e-1, 3e-2, 1e-2):
        worst = 0.0
        for _ in range(200):
            x = rng.standard_normal(2) * 2
            Jh = np.stack(
                [(g(x + h * e) - g(x - h * e)) / (2 * h) for e in np.eye(2)], axis=1
            )
            err = np.linalg.norm(Jh.T @ Jh - jac(x).T @ jac(x), 2)
            worst = max(worst, err)
        bound = finite_difference_error(2, h, lipschitz=M1, lam_min=1.0, third_bound=M3)
        assert worst <= bound
        looser = finite_difference_error(
            2, h, lipschitz=M1, lam_min=1.0, hessian_bound=M2
        )
        assert bound <= looser + 1e-15


def test_max_step_keeps_the_difference_error_within_a_fifth_of_eta_over_p():
    p, eta, M1, M3, lam = 8, 0.5, 2.0, 3.0, 0.2
    h = max_step_for_price(eta, p, lipschitz=M1, third_bound=M3, lam_min=lam)
    eps_h = finite_difference_error(p, h, lipschitz=M1, lam_min=lam, third_bound=M3)
    assert eps_h <= eta / (5 * p) * (1 + 1e-12)


def test_calibration_message_keeps_the_sandwich_and_counts_its_bits():
    rng = np.random.default_rng(51)
    p, lam, Lam, eq, ep, em = 5, 0.5, 4.0, 0.05, 0.2, 0.3
    w = rng.uniform((1 - em) * lam, (1 + ep) * Lam * 0.99, p)
    U, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Qh = (U * w) @ U.T
    msg = calibration_message(
        p, lam_min=lam, lam_max=Lam, eps_q=eq, eps_plus=ep, eps_minus=em
    )
    Qt = quantize_weight(Qh, msg)
    assert np.array_equal(Qt, Qt.T)
    Wi = np.linalg.inv(np.linalg.cholesky(Qh))
    rel = np.linalg.eigvalsh(Wi @ Qt @ Wi.T)
    assert rel.min() >= 1 - eq - 1e-12 and rel.max() <= 1 + eq + 1e-12
    assert msg.bits == p * (p + 1) // 2 * math.ceil(math.log2(2 * msg.levels + 1))
    with pytest.raises(ValueError):
        quantize_weight(Qh * 1e3, msg)


# ---------------------------------------------------------------------------
# attention_analytic exposes its samples without changing its operator
# ---------------------------------------------------------------------------


def _old_attention_operator(keys, q, n_probe, seed):
    """The operator exactly as written before point_factors existed."""
    d = keys.shape[1]
    z = (q @ keys.T) / np.sqrt(d)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    s_all = keys.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(s_all, size=min(n_probe, s_all), replace=False)
    M = np.zeros((d, d))
    for s in idx:
        e = np.zeros(s_all)
        e[s] = 1.0
        w = p * (e[None, :] - p[:, s : s + 1])
        a = (w**2).sum(axis=1)
        M += (q * a[:, None]).T @ q / d
    return 0.5 * (M + M.T)


def test_attention_point_factors_average_to_its_operator():
    rng = np.random.default_rng(61)
    keys, q = rng.standard_normal((90, 16)), rng.standard_normal((40, 16))
    op = AttentionAnalyticReadOperator(n_probe_keys=24, seed=3)
    M = op.operator(keys, queries=q)
    assert np.array_equal(M, _old_attention_operator(keys, q, 24, 3))
    F = op.point_factors(keys, queries=q)
    assert len(F) == 24
    assert np.allclose(sum(f.T @ f for f in F), M, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_price_from_factors_exit_codes(tmp_path, capsys):
    rng = np.random.default_rng(71)
    q = rng.standard_normal((20000, 4))
    path = tmp_path / "q.npy"
    np.save(path, q)
    out = tmp_path / "price.json"
    rc = main(
        [
            "observer",
            "price",
            "--factors",
            str(path),
            "--nu-plus",
            "40",
            "--linear",
            "--out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert rc == 0 and doc["status"] == CERTIFIED and doc["kind"] == "linear"
    assert doc["price_bits_per_vector"] > 0
    assert "regret bound" in doc["price_scope"]
    assert doc["invocation"]["argv"][1:3] == ["observer", "price"]
    rc = main(["observer", "price", "--factors", str(path), "--nu-plus", "0.5"])
    assert rc == 1
    assert REFUTED in capsys.readouterr().out


def test_cli_price_from_a_contract_splits_delta_and_combines(tmp_path):
    rng = np.random.default_rng(72)
    D = 8
    x = rng.standard_normal((500, D))
    queries = rng.standard_normal((5000, D))
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "q.npy", queries)
    contract = ObserverContract.from_dict(
        {
            "schema": "turboquant-pro/observer-contract",
            "profile": "tqp-observer/1",
            "observer": "mix",
            "target": "embedding",
            "consumers": [
                {"name": "search", "metric": "topk_cosine", "config": {"k": 10}},
                {
                    "name": "attn",
                    "metric": "read_operator",
                    "config": {"provider": "attention_analytic", "n_probe_keys": 64},
                },
                {
                    "name": "all",
                    "metric": "read_operator",
                    "config": {"provider": "identity"},
                },
            ],
        }
    )
    cpath = tmp_path / "c.json"
    cpath.write_text(json.dumps(contract.as_dict()))
    out = tmp_path / "o.json"
    rc = main(
        [
            "observer",
            "price",
            "--observer",
            str(cpath),
            "--artifact",
            str(tmp_path / "x.npy"),
            "--queries",
            str(tmp_path / "q.npy"),
            "--nu-plus",
            "search=40",
            "--nu-plus",
            "attn=200",
            "--out",
            str(out),
            "--format",
            "json",
        ]
    )
    doc = json.loads(out.read_text())
    kinds = {c["label"]: c["kind"] for c in doc["consumers"]}
    assert kinds == {"search": "linear", "attn": "points", "all": "exact"}
    assert doc["delta_per_sampled_consumer"] == pytest.approx(0.025)
    sampled = [c for c in doc["consumers"] if c["kind"] in ("linear", "points")]
    assert doc["band"]["eps_minus"] == max(c["band"]["eps_minus"] for c in sampled)
    # a nonlinear consumer is in the mix, so the price is a learning part only
    assert "learning part" in doc["price_scope"]
    assert rc == (0 if doc["status"] == CERTIFIED else 1)


def test_cli_price_needs_a_constant_for_every_sampled_consumer(tmp_path, capsys):
    x = np.random.default_rng(73).standard_normal((50, 4))
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "q.npy", x)
    contract = ObserverContract.from_dict(
        {
            "schema": "turboquant-pro/observer-contract",
            "profile": "tqp-observer/1",
            "observer": "s",
            "target": "embedding",
            "consumers": [
                {"name": "search", "metric": "topk_cosine", "config": {"k": 5}}
            ],
        }
    )
    cpath = tmp_path / "c.json"
    cpath.write_text(json.dumps(contract.as_dict()))
    rc = main(
        [
            "observer",
            "price",
            "--observer",
            str(cpath),
            "--artifact",
            str(tmp_path / "x.npy"),
            "--queries",
            str(tmp_path / "q.npy"),
            "--nu-plus",
            "other=3",
        ]
    )
    assert rc == 2
    assert "no --nu-plus for sampled consumer(s) search" in capsys.readouterr().err


def test_cli_probes(capsys):
    rc = main(
        ["observer", "probes", "--dim", "64", "--target-bits", "4", "--nu-plus", "3"]
    )
    assert rc == 0
    assert "Corollary 3(i)" in capsys.readouterr().out
    rc = main(
        [
            "observer",
            "probes",
            "--dim",
            "8",
            "--target-bits",
            "1",
            "--nu-plus",
            "1",
            "--finite-differences",
            "--lipschitz",
            "2",
            "--third-bound",
            "1",
            "--lam-min",
            "0.5",
            "--format",
            "json",
        ]
    )
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0 and doc["statement"] == "Corollary 3(ii)" and doc["max_step"] > 0
    assert doc["consumer_queries"] == 2 * 8 * doc["corollary"]
