# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Leakage to a holder (rate-leakage-distortion series, geometric-observation):
paper I's printed numbers (Section X), an optimality check by feasible
perturbation with a self-test that it rejects a design that is not optimal,
the costs against a Monte Carlo of the Gaussian test channel, the blocks of
paper III (pooling never cheaper) and paper II (a sequence against its
spectrum, Kolmogorov's AR(1) rate), and the CLI."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from turboquant_pro.cli import main
from turboquant_pro.leakage import (
    NO_TRADEOFF,
    TRADEOFF,
    Block,
    Holder,
    described_components,
    fit_holder,
    frontier,
    frontier_point,
    linear_leakage,
    pooling_gain,
    sequence_gain,
    spectral_blocks,
    tradeoff_report,
    water_fill,
)

# ---------------------------------------------------------------------------
# Paper I, Section X: the printed numbers
# ---------------------------------------------------------------------------


def _scalar_instance(rho2, tau2):
    """T = (Y, V), unit variances, correlation rho; W = 1 on Y; S = V + U."""
    r = math.sqrt(rho2)
    Sigma = np.array([[1.0, r], [r, 1.0]])
    return Block.from_model(
        np.diag([1.0, 0.0]), Sigma, Holder(np.array([[0.0, 1.0]]), np.array([[tau2]]))
    )


def test_scalar_instance_excesses():
    # "At (rho^2, tau^2, D) = (0.75, 0.5, 0.3) the rate-optimal channel pays
    # 0.0349 bits of excess leakage and the leakage-optimal channel 0.0400 bits
    # of excess rate."
    rep = tradeoff_report(frontier([_scalar_instance(0.75, 0.5)], D=0.3))
    assert rep["verdict"] == TRADEOFF and rep["solver_ok"]
    assert rep["excess_leakage_bits"] == pytest.approx(0.0349, abs=6e-5)
    assert rep["rate_premium_bits"] == pytest.approx(0.0400, abs=6e-5)


@pytest.mark.parametrize("rho2,tau2,L", [(0.6, 0.2, 1.1880), (0.75, 0.5, 1.2105)])
def test_scalar_instance_least_leakage(rho2, tau2, L):
    # "(rho^2, tau^2) = (3/5, 1/5) and (3/4, 1/2) have L = 1.1880 and 1.2105
    # bits at D = 0.1."
    pt = frontier_point([_scalar_instance(rho2, tau2)], alpha=0.0, D=0.1)
    assert pt.leakage == pytest.approx(L, abs=6e-5)


def _coupled():
    """Sigma_Y = diag(1, 0.4), W = I, Var V = 1, Cov(Y, V) = 0.6 u, u at 50
    degrees, S = V + U with Var U = 0.05, so D_max = 1.4."""
    u = np.array([math.cos(math.radians(50)), math.sin(math.radians(50))])
    Sigma = np.zeros((3, 3))
    Sigma[:2, :2] = np.diag([1.0, 0.4])
    Sigma[:2, 2] = Sigma[2, :2] = 0.6 * u
    Sigma[2, 2] = 1.0
    holder = Holder(np.array([[0.0, 0.0, 1.0]]), np.array([[0.05]]))
    return Block.from_model(np.diag([1.0, 1.0, 0.0]), Sigma, holder), Sigma


def _y_reduction_angle(block, Sigma, pt):
    Delta = block.root @ pt.X[0] @ block.root
    w, V = np.linalg.eigh(Sigma[:2, :2] - Delta[:2, :2])
    v = V[:, np.argmax(w)]
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


@pytest.mark.parametrize(
    "D,deg", [(1.38, 47.48), (1.3, 46.76), (1.2, 45.07), (1.1, 41.15)]
)
def test_coupled_instance_leakage_optimal_direction(D, deg):
    block, Sigma = _coupled()
    pt = frontier_point([block], alpha=0.0, D=D)
    assert _y_reduction_angle(block, Sigma, pt) == pytest.approx(deg, abs=0.006)


@pytest.mark.parametrize("D", [0.8, 1.0, 1.3])
def test_coupled_instance_rate_optimal_reads_the_first_axis(D):
    block, Sigma = _coupled()
    pt = frontier_point([block], alpha=1.0, D=D)
    a = _y_reduction_angle(block, Sigma, pt)
    assert min(a, 180.0 - a) < 1e-6


def _component(sigma2, rho2, tau2):
    c = math.sqrt(rho2 * sigma2)
    return Block.from_model(
        np.diag([1.0, 0.0]),
        np.array([[sigma2, c], [c, 1.0]]),
        Holder(np.array([[0.0, 1.0]]), np.array([[tau2]])),
    )


def test_independent_components_one_level():
    # "At D = 1.4 the rate-optimal description pays R = L = 0.1610 bits on
    # component 1 and the leakage-optimal description pays L = 0.0197 and
    # R = 0.3272 bits on component 2."
    blocks = [_component(1.0, 0.0, 1.0), _component(0.6, 0.9, 0.05)]
    r = frontier_point(blocks, alpha=1.0, D=1.4)
    assert r.rate_blocks == pytest.approx([0.1610, 0.0], abs=6e-5)
    assert r.leakage_blocks == pytest.approx([0.1610, 0.0], abs=6e-5)
    lk = frontier_point(blocks, alpha=0.0, D=1.4)
    assert lk.rate_blocks == pytest.approx([0.0, 0.3272], abs=6e-5)
    assert lk.leakage_blocks == pytest.approx([0.0, 0.0197], abs=6e-5)


def test_blocks_equal_the_block_diagonal_problem():
    blocks = [_component(1.0, 0.0, 1.0), _component(0.6, 0.9, 0.05)]
    Sigma = np.zeros((4, 4))
    Sigma[:2, :2] = blocks[0].root @ blocks[0].root
    Sigma[2:, 2:] = blocks[1].root @ blocks[1].root
    H = np.zeros((2, 4))
    H[0, 1] = H[1, 3] = 1.0
    joint = Block.from_model(
        np.diag([1.0, 0.0, 1.0, 0.0]), Sigma, Holder(H, np.diag([1.0, 0.05]))
    )
    for a in (1.0, 0.4, 0.0):
        sep = frontier_point(blocks, alpha=a, D=1.4)
        one = frontier_point([joint], alpha=a, D=1.4)
        assert sep.rate == pytest.approx(one.rate, abs=1e-8)
        assert sep.leakage == pytest.approx(one.leakage, abs=1e-8)


# ---------------------------------------------------------------------------
# Optimality, by an independent test, and its self-test
# ---------------------------------------------------------------------------


def _random_block(p, r, seed, q_rank=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    Sigma = A @ A.T / p + 0.2 * np.eye(p)
    B = rng.standard_normal((q_rank or p, p))
    Q = B.T @ B / p
    H = rng.standard_normal((r, p))
    return Block.from_model(Q, Sigma, Holder(H, 0.3 * np.eye(r)))


def _improvable(block, X, alpha, tries=4000, seed=0):
    """Is there a feasible direction (same distortion, 0 < X <= I) that lowers
    alpha r + (1 - alpha) l? Independent of the solver: it only evaluates costs."""
    rng = np.random.default_rng(seed)
    p = block.dim

    def f(Y):
        return alpha * block.rate(Y) + (1 - alpha) * block.leakage(Y)

    base = f(X)
    Qt = block.Qt
    for _ in range(tries):
        E = rng.standard_normal((p, p))
        E = 0.5 * (E + E.T)
        E -= np.trace(Qt @ E) / np.trace(Qt @ Qt) * Qt
        for t in (1e-3, 1e-4):
            Y = X + t * E
            ev = np.linalg.eigvalsh(Y)
            if ev.min() <= 0 or ev.max() > 1:
                continue
            if f(Y) < base - 1e-12:
                return True
    return False


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("seed", [1, 2])
def test_frontier_point_cannot_be_improved(alpha, seed):
    block = _random_block(4, 2, seed)
    pt = frontier_point([block], alpha=alpha, D=0.4 * block.d_max)
    assert pt.converged and pt.monotone and pt.residual < 1e-8
    assert pt.distortion == pytest.approx(0.4 * block.d_max, rel=1e-10)
    assert not _improvable(block, pt.X[0], alpha)


def test_self_test_the_rate_optimal_design_is_improvable_for_leakage():
    # I0: the check must be able to fail. Where there is a trade-off, the
    # consumer-only design is not leakage-optimal, and the check must find a
    # feasible direction that leaks less.
    block = _random_block(4, 2, 1)
    D = 0.4 * block.d_max
    rep = tradeoff_report(frontier([block], D=D))
    assert rep["verdict"] == TRADEOFF and rep["excess_leakage_bits"] > 1e-3
    rate_opt = frontier_point([block], alpha=1.0, D=D)
    assert _improvable(block, rate_opt.X[0], 0.0)


def test_rate_optimal_is_classical_reverse_water_filling():
    block = _random_block(6, 3, 5)
    D = 0.3 * block.d_max
    lam = np.linalg.eigvalsh(block.Qt)
    lo, hi = 0.0, float(lam.max())
    for _ in range(200):
        th = 0.5 * (lo + hi)
        lo, hi = (th, hi) if np.minimum(th, lam).sum() < D else (lo, th)
    rate = 0.5 * np.log2(np.maximum(lam / th, 1.0)).sum()
    assert frontier_point([block], alpha=1.0, D=D).rate == pytest.approx(rate, abs=1e-9)


def test_a_blind_holder_leaves_no_tradeoff():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((4, 4))
    block = Block.from_model(np.eye(4), A @ A.T + np.eye(4))
    rep = tradeoff_report(frontier([block], D=0.5 * block.d_max))
    assert rep["verdict"] == NO_TRADEOFF
    pt = frontier_point([block], alpha=0.0, D=0.5 * block.d_max)
    assert pt.leakage == pytest.approx(pt.rate, abs=1e-12)


def test_rotation_and_holder_scaling_change_nothing():
    rng = np.random.default_rng(7)
    p = 4
    A = rng.standard_normal((p, p))
    Sigma, Q = A @ A.T + np.eye(p), np.diag([2.0, 1.0, 0.5, 0.0])
    H, N = rng.standard_normal((2, p)), np.diag([0.3, 0.7])
    rot, _ = np.linalg.qr(rng.standard_normal((p, p)))
    base = Block.from_model(Q, Sigma, Holder(H, N))
    turned = Block.from_model(
        rot @ Q @ rot.T, rot @ Sigma @ rot.T, Holder(3.0 * H @ rot.T, 9.0 * N)
    )
    D = 0.35 * base.d_max
    for a in (1.0, 0.5, 0.0):
        x, y = frontier_point([base], alpha=a, D=D), frontier_point(
            [turned], alpha=a, D=D
        )
        assert y.rate == pytest.approx(x.rate, abs=1e-8)
        assert y.leakage == pytest.approx(x.leakage, abs=1e-8)


def test_water_fill_caps_at_one():
    X = water_fill(np.diag([4.0, 1.0, 0.25, 0.0]))
    assert np.allclose(X, np.diag([0.25, 1.0, 1.0, 1.0]))


def test_described_components_match_the_design():
    block = _random_block(5, 2, 8, q_rank=2)
    pt = frontier_point([block], alpha=0.5, D=0.3 * block.d_max)
    comp = described_components(pt, 0, block)
    # paper I, Proposition 6: at most rank Q components are described
    assert comp["components"] <= 2
    assert np.sum(comp["bits"]) == pytest.approx(pt.rate, abs=1e-9)


# ---------------------------------------------------------------------------
# The costs against a Monte Carlo of the Gaussian test channel; the holder fit
# ---------------------------------------------------------------------------


def test_costs_match_a_simulated_test_channel_and_the_proxy():
    rng = np.random.default_rng(9)
    block = _random_block(3, 1, 9)
    pt = frontier_point([block], alpha=0.5, D=0.4 * block.d_max)
    R = block.root
    Sigma, Delta = R @ R, R @ pt.X[0] @ R
    Si = np.linalg.inv(Sigma)
    n = 400_000
    T = rng.standard_normal((n, 3)) @ R
    H = block.Ht @ np.linalg.inv(R)
    S = T @ H.T + rng.standard_normal((n, 1)) * math.sqrt(block.noise[0, 0])
    Vc = Delta - Delta @ Si @ Delta
    w, U = np.linalg.eigh(0.5 * (Vc + Vc.T))
    V = rng.standard_normal((n, 3)) @ (U * np.sqrt(np.clip(w, 0, None))).T
    T_hat = T @ (np.eye(3) - Delta @ Si).T + V
    proxy = linear_leakage(T, S, T_hat)
    assert proxy["rate_proxy_bits"] == pytest.approx(pt.rate, abs=0.01)
    assert proxy["leakage_proxy_bits"] == pytest.approx(pt.leakage, abs=0.01)


def test_fit_holder_recovers_the_model():
    rng = np.random.default_rng(10)
    A = rng.standard_normal((4, 4))
    Sigma = A @ A.T + np.eye(4)
    H = rng.standard_normal((2, 4))
    N = np.array([[0.5, 0.1], [0.1, 0.3]])
    T = rng.multivariate_normal(np.zeros(4), Sigma, 300_000)
    S = T @ H.T + rng.multivariate_normal(np.zeros(2), N, 300_000)
    holder, Sig = fit_holder(T, S)
    assert np.allclose(holder.H, H, atol=0.02)
    assert np.allclose(holder.noise, N, atol=0.01)
    assert np.allclose(Sig, Sigma, rtol=0.02, atol=0.02)


def test_a_noiseless_holder_is_refused():
    with pytest.raises(ValueError, match="without noise"):
        Holder(np.eye(2), np.diag([1.0, 0.0]))


# ---------------------------------------------------------------------------
# Paper III: pooling never costs less (Corollary 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
def test_pooling_gain_is_never_negative(alpha):
    rng = np.random.default_rng(11)
    p = 3
    H = rng.standard_normal((1, p))
    holder = Holder(H, np.array([[0.4]]))
    Q = np.diag([1.0, 0.6, 0.2])
    for _trial in range(6):
        Ks = []
        for _ in range(3):
            A = rng.standard_normal((p, p))
            Ks.append(A @ A.T / p + 0.1 * np.eye(p))
        lam = rng.dirichlet(np.ones(3))
        d_max = float(np.trace(Q @ sum(li * K for li, K in zip(lam, Ks))))
        rep = pooling_gain(Ks, lam, Q, D=0.4 * d_max, alpha=alpha, holder=holder)
        assert rep["solver_ok"]
        assert rep["gain_bits"] >= -1e-9


def test_identical_strata_gain_nothing():
    rng = np.random.default_rng(12)
    A = rng.standard_normal((3, 3))
    K = A @ A.T + np.eye(3)
    rep = pooling_gain([K, K, K], [0.2, 0.3, 0.5], np.eye(3), D=0.5 * np.trace(K))
    assert rep["gain_bits"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Paper II: a sequence against its spectrum
# ---------------------------------------------------------------------------


def _ar1(a, n, seed, p=1):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, p))
    x = np.empty_like(e)
    x[0] = e[0] / math.sqrt(1 - a * a)
    for t in range(1, n):
        x[t] = a * x[t - 1] + e[t]
    return x


def test_sequence_gain_of_an_ar1_is_kolmogorovs():
    # For D below the spectrum's minimum, R_seq = 1/2 log2(sigma_e^2 / D) and
    # R_alone = 1/2 log2(sigma_x^2 / D): the gain is 1/2 log2(1 / (1 - a^2)).
    a = 0.8
    x = _ar1(a, 200_000, 13)
    rep = sequence_gain(x, np.eye(1), D=0.1, nfft=64)
    assert rep["solver_ok"]
    assert rep["gain_bits"] == pytest.approx(0.5 * math.log2(1 / (1 - a * a)), abs=0.03)
    assert rep["memoryless_rate_bits"] == pytest.approx(
        0.5 * math.log2(1 / ((1 - a * a) * 0.1)), abs=0.03
    )


def test_white_noise_gains_nothing_from_the_sequence():
    x = np.random.default_rng(14).standard_normal((100_000, 3))
    rep = sequence_gain(x, np.eye(3), D=0.6, nfft=32)
    assert abs(rep["gain_bits"]) < 0.02


def test_spectral_holder_fit_recovers_a_filter():
    # S_t = h0 x_t + h1 x_{t-1} + u_t has H(w) = h0 + h1 exp(-i w) on the FFT grid
    x = _ar1(0.5, 200_000, 15)
    u = np.random.default_rng(16).standard_normal((x.shape[0], 1)) * 0.3
    s = 1.5 * x + 0.7 * np.vstack([x[:1] * 0, x[:-1]]) + u
    blocks, c = spectral_blocks(x, np.eye(1), 32, holder_sequence=s)
    w = 2 * np.pi * np.fft.fftfreq(32)
    for k in (0, 3, 8, 16):
        H_k = blocks[k].Ht[0, 0] / blocks[k].root[0, 0]
        assert abs(H_k - (1.5 + 0.7 * np.exp(-1j * w[k]))) < 0.03


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _paired(tmp_path, n=20_000, seed=17):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((n, 4)) @ np.diag([2.0, 1.0, 0.7, 0.3])
    S = T[:, 1:2] * 0.9 + 0.4 * rng.standard_normal((n, 1))
    np.save(tmp_path / "t.npy", T)
    np.save(tmp_path / "s.npy", S)
    return T, S


def test_cli_plan_leakage(tmp_path):
    _paired(tmp_path)
    out = tmp_path / "lk.json"
    rc = main(
        [
            "plan",
            "leakage",
            "--artifact",
            str(tmp_path / "t.npy"),
            "--holder",
            str(tmp_path / "s.npy"),
            "--relative-distortion",
            "0.3",
            "--out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert rc == 0 and doc["solver_ok"]
    assert doc["verdict"] in (TRADEOFF, NO_TRADEOFF)
    assert len(doc["frontier"]) == 5
    assert doc["frontier"][0]["rate_bits"] <= doc["frontier"][-1]["rate_bits"] + 1e-9
    assert (
        doc["frontier"][0]["leakage_bits"] >= doc["frontier"][-1]["leakage_bits"] - 1e-9
    )


def test_cli_plan_leakage_pca_adds_the_unread_distortion(tmp_path):
    _paired(tmp_path)
    out = tmp_path / "lk.json"
    rc = main(
        [
            "plan",
            "leakage",
            "--artifact",
            str(tmp_path / "t.npy"),
            "--holder",
            str(tmp_path / "s.npy"),
            "--relative-distortion",
            "0.5",
            "--pca",
            "2",
            "--out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert rc == 0
    unread = doc["distortion"]["unread_by_pca"]
    assert unread == pytest.approx(0.7**2 + 0.3**2, rel=0.05)
    assert doc["D"] == pytest.approx(doc["distortion"]["requested"] - unread)


def test_cli_plan_leakage_refuses_unpaired_rows(tmp_path, capsys):
    _paired(tmp_path)
    np.save(tmp_path / "s2.npy", np.zeros((10, 1)))
    rc = main(
        [
            "plan",
            "leakage",
            "--artifact",
            str(tmp_path / "t.npy"),
            "--holder",
            str(tmp_path / "s2.npy"),
            "--relative-distortion",
            "0.3",
        ]
    )
    assert rc == 2 and "paired row by row" in capsys.readouterr().err


def test_cli_plan_strata_and_sequence(tmp_path):
    rng = np.random.default_rng(18)
    a = rng.standard_normal((3000, 3)) * np.array([3.0, 0.5, 0.5])
    b = rng.standard_normal((3000, 3)) * np.array([0.5, 3.0, 0.5])
    np.save(tmp_path / "x.npy", np.vstack([a, b]))
    np.save(tmp_path / "l.npy", np.repeat(["a", "b"], 3000))
    out = tmp_path / "st.json"
    rc = main(
        [
            "plan",
            "strata",
            "--artifact",
            str(tmp_path / "x.npy"),
            "--labels",
            str(tmp_path / "l.npy"),
            "--relative-distortion",
            "0.2",
            "--out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert rc == 0 and doc["rate"]["gain_bits"] > 0.1
    np.save(tmp_path / "seq.npy", _ar1(0.9, 20_000, 19, p=2))
    out2 = tmp_path / "sq.json"
    rc = main(
        [
            "plan",
            "sequence",
            "--artifact",
            str(tmp_path / "seq.npy"),
            "--nfft",
            "32",
            "--relative-distortion",
            "0.05",
            "--out",
            str(out2),
        ]
    )
    doc2 = json.loads(out2.read_text())
    assert rc == 0 and doc2["gain_bits"] > 0.5


def test_cli_plan_strata_budget_is_that_of_the_centred_strata(tmp_path):
    # Strata that differ only in their means: Corollary 6's program sees two
    # identical centred covariances, so the saving is zero, the no-description
    # distortion is tr(Q K_bar) = 3, and the means appear only as tr(Q B) = 25.
    rng = np.random.default_rng(20)
    a = rng.standard_normal((4000, 3)) + np.array([5.0, 0.0, 0.0])
    b = rng.standard_normal((4000, 3)) - np.array([5.0, 0.0, 0.0])
    np.save(tmp_path / "x.npy", np.vstack([a, b]))
    np.save(tmp_path / "l.npy", np.repeat([0, 1], 4000))
    out = tmp_path / "st.json"
    rc = main(
        [
            "plan",
            "strata",
            "--artifact",
            str(tmp_path / "x.npy"),
            "--labels",
            str(tmp_path / "l.npy"),
            "--relative-distortion",
            "0.5",
            "--out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert rc == 0
    assert doc["distortion"]["no_description"] == pytest.approx(3.0, rel=0.05)
    assert doc["between_means"]["trace_Q_B"] == pytest.approx(25.0, rel=0.02)
    assert abs(doc["rate"]["gain_bits"]) < 0.01
