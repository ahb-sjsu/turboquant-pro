"""Tests for the hub anatomy vector and the hubdiff differential oracle."""

import numpy as np
import pytest

from turboquant_pro.anatomy import hub_anatomy, hub_differential, knn_exact
from turboquant_pro.cli import main


def _corpus(n=600, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_knn_exact_matches_bruteforce():
    base, q = _corpus(200), _corpus(40, seed=1)
    d, idx = knn_exact(base, q, 5)
    full = np.linalg.norm(q[:, None, :] - base[None, :, :], axis=2)
    ref = np.argsort(full, axis=1)[:, :5]
    assert (idx == ref).all()
    assert np.allclose(d, np.take_along_axis(full, ref, 1), atol=1e-4)


def test_knn_exact_excludes_self():
    base = _corpus(100)
    _, idx = knn_exact(base, base, 3, exclude_self=True)
    assert not any(idx[i, 0] == i for i in range(len(base)))


def test_hub_anatomy_detects_planted_centrality_hub():
    rng = np.random.default_rng(2)
    base = _corpus(500)
    mean_dir = base.mean(0)
    mean_dir /= np.linalg.norm(mean_dir)
    # Plant super-hubs at the centroid direction: classic centrality anatomy.
    base[:3] = mean_dir + 0.01 * rng.standard_normal((3, base.shape[1])).astype(
        np.float32
    )
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    doc = hub_anatomy(base, k=10)
    assert doc["battery"] == "corpus->corpus"
    # The anatomy IDENTIFIES the planted rows: they dominate the count tail
    # (mean count is k=10) and read as CENTRAL — the identification claim,
    # robust across dimensions, rather than an absolute-count magic number.
    assert doc["count_max"] >= 2 * doc["k"]
    hub_c, all_c = doc["hub_vs_all_median_centrality"]
    assert hub_c > all_c


def test_hub_differential_perfect_agreement():
    # Enough queries that the bottom count decile is non-empty among NN-1 rows
    # (with few queries most rows have count 0 and no NN can be an anti-hub).
    base, q = _corpus(300), _corpus(240, seed=3)
    _, idx = knn_exact(base, q, 10)
    doc = hub_differential(idx, idx, len(base), k=10)
    assert doc["recall_at_k"] == 1.0
    assert doc["anti_hub_recall"] == 1.0
    assert doc["hub_rank_corr"] == pytest.approx(1.0)
    assert doc["hub_set_jaccard"] == 1.0


def test_hub_differential_catches_anti_hub_collapse():
    """Aggregate recall stays high while anti-hub queries fail — the oracle's job."""
    rng = np.random.default_rng(4)
    base, q = _corpus(400), _corpus(200, seed=5)
    _, exact = knn_exact(base, q, 10)
    counts = np.bincount(exact.ravel(), minlength=len(base))
    anti_rows = counts <= np.quantile(counts, 0.10)
    approx = exact.copy()
    hit = 0
    for i in range(len(q)):  # corrupt only queries whose true NN is an anti-hub
        if anti_rows[exact[i, 0]]:
            approx[i] = rng.integers(0, len(base), size=10)
            hit += 1
    assert hit >= 5
    doc = hub_differential(exact, approx, len(base), k=10)
    assert doc["recall_at_k"] > 0.75  # the mean still looks tolerable
    assert doc["anti_hub_recall"] < 0.30  # the oracle sees the collapse
    assert doc["recall_at_k"] - doc["anti_hub_recall"] > 0.4


def test_cli_anatomy_and_hubdiff(tmp_path):
    base = _corpus(250)
    recon = (base + 0.02 * np.random.default_rng(6).standard_normal(base.shape)).astype(
        np.float32
    )
    b, r = tmp_path / "b.npy", tmp_path / "r.npy"
    np.save(b, base)
    np.save(r, recon)
    out = tmp_path / "anatomy.json"
    assert main(["anatomy", "--npy", str(b), "--k", "5", "--out", str(out)]) == 0
    assert out.exists()
    out2 = tmp_path / "hubdiff.json"
    rc = main(
        [
            "hubdiff",
            "--original",
            str(b),
            "--reconstructed",
            str(r),
            "--k",
            "5",
            "--out",
            str(out2),
        ]
    )
    assert rc == 0
    assert out2.exists()


def test_cli_hubdiff_gate_fails_on_bad_anti_recall(tmp_path):
    rng = np.random.default_rng(7)
    base, q = _corpus(300), _corpus(120, seed=8)
    _, exact = knn_exact(base, q, 10)
    counts = np.bincount(exact.ravel(), minlength=len(base))
    anti = counts <= np.quantile(counts, 0.10)
    approx = exact.copy()
    for i in range(len(q)):
        if anti[exact[i, 0]]:
            approx[i] = rng.integers(0, len(base), size=10)
    e, a = tmp_path / "e.npy", tmp_path / "a.npy"
    np.save(e, exact)
    np.save(a, approx)
    args = [
        "hubdiff",
        "--exact",
        str(e),
        "--approx",
        str(a),
        "--n-base",
        "300",
        "--min-anti-recall",
        "0.9",
    ]
    assert main(args) == 1  # gate fires
    assert main(args[:-2]) == 0  # ungated run reports and exits 0
