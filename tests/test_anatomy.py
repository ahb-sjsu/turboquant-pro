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
    # The auditable curve: 10 deciles, all perfect under identity.
    curve = doc["recall_by_count_decile"]
    assert len(curve) == 10
    assert all(c == 1.0 for c in curve if np.isfinite(c))
    # Provenance: the meter meters itself.
    assert doc["mode"] == "id_arrays"
    assert doc["corr_method"] == "spearman"


def test_anatomy_provenance_and_size_stable_fields():
    base = _corpus(400)
    doc = hub_anatomy(base, k=10)
    assert doc["estimator"] == "exact_knn_on_given_vectors"
    assert doc["corr_method"] == "spearman"
    # The stride is part of the fingerprint: it declares its own estimator
    # (s1 = exact content hash; sN = sampled, probabilistic sensitivity).
    assert doc["dataset_fingerprint"].startswith("400x32:float32:s1:")
    assert doc["query_fingerprint"] is None
    assert 0.0 <= doc["robin_hood_index"] <= 1.0
    assert 0.0 <= doc["frac_above_2k"] <= 1.0
    assert doc["mechanism"] in ("centrality", "density", "mixed", "unclear")
    assert isinstance(doc["prescription"], str) and doc["prescription"]
    # Fingerprint is deterministic and content-sensitive (exact at this size).
    assert doc["dataset_fingerprint"] == hub_anatomy(base, k=10)["dataset_fingerprint"]
    other = hub_anatomy(_corpus(400, seed=9), k=10)
    assert other["dataset_fingerprint"] != doc["dataset_fingerprint"]


def test_fingerprint_layout_normalized_and_stride_declared():
    from turboquant_pro.anatomy import _fingerprint

    x = _corpus(64, 16)
    # Layout-normalizing: a non-contiguous view and its contiguous copy hash
    # identically — the hash is over logical content, not memory bytes.
    view = np.asfortranarray(x)
    assert not view.flags["C_CONTIGUOUS"]
    assert _fingerprint(view) == _fingerprint(x)
    strided_view = x[::2]
    assert _fingerprint(strided_view) == _fingerprint(strided_view.copy())
    # Small arrays are hashed exactly and say so.
    assert ":s1:" in _fingerprint(x)
    # Above ~1 MiB the sample stride appears in the string: the fingerprint
    # is itself a claim, and a sampled claim must declare it.
    big = np.zeros((1024, 512), dtype=np.float32)  # 2 MiB -> stride 2
    assert ":s2:" in _fingerprint(big)


# --- The classifier's confusion matrix -----------------------------------
# One fixture per cell of the prescription pad. A classifier tested on one
# cell has no confusion matrix; these four killed two earlier designs
# (corpus-wide corr thresholds; symmetric percentile thresholds) before this
# one survived. Constructions:
#  - centrality: unit shell + points planted at the CENTER — the center
#    beats every shell row's nearest neighbour, the pure pipeline super-hub.
#  - density: off-center unit shells, each with points planted at its LOCAL
#    center, plus a plain shell at the origin occupying the central band.
#    Local centers are "too close to everything" locally (small d_k, huge
#    counts) while sitting mid-pack in corpus centrality. At global scope
#    that is the density signature — global anatomy cannot split local
#    centrality from density (that is STRATA's per-area job), and
#    CSLS/mutual proximity is the correct global remedy for both.
#  - mixed: the density corpus with the origin shell ALSO center-planted.
#  - unclear: a constant-density ring — negligible hubness, so mechanism
#    attribution must abstain (materiality guard), not guess.


def _shell(rng, n, d, center, planted=0, planted_spread=0.1):
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x += center
    if planted:
        x[:planted] = center + planted_spread * rng.standard_normal(
            (planted, d)
        ).astype(np.float32)
    return x


def _fx_centrality(seed, d=32):
    rng = np.random.default_rng(seed)
    x = _shell(rng, 500, d, np.zeros(d, dtype=np.float32))
    x[:5] = 0.05 * rng.standard_normal((5, d)).astype(np.float32)
    return x


def _fx_local_center_shells(seed, d=32, origin_planted=0, origin_n=154):
    rng = np.random.default_rng(seed)
    parts = []
    for j, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        c = np.zeros(d, dtype=np.float32)
        c[j] = 2.0 * sgn
        parts.append(_shell(rng, 123, d, c, planted=3))
    parts.append(
        _shell(rng, origin_n, d, np.zeros(d, dtype=np.float32), planted=origin_planted)
    )
    return np.vstack(parts)


def _fx_unclear(seed, d=16):
    rng = np.random.default_rng(seed)
    phi = rng.random(500) * 2.0 * np.pi
    x = np.zeros((500, d), dtype=np.float32)
    x[:, 0] = np.cos(phi)
    x[:, 1] = np.sin(phi)
    return x + 0.01 * rng.standard_normal((500, d)).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_confusion_matrix_centrality(seed):
    doc = hub_anatomy(_fx_centrality(seed), k=10)
    assert doc["mechanism"] == "centrality"
    assert "centering" in doc["prescription"]
    assert doc["hub_frac_central"] == 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_confusion_matrix_density(seed):
    doc = hub_anatomy(_fx_local_center_shells(seed), k=10)
    assert doc["mechanism"] == "density"
    assert "CSLS" in doc["prescription"]
    assert doc["hub_frac_dense_noncentral"] >= 0.75
    assert doc["hub_frac_central"] == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_confusion_matrix_mixed(seed):
    # The origin shell is enlarged so its center-planted counts strictly
    # dominate the local-center counts: the hub set then deterministically
    # contains both species instead of slicing their overlap arbitrarily.
    doc = hub_anatomy(
        _fx_local_center_shells(seed, origin_planted=4, origin_n=250), k=10
    )
    assert doc["mechanism"] == "mixed"
    assert doc["hub_frac_central"] >= 0.25
    assert doc["hub_frac_dense_noncentral"] >= 0.25


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_confusion_matrix_unclear_abstains(seed):
    doc = hub_anatomy(_fx_unclear(seed), k=10)
    assert doc["mechanism"] == "unclear"
    # The abstain is the materiality guard, and the prescription says so.
    assert doc["count_max"] < 25
    assert "no material hub tail" in doc["prescription"]


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
