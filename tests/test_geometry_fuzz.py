"""Tests for the geometry profile and the retrieval fuzzer (issue #162)."""

import json
from pathlib import Path

import numpy as np
import pytest

from turboquant_pro.cli import main
from turboquant_pro.fuzz import (
    CLASSES,
    MUTATORS,
    BundleError,
    Target,
    _kendall_tau_strict,
    evaluate,
    replay_bundle,
    run_campaign,
)
from turboquant_pro.geometry import fit_whitener, geometry_profile, reverse_knn_counts


def _corpus(n=300, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    # Anisotropic, so whitening is not the identity.
    x = rng.standard_normal((n, dim)) * np.linspace(3.0, 0.2, dim)
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ------------------------------------------------------------------ geometry


def test_whitener_roundtrip_and_unit_covariance():
    x = _corpus(2000, 8)
    w = fit_whitener(x, shrinkage=0.0)
    z = w.whiten(x)
    assert np.allclose(w.unwhiten(z), x, atol=1e-6)
    assert np.allclose(np.cov(z.T, bias=True), np.eye(8), atol=1e-6)
    assert np.allclose(w.radius(x), np.linalg.norm(z, axis=1))


def test_ledoit_wolf_intensity_is_in_unit_interval_and_shrinks_more_when_n_is_small():
    big = fit_whitener(_corpus(4000, 16))
    small = fit_whitener(_corpus(20, 16))
    assert 0.0 <= big.shrinkage <= 1.0 and 0.0 <= small.shrinkage <= 1.0
    assert small.shrinkage > big.shrinkage


def test_singular_covariance_stays_finite():
    x = _corpus(10, 32)  # n < d: S has rank <= 9
    w = fit_whitener(x)  # Ledoit-Wolf shrinkage makes it full rank
    assert w.shrinkage > 0 and w.n_floored == 0
    assert np.isfinite(w.radius(x)).all()
    w0 = fit_whitener(x, shrinkage=0.0)  # no shrinkage: the floor decides
    assert w0.n_floored > 0
    assert np.isfinite(w0.radius(x)).all()


def test_zero_variance_and_bad_shrinkage_are_errors():
    with pytest.raises(ValueError):
        fit_whitener(np.ones((10, 4)))
    with pytest.raises(ValueError):
        fit_whitener(_corpus(50, 4), shrinkage=1.5)


def test_reverse_knn_counts_exact_and_sampled():
    x = _corpus(400, 8)
    c, est = reverse_knn_counts(x, 5)
    assert est["estimator"] == "exact" and c.sum() == 400 * 5
    cs, est_s = reverse_knn_counts(x, 5, sample=200, seed=1)
    assert est_s["estimator"] == "sampled"
    assert np.isclose(cs.sum(), 400 * 5)  # scaled by n / sample
    assert np.corrcoef(c, cs)[0, 1] > 0.5


def test_geometry_profile_report():
    prof, w, counts = geometry_profile(_corpus(500, 16), k=10)
    assert prof["schema"] == "turboquant-pro/geometry-profile"
    assert prof["n_base"] == 500 and len(counts) == 500
    for key in ("central_hubs", "peripheral_hubs", "central_anti_hubs"):
        assert isinstance(prof[key], list)
    assert set(prof["region_counts"]) == {
        "central_hub",
        "central_anti_hub",
        "peripheral_hub",
        "peripheral_anti_hub",
    }
    assert prof["regularization"]["whitener_sha256"] == w.digest()
    assert prof["spectrum"]["effective_rank"] > 1.0
    assert 0.0 <= prof["hub_stability_k_2k"] <= 1.0


# ------------------------------------------------------------------ mutators


def test_shell_holds_the_whitened_radius_and_radial_scales_it():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(16)
    zs, ps = MUTATORS["shell"](z, np.random.default_rng(4))
    assert np.isclose(np.linalg.norm(zs), np.linalg.norm(z), rtol=1e-12)
    assert 0.0 <= ps["angle_rad"] <= np.pi / 4
    zr, pr = MUTATORS["radial"](z, np.random.default_rng(5))
    assert np.allclose(zr, pr["alpha"] * z)
    assert 0.25 <= pr["alpha"] <= 4.0


def test_kendall_tau_strict():
    a = np.arange(10.0)
    assert _kendall_tau_strict(a, a) == 1.0
    assert _kendall_tau_strict(a, -a) == -1.0


# ------------------------------------------------------------------ oracle


def test_certificate_is_never_violated_on_honest_data():
    """The strict floor is a theorem on each case's distances.

    A violation would be a bug, never a hard query.
    """
    corpus = _corpus(300, 16)
    t = Target(corpus, bits=2)
    rng = np.random.default_rng(7)
    for q in rng.standard_normal((25, 16)).astype(np.float32):
        o = evaluate(t, q, 10)
        assert o["certificate"]["applicable"]
        assert not o["certificate"]["violated"]
        assert o["classification"] in CLASSES
        assert o["classification"] != "certified_violation"


def test_inner_product_has_no_certificate():
    t = Target(_corpus(200, 16), bits=4, metric="inner_product")
    o = evaluate(t, _corpus(1, 16, seed=9)[0], 5)
    assert o["certificate"] == {"applicable": False}


# ------------------------------------------------------------------ campaign + replay


def _campaign(tmp, budget=40, seed=11, **kw):
    corpus = _corpus(300, 16)
    queries = _corpus(8, 16, seed=1)
    return run_campaign(
        corpus, queries, tmp, k=10, budget=budget, seed=seed, bits=2, **kw
    )


def test_campaign_writes_valid_bundles_and_replays_them(tmp_path):
    run = _campaign(tmp_path / "run")
    cases = sorted((tmp_path / "run" / "cases").iterdir())
    assert run["retained"] == len(cases) >= 1
    assert sum(run["by_class"].values()) == run["retained"]
    for c in cases:
        for f in (
            "case.json",
            "query.npy",
            "expected_exact.json",
            "observed_tqp.json",
            "environment.json",
            "checksums.txt",
            "corpus.npy",
            "replay.sh",
        ):
            assert (c / f).is_file(), f
        doc = json.loads((c / "case.json").read_text())
        assert doc["codec"]["bits"] == 2 and doc["campaign_seed"] == 11
        assert doc["classification"] in CLASSES
    for c in cases[:5]:
        r = replay_bundle(c)
        assert r["reproduced"], r["differences"]


def test_campaign_is_deterministic(tmp_path):
    a = _campaign(tmp_path / "a")
    b = _campaign(tmp_path / "b")
    for key in ("retained", "cells_covered", "by_class", "worst_recall_at_k"):
        assert a[key] == b[key]
    for ca, cb in zip(
        sorted((tmp_path / "a" / "cases").iterdir()),
        sorted((tmp_path / "b" / "cases").iterdir()),
    ):
        ja = json.loads((ca / "case.json").read_text())
        jb = json.loads((cb / "case.json").read_text())
        ja["corpus"].pop("path"), jb["corpus"].pop("path")
        assert ja == jb
        assert np.array_equal(np.load(ca / "query.npy"), np.load(cb / "query.npy"))


def test_campaign_refuses_a_nonempty_directory_and_unknown_mutators(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "keep").write_text("x")
    with pytest.raises(FileExistsError):
        _campaign(d)
    with pytest.raises(ValueError):
        _campaign(tmp_path / "y", mutators=("nope",))


def _one_case(tmp_path):
    _campaign(tmp_path / "run", budget=10)
    return sorted((tmp_path / "run" / "cases").iterdir())[0]


def _refresh_checksum(case: Path, name: str) -> None:
    import hashlib

    lines = (case / "checksums.txt").read_text().splitlines()
    out = []
    for line in lines:
        digest, fname = line.split()
        if fname == name:
            digest = hashlib.sha256((case / name).read_bytes()).hexdigest()
        out.append(f"{digest}  {fname}")
    (case / "checksums.txt").write_text("\n".join(out) + "\n")


def test_altered_file_fails_closed(tmp_path):
    case = _one_case(tmp_path)
    q = np.load(case / "query.npy")
    np.save(case / "query.npy", q + 1e-3)
    with pytest.raises(BundleError, match="checksum"):
        replay_bundle(case)


def test_missing_file_fails_closed(tmp_path):
    case = _one_case(tmp_path)
    (case / "expected_exact.json").unlink()
    with pytest.raises(BundleError, match="missing"):
        replay_bundle(case)


def test_missing_checksums_fails_closed(tmp_path):
    case = _one_case(tmp_path)
    (case / "checksums.txt").unlink()
    with pytest.raises(BundleError):
        replay_bundle(case)


def test_unknown_schema_fails_closed_even_with_a_valid_checksum(tmp_path):
    case = _one_case(tmp_path)
    doc = json.loads((case / "case.json").read_text())
    doc["schema_version"] = 99
    (case / "case.json").write_text(json.dumps(doc))
    _refresh_checksum(case, "case.json")
    with pytest.raises(BundleError, match="schema"):
        replay_bundle(case)


def test_a_different_corpus_fails_closed(tmp_path):
    case = _one_case(tmp_path)
    (case / "corpus.npy").unlink()
    lines = [
        ln
        for ln in (case / "checksums.txt").read_text().splitlines()
        if not ln.endswith("corpus.npy")
    ]
    (case / "checksums.txt").write_text("\n".join(lines) + "\n")
    with pytest.raises(BundleError, match="not embedded"):
        replay_bundle(case)
    with pytest.raises(BundleError, match="sha256"):
        replay_bundle(case, _corpus(300, 16, seed=5))
    assert replay_bundle(case, _corpus(300, 16))["reproduced"]


def test_minimized_synthetic_regression_case(tmp_path):
    """A minimized case: one far-out query on a 120-row corpus at 2 bits.

    The query sits beyond every corpus row in the whitened space (the
    peripheral region the fuzzer exists to reach); the case must replay to the
    recorded recall and classification.
    """
    corpus = _corpus(120, 8, seed=21)
    w = fit_whitener(corpus)
    q = w.unwhiten(6.0 * w.whiten(corpus[:1]))[0].astype(np.float32)
    run = run_campaign(
        corpus, q[None, :], tmp_path / "min", k=5, budget=0, seed=1, bits=2
    )
    assert run["retained"] == 1
    case = tmp_path / "min" / "cases" / "case-000000"
    doc = json.loads((case / "case.json").read_text())
    assert doc["coverage_inputs"]["mahalanobis_quantile"] == 1.0
    assert doc["coverage_signature"][0] == 5  # beyond the corpus
    r = replay_bundle(case)
    assert r["reproduced"], r["differences"]
    assert r["replayed"]["classification"] == doc["classification"]


# ------------------------------------------------------------------ CLI


def test_cli_geometry_profile_json(tmp_path, capsys):
    np.save(tmp_path / "c.npy", _corpus(200, 8))
    assert (
        main(
            [
                "geometry",
                "profile",
                "--embeddings",
                str(tmp_path / "c.npy"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "turboquant-pro/geometry-profile"
    from turboquant_pro.schemas import validate

    assert validate(doc)["kind"] == "turboquant-pro/geometry-profile"
    assert "invocation" in doc


def test_cli_fuzz_retrieval_and_replay(tmp_path, capsys):
    np.save(tmp_path / "c.npy", _corpus(300, 16))
    np.save(tmp_path / "q.npy", _corpus(6, 16, seed=1))
    out = tmp_path / "run"
    rc = main(
        [
            "fuzz",
            "retrieval",
            "--embeddings",
            str(tmp_path / "c.npy"),
            "--queries",
            str(tmp_path / "q.npy"),
            "--out",
            str(out),
            "--budget",
            "15",
            "--bits",
            "2",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    run = json.loads(capsys.readouterr().out)
    assert run["schema"] == "turboquant-pro/fuzz-run"
    case = sorted((out / "cases").iterdir())[0]
    assert main(["fuzz", "replay", str(case), "--format", "json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["reproduced"] is True
    from turboquant_pro.schemas import validate

    assert validate(run)["kind"] == "turboquant-pro/fuzz-run"
    assert validate(rep)["kind"] == "turboquant-pro/fuzz-replay"
    case_doc = json.loads((case / "case.json").read_text())
    assert validate(case_doc)["kind"] == "turboquant-pro/fuzz-case"
    np.save(case / "query.npy", np.load(case / "query.npy") * 2)
    assert main(["fuzz", "replay", str(case)]) == 2
