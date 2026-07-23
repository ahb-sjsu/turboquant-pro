"""STRATA Phase 1 tests: area-map identity, ABSTAIN, min-over-strata, CLI.

Phase-1 gates from docs/STRATA_RFC.md §2.2: causes registered ·
min-over-strata semantics tested · ABSTAIN paths tested · schema frozen by
golden fixtures (test_strata_golden.py).
"""

import json

import numpy as np
import pytest

from turboquant_pro.anatomy import knn_exact
from turboquant_pro.cli import main
from turboquant_pro.strata import (
    STRATA_CAUSES,
    AreaMap,
    AreaMapProfile,
    _area_class,
    _cause,
    build_area_map,
    report_exit_code,
    require_same_map,
    stratified_anatomy,
    stratified_hub_differential,
)


def _strata_corpus(seed=0, d=32):
    """Four separated areas exercising the §1.4 classes + one thin stratum.

    ring A/B (interleaved labels -> transit ~0.5, 'plain') · ring D
    (isolated, count-symmetric -> 'stub') · shell E with planted local
    center (local hubs, no transit -> 'NSHA') · blob C (15 rows -> ABSTAIN).
    """
    rng = np.random.default_rng(seed)

    def ring(n, offset_axis, offset):
        phi = rng.random(n) * 2 * np.pi
        x = np.zeros((n, d), dtype=np.float32)
        x[:, 0], x[:, 1] = np.cos(phi), np.sin(phi)
        x += 0.01 * rng.standard_normal((n, d)).astype(np.float32)
        x[:, offset_axis] += offset
        return x

    ring_ab = ring(240, 4, 0.0)
    ring_d = ring(100, 5, 8.0)
    shell_e = rng.standard_normal((100, d)).astype(np.float32)
    shell_e /= np.linalg.norm(shell_e, axis=1, keepdims=True)
    center_e = np.zeros(d, dtype=np.float32)
    center_e[6] = 8.0
    shell_e += center_e
    shell_e[:3] = center_e + 0.1 * rng.standard_normal((3, d)).astype(np.float32)
    blob_c = (0.5 * rng.standard_normal((15, d))).astype(np.float32)
    blob_c[:, 7] += 8.0
    x = np.vstack([ring_ab, ring_d, shell_e, blob_c])
    labels = ["A", "B"] * 120 + ["D"] * 100 + ["E"] * 100 + ["C"] * 15
    return x, labels


def _map_of(x, labels, seed=0):
    return build_area_map(x, "by:test", seed=seed, labels=np.asarray(labels))


# --- registries and identity ----------------------------------------------


def test_cause_registry_is_closed():
    for c in STRATA_CAUSES:
        assert _cause(c) == c
    with pytest.raises(KeyError):
        _cause("novel_cause_nobody_registered")


def test_incomplete_area_map_profile_matches_nothing_including_itself():
    p = AreaMapProfile(algorithm_id="kmeans")  # everything else unknown
    assert not p.is_complete
    assert not p.compatible(p)
    x, labels = _strata_corpus()
    complete = _map_of(x, labels).profile
    assert complete.is_complete
    assert complete.compatible(complete)
    assert not complete.compatible(p)


def test_area_map_digest_content_addressed():
    x, labels = _strata_corpus()
    m1, m2 = _map_of(x, labels), _map_of(x, labels)
    assert m1.digest == m2.digest
    m3 = _map_of(x, labels, seed=1)
    assert m3.digest != m1.digest  # seed is configuration


def test_require_same_map_refuses():
    with pytest.raises(ValueError, match="area_map_mismatch"):
        require_same_map("a" * 64, "b" * 64)


def test_area_map_artifact_roundtrip_and_tamper_detection():
    x, labels = _strata_corpus()
    m = _map_of(x, labels)
    assert AreaMap.from_json(m.to_json()).digest == m.digest
    doc = json.loads(m.to_json())
    doc["profile"]["seed"] = 999  # hand-edit without recomputing the digest
    with pytest.raises(ValueError, match="tampered|digest"):
        AreaMap.from_json(json.dumps(doc))


# --- classification predicates (§1.4, pure function, all classes) ----------


def test_area_class_predicates():
    thr = {"tau": 0.5, "N": 50.0}
    assert _area_class(0.2, 0.7, 120.0, 10, thr) == "backbone"
    assert _area_class(3.0, 0.1, 90.0, 10, thr) == "NSHA"
    assert _area_class(3.0, 0.7, 40.0, 10, thr) == "hub"
    assert _area_class(0.2, 0.1, 20.0, 10, thr) == "stub"
    assert _area_class(0.5, 0.6, 30.0, 10, thr) == "plain"


# --- stratified anatomy ----------------------------------------------------


def test_stratified_anatomy_verdicts_and_classes():
    x, labels = _strata_corpus()
    report = stratified_anatomy(x, _map_of(x, labels), k=10, n_min=50, q_min=50)
    assert report["schema"] == "tqp-strata-report/1"
    areas = {a["id"]: a for a in report["areas"]}
    assert areas["C"]["verdict"] == "ABSTAIN"
    assert areas["C"]["cause"] == "stratum_insufficient_n"
    # ABSTAIN rows carry no measurements: uncertain => no verdict, no numbers.
    assert "count_skew" not in areas["C"]
    assert areas["A"]["class"] == "plain" and areas["B"]["class"] == "plain"
    assert areas["A"]["tau_mean"] > 0.3  # interleaved labels: real transit
    assert areas["D"]["class"] == "stub"
    assert areas["D"]["tau_mean"] < 0.05
    assert areas["E"]["class"] == "NSHA"  # local hubs, no global transit
    assert areas["E"]["count_skew"] > 1.0
    assert report["summary"]["n_abstain"] == 1
    # Thresholds are FIELDS of the report, not constants.
    assert report["thresholds"]["n_min"] == 50


def test_stratified_anatomy_only_abstain_exit_semantics():
    x, labels = _strata_corpus()
    report = stratified_anatomy(x, _map_of(x, labels), k=10, n_min=10_000, q_min=10_000)
    assert report["summary"]["only_abstain"]
    assert report_exit_code(report) == 3
    assert report_exit_code(report, abstain_fails=True) == 1


# --- stratified hubdiff: min over strata -----------------------------------


def test_stratified_hubdiff_min_over_strata_catches_one_bad_stratum():
    """Aggregate recall stays green while ONE stratum collapses — the point."""
    rng = np.random.default_rng(3)
    x, labels = _strata_corpus()
    _, exact = knn_exact(x, x, 10, exclude_self=True)
    counts = np.bincount(exact.ravel(), minlength=len(x))
    anti = counts <= np.quantile(counts, 0.10)
    approx = exact.copy()
    qlab = np.asarray(labels)
    hit = 0
    for i in range(len(x)):  # corrupt anti-hub queries in stratum D only
        if qlab[i] == "D" and anti[exact[i, 0]]:
            approx[i] = rng.integers(0, len(x), size=10)
            hit += 1
    assert hit >= 3
    report = stratified_hub_differential(
        exact,
        approx,
        len(x),
        _map_of(x, labels),
        labels,
        k=10,
        min_anti_recall=0.9,
        n_min=50,
        q_min=50,
    )
    areas = {a["id"]: a for a in report["areas"]}
    assert areas["D"]["verdict"] == "fail"
    assert areas["D"]["cause"] == "stratum_anti_hub_gap"
    assert areas["A"]["verdict"] == "pass"
    assert areas["C"]["verdict"] == "ABSTAIN"
    # The aggregate over all queries would have sailed through the same gate.
    agg = np.mean(
        [len(set(e) & set(a)) / 10 for e, a in zip(exact.tolist(), approx.tolist())]
    )
    assert agg > 0.9
    assert report["summary"]["n_failed"] == 1
    assert report_exit_code(report) == 1


def test_stratified_hubdiff_passes_when_clean():
    x, labels = _strata_corpus()
    _, exact = knn_exact(x, x, 10, exclude_self=True)
    report = stratified_hub_differential(
        exact,
        exact,
        len(x),
        _map_of(x, labels),
        labels,
        k=10,
        min_anti_recall=0.9,
        n_min=50,
        q_min=50,
    )
    assert report["summary"]["n_failed"] == 0
    assert report_exit_code(report) == 0
    m = report["summary"]["min_over_strata_anti_hub_recall"]
    assert m == 1.0


# --- CLI -------------------------------------------------------------------


def test_cli_stratified_anatomy_and_map_reuse(tmp_path):
    x, labels = _strata_corpus()
    npy = tmp_path / "x.npy"
    np.save(npy, x)
    lab = tmp_path / "labels.txt"
    lab.write_text("\n".join(labels), encoding="utf-8")
    out = tmp_path / "strata.json"
    mp = tmp_path / "map.json"
    rc = main(
        [
            "anatomy",
            "--npy",
            str(npy),
            "--by",
            "language",
            "--labels",
            str(lab),
            "--k",
            "10",
            "--min-stratum-n",
            "50",
            "--min-stratum-q",
            "50",
            "--out",
            str(out),
            "--save-map",
            str(mp),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "tqp-strata-report/1"
    assert mp.exists()
    # Reuse the saved artifact: same digest, same report identity.
    out2 = tmp_path / "strata2.json"
    rc = main(
        [
            "anatomy",
            "--npy",
            str(npy),
            "--strata",
            str(mp),
            "--k",
            "10",
            "--min-stratum-n",
            "50",
            "--min-stratum-q",
            "50",
            "--out",
            str(out2),
        ]
    )
    assert rc == 0
    r2 = json.loads(out2.read_text(encoding="utf-8"))
    assert (
        r2["provenance"]["area_map_digest"] == report["provenance"]["area_map_digest"]
    )


def test_cli_stratified_anatomy_kmeans_and_abstain_exit(tmp_path):
    x, _ = _strata_corpus()
    npy = tmp_path / "x.npy"
    np.save(npy, x)
    rc = main(
        [
            "anatomy",
            "--npy",
            str(npy),
            "--strata",
            "kmeans:4",
            "--seed",
            "7",
            "--k",
            "10",
            "--min-stratum-n",
            "50",
            "--min-stratum-q",
            "50",
        ]
    )
    assert rc == 0
    # Impossible evidence floor: every stratum ABSTAINs -> exit 3; the CI
    # mapping --abstain-fails makes that a failure.
    args = [
        "anatomy",
        "--npy",
        str(npy),
        "--strata",
        "kmeans:4",
        "--seed",
        "7",
        "--k",
        "10",
        "--min-stratum-n",
        "99999",
    ]
    assert main(args) == 3
    assert main(args + ["--abstain-fails"]) == 1


def test_cli_stratified_hubdiff_gate(tmp_path):
    rng = np.random.default_rng(5)
    x, labels = _strata_corpus()
    _, exact = knn_exact(x, x, 10, exclude_self=True)
    counts = np.bincount(exact.ravel(), minlength=len(x))
    anti = counts <= np.quantile(counts, 0.10)
    approx = exact.copy()
    qlab = np.asarray(labels)
    for i in range(len(x)):
        if qlab[i] == "D" and anti[exact[i, 0]]:
            approx[i] = rng.integers(0, len(x), size=10)
    e, a = tmp_path / "e.npy", tmp_path / "a.npy"
    np.save(e, exact)
    np.save(a, approx)
    lab = tmp_path / "labels.txt"
    lab.write_text("\n".join(labels), encoding="utf-8")
    base_args = [
        "hubdiff",
        "--exact",
        str(e),
        "--approx",
        str(a),
        "--n-base",
        str(len(x)),
        "--labels",
        str(lab),
        "--min-stratum-q",
        "50",
        "--min-stratum-n",
        "50",
    ]
    assert main(base_args + ["--min-anti-recall", "0.9"]) == 1  # stratum D
    assert main(base_args) == 0  # ungated: report only


# --- the relational surface: hubness as digest-guarded SQL ------------------


def _duck_setup():
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from turboquant_pro import duckdb_ext as tq

    x, labels = _strata_corpus()
    amap = _map_of(x, labels)
    _, idx = knn_exact(x, x, 10, exclude_self=True)
    con = duckdb.connect()
    tq.attach_strata(con, idx, amap, query_labels=list(labels))
    return tq, con, x, labels, amap, idx


def test_sql_hub_census_is_in_degree():
    tq, con, x, _, _, idx = _duck_setup()
    top_id, top_nk = tq.hub_census(con).fetchall()[0]
    counts = np.bincount(idx.ravel(), minlength=len(x))
    assert top_nk == counts.max()
    assert counts[top_id] == counts.max()


def test_sql_transit_by_area_matches_report():
    tq, con, x, labels, amap, _ = _duck_setup()
    rows = {r[0]: r[1] for r in tq.transit_by_area(con).fetchall()}
    report = stratified_anatomy(x, amap, k=10, n_min=50, q_min=50)
    areas = {a["id"]: a for a in report["areas"]}
    # SQL transit is per-EDGE, the report's tau_mean is per-ROW: same
    # phenomenon, agreeing on the extremes.
    assert rows["A"] > 0.3 and rows["B"] > 0.3  # interleaved: real transit
    assert rows["D"] < 0.05 and areas["D"]["tau_mean"] < 0.05
    assert rows["E"] < 0.15


def test_sql_refuses_cross_map_joins():
    tq, con, x, labels, _, _ = _duck_setup()
    other = _map_of(x, labels, seed=1)  # different configuration => digest
    import pyarrow as pa

    con.register(
        "strata",
        pa.table(
            {
                "id": pa.array(np.arange(len(x), dtype=np.int64)),
                "area": pa.array(list(labels), type=pa.string()),
                "area_map_digest": pa.array([other.digest] * len(x)),
            }
        ),
    )
    with pytest.raises(ValueError, match="area_map_mismatch"):
        tq.transit_by_area(con)


def test_sql_strata_gate_returns_failing_rows():
    tq, con, x, labels, amap, idx = _duck_setup()
    rng = np.random.default_rng(3)
    counts = np.bincount(idx.ravel(), minlength=len(x))
    anti = counts <= np.quantile(counts, 0.10)
    approx = idx.copy()
    qlab = np.asarray(labels)
    for i in range(len(x)):
        if qlab[i] == "D" and anti[idx[i, 0]]:
            approx[i] = rng.integers(0, len(x), size=10)
    report = stratified_hub_differential(
        idx,
        approx,
        len(x),
        amap,
        labels,
        k=10,
        min_anti_recall=0.9,
        n_min=50,
        q_min=50,
    )
    failing = tq.strata_gate(con, report, min_anti_recall=0.9).fetchall()
    assert [r[0] for r in failing] == ["D"]  # rows returned => exit 1
