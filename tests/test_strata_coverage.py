"""Certificates expire, phase 2 (issue #177): strata coverage and the monitor.

The strata sketch finds data in regions the certificate did not cover, which
the moment check misses; the monitor re-runs the validity checks on its window
and alerts once on the move into STALE."""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro.cli import main
from turboquant_pro.monitor import QualityMonitor
from turboquant_pro.strata import build_area_map
from turboquant_pro.validity import (
    INCONCLUSIVE,
    STALE,
    VALID,
    check_validity,
    strata_coverage,
    strata_sketch,
    validity_section,
)

D = 16
CENTRES = np.eye(4, D) * 6.0  # four areas, well apart


def _mix(rng, n, new=0.0):
    """Rows from the four certified areas; a fraction ``new`` from a fifth."""
    k = rng.integers(0, 4, n)
    x = CENTRES[k] + rng.standard_normal((n, D))
    m = rng.random(n) < new
    x[m] = np.full(D, 3.0) + rng.standard_normal((int(m.sum()), D))
    return x.astype(np.float32), k


def _labels(k):
    return [f"a{v}" for v in k]


def _cert(x, k):
    return {"validity": validity_section(sample=x, labels=_labels(k))}


def test_exchangeable_data_stays_within_the_baseline():
    rng = np.random.default_rng(0)
    x, k = _mix(rng, 4000)
    sk = strata_sketch(x, labels=_labels(k))
    assert sk["rows"] == 2000 and all(a["covered"] for a in sk["areas"])
    fr = [strata_coverage(sk, _mix(rng, 2000)[0])["uncovered"] for _ in range(20)]
    # conformal radius: the expected uncovered fraction is at most 1 - q = 0.01
    assert np.mean(fr) <= sk["baseline_uncovered"] + 0.003
    r = check_validity(_cert(x, k), data=_mix(rng, 2000)[0])
    assert r["status"] == VALID and r["checks"]["strata_coverage"]["status"] == "ok"


def test_a_small_new_region_is_stale_where_the_moments_see_nothing():
    rng = np.random.default_rng(1)
    x, k = _mix(rng, 4000)
    r = check_validity(_cert(x, k), data=_mix(rng, 3000, new=0.10)[0])
    assert r["checks"]["data_coverage"]["status"] == "ok"  # the moments pass
    sc = r["checks"]["strata_coverage"]
    assert sc["status"] == "FAIL" and sc["interval"][0] > sc["limit"]
    assert sc["from"][0]["why"] == "beyond radius"
    assert r["status"] == STALE and r["action"] == "RECERTIFY"
    assert "did not cover" in r["reason"]


def test_mass_in_an_area_seen_thinly_is_uncovered():
    rng = np.random.default_rng(2)
    x, k = _mix(rng, 4000)
    keep = (k != 3) | (rng.random(len(k)) < 0.05)  # area a3 barely sampled
    sk = strata_sketch(x[keep], labels=_labels(k[keep]))
    thin = [a for a in sk["areas"] if not a["covered"]]
    assert [a["name"] for a in thin] == ["a3"] and thin[0]["radius"] is None
    y = CENTRES[3] + rng.standard_normal((500, D))
    sc = strata_coverage(sk, y)
    assert sc["status"] == "FAIL" and sc["from"][0] == {
        "area": "a3",
        "rows": 500,
        "why": "thin at issue",
    }


def test_too_few_rows_abstain_and_nothing_is_promised():
    rng = np.random.default_rng(3)
    x, k = _mix(rng, 4000)
    r = check_validity(_cert(x, k), data=_mix(rng, 12)[0])
    assert r["checks"]["strata_coverage"]["status"] == "abstain"
    assert r["checks"]["data_coverage"]["status"] == "abstain"  # noise floor
    assert r["status"] == INCONCLUSIVE and r["applicable"] is True
    assert r["reason"] == "too few rows to decide data and strata coverage"


def test_the_moment_check_decides_in_noise_units():
    """Twelve rows from the certified distribution: the plug-in divergence
    alone reaches the bar (sampling noise, not drift), so the check abstains;
    the floor it reports is the one measured on repeated exchangeable draws."""
    from turboquant_pro.validity import coverage_divergence, coverage_noise_floor

    rng = np.random.default_rng(8)
    g = rng.standard_normal((20000, D))
    cs = validity_section(sample=g)["coverage_sketch"]
    draws = [rng.standard_normal((50, D)) for _ in range(400)]
    measured = np.mean([coverage_divergence(cs, y) for y in draws])
    predicted = np.mean([coverage_noise_floor(cs, y) for y in draws])
    assert predicted == pytest.approx(measured, rel=0.15)
    r = check_validity({"validity": {"coverage_sketch": cs}}, data=g[:12])
    assert r["checks"]["data_coverage"]["status"] == "abstain"
    assert r["status"] == INCONCLUSIVE


def test_capabilities_certify_only_a_valid_status():
    from turboquant_pro.capabilities import CERTIFIED, CONDITIONAL, capabilities
    from turboquant_pro.cli import _sha256_array

    rng = np.random.default_rng(4)
    x, k = _mix(rng, 4000)
    doc = _cert(x, k)
    doc.update(passed=True, inputs={"original": {"sha256": _sha256_array(x)}})
    ok = capabilities(x, [("c.json", doc)], data=_mix(rng, 2000)[0])
    assert ok.items[0].status == CERTIFIED
    few = capabilities(x, [("c.json", doc)], data=_mix(rng, 12)[0])
    assert few.items[0].status == CONDITIONAL
    assert few.items[0].reason.startswith(INCONCLUSIVE)


def test_an_area_map_of_another_corpus_is_refused():
    rng = np.random.default_rng(5)
    x, _ = _mix(rng, 400)
    other, _ = _mix(rng, 400)
    amap = build_area_map(other, "kmeans:4", seed=0)
    with pytest.raises(ValueError, match="another corpus"):
        strata_sketch(x, amap)
    assert strata_sketch(other, amap)["area_map_sha256"] == amap.digest


def test_certify_strata_and_verify_find_the_new_region(tmp_path, capsys):
    jsonschema = pytest.importorskip("jsonschema")
    from turboquant_pro.schemas import load_schema

    rng = np.random.default_rng(6)
    x, _ = _mix(rng, 2000)
    po, pr, cert = tmp_path / "o.npy", tmp_path / "r.npy", tmp_path / "c.json"
    np.save(po, x)
    np.save(pr, (x + 0.01 * rng.standard_normal(x.shape)).astype(np.float32))
    argv = ["certify", "--original", str(po), "--reconstructed", str(pr)]
    argv += ["--anchors", "16", "--strata", "kmeans:4", "--out", str(cert)]
    assert main(argv) in (0, 1)
    doc = json.loads(cert.read_text(encoding="utf-8"))
    jsonschema.validate(doc, load_schema("rank_certificate.schema.json"))
    assert len(doc["validity"]["strata_sketch"]["areas"]) == 4
    capsys.readouterr()
    data = tmp_path / "d.npy"
    np.save(data, _mix(rng, 3000)[0])
    assert main(["verify", str(cert), "--data", str(data), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["validity"]["status"] == VALID
    np.save(data, _mix(rng, 3000, new=0.15)[0])
    assert main(["verify", str(cert), "--data", str(data)]) == 1
    out = capsys.readouterr().out
    assert "strata coverage" in out and "beyond radius" in out and "STALE" in out


def test_the_monitor_alerts_once_when_the_certificate_goes_stale():
    rng = np.random.default_rng(7)
    x, k = _mix(rng, 4000)
    seen = []
    mon = QualityMonitor(
        quality_floor=0.0,
        certificate=_cert(x, k),
        validity_window=1000,
        validity_every=250,
        alert_callback=seen.append,
    )
    for row in _mix(rng, 1000)[0]:
        mon.record(row, row)
    m = mon.metrics_dict()
    assert m["turboquant_certificate_valid"] == 1
    assert np.isnan(m["turboquant_certificate_operator_overlap"])  # not measured
    assert 0 <= m["turboquant_certificate_uncovered_fraction"] < 0.05
    for row in _mix(rng, 2000, new=0.3)[0]:
        mon.record(row, row)
    m = mon.metrics_dict()
    assert m["turboquant_certificate_stale"] == 1
    stale = [a for a in seen if "certificate" in a]
    assert len(stale) == 1 and "did not cover" in stale[0]["reasons"][0]


def test_the_monitor_refuses_a_certificate_with_nothing_to_check():
    with pytest.raises(ValueError, match="no validity section"):
        QualityMonitor(certificate={"passed": True})
    assert "turboquant_certificate_valid" not in QualityMonitor().metrics_dict()
