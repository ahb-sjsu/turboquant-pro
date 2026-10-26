# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Certified capability discovery (issue #178): what an artifact is currently
certified to be used for, from the certificates that are about it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro.capabilities import (
    CERTIFIED,
    CONDITIONAL,
    NOT_CERTIFIED,
    capabilities,
)
from turboquant_pro.cli import main
from turboquant_pro.observer import ObserverContract

D = 16


def _pair(seed=0, n=96):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D)).astype(np.float32)
    y = (x + 0.01 * rng.standard_normal((n, D))).astype(np.float32)
    return x, y


def _contract(name="svc", provider="identity"):
    return ObserverContract.from_dict(
        {
            "schema": "turboquant-pro/observer-contract",
            "profile": "tqp-observer/1",
            "observer": name,
            "target": "embedding",
            "consumers": [
                {
                    "name": "retrieval",
                    "metric": "read_operator",
                    "config": {"provider": provider},
                }
            ],
        }
    )


def _certify(tmp_path, x, y, name="cert.json", extra=()):
    po, pr = tmp_path / f"o_{name}.npy", tmp_path / f"r_{name}.npy"
    np.save(po, x)
    np.save(pr, y)
    out = tmp_path / name
    argv = [
        "certify",
        "--original",
        str(po),
        "--reconstructed",
        str(pr),
        "--anchors",
        "16",
        "--out",
        str(out),
    ]
    argv.extend(extra)
    main(argv)
    return str(out), json.loads(out.read_text(encoding="utf-8"))


# ---- the three lists ----------------------------------------------------------


def test_a_certificate_about_this_artifact_with_a_sample_is_certified(tmp_path, capsys):
    x, y = _pair()
    c = _contract()
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c.as_dict()), encoding="utf-8")
    path, doc = _certify(tmp_path, x, y, extra=["--observer", str(cp)])
    capsys.readouterr()
    r = capabilities(x, [(path, doc)], contracts=[c], data=x)
    assert [i.status for i in r.items] == [CERTIFIED]
    assert r.items[0].name.startswith("svc: retrieval")
    assert "still applies" in r.items[0].reason
    assert r.artifact_sha256 == doc["inputs"]["original"]["sha256"]


def test_without_a_sample_nothing_is_promised(tmp_path, capsys):
    x, y = _pair()
    path, doc = _certify(tmp_path, x, y)
    capsys.readouterr()
    r = capabilities(x, [(path, doc)])
    assert r.by_status(CERTIFIED) == []
    assert len(r.by_status(CONDITIONAL)) == 1
    assert "nothing checked whether it still applies" in r.items[0].reason


def test_a_certificate_for_another_artifact_is_not_counted(tmp_path, capsys):
    x, y = _pair()
    other, _ = _pair(seed=5)
    path, doc = _certify(tmp_path, x, y)
    capsys.readouterr()
    r = capabilities(other, [(path, doc)], data=other)
    assert r.items == []
    assert len(r.other_artifacts) == 1
    assert r.other_artifacts[0]["certificate"] == path
    assert "different artifact" in r.explain()


def test_drifted_data_makes_a_capability_conditional(tmp_path, capsys):
    x, y = _pair()
    path, doc = _certify(tmp_path, x, y, extra=["--reference", "identity"])
    capsys.readouterr()
    r = capabilities(x, [(path, doc)], data=x + 5.0)
    assert [i.status for i in r.items] == [CONDITIONAL]
    assert "STALE" in r.items[0].reason and "RECERTIFY" in r.items[0].reason


def test_a_contract_with_no_certificate_says_what_to_certify_next(tmp_path, capsys):
    x, y = _pair()
    path, doc = _certify(tmp_path, x, y)
    capsys.readouterr()
    wanted = _contract(name="analytics")
    r = capabilities(x, [(path, doc)], contracts=[wanted], data=x)
    names = {i.status for i in r.items}
    assert NOT_CERTIFIED in names
    nc = r.by_status(NOT_CERTIFIED)[0]
    assert nc.name.startswith("analytics")
    assert "tqp certify --observer" in nc.reason
    assert nc.observer_sha256 == wanted.digest()


def test_a_failing_certificate_is_not_a_capability(tmp_path, capsys):
    x, y = _pair()
    path, doc = _certify(tmp_path, x, y)
    capsys.readouterr()
    doc["passed"] = False
    doc["interpretation"] = "VACUOUS: exact reranking required"
    r = capabilities(x, [(path, doc)], data=x)
    assert [i.status for i in r.items] == [NOT_CERTIFIED]
    assert "exact reranking required" in r.items[0].reason


def test_the_report_serialises_and_reads(tmp_path, capsys):
    x, y = _pair()
    path, doc = _certify(tmp_path, x, y)
    capsys.readouterr()
    r = capabilities(x, [(path, doc)], contracts=[_contract()], data=x)
    d = r.as_dict()
    assert d["schema"] == "turboquant-pro/capability-report"
    assert set(d) >= {"certified", "conditional", "not_certified"}
    json.dumps(d)
    text = r.explain()
    assert "CAPABILITIES of" in text and "not certified:" in text


def test_the_report_says_which_search_scorer_it_covers(tmp_path, capsys):
    """A rank certificate is computed on the float reconstruction, so it covers
    the reference scorer; the SIMD kernel's single-stage ranking is not covered."""
    x, y = _pair()
    path, doc = _certify(tmp_path, x, y)
    capsys.readouterr()
    r = capabilities(x, [(path, doc)], contracts=[_contract()], data=x)
    scope = r.as_dict()["scorer_scope"]
    assert scope["certified_on"] == "exact-float"
    assert any("exact rerank" in c for c in scope["covered"])
    assert any("kernel" in c for c in scope["not_covered"])
    assert "Not covered: the single-stage ranking" in r.explain()


# ---- CLI ----------------------------------------------------------------------


def test_cli_lists_capabilities_and_exits_on_none(tmp_path, capsys):
    x, y = _pair()
    c = _contract()
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c.as_dict()), encoding="utf-8")
    path, _ = _certify(tmp_path, x, y, extra=["--observer", str(cp)])
    art = tmp_path / "art.npy"
    np.save(art, x)
    capsys.readouterr()
    out = tmp_path / "caps.json"
    rc = main(
        [
            "capabilities",
            "--artifact",
            str(art),
            "--certificate",
            path,
            "--observer",
            str(cp),
            "--data",
            str(art),
            "--out",
            str(out),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["certified"]) == 1 and doc["not_certified"] == []
    capsys.readouterr()
    # an artifact with no certificate about it: nothing certified, exit 1
    other = tmp_path / "other.npy"
    np.save(other, _pair(seed=9)[0])
    assert main(["capabilities", "--artifact", str(other), "--certificate", path]) == 1
    capsys.readouterr()
    assert main(["capabilities", "--artifact", str(art)]) == 2
    assert "at least one --certificate or --observer" in capsys.readouterr().err


def test_cli_refuses_an_unreadable_certificate(tmp_path, capsys):
    x, _ = _pair()
    art = tmp_path / "art.npy"
    np.save(art, x)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert (
        main(["capabilities", "--artifact", str(art), "--certificate", str(bad)]) == 2
    )
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_cli_formats(tmp_path, capsys, fmt):
    x, y = _pair()
    path, _ = _certify(tmp_path, x, y)
    art = tmp_path / "art.npy"
    np.save(art, x)
    capsys.readouterr()
    main(
        [
            "capabilities",
            "--artifact",
            str(art),
            "--certificate",
            path,
            "--data",
            str(art),
            "--format",
            fmt,
        ]
    )
    out = capsys.readouterr().out
    assert ("CAPABILITIES of" in out) if fmt == "text" else json.loads(out)
