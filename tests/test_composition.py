# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Pipeline certificate composition (issue #182): the chain must connect, the
distortions multiply, and a trimmed composition is not called unconditional."""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro.cli import main
from turboquant_pro.composition import ChainError, compose

D = 16
N = 96


def _stages(seed=0, noise=(0.01, 0.01)):
    """A source and one array per stage, each a noisier view of the last."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D)).astype(np.float32)
    out = [x]
    for s in noise:
        out.append((out[-1] + s * rng.standard_normal((N, D))).astype(np.float32))
    return out


def _certify(tmp_path, a, b, name):
    pa, pb = tmp_path / f"a_{name}.npy", tmp_path / f"b_{name}.npy"
    np.save(pa, a)
    np.save(pb, b)
    out = tmp_path / name
    main(
        [
            "certify",
            "--original",
            str(pa),
            "--reconstructed",
            str(pb),
            "--anchors",
            "64",
            "--out",
            str(out),
        ]
    )
    return str(out), json.loads(out.read_text(encoding="utf-8"))


def _chain(tmp_path, capsys, **kw):
    arrays = _stages(**kw)
    certs = [
        _certify(tmp_path, arrays[i], arrays[i + 1], f"s{i + 1}.json")
        for i in range(len(arrays) - 1)
    ]
    capsys.readouterr()
    return arrays, certs


# ---- the chain ----------------------------------------------------------------


def test_a_connected_chain_composes(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    r = compose(certs, arrays[0])
    assert len(r.stages) == 2
    assert r.product_kappa == pytest.approx(
        certs[0][1]["certificate"]["kappa"] * certs[1][1]["certificate"]["kappa"]
    )
    assert r.product_kappa >= max(s["kappa"] for s in r.stages)
    assert r.metric == "cosine"
    assert r.weakest in (certs[0][0], certs[1][0])
    json.dumps(r.as_dict())


def test_the_chain_floor_is_no_higher_than_any_stage(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    r = compose(certs, arrays[0])
    for st in r.stages:
        assert r.tau_floor <= st["tau_floor"] + 1e-12


def test_a_chain_that_does_not_connect_is_refused(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    other = _stages(seed=7)
    stray = _certify(tmp_path, other[1], other[2], "stray.json")
    capsys.readouterr()
    with pytest.raises(ChainError, match="does not connect"):
        compose([certs[0], stray], arrays[0])


def test_a_source_that_is_not_the_first_stages_input_is_refused(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    with pytest.raises(ChainError, match="not what the first stage"):
        compose(certs, _stages(seed=3)[0])


def test_a_failing_stage_stops_the_composition(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    certs[1][1]["passed"] = False
    certs[1][1]["interpretation"] = "VACUOUS: exact reranking required"
    with pytest.raises(ChainError, match="does not pass on its own"):
        compose(certs, arrays[0])


def test_mismatched_metrics_are_refused(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    certs[1][1]["params"]["metric"] = "l2"
    with pytest.raises(ChainError, match="different metrics"):
        compose(certs, arrays[0])


def test_one_stage_is_not_a_chain(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    with pytest.raises(ChainError, match="at least two stages"):
        compose(certs[:1], arrays[0])


def test_a_document_that_is_not_a_certificate_is_refused(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    bad = ("x.json", {"schema": "turboquant-pro/verification", "passed": True})
    with pytest.raises(ChainError, match="not a rank certificate"):
        compose([certs[0], bad], arrays[0])


# ---- what it says -------------------------------------------------------------


def test_a_trimmed_composition_is_not_called_unconditional(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    r = compose(certs, arrays[0])
    assert r.unconditional is False
    assert "tail penalty" in r.interpretation
    assert "tail penalty" in r.explain()
    # Trimmed stages record a nonzero tail, so declaring them strict is refused.
    with pytest.raises(ChainError):
        compose(certs, arrays[0], unconditional=True)


def test_stages_over_different_anchor_samples_are_refused(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    certs[1][1]["params"]["seed"] = certs[1][1]["params"]["seed"] + 1
    with pytest.raises(ChainError):
        compose(certs, arrays[0])


def test_the_chain_floor_pays_the_summed_tail(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    r = compose(certs, arrays[0])
    tails = sum(c[1]["certificate"]["tail_fraction"] for c in certs)
    assert tails > 0
    assert r.tau_floor == pytest.approx(1.0 - 2.0 * r.mu_hat - 2.0 * min(1.0, tails))


def test_a_stage_without_a_tail_record_is_refused(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    del certs[0][1]["certificate"]["tail_fraction"]
    with pytest.raises(ChainError):
        compose(certs, arrays[0])


def test_a_floor_the_chain_cannot_clear_fails(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    assert compose(certs, arrays[0], min_tau=0.999999).passed is False
    r = compose(certs, arrays[0], min_tau=0.999999)
    assert "validate the chain or rerank" in r.interpretation


def test_the_report_reads(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    text = compose(certs, arrays[0]).explain()
    assert "PIPELINE CERTIFICATE" in text
    assert "Product kappa" in text and "Chain floor" in text


# ---- CLI ----------------------------------------------------------------------


def test_cli_composes_and_reports(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    src = tmp_path / "src.npy"
    np.save(src, arrays[0])
    out = tmp_path / "chain.json"
    rc = main(
        [
            "compose",
            "--certificate",
            certs[0][0],
            "--certificate",
            certs[1][0],
            "--source",
            str(src),
            "--out",
            str(out),
            "--format",
            "json",
        ]
    )
    assert rc in (0, 1)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "turboquant-pro/composition-certificate"
    assert len(doc["stages"]) == 2
    assert doc["unconditional"] is False
    capsys.readouterr()


def test_cli_exits_two_on_a_broken_chain(tmp_path, capsys):
    arrays, certs = _chain(tmp_path, capsys)
    other = _stages(seed=11)
    stray = _certify(tmp_path, other[1], other[2], "stray.json")
    src = tmp_path / "src.npy"
    np.save(src, arrays[0])
    capsys.readouterr()
    rc = main(
        [
            "compose",
            "--certificate",
            certs[0][0],
            "--certificate",
            stray[0],
            "--source",
            str(src),
        ]
    )
    assert rc == 2
    assert "does not connect" in capsys.readouterr().err


def test_cli_refuses_an_unreadable_certificate(tmp_path, capsys):
    src = tmp_path / "src.npy"
    np.save(src, _stages()[0])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["compose", "--certificate", str(bad), "--source", str(src)]) == 2
    assert "cannot read" in capsys.readouterr().err
