# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""`tqp verify` — schema/self-consistency + independent recompute of a
certificate.json emitted by `tqp certify`. Covers the happy path (schema-only
and recompute), and the three ways a certificate can fail to verify: a tampered
recorded floor, an input-hash mismatch, and a malformed schema."""

from __future__ import annotations

import argparse
import json

import numpy as np

from turboquant_pro import cli


def _ns(**kw):
    return argparse.Namespace(**kw)


def _make_cert(tmp_path, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    orig = rng.standard_normal((80, 24)).astype(np.float32)
    recon = (orig + noise * rng.standard_normal((80, 24))).astype(np.float32)
    op, rp, certp = tmp_path / "o.npy", tmp_path / "r.npy", tmp_path / "cert.json"
    np.save(op, orig)
    np.save(rp, recon)
    cli._cmd_certify(
        _ns(
            original=str(op),
            reconstructed=str(rp),
            metric="cosine",
            anchors=50,
            seed=0,
            min_tau=None,
            out=str(certp),
            format="json",
        )
    )
    assert certp.exists()
    return op, rp, certp


def _verify(certp, original=None, reconstructed=None):
    return cli._cmd_verify(
        _ns(
            certificate=str(certp),
            original=str(original) if original else None,
            reconstructed=str(reconstructed) if reconstructed else None,
            atol=1e-6,
            rtol=1e-4,
            out=None,
            format="text",
        )
    )


def test_verify_schema_only_ok(tmp_path):
    _, _, certp = _make_cert(tmp_path)
    assert _verify(certp) == 0


def test_verify_recompute_matches(tmp_path):
    op, rp, certp = _make_cert(tmp_path)
    assert _verify(certp, op, rp) == 0


def test_verify_detects_tampered_floor(tmp_path):
    op, rp, certp = _make_cert(tmp_path)
    doc = json.loads(certp.read_text())
    doc["certificate"]["tau_floor"] = -0.5  # bogus but in-range, so schema still OK
    certp.write_text(json.dumps(doc))
    assert _verify(certp, op, rp) == 1  # recompute exposes the mismatch


def test_verify_detects_hash_mismatch(tmp_path):
    op, rp, certp = _make_cert(tmp_path)
    other = tmp_path / "other.npy"
    np.save(
        other, np.random.default_rng(9).standard_normal((80, 24)).astype(np.float32)
    )
    assert _verify(certp, other, rp) == 1  # inputs don't match recorded hashes


def test_verify_bad_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope", "certificate": {}}))
    assert _verify(bad) == 1


def test_verify_missing_file(tmp_path):
    assert _verify(tmp_path / "nope.json") == 2  # IO error -> exit 2
