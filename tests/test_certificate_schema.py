"""The rank-certificate JSON artifact is schema-locked and drift-guarded.

`tqp certify` emits a provenance-stamped certificate; this suite pins it:

- the shipped schema is itself a valid JSON Schema;
- a freshly emitted certificate validates against it;
- a tiny deterministic corpus reproduces a committed golden certificate;
- non-finite measurements serialize as JSON ``null`` (never bare ``NaN``).

See ``docs/CERTIFICATE_SPEC.md`` for the compatibility promise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from turboquant_pro.cli import _json_safe, main
from turboquant_pro.schemas import load_schema, schema_path

GOLDEN = Path(__file__).parent / "fixtures" / "rank_certificate_golden.json"
SCHEMA_NAME = "rank_certificate.schema.json"


def _deterministic_pair():
    """A fixed tiny (original, reconstructed) pair — the golden corpus.

    A near-lossless reconstruction (small fixed perturbation) so the certificate
    is non-vacuous and certifies a positive floor — the happy path the golden
    anchors. The vacuous/null path is covered separately.
    """
    orig = np.random.default_rng(0).standard_normal((128, 32)).astype(np.float32)
    noise = np.random.default_rng(1).standard_normal((128, 32)).astype(np.float32)
    recon = (orig + 0.02 * noise).astype(np.float32)
    return orig, recon


def _certify_to_json(tmp_path, orig, recon, **flags) -> tuple[dict, str]:
    """Run ``tqp certify`` on the arrays; return (parsed doc, raw text)."""
    o = tmp_path / "orig.npy"
    r = tmp_path / "recon.npy"
    np.save(o, orig)
    np.save(r, recon)
    out = tmp_path / "cert.json"
    argv = [
        "certify",
        "--original",
        str(o),
        "--reconstructed",
        str(r),
        "--out",
        str(out),
    ]
    for k, v in flags.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    main(argv)
    text = out.read_text(encoding="utf-8")
    return json.loads(text), text


# --------------------------------------------------------------------------- #
# Schema validity                                                             #
# --------------------------------------------------------------------------- #


def test_schema_is_itself_valid():
    jsonschema = pytest.importorskip("jsonschema")
    schema = load_schema(SCHEMA_NAME)
    # Raises SchemaError if the schema is malformed.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_shipped_schema_is_discoverable():
    # importlib.resources path resolves whether installed or in-tree.
    assert schema_path(SCHEMA_NAME).is_file()


# --------------------------------------------------------------------------- #
# Emitted certificate conforms                                                #
# --------------------------------------------------------------------------- #


def test_emitted_certificate_matches_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    orig, recon = _deterministic_pair()
    doc, text = _certify_to_json(tmp_path, orig, recon, seed=0, anchors=64)
    jsonschema.validate(doc, load_schema(SCHEMA_NAME))
    assert "NaN" not in text and "Infinity" not in text  # spec-valid JSON


def test_vacuous_certificate_still_valid_and_null(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    # An all-identical corpus has degenerate (zero) distances -> non-finite
    # distortion -> vacuous certificate with null floors, but still valid JSON.
    row = np.random.default_rng(1).standard_normal((1, 16)).astype(np.float32)
    orig = np.repeat(row, 32, axis=0)
    doc, text = _certify_to_json(tmp_path, orig, orig, seed=0, anchors=32)
    jsonschema.validate(doc, load_schema(SCHEMA_NAME))
    assert "NaN" not in text and "Infinity" not in text
    # At least one floor is null (non-finite), proving the null path round-trips.
    cert = doc["certificate"]
    assert cert["tau_floor"] is None or cert["spearman_floor"] is None


# --------------------------------------------------------------------------- #
# Golden fixture                                                              #
# --------------------------------------------------------------------------- #


def test_certificate_golden(tmp_path):
    orig, recon = _deterministic_pair()
    doc, _ = _certify_to_json(tmp_path, orig, recon, seed=0, anchors=64)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    # Stable format identity.
    assert doc["schema"] == golden["schema"]
    assert doc["schema_version"] == golden["schema_version"]
    assert doc["params"] == golden["params"]
    assert doc["passed"] == golden["passed"]

    # Deterministic measured values (approx — guards against silent drift while
    # tolerating cross-platform float noise in the distance-ratio statistic).
    gc = golden["certificate"]
    dc = doc["certificate"]
    assert dc["n_pairs"] == gc["n_pairs"]
    assert dc["vacuous"] == gc["vacuous"]
    for k in ("kappa", "mu_hat", "tau_floor", "spearman_floor"):
        assert dc[k] == pytest.approx(gc[k], rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------- #
# Non-finite -> null                                                          #
# --------------------------------------------------------------------------- #


def test_json_safe_maps_non_finite_to_none():
    out = _json_safe(
        {"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": -float("inf")}}
    )
    assert out == {"a": None, "b": [None, 1.0], "c": {"d": None}}
    # And the result serializes under strict JSON.
    json.dumps(out, allow_nan=False)


def test_json_safe_preserves_finite_and_keys():
    doc = {"kappa": 1.5, "n": 3, "ok": True, "s": "x", "shape": [64, 16]}
    assert _json_safe(doc) == doc


# --------------------------------------------------------------------------- #
# Richer envelope (additive: task / environment / limitations / --html)       #
# --------------------------------------------------------------------------- #


def test_richer_envelope_validates_and_is_additive(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    orig, recon = _deterministic_pair()
    o, r = tmp_path / "o.npy", tmp_path / "r.npy"
    out, htm = tmp_path / "c.json", tmp_path / "c.html"
    np.save(o, orig)
    np.save(r, recon)
    main(
        [
            "certify",
            "--original",
            str(o),
            "--reconstructed",
            str(r),
            "--out",
            str(out),
            "--anchors",
            "64",
            "--seed",
            "0",
            "--task",
            "recall@10 >= 0.9",
            "--environment",
            "--limitation",
            "sampled bound",
            "--limitation",
            "cosine metric only",
            "--html",
            str(htm),
        ]
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    # Still a valid v1 certificate — the new sections are additive.
    jsonschema.validate(doc, load_schema(SCHEMA_NAME))
    assert doc["schema_version"] == 1
    assert doc["task"] == {"kind": "retrieval", "target": "recall@10 >= 0.9"}
    assert "tool_version" in doc["environment"] and "git_commit" in doc["environment"]
    assert doc["limitations"] == ["sampled bound", "cosine metric only"]
    # HTML report written and self-contained.
    text = htm.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and "certificate" in text


def test_base_certificate_omits_optional_sections(tmp_path):
    """Without the flags, the certificate has none of the optional sections."""
    orig, recon = _deterministic_pair()
    doc, _ = _certify_to_json(tmp_path, orig, recon, seed=0, anchors=64)
    for k in ("task", "environment", "limitations"):
        assert k not in doc


# --------------------------------------------------------------------------- #
# The reference section: which read operator a consumer-relative number is    #
# computed against. Additive, so schema_version stays 1.                      #
# --------------------------------------------------------------------------- #


def _certify_with_reference(tmp_path, provider, extra=None):
    """Run certify with --reference; return (returncode, parsed doc or None)."""
    orig, recon = _deterministic_pair()
    o, r = tmp_path / "orig.npy", tmp_path / "recon.npy"
    np.save(o, orig)
    np.save(r, recon)
    out = tmp_path / "cert.json"
    argv = [
        "certify",
        "--original",
        str(o),
        "--reconstructed",
        str(r),
        "--out",
        str(out),
        "--reference",
        provider,
    ] + list(extra or [])
    rc = main(argv)
    if not out.exists():
        return rc, None
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_reference_section_validates_and_is_additive(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    rc, doc = _certify_with_reference(tmp_path, "identity")
    assert rc in (0, 1)
    jsonschema.validate(doc, load_schema(SCHEMA_NAME))
    assert doc["schema_version"] == 1

    ref = doc["reference"]
    assert ref["provider"] == "identity"
    assert ref["exact"] is True
    assert ref["dim"] == 32
    assert len(ref["operator_sha256"]) == 64


def test_identity_reference_equals_reconstruction_error(tmp_path):
    """The null reference makes the metric's degenerate case explicit."""
    _, doc = _certify_with_reference(tmp_path, "identity")
    ref = doc["reference"]
    assert ref["consumer_distortion"] == pytest.approx(
        ref["reconstruction_distortion"], rel=1e-9
    )


def test_a_real_reference_differs_from_reconstruction(tmp_path):
    """A consumer that reads a subspace must not agree with the trace, or the
    section would be recording nothing."""
    basis = np.linalg.qr(np.random.default_rng(6).standard_normal((32, 3)))[0]
    _, doc = _certify_with_reference(
        tmp_path,
        "declared",
        ["--reference-config", json.dumps({"matrix": (basis @ basis.T).tolist()})],
    )
    ref = doc["reference"]
    assert ref["provider"] == "declared"
    assert ref["consumer_distortion"] < ref["reconstruction_distortion"]
    assert ref["effective_rank"] == pytest.approx(3.0, abs=1e-6)


def test_operator_hash_binds_the_number_to_its_operator(tmp_path):
    """Two different references must not be confusable after the fact."""
    hashes = []
    for i in (0, 1):
        basis = np.linalg.qr(np.random.default_rng(20 + i).standard_normal((32, 3)))[0]
        d = tmp_path / f"h{i}"
        d.mkdir()
        _, doc = _certify_with_reference(
            d,
            "declared",
            [
                "--reference-config",
                json.dumps({"matrix": (basis @ basis.T).tolist()}),
            ],
        )
        hashes.append(doc["reference"]["operator_sha256"])
    assert hashes[0] != hashes[1]


def test_unknown_reference_fails_loudly_rather_than_dropping_the_section(
    tmp_path,
):
    """A silently missing section is indistinguishable from one never asked
    for, which would let a consumer-relative claim lose its reference."""
    rc, _ = _certify_with_reference(tmp_path, "no_such_provider")
    assert rc == 2


def test_base_certificate_still_omits_the_reference(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    orig, recon = _deterministic_pair()
    doc, _ = _certify_to_json(tmp_path, orig, recon)
    assert "reference" not in doc
    jsonschema.validate(doc, load_schema(SCHEMA_NAME))
