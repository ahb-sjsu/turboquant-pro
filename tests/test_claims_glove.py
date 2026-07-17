"""The canonical GloVe recall claim is executable and gated in CI.

The central Track-1 embedding result — a fixed compression recipe preserves
**recall@10** (the metric retrieval consumes) at real compression — is not just
prose in a notebook: ``tqp replay`` runs it and checks the measured recall and
compression against declared floors, exiting non-zero on regression.

This suite runs the **hermetic** ``--small`` path against the real
``claims.yaml``: a tiny *real* glove-100-angular subset bundled at
``benchmarks/fixtures/glove_tiny.npz`` (no network, no GPU, numpy + core only).
The full 1.18M-vector run (``full_command``) is Atlas/local-only and not
exercised here. See ``benchmarks/canonical_glove.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from turboquant_pro.cli import main

REPO = Path(__file__).resolve().parents[1]
CLAIMS = REPO / "claims.yaml"
CLAIM = "embedding_glove_recall"


def test_claim_is_declared_and_executable():
    spec = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    claim = spec["claims"][CLAIM]
    assert "--small" in claim["command"]
    assert "--full" in claim["full_command"]
    # Acceptance floors are the consumer metric (reranked recall) + compression,
    # never reconstruction cosine.
    assert "recall_at_10_rerank_min" in claim["expected"]
    assert "compression_ratio_min" in claim["expected"]
    assert not any("cosine" in k for k in claim["expected"])


def test_bundled_fixture_present():
    assert (REPO / "benchmarks" / "fixtures" / "glove_tiny.npz").is_file()


def test_replay_small_reproduces_and_gates(capsys):
    """`tqp replay` runs the hermetic claim and returns reproduced (exit 0)."""
    out = REPO / "results.json"
    if out.exists():
        out.unlink()
    try:
        rc = main(
            [
                "replay",
                CLAIM,
                "--claims",
                str(CLAIMS),
                "--cwd",
                str(REPO),
                "--json",
            ]
        )
        report = json.loads(capsys.readouterr().out)
        assert rc == 0, report
        row = report["claims"][0]
        assert row["verdict"] == "reproduced", row
        assert report["summary"]["reproduced"] == 1
        # The gated metrics really were measured (not just present).
        assert row["measured"]["recall_at_10_rerank"] is not None
        assert row["measured"]["compression_ratio"] is not None
    finally:
        if out.exists():
            out.unlink()
