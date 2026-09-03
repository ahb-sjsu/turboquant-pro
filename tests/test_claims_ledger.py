# TurboQuant Pro: CLAIMS.md <-> claims.yaml consistency gate.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""The two claim ledgers must agree.

``claims.yaml`` is the machine-readable ledger ``tqp replay`` reads;
``CLAIMS.md`` is the human table. Historically they drifted (different row
counts, "Reproducible" in one and "needs-local-run" in the other). This test
makes the drift a CI failure:

  1. every ``claims.yaml`` id appears as a **Ledger id** in a CLAIMS.md table row;
  2. every Ledger id in CLAIMS.md exists in ``claims.yaml``;
  3. the leading status word of each CLAIMS.md row equals the yaml ``status``;
  4. statuses come from the declared vocabulary, and ``executable`` <=> has a
     ``command``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
CLAIMS_MD = REPO / "CLAIMS.md"
CLAIMS_YAML = REPO / "claims.yaml"

VOCAB = {
    "executable",
    "reproducible",
    "needs-local-run",
    "partial",
    "experimental",
    "reported",
    "retracted",
}


def _yaml_claims() -> dict:
    return yaml.safe_load(CLAIMS_YAML.read_text(encoding="utf-8"))["claims"]


def _md_rows() -> dict[str, str]:
    """Map Ledger id -> normalized status word, from every CLAIMS.md table row."""
    rows: dict[str, str] = {}
    header_cols: list[str] | None = None
    for line in CLAIMS_MD.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            header_cols = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header_cols is None:
            header_cols = cells
            continue
        if all(set(c) <= set(":-") for c in cells):  # the |---|---| separator
            continue
        if "Ledger id" not in header_cols or "Status" not in header_cols:
            continue
        cid_cell = cells[header_cols.index("Ledger id")]
        status_cell = cells[header_cols.index("Status")]
        m = re.search(r"`([a-z0-9_]+)`", cid_cell)
        assert m, f"CLAIMS.md row has no backticked ledger id: {cid_cell!r}"
        word = re.sub(r"[*_]", "", status_cell).split("(")[0].strip().lower()
        rows[m.group(1)] = word
    return rows


def test_every_yaml_claim_has_a_claims_md_row():
    missing = set(_yaml_claims()) - set(_md_rows())
    assert not missing, f"claims.yaml ids absent from CLAIMS.md: {sorted(missing)}"


def test_every_claims_md_row_exists_in_yaml():
    extra = set(_md_rows()) - set(_yaml_claims())
    assert not extra, f"CLAIMS.md ids absent from claims.yaml: {sorted(extra)}"


def test_statuses_agree_and_use_the_vocabulary():
    y = _yaml_claims()
    md = _md_rows()
    bad_vocab = {cid: c["status"] for cid, c in y.items() if c["status"] not in VOCAB}
    assert not bad_vocab, f"claims.yaml statuses outside the vocabulary: {bad_vocab}"
    mismatch = {
        cid: (md[cid], y[cid]["status"])
        for cid in set(md) & set(y)
        if md[cid] != y[cid]["status"]
    }
    assert not mismatch, f"CLAIMS.md vs claims.yaml status (md, yaml): {mismatch}"


def test_executable_iff_command():
    for cid, c in _yaml_claims().items():
        has_cmd = "command" in c
        is_exec = c["status"] == "executable"
        cmd_state = "present" if has_cmd else "absent"
        assert (
            has_cmd == is_exec
        ), f"{cid}: status={c['status']!r} but command {cmd_state}"
        if not has_cmd:
            assert "reference" in c, f"{cid}: reference claim needs a `reference`"
