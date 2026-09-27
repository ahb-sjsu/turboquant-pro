"""The artifact registry and the invocation block.

Every document the package writes is a registered kind; a kind validates only
against a schema that ships, and says "no schema shipped" otherwise; every JSON
document the CLI emits records how it was produced.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import turboquant_pro
from turboquant_pro import invocation as INV
from turboquant_pro.cli import main
from turboquant_pro.schemas import (
    KINDS,
    REGISTRY,
    identify,
    load_schema,
    validate,
)

jsonschema = pytest.importorskip("jsonschema")

PKG = Path(turboquant_pro.__file__).parent


# ---- the registry is complete and honest ------------------------------------


def test_every_shipped_schema_is_registered_once_and_is_a_valid_schema():
    shipped = {
        p.name
        for p in files("turboquant_pro.schemas").iterdir()
        if p.name.endswith(".schema.json")
    }
    used = [k.schema_file for k in KINDS if k.schema_file]
    assert len(used) == len(set(used))
    # invocation.schema.json is a definition embedded in other schemas, not a kind
    assert set(used) == shipped - {"invocation.schema.json"}
    for name in shipped:
        jsonschema.Draft202012Validator.check_schema(load_schema(name))


def test_a_schema_file_names_the_kind_it_is_registered_for():
    for k in KINDS:
        if k.schema_file is None:
            continue
        const = load_schema(k.schema_file).get("properties", {}).get("schema") or {}
        if "const" in const:
            assert const["const"] == k.id, k.schema_file


_ID = re.compile(
    r'(?:"schema":\s*|\b[A-Z_]*SCHEMA[A-Z_]*\s*=\s*|\bSCHEMA_ID\s*=\s*)"([^"]+)"'
)


def test_every_schema_id_the_source_writes_is_registered():
    """Scan the package for every schema id it can write, so a new artifact
    kind cannot ship unregistered."""
    found = {}
    for py in PKG.rglob("*.py"):
        for m in _ID.finditer(py.read_text(encoding="utf-8")):
            found.setdefault(m.group(1), py.relative_to(PKG).as_posix())
    missing = {sid: where for sid, where in found.items() if sid not in REGISTRY}
    assert not missing, f"unregistered artifact kinds: {missing}"


def test_the_embedded_invocation_definitions_equal_the_canonical_one():
    canon = load_schema("invocation.schema.json")
    for k in KINDS:
        if k.schema_file is None:
            continue
        inv = load_schema(k.schema_file)["properties"].get("invocation")
        if inv is None:
            continue
        for key in ("type", "required", "properties", "additionalProperties"):
            assert inv[key] == canon[key], (k.schema_file, key)


# ---- identification and validation ------------------------------------------


def test_schemaless_outputs_are_identified_by_their_fields_and_say_so():
    from turboquant_pro.a2_probe import A2ProbeResult
    from turboquant_pro.telemetry.metrics import reading

    probe = A2ProbeResult("cosine", 0.9, 0.8, 1.0, 1.2, "polar", 0.1).as_dict()
    assert identify(probe)[0].id == "turboquant-pro/a2-probe"
    assert identify(probe)[1] == "fields"
    r = reading("search.qps", 3.0)
    v = validate(r)
    assert v["kind"] == "turboquant-pro/metric-reading" and v["status"] == "valid"


def test_an_unknown_or_unregistered_document_is_unrecognized_not_guessed():
    assert validate({"a": 1})["status"] == "unrecognized"
    # a schema field that is not registered is never matched by shape
    doc = {"schema": "someone-else/thing", "name": 1, "unit": 1, "aggregation": 1,
           "window_s": 1, "kind": 1, "value": 1}  # fmt: skip
    assert validate(doc)["kind"] is None


def test_a_kind_without_a_schema_is_never_reported_valid():
    v = validate({"schema": "turboquant-pro/index-search", "results": []})
    assert v["status"] == "no schema shipped"


def test_an_invalid_document_reports_where():
    v = validate({"schema": "turboquant-pro/rank-certificate", "schema_version": 1})
    assert v["status"] == "invalid"
    assert v["errors"] and all("path" in e and "message" in e for e in v["errors"])


# ---- the invocation block -----------------------------------------------------


def _pair(tmp_path, n=64, d=16):
    x = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
    o, r = tmp_path / "o.npy", tmp_path / "r.npy"
    np.save(o, x)
    np.save(r, x + 0.01 * np.random.default_rng(1).standard_normal(x.shape))
    return str(o), str(r)


def test_every_cli_document_records_its_invocation_and_validates(tmp_path, capsys):
    """Run the CLI's emitters end to end: each document is a registered kind,
    carries an invocation that reproduces it, and is valid wherever a schema
    ships (never invalid)."""
    o, r = _pair(tmp_path, n=400, d=16)
    idx = str(tmp_path / "i.tqe")
    runs = {
        "certify": ["certify", "--original", o, "--reconstructed", r],
        "anatomy": ["anatomy", "--npy", o, "--k", "5"],
        "hubdiff": ["hubdiff", "--original", o, "--reconstructed", r, "--k", "5"],
        "index search": ["index", "search", idx, "--queries", o, "--k", "3"],
        "index info": ["index", "info", idx],
        "index drift": ["index", "drift", idx, "--embeddings", o],
        "index certify": ["index", "certify", idx, "--sample", "100"],
    }
    assert main(["index", "create", "--embeddings", o, "--out", idx]) == 0
    capsys.readouterr()
    for name, argv in runs.items():
        out = str(tmp_path / f"{name.replace(' ', '_')}.json")
        main([*argv, "--out", out])
        capsys.readouterr()
        doc = json.loads(Path(out).read_text(encoding="utf-8"))
        inv = doc["invocation"]
        assert inv["argv"] == ["tqp", *argv, "--out", out], name
        jsonschema.validate(inv, load_schema("invocation.schema.json"))
        v = validate(doc)
        assert v["kind"] is not None, name
        assert v["status"] in ("valid", "no schema shipped"), (name, v)


def test_only_a_cli_run_stamps_an_invocation(capsys):
    from turboquant_pro import cli

    cli._emit_doc({"schema": "x"}, None, "json", "")
    assert "invocation" not in json.loads(capsys.readouterr().out)
    assert cli._ARGV is None  # main() leaves no argv behind


def test_the_commit_is_the_package_source_not_the_shell_repository(monkeypatch):
    INV.source_commit.cache_clear()
    calls = []

    def fake_git(args, cwd):
        calls.append((tuple(args), Path(cwd)))
        if args[0] == "rev-parse" and args[1] == "--show-toplevel":
            return str(Path(cwd).parent)  # the package's own checkout
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        return " M turboquant_pro/cli.py"  # status: a local change

    monkeypatch.setattr(INV, "_git", fake_git)
    monkeypatch.chdir(Path.home())  # the shell is somewhere else entirely
    assert INV.source_commit() == ("abc123", True)
    assert all(cwd == PKG.resolve() for _, cwd in calls)

    INV.source_commit.cache_clear()
    monkeypatch.setattr(INV, "_git", lambda args, cwd: "/some/other/repo")
    assert INV.source_commit() == (None, None)  # not this package's source tree
    INV.source_commit.cache_clear()


def test_a_weight_plan_records_its_invocation_and_stays_valid(tmp_path, capsys):
    from tests.test_weight_plan import _table

    cp, out = tmp_path / "costs.json", tmp_path / "plan.json"
    cp.write_text(json.dumps(_table(4, n=6, levels=(3, 4, 5, 6, 8)).as_dict()))
    argv = ["plan", "weights", "--costs", str(cp), "--bits-per-weight", "4.5",
            "--out", str(out)]  # fmt: skip
    assert main(argv) == 0
    capsys.readouterr()
    doc = json.loads(out.read_text())
    assert doc["invocation"]["argv"] == ["tqp", *argv]
    assert validate(doc)["status"] == "valid"
