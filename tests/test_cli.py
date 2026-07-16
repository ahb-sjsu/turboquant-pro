"""Rigorous tests for the ``tqp`` CLI dispatcher.

Covers dispatch, exit-code contracts (0 ok / 1 conformance-fail / 2 stub-or-guard),
argument parsing, output content, and error handling (unknown plugin, raising
factory, bad target). The ``trace`` subcommand tests skip cleanly where torch /
transformers are absent (e.g. minimal CI), and never touch the network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro import plugins
from turboquant_pro.cli import build_parser, main

IN_TREE = {"per_channel", "polar"}


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def temp_plugin():
    """Register throwaway plugins and clean the registry afterwards."""
    added: list[str] = []

    def _add(spec: plugins.PluginSpec) -> str:
        plugins.register(spec, overwrite=True)
        added.append(spec.name)
        return spec.name

    yield _add
    for name in added:
        plugins._REGISTRY.pop(name, None)


class _BrokenQuantizer:
    """Satisfies the protocol but fails the roundtrip check."""

    def compress(self, x, **kwargs):
        return ("broken", x.shape)

    def decompress(self, c):
        return np.zeros(1, dtype=np.float32)  # wrong shape on purpose


def _broken_factory(**config):
    return _BrokenQuantizer()


def _raising_factory(**config):
    raise RuntimeError("factory boom")


# ------------------------------------------------------------------ version
def test_version_ok(capsys):
    assert main(["version"]) == 0
    assert "turboquant-pro" in capsys.readouterr().out


def test_main_returns_handler_int():
    # main() must propagate the subcommand's return code verbatim
    assert isinstance(main(["version"]), int)
    assert main(["probe"]) == 2  # a real handler's usage-error code


# ------------------------------------------------------------------ plugin list
def test_plugin_list_shows_in_tree(capsys):
    assert main(["plugin", "list"]) == 0
    out = capsys.readouterr().out
    assert IN_TREE <= set(out.split())
    assert "TARGETS" in out  # header rendered


def test_plugin_list_verbose_descriptions(capsys):
    assert main(["plugin", "list", "-v"]) == 0
    assert "PerChannelKV" in capsys.readouterr().out


def test_plugin_list_target_filter(capsys):
    assert main(["plugin", "list", "--target", "kv_key"]) == 0
    out = capsys.readouterr().out
    assert "per_channel" in out and "polar" not in out


def test_plugin_list_unknown_target_is_clean(capsys):
    assert main(["plugin", "list", "--target", "embedding"]) == 0
    assert "no quantizer plugins registered" in capsys.readouterr().out


# ------------------------------------------------------------------ conformance
def test_conformance_in_tree_passes(capsys):
    rc = main(["plugin", "conformance"])
    out = capsys.readouterr().out
    assert "per_channel" in out and "polar" in out
    assert rc == 0, out


def test_conformance_single_named_plugin(capsys):
    assert main(["plugin", "conformance", "per_channel"]) == 0
    out = capsys.readouterr().out
    assert "per_channel" in out and "polar" not in out


def test_conformance_target_filter(capsys):
    assert main(["plugin", "conformance", "--target", "kv_value"]) == 0
    out = capsys.readouterr().out
    assert "polar" in out and "per_channel" not in out


def test_conformance_failing_plugin_exits_1(capsys, temp_plugin):
    temp_plugin(
        plugins.PluginSpec(
            name="_broken",
            factory=_broken_factory,
            targets=frozenset({plugins.TARGET_KV_KEY}),
        )
    )
    rc = main(["plugin", "conformance", "_broken"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out or "ERROR" in out


def test_conformance_raising_factory_reported_not_crash(capsys, temp_plugin):
    temp_plugin(
        plugins.PluginSpec(
            name="_raiser",
            factory=_raising_factory,
            targets=frozenset({plugins.TARGET_KV_KEY}),
        )
    )
    rc = main(["plugin", "conformance", "_raiser"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ERROR" in out and "factory boom" in out


def test_conformance_one_fail_one_pass_is_1(capsys, temp_plugin):
    temp_plugin(
        plugins.PluginSpec(
            name="_broken2",
            factory=_broken_factory,
            targets=frozenset({plugins.TARGET_KV_KEY}),
        )
    )
    # explicit good + bad -> overall non-zero
    rc = main(["plugin", "conformance", "per_channel", "_broken2"])
    assert rc == 1


def test_conformance_unknown_plugin_reports_exit_1(capsys):
    rc = main(["plugin", "conformance", "does_not_exist"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ERROR" in out


def test_conformance_shape_override(capsys, temp_plugin):
    # --shape drives the sample; a passthrough quantizer must round-trip exactly
    class _Identity:
        def compress(self, x, **k):
            return x

        def decompress(self, c):
            return c

    temp_plugin(
        plugins.PluginSpec(
            name="_id",
            factory=lambda **c: _Identity(),
            targets=frozenset({plugins.TARGET_EMBEDDING}),
        )
    )
    assert main(["plugin", "conformance", "_id", "--shape", "64,32"]) == 0


# ------------------------------------------------------------------ trace
def test_trace_bad_target_exits_2(capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    rc = main(["trace", "some/model", "--target", "bogus"])
    assert rc == 2
    assert "unknown --target" in capsys.readouterr().err


def _patch_transformers(monkeypatch, transformers, module_factory):
    """Route AutoConfig/AutoModel to a local meta-device module — no network."""
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: object()),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_config",
        staticmethod(lambda *a, **k: module_factory()),
    )


def test_trace_meta_model_summary(capsys, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    # tiny real module on meta device; no network, no weights
    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.k_proj = torch.nn.Linear(8, 8)
            self.v_proj = torch.nn.Linear(8, 8)
            self.norm = torch.nn.LayerNorm(8)

    _patch_transformers(monkeypatch, transformers, _Tiny)
    rc = main(["trace", "fake/model", "--target", "weight"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "regime distribution" in out
    assert "discipline family distribution" in out


def test_trace_verbose_per_tensor(capsys, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)

    _patch_transformers(monkeypatch, transformers, _Tiny)
    assert main(["trace", "fake/model", "-v"]) == 0
    assert "q_proj" in capsys.readouterr().out


# ------------------------------------------------------------------ probe
def test_probe_demo_isotropic_json(capsys):
    assert main(["probe", "--demo", "isotropic", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {
        "recommendation",
        "spearman_polar",
        "spearman_per_channel",
        "margin",
    } <= set(data)
    assert data["recommendation"] in {"polar", "per_channel"}


def test_probe_dc_offset_recommends_per_channel(capsys):
    # the v1.2.0 KV-keys regime: on attention logits the polar (per-vector
    # direction) quotient collapses while per-channel affine tracks the ranking
    rc = main(
        ["probe", "--demo", "dc_offset", "--consumer", "attention_logits", "--json"]
    )
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["recommendation"] == "per_channel"
    assert data["spearman_per_channel"] > data["spearman_polar"]


def test_probe_text_output(capsys):
    assert main(["probe", "--demo", "isotropic"]) == 0
    out = capsys.readouterr().out
    assert "recommend:" in out and "spearman(polar)" in out


def test_probe_requires_input(capsys):
    assert main(["probe"]) == 2
    assert "needs input" in capsys.readouterr().err


def test_probe_bad_consumer_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["probe", "--demo", "isotropic", "--consumer", "bogus"]
        )


def test_probe_npy_roundtrip(capsys, tmp_path):
    p = tmp_path / "keys.npy"
    np.save(p, np.random.default_rng(1).standard_normal((256, 32)).astype(np.float32))
    assert main(["probe", "--npy", str(p), "--json"]) == 0
    assert "recommendation" in capsys.readouterr().out


def test_probe_npy_ndim_flatten(capsys, tmp_path):
    p = tmp_path / "kv.npy"
    # (batch, heads, head_dim) — probe flattens all but the last axis to rows
    np.save(p, np.random.default_rng(2).standard_normal((4, 8, 64)).astype(np.float32))
    assert main(["probe", "--npy", str(p)]) == 0
    assert "reshaped" in capsys.readouterr().out


def test_probe_missing_npy(capsys, tmp_path):
    assert main(["probe", "--npy", str(tmp_path / "nope.npy")]) == 2
    assert "cannot load" in capsys.readouterr().err


def test_probe_too_small_batch_exits_2(capsys, tmp_path):
    p = tmp_path / "tiny.npy"
    np.save(p, np.ones((2, 8), dtype=np.float32))  # probe_quotient needs n >= 4
    assert main(["probe", "--npy", str(p)]) == 2
    assert "n >= 4" in capsys.readouterr().err


# ------------------------------------------------------------------ monitor
def _save_pair(tmp_path, orig, recon, prefix="p"):
    o, r = tmp_path / f"{prefix}_o.npy", tmp_path / f"{prefix}_r.npy"
    np.save(o, orig)
    np.save(r, recon)
    return str(o), str(r)


def test_monitor_healthy_json_exit_0(capsys, tmp_path):
    x = np.random.default_rng(0).standard_normal((128, 64)).astype(np.float32)
    o, r = _save_pair(tmp_path, x, x)  # perfect reconstruction
    assert main(["monitor", "--original", o, "--reconstructed", r]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["turboquant_quality_mean_cosine"] > 0.99
    assert m["turboquant_quality_is_healthy"] == 1


def test_monitor_unhealthy_exit_1(capsys, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 64)).astype(np.float32)
    y = rng.standard_normal((128, 64)).astype(np.float32)  # unrelated
    o, r = _save_pair(tmp_path, x, y)
    rc = main(["monitor", "--original", o, "--reconstructed", r, "--floor", "0.95"])
    m = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert m["turboquant_quality_is_healthy"] == 0


def test_monitor_prometheus_format(capsys, tmp_path):
    x = np.ones((16, 8), dtype=np.float32)
    o, r = _save_pair(tmp_path, x, x)
    rc = main(
        ["monitor", "--original", o, "--reconstructed", r, "--format", "prometheus"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # exposition format: one HELP/TYPE-style comment + a `name value` line each
    assert "# TYPE turboquant_quality_mean_cosine gauge" in out
    assert "turboquant_quality_is_healthy 1" in out


def test_monitor_text_format(capsys, tmp_path):
    x = np.ones((16, 8), dtype=np.float32)
    o, r = _save_pair(tmp_path, x, x)
    rc = main(["monitor", "--original", o, "--reconstructed", r, "--format", "text"])
    assert rc == 0
    assert "mean_cosine" in capsys.readouterr().out


def test_monitor_shape_mismatch_exit_2(capsys, tmp_path):
    o, r = _save_pair(
        tmp_path, np.ones((8, 4), np.float32), np.ones((8, 5), np.float32)
    )
    assert main(["monitor", "--original", o, "--reconstructed", r]) == 2
    assert "shape mismatch" in capsys.readouterr().err


def test_monitor_requires_both_inputs():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["monitor", "--original", "o.npy"])


# ------------------------------------------------------------------ certify
def test_certify_faithful_run_passes(capsys, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 48)).astype(np.float32)
    recon = x + 1e-4 * rng.standard_normal((256, 48))  # near-perfect
    o, r = _save_pair(tmp_path, x, recon)
    rc = main(["certify", "--original", o, "--reconstructed", r])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["passed"] is True
    assert doc["certificate"]["tau_floor"] > 0.0
    assert doc["certificate"]["vacuous"] is False


def test_certify_provenance_fields(capsys, tmp_path):
    x = np.random.default_rng(1).standard_normal((64, 16)).astype(np.float32)
    o, r = _save_pair(tmp_path, x, x, prefix="prov")
    main(["certify", "--original", o, "--reconstructed", r])
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "turboquant-pro/rank-certificate"
    assert doc["schema_version"] == 1
    assert doc["tool_version"]
    assert doc["created_utc"]
    for side in ("original", "reconstructed"):
        info = doc["inputs"][side]
        assert len(info["sha256"]) == 64  # full sha256 hex
        assert info["shape"] == [64, 16]
    assert doc["params"] == {"metric": "cosine", "n_anchors": 200, "seed": 0}


def test_certify_identical_inputs_share_hash(capsys, tmp_path):
    x = np.random.default_rng(2).standard_normal((32, 8)).astype(np.float32)
    o, r = _save_pair(tmp_path, x, x, prefix="same")
    main(["certify", "--original", o, "--reconstructed", r])
    doc = json.loads(capsys.readouterr().out)
    ins = doc["inputs"]
    assert ins["original"]["sha256"] == ins["reconstructed"]["sha256"]


def test_certify_min_tau_gate_fails(capsys, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 48)).astype(np.float32)
    recon = x + 1e-4 * rng.standard_normal((256, 48))
    o, r = _save_pair(tmp_path, x, recon, prefix="gate")
    # an unreachable floor forces the gate to fail even on a faithful run
    rc = main(["certify", "--original", o, "--reconstructed", r, "--min-tau", "1.5"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert doc["passed"] is False
    assert "FAIL" in doc["interpretation"]


def test_certify_out_writes_file(capsys, tmp_path):
    x = np.random.default_rng(3).standard_normal((64, 16)).astype(np.float32)
    o, r = _save_pair(tmp_path, x, x, prefix="out")
    out = tmp_path / "certificate.json"
    rc = main(["certify", "--original", o, "--reconstructed", r, "--out", str(out)])
    assert rc == 0
    assert f"wrote {out}" in capsys.readouterr().out
    doc = json.loads(out.read_text())
    assert doc["schema"] == "turboquant-pro/rank-certificate"
    assert "certificate" in doc


def test_certify_text_format(capsys, tmp_path):
    x = np.random.default_rng(4).standard_normal((64, 16)).astype(np.float32)
    o, r = _save_pair(tmp_path, x, x, prefix="txt")
    rc = main(["certify", "--original", o, "--reconstructed", r, "--format", "text"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Kendall" in out and "tau" in out


def test_certify_l2_metric(capsys, tmp_path):
    x = np.random.default_rng(5).standard_normal((64, 16)).astype(np.float32)
    o, r = _save_pair(tmp_path, x, x, prefix="l2")
    main(["certify", "--original", o, "--reconstructed", r, "--metric", "l2"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["params"]["metric"] == "l2"


def test_certify_shape_mismatch_exit_2(capsys, tmp_path):
    o, r = _save_pair(
        tmp_path, np.ones((8, 4), np.float32), np.ones((8, 5), np.float32), prefix="mm"
    )
    assert main(["certify", "--original", o, "--reconstructed", r]) == 2
    assert "shape mismatch" in capsys.readouterr().err


def test_certify_requires_inputs():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["certify", "--original", "o.npy"])


# ------------------------------------------------------------------ plan embeddings
def _save_embeddings(tmp_path, seed=0, n=400, dim=64):
    x = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    p = tmp_path / "emb.npy"
    np.save(p, x)
    return str(p)


def test_plan_embeddings_json(capsys, tmp_path):
    p = _save_embeddings(tmp_path)
    rc = main(["plan", "embeddings", "--embeddings", p, "--sample", "50"])
    doc = json.loads(capsys.readouterr().out)
    assert rc in (0, 1)  # 0 unless the preview is vacuous on this data
    assert doc["schema"] == "turboquant-pro/embedding-plan"
    assert "recommended" in doc and "alternatives" in doc
    assert "certificate_preview" in doc  # rank floor, not cosine, is acceptance
    assert "cosine" in doc["note"]  # scope note names the cosine caveat


def test_plan_embeddings_leads_with_rank_not_cosine(capsys, tmp_path):
    p = _save_embeddings(tmp_path)
    main(
        ["plan", "embeddings", "--embeddings", p, "--sample", "50", "--format", "text"]
    )
    out = capsys.readouterr().out
    assert "rank floor" in out or "rank certificate preview" in out
    assert "diagnostic only" in out  # cosine is demoted to a diagnostic


def test_plan_embeddings_byte_budget_unmet(capsys, tmp_path):
    p = _save_embeddings(tmp_path)
    rc = main(
        [
            "plan",
            "embeddings",
            "--embeddings",
            p,
            "--sample",
            "50",
            "--max-bytes-per-vector",
            "0.5",  # unsatisfiable
        ]
    )
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1 and doc["passed"] is False
    assert any("no recipe fits" in f for f in doc["risk_flags"])


def test_plan_embeddings_out_file(capsys, tmp_path):
    p = _save_embeddings(tmp_path)
    out = tmp_path / "plan.json"
    main(["plan", "embeddings", "--embeddings", p, "--sample", "50", "--out", str(out)])
    assert f"wrote {out}" in capsys.readouterr().out
    assert json.loads(out.read_text())["schema"] == "turboquant-pro/embedding-plan"


def test_plan_embeddings_missing_file(capsys, tmp_path):
    assert main(["plan", "embeddings", "--embeddings", str(tmp_path / "no.npy")]) == 2
    assert "cannot load" in capsys.readouterr().err


def test_plan_embeddings_requires_arg():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan", "embeddings"])


# ------------------------------------------------------------------ plan kv
def test_plan_kv_registry_model(capsys):
    rc = main(["plan", "kv", "--model", "llama-3-8b"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["schema"] == "turboquant-pro/kv-plan"
    assert doc["policy"]["key_bits"] and doc["policy"]["value_bits"]
    assert "risk_flags" in doc


def test_plan_kv_extreme_flags_key_risk(capsys):
    main(["plan", "kv", "--model", "llama-3-8b", "--target", "extreme"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["policy"]["key_bits"] < 4
    assert any("4-bit" in f for f in doc["risk_flags"])  # KV-keys risk surfaced


def test_plan_kv_context_override(capsys):
    main(["plan", "kv", "--model", "llama-3-8b", "--context", "4096"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["policy"]["max_seq_len"] == 4096


def test_plan_kv_unresolved_model_exit_2(capsys):
    import importlib.util

    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers present -> from_pretrained would query HF hub")
    assert main(["plan", "kv", "--model", "___no_such_model___"]) == 2
    assert "Could not resolve" in capsys.readouterr().err


# ------------------------------------------------------------------ replay
def _write_claims(tmp_path, claims: dict):
    import yaml

    p = tmp_path / "claims.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "claims": claims}))
    return str(p)


def _emit_cmd(tmp_path, payload: dict):
    """A portable command that writes results.json with ``payload``."""
    import sys

    script = tmp_path / "emit.py"
    script.write_text(
        "import json\n" f"open('results.json','w').write(json.dumps({payload!r}))\n"
    )
    return f"{sys.executable} {script}"


def test_replay_list(capsys, tmp_path):
    pytest.importorskip("yaml")
    cf = _write_claims(
        tmp_path,
        {"c1": {"track": "embedding", "status": "reproducible", "hardware": "cpu"}},
    )
    assert main(["replay", "--list", "--claims", cf]) == 0
    assert "c1" in capsys.readouterr().out


def test_replay_dry_run(capsys, tmp_path):
    pytest.importorskip("yaml")
    cf = _write_claims(
        tmp_path,
        {"c1": {"track": "embedding", "command": "echo hi", "outputs": []}},
    )
    rc = main(["replay", "c1", "--claims", cf, "--dry-run", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["claims"][0]["verdict"] == "dry_run"


def test_replay_reproduced(capsys, tmp_path):
    pytest.importorskip("yaml")
    cmd = _emit_cmd(tmp_path, {"metric": 0.95})
    cf = _write_claims(
        tmp_path,
        {
            "c1": {
                "track": "embedding",
                "command": cmd,
                "outputs": ["results.json"],
                "expected": {"metric_min": 0.90},
            }
        },
    )
    rc = main(["replay", "c1", "--claims", cf, "--cwd", str(tmp_path), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["claims"][0]["verdict"] == "reproduced"
    assert doc["summary"]["reproduced"] == 1


def test_replay_regressed(capsys, tmp_path):
    pytest.importorskip("yaml")
    cmd = _emit_cmd(tmp_path, {"metric": 0.50})
    cf = _write_claims(
        tmp_path,
        {
            "c1": {
                "command": cmd,
                "outputs": ["results.json"],
                "expected": {"metric_min": 0.90},  # 0.50 < 0.90
            }
        },
    )
    rc = main(["replay", "c1", "--claims", cf, "--cwd", str(tmp_path), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert doc["claims"][0]["verdict"] == "regressed"


def test_replay_command_error(capsys, tmp_path):
    pytest.importorskip("yaml")
    import sys

    cf = _write_claims(
        tmp_path,
        {"c1": {"command": f"{sys.executable} -c 'import sys; sys.exit(3)'"}},
    )
    rc = main(["replay", "c1", "--claims", cf, "--cwd", str(tmp_path), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert doc["claims"][0]["verdict"] == "error"
    assert doc["claims"][0]["exit_code"] == 3


def test_replay_manual_reference_claim(capsys, tmp_path):
    pytest.importorskip("yaml")
    cf = _write_claims(
        tmp_path,
        {"c1": {"track": "kv", "reference": "notebooks/claims/foo.ipynb"}},
    )
    rc = main(["replay", "c1", "--claims", cf, "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["claims"][0]["verdict"] == "manual"


def test_replay_all_track_filter(capsys, tmp_path):
    pytest.importorskip("yaml")
    cf = _write_claims(
        tmp_path,
        {
            "e1": {"track": "embedding", "reference": "x"},
            "k1": {"track": "kv", "reference": "y"},
        },
    )
    rc = main(["replay", "all", "--track", "kv", "--claims", cf, "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    ids = [r["id"] for r in doc["claims"]]
    assert ids == ["k1"]


def test_replay_unknown_claim(capsys, tmp_path):
    pytest.importorskip("yaml")
    cf = _write_claims(tmp_path, {"c1": {"reference": "x"}})
    assert main(["replay", "nope", "--claims", cf]) == 2
    assert "unknown claim" in capsys.readouterr().err


def test_replay_missing_claims_file(capsys, tmp_path):
    pytest.importorskip("yaml")
    assert main(["replay", "all", "--claims", str(tmp_path / "none.yaml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_replay_requires_target(capsys, tmp_path):
    pytest.importorskip("yaml")
    cf = _write_claims(tmp_path, {"c1": {"reference": "x"}})
    assert main(["replay", "--claims", cf]) == 2
    assert "specify a claim id" in capsys.readouterr().err


# ------------------------------------------------------------------ argparse
def test_no_subcommand_errors():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_plugin_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plugin"])


def test_unknown_command_errors():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nope"])


def test_all_expected_commands_present():
    parser = build_parser()
    # dig the registered subcommand names out of the subparsers action
    names = set()
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices:
            names.update(action.choices)
    expected = {
        "version",
        "plugin",
        "trace",
        "plan",
        "certify",
        "replay",
        "monitor",
        "probe",
    }
    assert expected <= names


def test_conformance_defaults():
    ns = build_parser().parse_args(["plugin", "conformance"])
    assert ns.heads == 4 and ns.seq == 96 and ns.dim == 64
    assert ns.names == [] and ns.shape is None
