"""Rigorous tests for the ``tqp`` CLI dispatcher.

Covers dispatch, exit-code contracts (0 ok / 1 conformance-fail / 2 stub-or-guard),
argument parsing, output content, and error handling (unknown plugin, raising
factory, bad target). The ``trace`` subcommand tests skip cleanly where torch /
transformers are absent (e.g. minimal CI), and never touch the network.
"""

from __future__ import annotations

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
    assert main(["certify"]) == 2


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


# ------------------------------------------------------------------ stubs
@pytest.mark.parametrize("cmd", ["plan", "certify", "replay", "monitor", "probe"])
def test_stubs_exit_2_with_roadmap(cmd, capsys):
    assert main([cmd]) == 2
    out = capsys.readouterr().out
    assert "not implemented yet" in out and "roadmap" in out


@pytest.mark.parametrize("cmd", ["plan", "certify", "replay", "monitor", "probe"])
def test_stubs_swallow_positional_args(cmd):
    # stubs declare a nargs="*" positional sink, so extra positionals are
    # absorbed and still exit 2 (unknown -flags are correctly rejected instead)
    assert main([cmd, "foo", "bar"]) == 2


@pytest.mark.parametrize("cmd", ["plan", "certify", "replay", "monitor", "probe"])
def test_stubs_reject_unknown_flags(cmd):
    with pytest.raises(SystemExit):
        main([cmd, "--nonexistent-flag"])


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
