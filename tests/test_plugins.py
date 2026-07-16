"""
Tests for the quantizer plugin protocol, registry, and conformance kit (P0 of
docs/DESIGN_hardware_and_plugins.md).

The exit criterion for P0 is dogfood: the in-tree formats must register and
pass their own conformance suite through exactly the interface an out-of-tree
package would use.

Usage:
    pytest tests/test_plugins.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

import turboquant_pro.plugins as plugins
from turboquant_pro.plugin_conformance import assert_conformance, run_conformance
from turboquant_pro.plugins import (
    TARGET_KV_KEY,
    TARGET_KV_VALUE,
    PluginSpec,
    Quantizer,
    affine_params,
    available_plugins,
    create,
    get_plugin,
    native_dtype,
)

RNG = np.random.default_rng(7)
H, S, D = 4, 96, 64
THETA = 1e6


def _keys(offset_scale: float = 4.0) -> np.ndarray:
    off = RNG.uniform(-offset_scale, offset_scale, size=(1, H, 1, D))
    return (off + RNG.standard_normal((1, H, S, D))).astype(np.float32)


class TestRegistry:
    def test_in_tree_plugins_registered(self) -> None:
        specs = available_plugins()
        assert "per_channel" in specs and "polar" in specs
        assert TARGET_KV_KEY in specs["per_channel"].targets
        assert TARGET_KV_VALUE in specs["polar"].targets

    def test_filter_by_target(self) -> None:
        assert "polar" not in available_plugins(target=TARGET_KV_KEY)
        assert "polar" in available_plugins(target=TARGET_KV_VALUE)

    def test_create_returns_quantizer_protocol(self) -> None:
        q = create("per_channel", head_dim=D, n_heads=H, nf4_asym=True)
        assert isinstance(q, Quantizer)
        q2 = create("polar", head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=0)
        assert isinstance(q2, Quantizer)

    def test_unknown_plugin_lists_available(self) -> None:
        with pytest.raises(KeyError, match="per_channel"):
            get_plugin("no_such_plugin")

    def test_name_collision_rejected(self) -> None:
        spec = available_plugins()["per_channel"]
        with pytest.raises(ValueError, match="already registered"):
            plugins.register(spec)

    def test_spec_validation(self) -> None:
        with pytest.raises(ValueError, match="unknown targets"):
            PluginSpec(name="x", factory=lambda: None, targets=frozenset({"bogus"}))
        with pytest.raises(ValueError, match="tier"):
            PluginSpec(
                name="x",
                factory=lambda: None,
                targets=frozenset({TARGET_KV_KEY}),
                tier="production",
            )

    def test_entry_point_discovery(self, monkeypatch) -> None:
        """An out-of-tree package's entry point lands in the registry; a
        broken one is skipped without taking the registry down."""

        class _FakeEP:
            def __init__(self, name, obj):
                self.name, self._obj = name, obj

            def load(self):
                if isinstance(self._obj, Exception):
                    raise self._obj
                return self._obj

        good = PluginSpec(
            name="fake_external",
            factory=lambda **kw: create("per_channel", **kw),
            targets=frozenset({TARGET_KV_KEY}),
        )
        eps = [_FakeEP("good", good), _FakeEP("broken", RuntimeError("boom"))]
        monkeypatch.setattr(plugins.metadata, "entry_points", lambda group: eps)
        monkeypatch.setitem(plugins.__dict__, "_ENTRY_POINTS_LOADED", False)
        try:
            new = plugins.load_entry_point_plugins(force=True)
            assert new == ["fake_external"]
            assert "fake_external" in available_plugins()
        finally:
            plugins._REGISTRY.pop("fake_external", None)


class TestConformanceInTree:
    """The P0 exit criterion: every in-tree format passes its own suite."""

    @pytest.mark.parametrize(
        "config",
        [
            {"bits": 4},
            {"bits": 3},
            {"nf4": True},
            {"nf4_asym": True},
            {"nf4_asym": True, "outlier_frac": 0.02},
            {
                "nf4_asym": True,
                "outlier_frac": 0.02,
                "zero_point": "sparse",
                "rope_theta": THETA,
            },
        ],
        ids=["uniform4", "uniform3", "nf4", "nf4_asym", "outliers", "zp_sparse"],
    )
    def test_per_channel_variants(self, config) -> None:
        q = create("per_channel", head_dim=D, n_heads=H, **config)
        report = assert_conformance(q, _keys())
        assert report.results["affine"].startswith("pass"), report

    def test_per_channel_bias_zero_point(self) -> None:
        bias = RNG.uniform(-2, 2, size=(H, D)).astype(np.float32)
        q = create(
            "per_channel",
            head_dim=D,
            n_heads=H,
            nf4_asym=True,
            outlier_frac=0.02,
            zero_point="bias",
            rope_theta=THETA,
            k_bias=bias,
        )
        assert_conformance(q, _keys(), compress_kwargs={"position_start": 100})

    def test_per_channel_nuq_degrades_not_fails(self) -> None:
        """nuq has no fused form: affine must SKIP (never FAIL) and the
        roundtrip must still pass -- the documented graceful degrade."""
        q = create("per_channel", head_dim=D, n_heads=H, nuq=True)
        report = assert_conformance(q, _keys())
        assert report.results["affine"].startswith("skip")

    def test_polar_values(self) -> None:
        q = create("polar", head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=0)
        report = assert_conformance(q, RNG.standard_normal((1, H, S, D)))
        assert report.results["affine"].startswith("skip")
        assert native_dtype(q) is None


class TestConformanceCatchesViolations:
    """The kit must reject a plugin that lies about the affine contract."""

    def test_wrong_grid_params_fail(self) -> None:
        q = create("per_channel", head_dim=D, n_heads=H, nf4_asym=True)

        class Liar:
            def __init__(self, inner):
                self._inner = inner

            def compress(self, x, **kw):
                return self._inner.compress(x, **kw)

            def decompress(self, c):
                return self._inner.decompress(c)

            def codes(self, c):
                return self._inner.codes(c)

            def grid_params(self, c):
                mu, w, g = self._inner.grid_params(c)
                return mu + 1.0, w, g  # wrong zero-point

        report = run_conformance(Liar(q), _keys())
        assert not report.passed
        assert report.results["affine"].startswith("FAIL")

    def test_broken_roundtrip_fail(self) -> None:
        class Zero:
            def compress(self, x, **kw):
                return x.shape

            def decompress(self, c):
                return np.zeros(c, dtype=np.float32)

        report = run_conformance(Zero(), _keys())
        assert not report.passed
        assert report.results["roundtrip"].startswith("FAIL")


class TestAffineHelpers:
    def test_affine_params_absent_is_none(self) -> None:
        q = create("polar", head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=0)
        c = q.compress(RNG.standard_normal((1, H, S, D)).astype(np.float32))
        assert affine_params(q, c) is None

    def test_affine_matches_fused_reference(self) -> None:
        """The plugin capability surface is byte-compatible with what the M4
        fused path consumes (PreparedPCKBlock's inputs)."""
        from turboquant_pro.kv_fused_pck import _codes, _grid_params

        q = create(
            "per_channel", head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.02
        )
        x = _keys()
        c = q.compress(x)
        mu, w, g = affine_params(q, c)
        mu2, w2, g2 = _grid_params(q._q, c)
        assert np.array_equal(mu, mu2) and np.array_equal(w, w2)
        assert np.array_equal(np.asarray(g), np.asarray(g2))
        assert np.array_equal(q.codes(c), _codes(c))


class TestResolvePlugins:
    """P4 exit: model in, named recipe out -- operator_trace disciplines
    resolved to registered plugin names, both quantization targets."""

    def _model(self):
        torch = pytest.importorskip("torch")
        nn = torch.nn

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32, bias=False)
                self.k_proj = nn.Linear(32, 32, bias=False)
                self.v_proj = nn.Linear(32, 32, bias=False)
                self.o_proj = nn.Linear(32, 32, bias=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()
                self.up_proj = nn.Linear(32, 64, bias=False)

        return M()

    def test_kv_keys_resolve_to_per_channel_family(self):
        from turboquant_pro.plugins import resolve_plugins

        rec = resolve_plugins(self._model(), target="kv_activation")
        k = next(v for n, v in rec.items() if "k_proj" in n)
        assert k["family"] == "per_channel" and k["protect_dc"]
        assert k["plugins"][0] == "per_channel"

    def test_weights_resolve_to_gptq_awq_family_when_registered(self):
        import sys

        for pkg in ("plugins/tqp-gptq-awq", "plugins/tqp-bnb"):
            sys.path.insert(0, pkg)
        from tqp_bnb.plugin import SPEC_NF4
        from tqp_gptq_awq.plugin import SPEC_AWQ, SPEC_GPTQ

        from turboquant_pro.plugins import register, resolve_plugins

        for spec in (SPEC_GPTQ, SPEC_AWQ, SPEC_NF4):
            try:
                register(spec)
            except ValueError:
                pass
        rec = resolve_plugins(self._model(), target="weight")
        q = next(v for n, v in rec.items() if "q_proj" in n)
        assert q["family"] == "symmetric"
        assert {"gptq", "awq", "bnb_nf4"} <= set(q["plugins"])
