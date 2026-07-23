"""Tests for the vLLM V1 connector scaffold (no vLLM required).

These validate the protocol surface, the quantized round-trip through the
block store, and factory registration against a stubbed vLLM — the real-engine
end-to-end lane runs in CI with pinned vLLM versions (ROADMAP_2.0 M1).
"""

import sys
import types

import numpy as np
import pytest

from turboquant_pro.connectors import (
    TurboQuantBlockStore,
    TurboQuantKVConnector,
    register,
)


def _kv_block(n_heads=4, tokens=64, head_dim=32, seed=0):
    rng = np.random.default_rng(seed)
    off = rng.uniform(-3.0, 3.0, size=(n_heads, 1, head_dim))
    k = (off + rng.standard_normal((n_heads, tokens, head_dim))).astype(np.float32)
    v = rng.standard_normal((n_heads, tokens, head_dim)).astype(np.float32)
    return k, v


def test_store_roundtrip_preserves_kv_within_quant_tolerance():
    store = TurboQuantBlockStore()
    k, v = _kv_block()
    store.save("req-1", "layer.0", k, v)
    out = store.load("req-1", "layer.0")
    assert out is not None
    k2, v2 = out
    assert k2.shape == k.shape and v2.shape == v.shape
    # Quantized, not lossless — but the consumer-facing signal must hold:
    # relative Frobenius error small and per-head correlation near 1.
    assert np.linalg.norm(k2 - k) / np.linalg.norm(k) < 0.25
    assert np.linalg.norm(v2 - v) / np.linalg.norm(v) < 0.35
    for h in range(k.shape[0]):
        c = np.corrcoef(k[h].ravel(), k2[h].ravel())[0, 1]
        assert c > 0.97


def test_store_matched_tokens_and_evict():
    store = TurboQuantBlockStore()
    k, v = _kv_block(tokens=48)
    store.save("req-9", "layer.0", k, v)
    store.save("req-9", "layer.1", k, v)
    assert store.matched_tokens("req-9") == 48
    assert store.matched_tokens("unknown") == 0
    assert store.evict("req-9") == 2
    assert store.matched_tokens("req-9") == 0
    assert store.load("req-9", "layer.0") is None


def test_connector_protocol_surface_without_vllm():
    conn = TurboQuantKVConnector(vllm_config=None, role="worker")
    for meth in (
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
        "build_connector_meta",
        "request_finished",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
        "wait_for_save",
        "get_finished",
    ):
        assert callable(getattr(conn, meth)), meth


def test_connector_save_then_scheduler_match_then_finish_evicts():
    conn = TurboQuantKVConnector(vllm_config=None, role="worker")
    k, v = _kv_block(tokens=32)
    kv_layer = np.stack([k, v])
    meta = types.SimpleNamespace(request_id="req-7")
    conn.save_kv_layer("layer.3", kv_layer, meta)
    req = types.SimpleNamespace(request_id="req-7")
    extra, pending = conn.get_num_new_matched_tokens(req, num_computed_tokens=8)
    assert (extra, pending) == (24, False)
    extra, _ = conn.get_num_new_matched_tokens(req, num_computed_tokens=40)
    assert extra == 0  # never negative
    conn.get_finished({"req-7"})
    assert conn.store.matched_tokens("req-7") == 0


def test_register_against_stubbed_vllm(monkeypatch):
    calls = []

    class _Factory:
        @staticmethod
        def register_connector(name, module, cls):
            calls.append((name, module, cls))

    pkgs = {
        "vllm": types.ModuleType("vllm"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.kv_transfer": types.ModuleType(
            "vllm.distributed.kv_transfer"
        ),
        "vllm.distributed.kv_transfer.kv_connector": types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector"
        ),
    }
    fac = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.factory")
    fac.KVConnectorFactory = _Factory
    pkgs["vllm.distributed.kv_transfer.kv_connector.factory"] = fac
    for name, mod in pkgs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    assert register() is True
    assert calls == [
        (
            "TurboQuantConnector",
            "turboquant_pro.connectors.vllm_v1",
            "TurboQuantKVConnector",
        )
    ]


def test_register_without_vllm_returns_false():
    for m in list(sys.modules):
        if m.startswith("vllm"):
            pytest.skip("real vLLM present; skip the absence path")
    assert register() is False
