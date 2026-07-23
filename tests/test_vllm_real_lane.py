"""Pinned-vLLM conformance lane (runs only where vLLM is installed).

This is the alpha-gate smoke from the 2.0 roadmap: against a REAL, pinned
vLLM (see ``.github/workflows/vllm-lane.yml`` and ``COMPATIBILITY_2.0.md``),
verify that the connector genuinely subclasses the V1 base (not the shim),
satisfies its abstract surface, registers through the real factory, and that
identity extraction reads a real vLLM config. Engine execution (GPU) is the
beta gate; this lane pins the *protocol*, which is what upstream marks
experimental and most likely to drift.
"""

import importlib
import sys

import pytest

vllm = pytest.importorskip("vllm")


def _reload_connector():
    """Re-import with vLLM present so the real base is bound, not the shim."""
    for m in list(sys.modules):
        if m.startswith("turboquant_pro.connectors"):
            del sys.modules[m]
    return importlib.import_module("turboquant_pro.connectors.vllm_v1")


def test_connector_binds_the_real_v1_base():
    mod = _reload_connector()
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
    )

    assert issubclass(mod.TurboQuantKVConnector, KVConnectorBase_V1)
    # Abstract-surface check: every abstract method of the pinned base must be
    # implemented — instantiation fails loudly on drift.
    abstract = getattr(KVConnectorBase_V1, "__abstractmethods__", frozenset())
    missing = [
        name
        for name in abstract
        if getattr(mod.TurboQuantKVConnector, name, None)
        is getattr(KVConnectorBase_V1, name, None)
    ]
    assert not missing, f"unimplemented abstract methods under this pin: {missing}"


def test_register_with_real_factory():
    mod = _reload_connector()
    assert mod.register() is True
    # Idempotent re-registration must not raise.
    assert mod.register() is True


def test_identity_extraction_from_real_engine_config():
    from turboquant_pro.connectors.identity import KVIdentityProfile

    engine_args = vllm.EngineArgs(model="facebook/opt-125m", load_format="dummy")
    cfg = engine_args.create_engine_config()
    p = KVIdentityProfile.from_vllm_config(cfg, quant={"key": "per_channel"})
    assert p.model_repo == "facebook/opt-125m"
    assert p.n_layers and p.n_heads and p.block_size
    # Non-derivable facts MUST remain unknown -> profile incomplete -> no reuse.
    assert p.weight_fingerprint is None
    assert not p.is_complete
