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
    import types

    from turboquant_pro.connectors.identity import KVIdentityProfile

    try:
        engine_args = vllm.EngineArgs(model="facebook/opt-125m", load_format="dummy")
        cfg = engine_args.create_engine_config()
    except RuntimeError:
        # CPU-only runner under a CUDA wheel: platform resolution fails before
        # any config exists. The full engine-config path is the GPU beta lane;
        # here we still validate extraction against the REAL HF config object
        # (attribute names are the drift surface for this test).
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained("facebook/opt-125m")
        cfg = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                model="facebook/opt-125m", revision="main", hf_config=hf
            ),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=types.SimpleNamespace(cache_dtype="auto", block_size=16),
        )
    p = KVIdentityProfile.from_vllm_config(cfg, quant={"key": "per_channel"})
    assert p.model_repo == "facebook/opt-125m"
    assert p.n_layers and p.n_heads and p.block_size
    # Non-derivable facts MUST remain unknown -> profile incomplete -> no reuse.
    assert p.weight_fingerprint is None
    assert not p.is_complete
