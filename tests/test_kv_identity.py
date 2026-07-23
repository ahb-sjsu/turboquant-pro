"""Tests for the KV identity profile (roadmap P1-M1) + store persistence gates.

The one unacceptable failure is wrong-prefix reuse; these tests pin the
structural defenses: incomplete profiles match nothing (including themselves),
content addresses cannot be minted under uncertainty, imports refuse
mismatched or unknown identity wholesale, and per-record corruption downgrades
to a miss rather than a decode.
"""

import dataclasses
import types

import numpy as np
import pytest

from turboquant_pro.connectors import (
    IncompatibleProfile,
    KVIdentityProfile,
    TurboQuantBlockStore,
    prefix_block_hashes,
)


def _complete(**over) -> KVIdentityProfile:
    base = dict(
        model_repo="org/model",
        model_revision="abc123",
        weight_fingerprint="wf:1",
        tokenizer_fingerprint="tf:1",
        architecture="LlamaForCausalLM",
        adapter_identity="",
        rope={"theta": 10000.0, "scaling": None, "max_position_embeddings": 4096},
        attention_backend="flash",
        kv_layout_version="v1",
        sliding_window=0,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        head_dim=32,
        tp_size=1,
        pp_size=1,
        kv_dtype="auto",
        block_size=16,
        quant={"key": "per_channel", "value": "polar"},
        encoder_version="test",
    )
    base.update(over)
    return KVIdentityProfile(**base)


def test_digest_deterministic_and_field_sensitive():
    a, b = _complete(), _complete()
    assert a.is_complete and a.digest() == b.digest()
    assert a.compatible(b)
    c = _complete(model_revision="def456")
    assert a.digest() != c.digest()
    assert not a.compatible(c)
    d = _complete(quant={"key": "per_channel", "value": "polar", "bits": 3})
    assert a.digest() != d.digest()  # discipline parameters are identity


def test_incomplete_profile_matches_nothing_including_itself():
    p = _complete(weight_fingerprint=None)
    assert not p.is_complete
    assert "weight_fingerprint" in p.unknown_fields
    assert not p.compatible(p)  # uncertain => miss, even against itself
    assert not p.compatible(_complete())
    assert not _complete().compatible(p)
    with pytest.raises(IncompatibleProfile, match="unknown fields"):
        _complete().require_compatible(p)


def test_require_compatible_names_the_differing_fields():
    with pytest.raises(IncompatibleProfile, match="model_revision"):
        _complete().require_compatible(_complete(model_revision="other"))


def test_prefix_block_hashes_chain_and_diverge():
    p = _complete()
    ids = list(range(64))
    h = prefix_block_hashes(p, ids, block_size=16)
    assert len(h) == 4
    # Shared prefix shares leading hashes; divergence changes the tail only.
    ids2 = ids[:32] + [999] + ids[33:]
    h2 = prefix_block_hashes(p, ids2, block_size=16)
    assert h2[:2] == h[:2] and h2[2:] != h[2:]
    # A different profile changes EVERY hash.
    h3 = prefix_block_hashes(_complete(model_revision="other"), ids, block_size=16)
    assert all(x != y for x, y in zip(h, h3))
    # Partial trailing blocks are not addressable.
    assert len(prefix_block_hashes(p, ids[:70], block_size=16)) == 4


def test_prefix_hashes_refuse_incomplete_profiles():
    with pytest.raises(IncompatibleProfile, match="incomplete"):
        prefix_block_hashes(_complete(tokenizer_fingerprint=None), [1, 2, 3])


def test_from_vllm_config_leaves_unknowables_none():
    hf = types.SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        sliding_window=None,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
    )
    cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model="org/m", revision="r1", hf_config=hf),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        cache_config=types.SimpleNamespace(cache_dtype="auto", block_size=16),
    )
    p = KVIdentityProfile.from_vllm_config(cfg, quant={"key": "per_channel"})
    assert p.model_repo == "org/m" and p.n_kv_heads == 4
    # The fingerprints are NOT derivable from config: they must stay unknown,
    # which keeps the profile incomplete and reuse impossible until supplied.
    assert p.weight_fingerprint is None and p.tokenizer_fingerprint is None
    assert not p.is_complete
    complete = dataclasses.replace(
        p,
        weight_fingerprint="wf",
        tokenizer_fingerprint="tf",
        attention_backend="flash",
        kv_layout_version="v1",
    )
    assert complete.is_complete


def _stored(profile):
    store = TurboQuantBlockStore(profile=profile)
    rng = np.random.default_rng(0)
    k = rng.standard_normal((4, 32, 32)).astype(np.float32)
    store.save("req-1", "layer.0", k, k)
    store.save("req-1", "layer.1", k, k)
    return store


def test_export_import_roundtrip_with_matching_profiles():
    p = _complete()
    src = _stored(p)
    state = src.export_state()
    dst = TurboQuantBlockStore(profile=_complete())
    assert dst.import_state(state) == 2
    assert dst.matched_tokens("req-1") == 32
    assert dst.load("req-1", "layer.0") is not None


def test_export_refused_without_complete_profile():
    with pytest.raises(IncompatibleProfile):
        _stored(None).export_state()
    with pytest.raises(IncompatibleProfile):
        _stored(_complete(weight_fingerprint=None)).export_state()


def test_import_refused_on_profile_mismatch():
    state = _stored(_complete()).export_state()
    dst = TurboQuantBlockStore(profile=_complete(model_revision="other"))
    with pytest.raises(IncompatibleProfile, match="digest mismatch"):
        dst.import_state(state)


def test_import_skips_corrupt_records_as_misses():
    state = _stored(_complete()).export_state()
    key = next(iter(state["records"]))
    state["records"][key]["blob"] = state["records"][key]["blob"][:-3] + b"xxx"
    dst = TurboQuantBlockStore(profile=_complete())
    assert dst.import_state(state) == 1  # the intact record still arrives


def test_corrupt_in_store_record_loads_as_miss():
    store = _stored(_complete())
    with store._lock:
        rec = store._records[("req-1", "layer.0")]
        object.__setattr__(rec, "key_payload", b"garbage")
    assert store.load("req-1", "layer.0") is None  # miss, no exception
    assert store.load("req-1", "layer.1") is not None
