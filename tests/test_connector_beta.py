"""Beta-tranche tests: metrics, async saves + backpressure, atomic persistence.

The observability claims are tested, not asserted: every miss has a cause,
byte accounting distinguishes logical from physical, backpressure is bounded
and counted, and the on-disk store is invisible until its COMMIT lands.
"""

import numpy as np
import pytest

from turboquant_pro.connectors import (
    IncompatibleProfile,
    KVIdentityProfile,
    TurboQuantBlockStore,
)
from turboquant_pro.connectors.metrics import ConnectorMetrics


def _profile(**over) -> KVIdentityProfile:
    base = dict(
        model_repo="org/model",
        model_revision="abc",
        weight_fingerprint="wf",
        tokenizer_fingerprint="tf",
        architecture="X",
        adapter_identity="",
        rope={"theta": 1e4},
        attention_backend="flash",
        kv_layout_version="v1",
        sliding_window=0,
        n_layers=2,
        n_heads=4,
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


def _kv(tokens=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((4, tokens, 32)).astype(np.float32)


def test_metrics_counters_and_causes():
    store = TurboQuantBlockStore()
    k = _kv()
    store.save("r1", "l0", k, k)
    assert store.load("r1", "l0") is not None
    assert store.load("r1", "nope") is None  # empty miss
    store.evict("r1")
    m = store.metrics.to_dict()
    assert m["saves"] == 1 and m["hits"] == 1 and m["misses_empty"] == 1
    assert m["evictions"] == 1
    assert m["bytes_logical"] > m["bytes_physical"] > 0  # compression measured
    assert m["effective_expansion"] > 1.5
    assert m["hit_rate"] == pytest.approx(0.5)


def test_metrics_prometheus_exposition():
    m = ConnectorMetrics()
    m.inc("saves")
    m.miss("corrupt")
    text = m.to_prometheus()
    assert "# TYPE tqp_kv_saves counter" in text
    assert "tqp_kv_misses_corrupt 1" in text
    assert 'tqp_kv_save_latency_seconds{quantile="0.99"}' in text
    with pytest.raises(KeyError):
        m.miss("unnamed_cause")  # every miss must have a registered cause


def test_async_saves_flush_barrier():
    store = TurboQuantBlockStore(async_saves=True, queue_depth=8)
    k = _kv()
    for i in range(12):
        assert store.save_async(f"r{i}", "l0", k, k)
    store.flush()
    assert store.metrics.to_dict()["saves"] == 12
    assert store.matched_tokens("r11") == 32
    store.close()


def test_backpressure_drop_policy_sheds_and_counts():
    store = TurboQuantBlockStore(async_saves=True, queue_depth=1, backpressure="drop")
    # Stall the worker by filling the queue faster than tiny saves drain it.
    k = _kv(tokens=256)
    results = [store.save_async(f"r{i}", "l0", k, k) for i in range(30)]
    store.flush()
    m = store.metrics.to_dict()
    dropped = results.count(False)
    assert m["backpressure_dropped"] == dropped
    assert m["saves"] == 30 - dropped
    store.close()


def test_persistence_roundtrip_no_pickle(tmp_path):
    src = TurboQuantBlockStore(profile=_profile())
    k = _kv()
    src.save("r1", "l0", k, k)
    src.save("r1", "l1", k, k)
    n = src.save_to_dir(str(tmp_path / "store"))
    assert n == 2
    dst = TurboQuantBlockStore(profile=_profile())
    assert dst.load_from_dir(str(tmp_path / "store")) == 2
    got = dst.load("r1", "l0")
    want = src.load("r1", "l0")
    np.testing.assert_allclose(got[0], want[0], atol=1e-6)
    # The blobs must be numpy archives, not pickles.
    rec_files = list((tmp_path / "store").glob("*.rec"))
    assert rec_files and rec_files[0].read_bytes()[:2] == b"PK"  # zip magic


def test_persistence_without_commit_marker_is_invisible(tmp_path):
    src = TurboQuantBlockStore(profile=_profile())
    src.save("r1", "l0", _kv(), _kv())
    src.save_to_dir(str(tmp_path / "store"))
    (tmp_path / "store" / "COMMIT").unlink()  # simulate crash before commit
    dst = TurboQuantBlockStore(profile=_profile())
    with pytest.raises(IncompatibleProfile, match="COMMIT"):
        dst.load_from_dir(str(tmp_path / "store"))


def test_persistence_tampered_record_skipped_others_load(tmp_path):
    src = TurboQuantBlockStore(profile=_profile())
    src.save("r1", "l0", _kv(), _kv())
    src.save("r1", "l1", _kv(), _kv())
    src.save_to_dir(str(tmp_path / "store"))
    victim = sorted((tmp_path / "store").glob("*.rec"))[0]
    victim.write_bytes(victim.read_bytes()[:-8] + b"deadbeef")
    dst = TurboQuantBlockStore(profile=_profile())
    assert dst.load_from_dir(str(tmp_path / "store")) == 1
    assert dst.metrics.to_dict()["integrity_failures"] == 1


def test_persistence_profile_mismatch_refuses_directory(tmp_path):
    src = TurboQuantBlockStore(profile=_profile())
    src.save("r1", "l0", _kv(), _kv())
    src.save_to_dir(str(tmp_path / "store"))
    dst = TurboQuantBlockStore(profile=_profile(model_revision="other"))
    with pytest.raises(IncompatibleProfile, match="digest"):
        dst.load_from_dir(str(tmp_path / "store"))
