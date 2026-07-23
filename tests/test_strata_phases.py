"""STRATA Phases 2-4: remedies, per-area operating points, blast radius."""

import numpy as np
import pytest

from turboquant_pro.strata import build_area_map
from turboquant_pro.strata_ops import (
    StratifiedIndex,
    allocate_by_fragility,
    area_scoped_contract_key,
    stale_set,
)
from turboquant_pro.strata_remedies import (
    AreaCentering,
    csls_rescore,
    mutual_proximity_scalar,
    pack_hubness_trailer,
    unpack_hubness_trailer,
)


def _corpus(n=400, d=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels = ["A" if i % 2 == 0 else "B" for i in range(n)]
    return x, labels


# --- Phase 2: r_k scalar + trailer -----------------------------------------


def test_mutual_proximity_scalar_ranges_and_hub_bias():
    x, _ = _corpus()
    r = mutual_proximity_scalar(x, k=10)
    assert r.shape == (len(x),) and r.dtype == np.float32
    assert (r > -1.0).all() and (r < 1.0).all()
    # A planted near-duplicate cluster has higher local similarity than the
    # population — the scalar sees exactly what CSLS corrects.
    y = x.copy()
    y[:12] = y[0] + 0.01 * np.random.default_rng(1).standard_normal((12, x.shape[1]))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    r2 = mutual_proximity_scalar(y, k=10)
    assert r2[:12].mean() > np.median(r2) + 0.1


def test_csls_rescore_demotes_high_rk_candidates():
    scores = np.array([0.9, 0.9, 0.9])
    rk = np.array([0.8, 0.2, 0.5], dtype=np.float32)
    out = csls_rescore(scores, rk)
    assert out.argmax() == 1 and out.argmin() == 0


def test_hubness_trailer_roundtrip_and_tamper():
    r = np.linspace(-0.5, 0.9, 257, dtype=np.float32)
    blob = pack_hubness_trailer(r, 10, estimator="exact_knn")
    doc = unpack_hubness_trailer(blob)
    assert doc["k"] == 10 and doc["estimator"] == "exact_knn"
    assert np.allclose(doc["r_k"], r)
    bad = bytearray(blob)
    bad[9] ^= 0xFF  # flip a payload byte: the trailer's own hash must catch it
    with pytest.raises(ValueError, match="hash"):
        unpack_hubness_trailer(bytes(bad))
    with pytest.raises(KeyError):
        pack_hubness_trailer(r, 10, estimator="vibes")


def test_area_centering_identity_and_apply():
    x, labels = _corpus()
    amap = build_area_map(x, "by:test", labels=np.asarray(labels))
    cent = AreaCentering.fit(x, amap)
    y = cent.apply(x, labels)
    lab = np.asarray(labels)
    for a in ("A", "B"):  # per-area means are (near) removed, rows unit-norm
        assert np.linalg.norm(y[lab == a].mean(0)) < np.linalg.norm(x[lab == a].mean(0))
    assert np.allclose(np.linalg.norm(y, axis=1), 1.0, atol=1e-5)
    # Centroids are encoder parameters: digest moves when they move.
    d1 = cent.params_digest
    cent.centroids["A"] = cent.centroids["A"] + 0.1
    assert cent.params_digest != d1
    with pytest.raises(KeyError):
        cent.apply(x[:4], ["A", "A", "Z", "B"])


# --- Phase 3: fragility allocation + identity-gated stratified index --------


def _fake_hubdiff(anti):
    return {
        "areas": [
            {"id": a, "verdict": "pass", "anti_hub_recall": v, "n_queries": 5000}
            for a, v in anti.items()
        ]
    }


def test_allocate_by_fragility_spends_on_the_weak():
    counts = {"A": 1000, "B": 1000, "C": 1000, "D": 1000}
    rep = _fake_hubdiff({"A": 0.95, "B": 0.70, "C": 0.90, "D": 0.60})
    bits = allocate_by_fragility(rep, counts, budget_bits_per_row=3.0)
    assert bits["D"] >= bits["B"] >= bits["A"]
    assert bits["D"] == 4
    mean = sum(bits[a] * counts[a] for a in counts) / sum(counts.values())
    assert mean <= 3.0 + 1e-9


def test_stratified_index_search_and_digest_refusal():
    x, labels = _corpus(n=600)
    amap = build_area_map(x, "by:test", labels=np.asarray(labels))
    idx = StratifiedIndex.build(x, amap, {"A": 4, "B": 2}, output_dim=16, seed=1)
    assert set(idx.areas) == {"A", "B"}
    assert idx.metadata("A")["area_codec_params"]["bits"] == 4
    assert 2.0 < idx.bits_per_row_mean() < 4.0
    ids, sc = idx.search(x[:8], k=5, area_map_digest=amap.digest)
    assert ids.shape == (8, 5)
    # Self-retrieval sanity: each query's own row is its top hit.
    assert (ids[:, 0] == np.arange(8)).mean() >= 0.75
    with pytest.raises(ValueError, match="area_map_mismatch"):
        idx.search(x[:2], k=5, area_map_digest="0" * 64)


# --- Phase 4: contract key + stale sets ------------------------------------


def test_area_scoped_contract_key_extends_not_replaces():
    base = {"index_fingerprint": "f", "metric": "cosine", "k": 10}
    key = area_scoped_contract_key(base, "d" * 64, "english")
    assert key["index_fingerprint"] == "f" and key["area_id"] == "english"
    assert "area_id" not in base  # non-mutating


def test_stale_sets_match_section5_table():
    x, labels = _corpus()
    amap = build_area_map(x, "by:test", labels=np.asarray(labels))
    adj = {"A": {"B"}}
    assert stale_set("mutation", amap, area="A", adjacency=adj) == {"A", "B"}
    assert stale_set("mutation", amap, area="B") == {"B"}
    assert stale_set("map_recompute", amap) == {"A", "B"}
    assert stale_set("operating_point", amap, area="A") == {"A"}
    with pytest.raises(KeyError):
        stale_set("vibes", amap, area="A")
    with pytest.raises(KeyError):
        stale_set("mutation", amap, area="nope")
