"""Runtime safe fallback: fragile signals back off, wide margins run cheap.

Each evaluator is exercised on both branches with the real instruments, plus the
adaptive TQEIndex.search escalation. Pure-numpy; runs in CI.
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import (
    RuntimeDecision,
    TQEIndex,
    TQPRuntimePolicy,
    certificate_from_embeddings,
)
from turboquant_pro.runtime_policy import (
    KEEP_ROUTER_FP16,
    LOG_TAU_OR_FP16,
    PER_CHANNEL_OR_FP16,
    PROCEED,
    RECALIBRATE_OR_DISABLE_POLAR,
    REFIT_OR_MIGRATE,
    REQUIRE_EXACT_RERANK,
    RERANK_MORE,
)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Retrieval                                                                    #
# --------------------------------------------------------------------------- #


def test_retrieval_tied_escalates():
    p = TQPRuntimePolicy(retrieval_gap_floor=0.01)
    tied = np.tile(np.linspace(1.0, 0.99, 10), (50, 1))  # boundary gap ~0.001
    d = p.evaluate_retrieval(tied)
    assert d.conservative and d.action == RERANK_MORE
    assert d.params["oversample"] == p.rerank_oversample


def test_retrieval_wide_proceeds():
    p = TQPRuntimePolicy(retrieval_gap_floor=0.01)
    wide = np.tile(np.linspace(1.0, 0.0, 10), (50, 1))  # boundary gap ~0.11
    d = p.evaluate_retrieval(wide)
    assert not d.conservative and d.action == PROCEED


# --------------------------------------------------------------------------- #
# Certificate                                                                  #
# --------------------------------------------------------------------------- #


def test_certificate_vacuous_requires_rerank():
    p = TQPRuntimePolicy()
    row = RNG.standard_normal((1, 16)).astype(np.float32)
    ident = np.repeat(row, 40, axis=0)  # degenerate -> vacuous certificate
    cert = certificate_from_embeddings(ident, ident, n_anchors=32, seed=0)
    d = p.evaluate_certificate(cert)
    assert d.conservative and d.action == REQUIRE_EXACT_RERANK


def test_certificate_strong_proceeds():
    p = TQPRuntimePolicy(min_tau_floor=0.5)
    orig = RNG.standard_normal((256, 32)).astype(np.float32)
    recon = orig + 0.01 * RNG.standard_normal(orig.shape).astype(np.float32)
    cert = certificate_from_embeddings(orig, recon, n_anchors=64, seed=0)
    d = p.evaluate_certificate(cert)
    assert not d.conservative and d.action == PROCEED


# --------------------------------------------------------------------------- #
# KV keys                                                                      #
# --------------------------------------------------------------------------- #


def test_kv_keys_unknown_regime_is_conservative():
    p = TQPRuntimePolicy()
    d = p.evaluate_kv_keys(regime="unknown")
    assert d.conservative and d.action == PER_CHANNEL_OR_FP16


def test_kv_keys_dc_offset_recommends_per_channel():
    # DC-offset keys (the v1.2.0 regime): the (A2) probe should reject polar.
    p = TQPRuntimePolicy()
    offset = RNG.uniform(-6.0, 6.0, size=(1, 64))
    keys = (offset + 0.15 * RNG.standard_normal((512, 64))).astype(np.float32)
    d = p.evaluate_kv_keys(keys, bits=4, seed=0)
    assert d.action in (PER_CHANNEL_OR_FP16, PROCEED)
    assert d.measured["recommendation"] in ("polar", "per_channel")


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #


def test_routing_tiny_margins_keep_fp16():
    p = TQPRuntimePolicy(routing_margin_floor=0.02)
    logits = (RNG.standard_normal((2000, 32)) * 0.001).astype(np.float64)
    d = p.evaluate_routing(logits, k=1)
    assert d.conservative and d.action == KEEP_ROUTER_FP16


def test_routing_wide_margins_proceed():
    p = TQPRuntimePolicy(routing_margin_floor=0.02)
    logits = (RNG.standard_normal((2000, 32)) * 10.0).astype(np.float64)
    d = p.evaluate_routing(logits, k=1)
    assert not d.conservative and d.action == PROCEED


# --------------------------------------------------------------------------- #
# Decay                                                                        #
# --------------------------------------------------------------------------- #


def test_decay_slow_channels_use_log_tau():
    p = TQPRuntimePolicy(decay_slow_fraction_ceiling=0.02)
    slow = RNG.uniform(0.92, 0.999, size=4096)  # mostly slow (a > 0.9)
    d = p.evaluate_decay(slow, seq_len=1024)
    assert d.conservative and d.action == LOG_TAU_OR_FP16


def test_decay_fast_channels_proceed():
    p = TQPRuntimePolicy(decay_slow_fraction_ceiling=0.02)
    fast = RNG.uniform(0.0, 0.5, size=4096)
    d = p.evaluate_decay(fast)
    assert not d.conservative and d.action == PROCEED


# --------------------------------------------------------------------------- #
# A2 drift                                                                     #
# --------------------------------------------------------------------------- #


def test_a2_norm_dominated_recalibrates():
    p = TQPRuntimePolicy(radial_drift_floor=0.15)
    d = p.evaluate_a2(median_tangential_fraction=0.05)
    assert d.conservative and d.action == RECALIBRATE_OR_DISABLE_POLAR


def test_a2_tangential_proceeds():
    p = TQPRuntimePolicy(radial_drift_floor=0.15)
    d = p.evaluate_a2(median_tangential_fraction=0.85)
    assert not d.conservative and d.action == PROCEED


# --------------------------------------------------------------------------- #
# Index drift                                                                  #
# --------------------------------------------------------------------------- #


def test_index_drift_stale_refits():
    p = TQPRuntimePolicy()
    corpus = RNG.standard_normal((800, 48)).astype(np.float32)
    idx = TQEIndex.create(corpus, output_dim=24, bits=4)
    shifted = (RNG.standard_normal((200, 48)) * 3.0 + 5.0).astype(np.float32)
    d = p.evaluate_index_drift(idx.drift(shifted))
    assert d.conservative and d.action == REFIT_OR_MIGRATE
    d2 = p.evaluate_index_drift(idx.drift(corpus[:200]))
    assert not d2.conservative and d2.action == PROCEED


# --------------------------------------------------------------------------- #
# Aggregate + adaptive index search                                           #
# --------------------------------------------------------------------------- #


def test_evaluate_all_collects_decisions():
    p = TQPRuntimePolicy()
    decisions = p.evaluate_all(
        regime="unknown",
        gate_logits=(RNG.standard_normal((500, 16)) * 0.001),
        decays=RNG.uniform(0.95, 0.999, size=1024),
    )
    assert len(decisions) == 3
    assert all(isinstance(d, RuntimeDecision) for d in decisions)
    assert all(d.conservative for d in decisions)


def _rank_corpus(n=1500, dim=48, seed=0):
    rng = np.random.default_rng(seed)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    coeffs = rng.standard_normal((n, rank)) * np.linspace(1.0, 0.3, rank)
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def test_adaptive_search_escalates_when_tied():
    corpus = _rank_corpus()
    idx = TQEIndex.create(corpus, output_dim=32, bits=4)
    q = corpus[:60]

    def recall(ids):
        gt = np.arange(60)  # each query is its own vector; id == position here
        return float(np.mean([qi in row for qi, row in zip(gt, ids)]))

    single, _ = idx.search(q, k=10)
    # A policy with a high gap floor treats every boundary as tied -> escalates.
    always = TQPRuntimePolicy(retrieval_gap_floor=1e9, rerank_oversample=15)
    adaptive, _ = idx.search(q, k=10, policy=always)
    # A policy that never triggers leaves the single-pass result unchanged.
    never = TQPRuntimePolicy(retrieval_gap_floor=0.0)
    passthrough, _ = idx.search(q, k=10, policy=never)

    np.testing.assert_array_equal(single, passthrough)
    assert recall(adaptive) >= recall(single)  # escalation never hurts recall
    assert recall(adaptive) > 0.95  # exact rerank recovers the true neighbours
