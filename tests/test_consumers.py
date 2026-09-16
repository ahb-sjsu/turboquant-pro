"""The consumer-metric contract: what a compression decision is allowed to read.

`turboquant_pro.consumers` is the third extension point beside `plugins` (the
codec) and `read_operators` (the operator it is judged against). This suite pins
the registry semantics, each built-in metric's direction and identity behaviour,
the alignment rule that makes the false-clear diagnostic meaningful, and the
error paths that must raise rather than substitute a metric the caller did not
ask for.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import consumers as cm
from turboquant_pro.consumers import (
    AttentionScoreConsumer,
    ConsumerSpec,
    DeclaredConsumer,
    TopKRetrievalConsumer,
    available_consumers,
    create_consumer,
    get_consumer,
    nominal_per_item,
    register_consumer,
    row_cosine,
)

EMB = frozenset({"embedding"})


@pytest.fixture
def clean_registry():
    """Restore the registry so a test's registrations do not leak."""
    before = dict(cm._REGISTRY)
    loaded = cm._ENTRY_POINTS_LOADED
    yield
    cm._REGISTRY.clear()
    cm._REGISTRY.update(before)
    cm._ENTRY_POINTS_LOADED = loaded


def _corpus(n=300, d=32, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _quantize(x, bits):
    """A plain per-row uniform round-trip, for degradation tests."""
    f = np.asarray(x, dtype=np.float32)
    lo = f.min(axis=1, keepdims=True)
    hi = f.max(axis=1, keepdims=True)
    levels = (1 << bits) - 1
    scale = np.maximum((hi - lo) / levels, 1e-20)
    codes = np.rint((f - lo) / scale).clip(0, levels)
    return (codes * scale + lo).astype(np.float32)


# ------------------------------------------------------------------ #
# The registry                                                        #
# ------------------------------------------------------------------ #


class TestConsumerSpec:
    def test_a_name_with_a_separator_is_rejected(self):
        with pytest.raises(ValueError, match="invalid consumer name"):
            ConsumerSpec(name="a/b", factory=lambda **_: None, targets=EMB)

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="invalid consumer name"):
            ConsumerSpec(name="", factory=lambda **_: None, targets=EMB)

    def test_an_unknown_target_is_rejected(self):
        with pytest.raises(ValueError, match="unknown targets"):
            ConsumerSpec(
                name="x", factory=lambda **_: None, targets=frozenset({"telemetry"})
            )

    def test_an_unknown_evidence_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown evidence_kind"):
            ConsumerSpec(
                name="x",
                factory=lambda **_: None,
                targets=EMB,
                evidence_kind="vibes",
            )

    def test_the_three_evidence_kinds_are_accepted(self):
        for kind in ("certificate", "statistical", "analytic"):
            ConsumerSpec(
                name="x", factory=lambda **_: None, targets=EMB, evidence_kind=kind
            )


class TestRegistry:
    def test_a_collision_raises_unless_overwrite(self, clean_registry):
        spec = ConsumerSpec(name="t_dup", factory=lambda **_: None, targets=EMB)
        register_consumer(spec)
        with pytest.raises(ValueError, match="already registered"):
            register_consumer(spec)
        assert register_consumer(spec, overwrite=True) is spec

    def test_an_unknown_consumer_names_what_is_available_and_says_it_abstains(self):
        with pytest.raises(KeyError) as e:
            get_consumer("clairvoyance")
        message = str(e.value)
        assert "clairvoyance" in message
        assert "topk_inner_product" in message
        assert "abstains" in message

    def test_available_consumers_filters_by_target_and_sorts(self):
        emb = available_consumers(target="embedding")
        keys = available_consumers(target="kv_key")
        assert list(emb) == sorted(emb)
        assert "topk_inner_product" in emb
        assert "attention_softmax" not in emb
        assert "attention_softmax" in keys
        assert "read_operator" in emb and "read_operator" in keys

    def test_a_factory_returning_the_wrong_shape_of_object_is_refused(
        self, clean_registry
    ):
        register_consumer(
            ConsumerSpec(name="t_bad", factory=lambda **_: object(), targets=EMB)
        )
        with pytest.raises(TypeError, match="has no"):
            create_consumer("t_bad")

    def test_the_builtin_specs_declare_themselves_honestly(self):
        spec = get_consumer("read_operator")
        assert spec.exact is True
        assert spec.evidence_kind == "analytic"
        assert get_consumer("declared").exact is False


class TestEntryPoints:
    """One broken out-of-tree package must not take the registry down."""

    def _fake_eps(self, monkeypatch, entries):
        class _EP:
            def __init__(self, name, loader):
                self.name = name
                self._loader = loader

            def load(self):
                return self._loader()

        monkeypatch.setattr(
            cm.metadata,
            "entry_points",
            lambda group=None: [_EP(n, ld) for n, ld in entries],
        )

    def test_a_spec_an_iterable_and_a_callable_all_register(
        self, clean_registry, monkeypatch
    ):
        one = ConsumerSpec(name="ep_one", factory=lambda **_: None, targets=EMB)
        two = ConsumerSpec(name="ep_two", factory=lambda **_: None, targets=EMB)
        three = ConsumerSpec(name="ep_three", factory=lambda **_: None, targets=EMB)
        self._fake_eps(
            monkeypatch,
            [
                ("a", lambda: one),
                ("b", lambda: [two]),
                ("c", lambda: (lambda: three)),
            ],
        )
        cm._ENTRY_POINTS_LOADED = False
        new = cm.load_entry_point_consumers(force=True)
        assert set(new) == {"ep_one", "ep_two", "ep_three"}

    def test_a_broken_entry_point_is_skipped_not_fatal(
        self, clean_registry, monkeypatch
    ):
        def boom():
            raise ImportError("this package is broken")

        good = ConsumerSpec(name="ep_good", factory=lambda **_: None, targets=EMB)
        self._fake_eps(monkeypatch, [("bad", boom), ("good", lambda: good)])
        cm._ENTRY_POINTS_LOADED = False
        new = cm.load_entry_point_consumers(force=True)
        assert new == ["ep_good"]

    def test_discovery_failure_returns_empty_rather_than_raising(
        self, clean_registry, monkeypatch
    ):
        def boom(group=None):
            raise RuntimeError("importlib.metadata is unhappy")

        monkeypatch.setattr(cm.metadata, "entry_points", boom)
        cm._ENTRY_POINTS_LOADED = False
        assert cm.load_entry_point_consumers(force=True) == []


# ------------------------------------------------------------------ #
# Retrieval                                                           #
# ------------------------------------------------------------------ #


class TestTopKRetrieval:
    def test_an_identity_round_trip_loses_no_neighbours(self):
        x = _corpus()
        for metric in ("inner_product", "cosine", "l2"):
            c = create_consumer(f"topk_{metric}", k=10, n_queries=40)
            assert np.allclose(c.per_item(x, x), 1.0), metric

    def test_one_score_per_query_not_per_row(self):
        c = create_consumer("topk_inner_product", k=5, n_queries=37)
        x = _corpus(n=400)
        assert c.per_item(x, x).shape == (37,)

    def test_an_explicit_query_sample_is_used_and_recorded(self):
        c = TopKRetrievalConsumer(k=5, n_queries=20)
        x = _corpus(n=200, d=16)
        q = _corpus(n=20, d=16, seed=5)
        scores = c.per_item(x, x, queries=q)
        assert scores.shape == (20,)
        assert c.corpus_as_queries is False

    def test_without_queries_the_corpus_stands_in_and_says_so(self):
        c = TopKRetrievalConsumer(k=5, n_queries=20)
        x = _corpus(n=200, d=16)
        c.per_item(x, x)
        assert c.corpus_as_queries is True

    def test_quality_degrades_monotonically_with_fewer_bits(self):
        x = _corpus(n=500, d=48, seed=3)
        c = create_consumer("topk_inner_product", k=10, n_queries=100)
        eight = float(c.per_item(x, _quantize(x, 8)).mean())
        four = float(c.per_item(x, _quantize(x, 4)).mean())
        two = float(c.per_item(x, _quantize(x, 2)).mean())
        assert eight >= four >= two
        assert eight > two

    def test_k_larger_than_the_corpus_clamps_rather_than_crashing(self):
        c = TopKRetrievalConsumer(k=1000, n_queries=5)
        x = _corpus(n=30, d=8)
        assert np.allclose(c.per_item(x, x), 1.0)

    def test_a_shape_mismatch_raises(self):
        c = TopKRetrievalConsumer(k=5, n_queries=5)
        with pytest.raises(ValueError, match="shape mismatch"):
            c.per_item(_corpus(n=50, d=8), _corpus(n=50, d=9))

    def test_queries_of_the_wrong_width_raise(self):
        c = TopKRetrievalConsumer(k=5, n_queries=5)
        x = _corpus(n=50, d=8)
        with pytest.raises(ValueError, match="dimensional"):
            c.per_item(x, x, queries=_corpus(n=5, d=16))

    def test_an_unknown_retrieval_metric_raises(self):
        c = TopKRetrievalConsumer(k=5, n_queries=5, metric="manhattan")
        with pytest.raises(ValueError, match="unknown retrieval metric"):
            c.per_item(_corpus(n=40, d=8), _corpus(n=40, d=8))

    def test_the_metric_name_carries_k_and_the_geometry(self):
        c = TopKRetrievalConsumer(k=25, metric="l2")
        assert c.name == "recall@25/l2"
        assert c.higher_is_better is True


# ------------------------------------------------------------------ #
# Attention keys                                                      #
# ------------------------------------------------------------------ #


class TestAttentionScore:
    def _pair(self, n=200, d=32, nq=16, seed=11):
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal((n, d)).astype(np.float32),
            rng.standard_normal((nq, d)).astype(np.float32),
        )

    def test_a_head_without_its_queries_is_refused_not_guessed(self):
        keys, _ = self._pair()
        c = create_consumer("attention_softmax")
        with pytest.raises(ValueError, match="queries"):
            c.per_item(keys, keys)

    def test_identity_is_perfect_in_every_mode(self):
        keys, q = self._pair()
        assert np.allclose(
            AttentionScoreConsumer("spearman").per_item(keys, keys, queries=q), 1.0
        )
        assert np.allclose(
            AttentionScoreConsumer("topk", k=8).per_item(keys, keys, queries=q), 1.0
        )
        kl = AttentionScoreConsumer("kl").per_item(keys, keys, queries=q)
        assert np.allclose(kl, 0.0, atol=1e-9)

    def test_kl_is_a_divergence_so_lower_is_better(self):
        c = AttentionScoreConsumer("kl")
        assert c.higher_is_better is False
        assert c.name == "attention_softmax_kl"
        keys, q = self._pair()
        worse = c.per_item(keys, _quantize(keys, 2), queries=q)
        better = c.per_item(keys, _quantize(keys, 8), queries=q)
        assert float(worse.mean()) > float(better.mean())

    def test_spearman_and_topk_are_similarities_so_higher_is_better(self):
        for mode in ("spearman", "topk"):
            c = AttentionScoreConsumer(mode)
            assert c.higher_is_better is True
        keys, q = self._pair()
        c = AttentionScoreConsumer("spearman")
        assert float(c.per_item(keys, _quantize(keys, 8), queries=q).mean()) > float(
            c.per_item(keys, _quantize(keys, 2), queries=q).mean()
        )

    def test_the_topk_name_carries_k(self):
        assert AttentionScoreConsumer("topk", k=4).name == "attention_top4_recall"

    def test_an_unknown_mode_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown attention mode"):
            AttentionScoreConsumer("telepathy")

    def test_queries_of_the_wrong_width_raise(self):
        keys, _ = self._pair(d=32)
        bad = np.zeros((4, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="dimensional"):
            AttentionScoreConsumer("spearman").per_item(keys, keys, queries=bad)

    def test_a_caller_supplied_scale_is_honoured(self):
        keys, q = self._pair()
        recon = _quantize(keys, 3)
        c = AttentionScoreConsumer("kl")
        sharp = float(c.per_item(keys, recon, queries=q, scale=1.0).mean())
        flat = float(c.per_item(keys, recon, queries=q, scale=0.01).mean())
        assert sharp > flat, "a flatter softmax is less sensitive to key error"

    def test_the_query_sample_is_capped(self):
        keys, q = self._pair(nq=50)
        c = AttentionScoreConsumer("spearman", n_queries=7)
        assert c.per_item(keys, keys, queries=q).shape == (7,)


# ------------------------------------------------------------------ #
# Read-operator distortion                                            #
# ------------------------------------------------------------------ #


class TestReadOperatorConsumer:
    def test_it_is_a_distortion_so_lower_is_better(self):
        c = create_consumer("read_operator", operator="identity")
        assert c.higher_is_better is False
        assert c.name == "consumer_distortion/identity"

    def test_its_mean_is_the_trace_the_theory_names(self):
        from turboquant_pro.read_operators import consumer_distortion, error_covariance

        x = _corpus(n=400, d=24, seed=8)
        recon = _quantize(x, 3)
        c = create_consumer("read_operator", operator="identity")
        per_item = c.per_item(x, recon)
        sigma = error_covariance(x, recon)
        expected = consumer_distortion(np.eye(x.shape[1]), sigma)
        assert float(per_item.mean()) == pytest.approx(expected, rel=1e-9)

    def test_error_in_a_direction_the_consumer_ignores_costs_nothing(self):
        d = 8
        P = np.zeros((d, d))
        P[0, 0] = 1.0  # the consumer reads only the first channel
        x = _corpus(n=200, d=d, seed=2)
        blind = x.copy()
        blind[:, 1:] += np.random.default_rng(4).standard_normal((200, d - 1)) * 5.0
        c = create_consumer("read_operator", operator="declared", matrix=P)
        assert float(c.per_item(x, blind).mean()) == pytest.approx(0.0, abs=1e-9)

        seen = x.copy()
        seen[:, 0] += 1.0 + np.random.default_rng(5).standard_normal(200)
        assert float(c.per_item(x, seen).mean()) > 0.1

    def test_an_analytic_operator_still_needs_the_consumers_queries(self):
        c = create_consumer("read_operator", operator="attention_analytic")
        x = _corpus(n=64, d=16)
        with pytest.raises(ValueError, match="queries"):
            c.per_item(x, x)

    def test_an_unregistered_operator_raises(self):
        with pytest.raises(KeyError):
            create_consumer("read_operator", operator="telepathy")

    def test_a_shape_mismatch_raises(self):
        c = create_consumer("read_operator", operator="identity")
        with pytest.raises(ValueError, match="shape mismatch"):
            c.per_item(_corpus(n=20, d=8), _corpus(n=20, d=9))


# ------------------------------------------------------------------ #
# Declared                                                            #
# ------------------------------------------------------------------ #


class TestDeclaredConsumer:
    def test_it_needs_a_callable(self):
        with pytest.raises(TypeError, match="callable"):
            DeclaredConsumer(fn="not a function")

    def test_the_name_and_direction_are_the_callers(self):
        c = create_consumer(
            "declared",
            fn=lambda o, r, **_: np.abs(o - r).mean(axis=1),
            name="mean_abs_error",
            higher_is_better=False,
        )
        x = _corpus(n=50, d=8)
        assert c.name == "mean_abs_error"
        assert c.higher_is_better is False
        assert c.per_item(x, x).shape == (50,)
        assert np.allclose(c.per_item(x, x), 0.0)

    def test_the_result_is_flattened_to_one_score_per_item(self):
        c = DeclaredConsumer(fn=lambda o, r, **_: np.zeros((len(o), 1)))
        assert c.per_item(_corpus(n=12, d=4), _corpus(n=12, d=4)).shape == (12,)


# ------------------------------------------------------------------ #
# The alignment rule                                                  #
# ------------------------------------------------------------------ #


class TestNominalAlignment:
    def test_row_cosine_is_one_for_identical_rows(self):
        x = _corpus(n=40, d=8)
        assert np.allclose(row_cosine(x, x), 1.0)

    def test_row_cosine_refuses_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            row_cosine(_corpus(n=10, d=4), _corpus(n=10, d=5))

    def test_a_zero_row_does_not_divide_by_zero(self):
        x = _corpus(n=10, d=4)
        x[0] = 0.0
        out = row_cosine(x, x)
        assert np.isfinite(out).all()

    @pytest.mark.parametrize(
        "name,config,context",
        [
            ("topk_inner_product", {"k": 5, "n_queries": 20}, {}),
            ("topk_l2", {"k": 5, "n_queries": 20}, {}),
            ("attention_softmax", {"n_queries": 20}, "queries"),
            ("attention_topk", {"k": 4, "n_queries": 20}, "queries"),
            ("read_operator", {"operator": "identity"}, {}),
        ],
    )
    def test_the_cheap_metric_scores_the_same_items_as_the_consumer(
        self, name, config, context
    ):
        target = "kv_key" if name.startswith("attention") else "embedding"
        d = 24
        x = _corpus(n=200, d=d, seed=6)
        ctx = {}
        if context == "queries":
            ctx["queries"] = _corpus(n=20, d=d, seed=7)
        assert target  # the spec's target, asserted by the registry elsewhere
        c = create_consumer(name, **config)
        recon = _quantize(x, 4)
        per_item = c.per_item(x, recon, **ctx)
        nominal = nominal_per_item(c, x, recon, **ctx)
        assert nominal is not None
        assert nominal.shape == per_item.shape

    def test_an_unalignable_pair_returns_none_rather_than_a_wrong_number(self):
        class Unalignable:
            name = "unalignable"
            higher_is_better = True

            def per_item(self, o, r, **_):
                return np.zeros(3)

        x = _corpus(n=10, d=4)
        assert nominal_per_item(Unalignable(), x, x) is not None  # falls back to rows
        assert nominal_per_item(Unalignable(), x, _corpus(n=10, d=5)) is None

    def test_a_raising_nominal_never_breaks_the_caller(self):
        class Angry:
            name = "angry"
            higher_is_better = True

            def per_item(self, o, r, **_):
                return np.zeros(len(o))

            def nominal_per_item(self, o, r, **_):
                raise RuntimeError("no")

        x = _corpus(n=10, d=4)
        assert nominal_per_item(Angry(), x, x) is None
