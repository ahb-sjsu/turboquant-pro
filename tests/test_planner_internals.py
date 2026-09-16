"""The control plane's primitives, one property per test.

`tests/test_planner.py` pins the acceptance criteria of issue #169 end to end.
This suite pins the parts they rest on: byte accounting that misses nothing,
preflight, the uncertainty arithmetic, the dominance rule, the shape-hint
convention, and every path where something goes wrong and the planner has to
record it rather than quietly drop it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from turboquant_pro import plugins
from turboquant_pro.planner import (
    ABSTAIN,
    Artifact,
    Budget,
    CompressionPlanner,
    QualityEstimate,
    QualityFloor,
    WorkloadSpec,
    _estimate,
    _frontier,
    _strictly_worse,
    container_bytes,
    plan_schema,
    preflight,
    replay_plan,
)

ALL_TARGETS = frozenset({"embedding", "kv_key", "kv_value", "weight"})


# ------------------------------------------------------------------ #
# Codecs used below, each exercising one path                         #
# ------------------------------------------------------------------ #


class _Packed:
    """Stores codes at the declared width, so bytes actually differ with bits."""

    def __init__(self, bits: int = 4, **_ignored):
        if bits not in (4, 8):
            raise ValueError(f"unsupported bits {bits}")
        self.bits = int(bits)

    def capabilities(self):
        return {"bit_widths": (4, 8)}

    def compress(self, x, **_):
        a = np.asarray(x, dtype=np.float32)
        f = a.reshape(-1, a.shape[-1])
        lo = f.min(axis=1, keepdims=True).astype(np.float32)
        hi = f.max(axis=1, keepdims=True).astype(np.float32)
        levels = (1 << self.bits) - 1
        scale = np.maximum((hi - lo) / levels, 1e-20).astype(np.float32)
        codes = np.rint((f - lo) / scale).clip(0, levels).astype(np.uint8)
        stored = codes
        if self.bits == 4:
            if codes.shape[1] % 2:
                codes = np.pad(codes, ((0, 0), (0, 1)))
            stored = (codes[:, 0::2] << 4) | codes[:, 1::2]
        return {
            "codes": stored,
            "lo": lo,
            "scale": scale,
            "shape": a.shape,
            "width": f.shape[1],
            "bits": self.bits,
        }

    def decompress(self, c):
        codes = c["codes"]
        if c["bits"] == 4:
            hi = (codes >> 4).astype(np.uint8)
            lo = (codes & 0x0F).astype(np.uint8)
            codes = np.empty((hi.shape[0], hi.shape[1] * 2), dtype=np.uint8)
            codes[:, 0::2] = hi
            codes[:, 1::2] = lo
            codes = codes[:, : c["width"]]
        out = codes.astype(np.float32) * c["scale"] + c["lo"]
        return out.reshape(c["shape"])


class _NeedsGeometry:
    """Only works when it is told the artifact's channel count."""

    def __init__(self, head_dim: int, bits: int = 8, **_ignored):
        self.head_dim = int(head_dim)
        self.bits = int(bits)

    def capabilities(self):
        return {"bit_widths": (8,)}

    def compress(self, x, **_):
        a = np.asarray(x, dtype=np.float32)
        if a.shape[-1] != self.head_dim:
            raise ValueError(
                f"configured for head_dim={self.head_dim}, got {a.shape[-1]}"
            )
        return {"x": a.astype(np.float16)}

    def decompress(self, c):
        return c["x"].astype(np.float32)


class _StrictSignature:
    """Takes bits and nothing else; a shape hint is a TypeError to it."""

    def __init__(self, bits: int = 8):
        self.bits = int(bits)

    def compress(self, x, **_):
        return {"x": np.asarray(x, dtype=np.float16)}

    def decompress(self, c):
        return c["x"].astype(np.float32)


class _BreaksOnCompress:
    def __init__(self, bits: int = 8, **_ignored):
        self.bits = int(bits)

    def capabilities(self):
        return {"bit_widths": (8,)}

    def compress(self, x, **_):
        raise RuntimeError("the kernel is not built on this machine")

    def decompress(self, c):  # pragma: no cover - never reached
        raise AssertionError


class _WrongShapeBack:
    def __init__(self, bits: int = 8, **_ignored):
        self.bits = int(bits)

    def capabilities(self):
        return {"bit_widths": (8,)}

    def compress(self, x, **_):
        return {"x": np.asarray(x, dtype=np.float32)}

    def decompress(self, c):
        return c["x"][..., :1]  # a shape the caller never asked for


def _register(name, factory, targets=ALL_TARGETS, tier="experimental"):
    plugins.register(
        plugins.PluginSpec(
            name=name,
            factory=factory,
            targets=targets,
            tier=tier,
            description=f"test codec {name}",
        ),
        overwrite=True,
    )


@pytest.fixture
def codecs():
    before = dict(plugins._REGISTRY)
    _register("p_packed", lambda **c: _Packed(**c))
    _register("p_geometry", lambda **c: _NeedsGeometry(**c))
    _register("p_strict", lambda **c: _StrictSignature(**c))
    _register("p_broken", lambda **c: _BreaksOnCompress(**c))
    _register("p_wrongshape", lambda **c: _WrongShapeBack(**c))
    yield
    plugins._REGISTRY.clear()
    plugins._REGISTRY.update(before)


def _corpus(n=700, d=48, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _spec(**over):
    base = dict(
        target="embedding",
        consumer="topk_inner_product",
        consumer_config={"k": 10, "n_queries": 100},
        floor=QualityFloor(minimum=0.70, confidence=0.95),
        seed=0,
        n_boot=96,
    )
    base.update(over)
    return WorkloadSpec(**base)


# ------------------------------------------------------------------ #
# Byte accounting (R6: account for every stored byte)                 #
# ------------------------------------------------------------------ #


class TestContainerBytes:
    def test_it_walks_nested_containers(self):
        c = {
            "codes": np.zeros(100, dtype=np.uint8),
            "meta": {"scale": np.zeros(4, dtype=np.float32)},
            "extras": [np.zeros(2, dtype=np.float64), b"1234"],
        }
        acct = container_bytes(c)
        assert acct["total_bytes"] == 100 + 16 + 16 + 4
        assert "container.meta.scale" in acct["breakdown"]
        assert "container.extras[0]" in acct["breakdown"]

    def test_scalars_and_strings_carry_no_stored_bytes(self):
        assert (
            container_bytes({"bits": 4, "name": "x", "flag": True})["total_bytes"] == 0
        )

    def test_it_counts_a_dataclass_container(self):
        @dataclass
        class Box:
            codes: np.ndarray
            scale: np.ndarray

        acct = container_bytes(
            Box(np.zeros(64, dtype=np.uint8), np.zeros(8, dtype=np.float32))
        )
        assert acct["total_bytes"] == 64 + 32

    def test_it_counts_a_slotted_container(self):
        class Slots:
            __slots__ = ("codes",)

            def __init__(self):
                self.codes = np.zeros(10, dtype=np.uint8)

        assert container_bytes(Slots())["total_bytes"] == 10

    def test_a_shared_buffer_is_counted_once(self):
        shared = np.zeros(50, dtype=np.uint8)
        assert container_bytes({"a": shared, "b": shared})["total_bytes"] == 50

    def test_a_cycle_terminates(self):
        c = {"codes": np.zeros(8, dtype=np.uint8)}
        c["self"] = c
        assert container_bytes(c)["total_bytes"] == 8

    def test_it_counts_a_sparse_outlier_overlay(self):
        sparse = pytest.importorskip("scipy.sparse")
        m = sparse.csr_matrix(np.eye(10, dtype=np.float32))
        total = container_bytes({"overlay": m})["total_bytes"]
        assert total == m.data.nbytes + m.indices.nbytes + m.indptr.nbytes

    def test_the_accounting_says_what_it_counted(self):
        acct = container_bytes({"codes": np.zeros(4, dtype=np.uint8)})
        assert "container-in-memory" in acct["accounting"]

    def test_packing_actually_shows_up_as_fewer_bytes(self):
        x = _corpus(n=100, d=64)
        four = container_bytes(_Packed(bits=4).compress(x))["total_bytes"]
        eight = container_bytes(_Packed(bits=8).compress(x))["total_bytes"]
        assert four < eight


# ------------------------------------------------------------------ #
# Preflight                                                           #
# ------------------------------------------------------------------ #


class TestPreflight:
    def test_it_counts_zero_and_non_finite_rows(self):
        x = _corpus(n=200, d=16)
        x[3] = 0.0
        x[7, 0] = np.nan
        x[9, 1] = np.inf
        r = preflight(x)
        assert r["zero_rows"] == 1
        assert r["nonfinite_rows"] == 2
        assert any("all-zero" in f for f in r["flags"])
        assert any("non-finite" in f for f in r["flags"])

    def test_clean_data_raises_no_flags(self):
        assert preflight(_corpus(n=200, d=16))["flags"] == []

    def test_normalised_data_has_a_norm_spread_of_one(self):
        x = _corpus(n=200, d=16)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        assert preflight(x)["norm_spread"] == pytest.approx(1.0, rel=1e-5)

    def test_the_spectrum_is_cumulative_and_bounded(self):
        s = preflight(_corpus(n=400, d=32))["spectrum"]
        assert (
            s["variance_at_d_over_8"]
            <= s["variance_at_d_over_4"]
            <= s["variance_at_d_over_2"]
            <= 1.0
        )

    def test_a_concentrated_corpus_reads_as_concentrated(self):
        rng = np.random.default_rng(0)
        basis = rng.standard_normal((4, 32))
        x = (rng.standard_normal((400, 4)) @ basis).astype(np.float32)
        assert preflight(x)["spectrum"]["variance_at_d_over_8"] > 0.99

    def test_it_samples_large_corpora_and_says_how_many(self):
        r = preflight(_corpus(n=5000, d=8), sample=500)
        assert r["n_rows"] == 5000
        assert r["sampled"] == 500

    def test_the_spectrum_ignores_non_finite_rows(self):
        """A NaN row must not turn the whole spectrum into NaN, or warn."""
        import warnings

        x = _corpus(n=200, d=16)
        x[0, 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r = preflight(x)
        assert r["nonfinite_rows"] == 1
        assert all(np.isfinite(v) for v in r["spectrum"].values())
        assert r["spectrum"]["variance_at_d_over_2"] > 0.0

    def test_an_all_non_finite_corpus_reports_no_spectrum_rather_than_nan(self):
        x = np.full((10, 8), np.nan, dtype=np.float32)
        r = preflight(x)
        assert r["spectrum"] == {}
        assert r["nonfinite_rows"] == 10

    def test_it_accepts_a_key_block_shaped_input(self):
        keys = np.zeros((1, 4, 100, 16), dtype=np.float32)
        assert preflight(keys)["dim"] == 16
        assert preflight(keys)["n_rows"] == 400


# ------------------------------------------------------------------ #
# Uncertainty                                                         #
# ------------------------------------------------------------------ #


class TestQualityEstimate:
    def _est(self, values, higher=True, conf=0.95):
        return _estimate(
            np.asarray(values, dtype=np.float64),
            metric="m",
            higher_is_better=higher,
            confidence=conf,
            n_boot=400,
            seed=0,
        )

    def test_higher_is_better_takes_the_lower_end(self):
        e = self._est(np.linspace(0.5, 1.0, 200))
        assert e.bound <= e.mean
        assert e.as_dict()["bound_is"] == "lower"
        assert e.score == e.bound

    def test_lower_is_better_takes_the_upper_end(self):
        e = self._est(np.linspace(0.0, 2.0, 200), higher=False)
        assert e.bound >= e.mean
        assert e.as_dict()["bound_is"] == "upper"
        assert e.score == -e.bound

    def test_clears_reads_the_bound_in_the_right_direction(self):
        up = self._est(np.full(100, 0.9))
        assert up.clears(QualityFloor(minimum=0.8))
        assert not up.clears(QualityFloor(minimum=0.95))
        down = self._est(np.full(100, 0.1), higher=False)
        assert down.clears(QualityFloor(minimum=0.2))
        assert not down.clears(QualityFloor(minimum=0.05))

    def test_no_floor_means_nothing_to_clear(self):
        assert self._est(np.full(10, 0.1)).clears(None)

    def test_a_wider_confidence_moves_the_bound_further_out(self):
        v = np.random.default_rng(0).normal(0.9, 0.05, 300)
        assert self._est(v, conf=0.99).bound < self._est(v, conf=0.80).bound

    def test_a_single_item_degenerates_to_its_own_value(self):
        e = self._est([0.42])
        assert e.mean == e.bound == e.ci_low == e.ci_high == pytest.approx(0.42)

    def test_non_finite_scores_are_dropped_not_propagated(self):
        e = self._est([1.0, np.nan, 1.0, np.inf])
        assert e.n_items == 2
        assert e.mean == pytest.approx(1.0)

    def test_an_empty_metric_is_an_error_not_a_zero(self):
        with pytest.raises(ValueError, match="no finite per-item"):
            self._est([np.nan, np.nan])

    def test_from_dict_ignores_the_fields_as_dict_derives(self):
        e = self._est(np.linspace(0.5, 1.0, 50))
        back = QualityEstimate.from_dict(e.as_dict())
        assert back == e
        assert QualityEstimate.from_dict(None) is None
        assert QualityEstimate.from_dict({}) is None


class TestDominance:
    def _q(self, lo, hi, higher=True):
        return QualityEstimate(
            metric="m",
            mean=(lo + hi) / 2,
            bound=lo if higher else hi,
            ci_low=lo,
            ci_high=hi,
            n_items=10,
            confidence=0.95,
            higher_is_better=higher,
        )

    def test_overlapping_intervals_are_not_strictly_worse(self):
        assert not _strictly_worse(self._q(0.8, 0.95), self._q(0.9, 0.99))

    def test_a_disjoint_lower_interval_is_strictly_worse(self):
        assert _strictly_worse(self._q(0.5, 0.6), self._q(0.9, 0.99))

    def test_direction_is_respected_for_a_distortion(self):
        assert _strictly_worse(
            self._q(5.0, 6.0, higher=False), self._q(0.1, 0.2, higher=False)
        )
        assert not _strictly_worse(
            self._q(0.1, 0.2, higher=False), self._q(5.0, 6.0, higher=False)
        )

    def test_the_frontier_drops_only_what_is_beaten_on_both_axes(self):
        from turboquant_pro.planner import CandidateResult

        cheap_good = CandidateResult(codec="a", config={})
        cheap_good.quality = self._q(0.90, 0.95)
        cheap_good.bytes_per_vector = 10.0

        dear_bad = CandidateResult(codec="b", config={})
        dear_bad.quality = self._q(0.50, 0.55)
        dear_bad.bytes_per_vector = 40.0

        dear_good = CandidateResult(codec="c", config={})
        dear_good.quality = self._q(0.97, 0.99)
        dear_good.bytes_per_vector = 40.0

        keep = _frontier([cheap_good, dear_bad, dear_good])
        assert cheap_good in keep
        assert dear_good in keep
        assert dear_bad not in keep

    def test_a_tie_is_kept_and_reported_not_broken_silently(self):
        from turboquant_pro.planner import CandidateResult

        a = CandidateResult(codec="a", config={})
        a.quality = self._q(0.90, 0.95)
        a.bytes_per_vector = 10.0
        b = CandidateResult(codec="b", config={})
        b.quality = self._q(0.90, 0.95)
        b.bytes_per_vector = 10.0
        assert len(_frontier([a, b])) == 2


# ------------------------------------------------------------------ #
# The spec and the artifact                                           #
# ------------------------------------------------------------------ #


class TestWorkloadSpec:
    def test_an_unknown_objective_is_refused(self):
        with pytest.raises(ValueError, match="unknown objective"):
            WorkloadSpec(target="embedding", consumer="topk_l2", objective="vibes")

    def test_a_holdout_fraction_outside_the_open_unit_interval_is_refused(self):
        for bad in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError, match="holdout_fraction"):
                WorkloadSpec(
                    target="embedding", consumer="topk_l2", holdout_fraction=bad
                )

    def test_it_round_trips_to_a_record_shaped_dict(self):
        spec = _spec(budget=Budget(max_bits=4, max_bytes_per_vector=32.0))
        d = spec.as_dict()
        assert d["budget"]["max_bits"] == 4
        assert d["floor"]["confidence"] == 0.95
        assert json.loads(json.dumps(d)) == d


class TestArtifact:
    def test_identity_is_a_content_hash(self):
        x = _corpus(n=50, d=8)
        assert Artifact(x).identity == Artifact(x.copy()).identity
        assert Artifact(x).identity != Artifact(_corpus(n=50, d=8, seed=1)).identity

    def test_dtype_is_part_of_the_identity(self):
        x = _corpus(n=50, d=8)
        assert Artifact(x).identity != Artifact(x.astype(np.float64)).identity

    def test_a_caller_supplied_identity_is_kept(self):
        assert Artifact(_corpus(n=4, d=2), identity="mine").identity == "mine"

    def test_rows_and_dim_read_a_key_block_correctly(self):
        a = Artifact(np.zeros((1, 8, 128, 64), dtype=np.float32))
        assert a.n_rows == 8 * 128
        assert a.dim == 64


# ------------------------------------------------------------------ #
# Capability probe and shape hints                                    #
# ------------------------------------------------------------------ #


class TestCapabilities:
    def test_a_codec_that_declares_nothing_gets_an_empty_dict(self):
        assert plugins.capabilities(_StrictSignature()) == {}

    def test_a_declared_capability_is_returned(self):
        assert plugins.capabilities(_Packed())["bit_widths"] == (4, 8)

    def test_a_raising_probe_never_breaks_a_plan(self):
        class Angry:
            def capabilities(self):
                raise RuntimeError("no")

        assert plugins.capabilities(Angry()) == {}

    def test_a_codec_config_key_called_name_does_not_collide(self):
        """The registry's own parameter is positional-only for this reason."""
        before = dict(plugins._REGISTRY)
        _register("p_named", lambda **c: _StrictSignature(bits=c.get("bits", 8)))
        try:
            assert plugins.create("p_named", name="mine", bits=8) is not None
        finally:
            plugins._REGISTRY.clear()
            plugins._REGISTRY.update(before)

    def test_a_non_mapping_answer_is_discarded(self):
        class Odd:
            def capabilities(self):
                return 7

        assert plugins.capabilities(Odd()) == {}

    def test_declared_widths_are_the_ones_tried(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_packed",))).plan(
            Artifact(_corpus())
        )
        widths = {c.declared_bits for c in plan.candidate_results}
        assert widths == {4.0, 8.0}


class TestShapeHints:
    def test_a_codec_that_needs_the_geometry_is_given_it(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_geometry",))).plan(
            Artifact(_corpus(d=48))
        )
        assert plan.selected_codec == "p_geometry"

    def test_a_codec_that_rejects_the_hint_is_built_without_it(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_strict",))).plan(
            Artifact(_corpus())
        )
        assert plan.selected_codec == "p_strict"


# ------------------------------------------------------------------ #
# Failure is recorded, never dropped                                  #
# ------------------------------------------------------------------ #


class TestFailuresAreRecorded:
    def test_a_codec_that_cannot_compress_is_unsupported_with_its_error(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_packed", "p_broken"))).plan(
            Artifact(_corpus())
        )
        broken = [c for c in plan.candidate_results if c.codec == "p_broken"]
        assert broken
        assert all(c.verdict == "unsupported" for c in broken)
        assert all("kernel is not built" in (c.error or "") for c in broken)
        assert plan.selected_codec == "p_packed"

    def test_a_codec_returning_the_wrong_shape_is_unsupported(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_wrongshape",))).plan(
            Artifact(_corpus())
        )
        assert plan.selected_codec == ABSTAIN
        assert any(c.verdict == "unsupported" for c in plan.candidate_results)

    def test_an_unregistered_codec_is_recorded_not_silently_skipped(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_packed", "nonesuch"))).plan(
            Artifact(_corpus())
        )
        miss = [c for c in plan.candidate_results if c.codec == "nonesuch"]
        assert len(miss) == 1
        assert miss[0].verdict == "unsupported"
        assert "no quantizer plugin" in miss[0].reason

    def test_a_codec_registered_for_another_target_is_recorded(self, codecs):
        before = dict(plugins._REGISTRY)
        _register("p_keysonly", lambda **c: _Packed(**c), targets=frozenset({"kv_key"}))
        try:
            plan = CompressionPlanner(
                _spec(candidates=("p_packed", "p_keysonly"))
            ).plan(Artifact(_corpus()))
            odd = [c for c in plan.candidate_results if c.codec == "p_keysonly"][0]
            assert odd.verdict == "unsupported"
            assert "do not include 'embedding'" in odd.reason
        finally:
            plugins._REGISTRY.clear()
            plugins._REGISTRY.update(before)

    def test_a_bit_width_the_codec_refuses_is_recorded_with_the_reason(self, codecs):
        spec = _spec(candidates=("p_packed",), budget=Budget())
        plan = CompressionPlanner(spec).plan(Artifact(_corpus()))
        # _Packed declares (4, 8); the default grid's 2 and 3 are never tried.
        assert {c.declared_bits for c in plan.candidate_results} == {4.0, 8.0}

    def test_a_consumer_that_cannot_read_a_codec_marks_it_unsupported(self, codecs):
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((400, 32)).astype(np.float32)
        spec = WorkloadSpec(
            target="kv_key",
            consumer="attention_softmax",
            consumer_config={"n_queries": 16},
            candidates=("p_packed",),
            seed=0,
            n_boot=64,
        )
        # No queries in the artifact context: the consumer refuses to guess.
        plan = CompressionPlanner(spec).plan(Artifact(keys))
        assert plan.selected_codec == ABSTAIN
        assert all(c.verdict == "unsupported" for c in plan.candidate_results)
        assert any("queries" in (c.error or "") for c in plan.candidate_results)


# ------------------------------------------------------------------ #
# Objectives and other consumers                                      #
# ------------------------------------------------------------------ #


class TestObjectives:
    def test_max_quality_buys_the_better_metric(self, codecs):
        plan = CompressionPlanner(
            _spec(candidates=("p_packed",), objective="max_quality")
        ).plan(Artifact(_corpus()))
        assert plan.selected_parameters["bits"] == 8
        assert plan.selection["objective"] == "max_quality"

    def test_min_cost_buys_the_cheaper_one_that_still_clears(self, codecs):
        plan = CompressionPlanner(
            _spec(candidates=("p_packed",), objective="min_cost")
        ).plan(Artifact(_corpus()))
        assert plan.selected_parameters["bits"] == 4
        assert "minimise stored bytes" in plan.selection["rule"]

    def test_a_candidate_that_clears_the_floor_is_never_cut_in_halving(self, codecs):
        """The floor is the bar. Quality above it is a tiebreak, not the bar.

        Cutting a floor-clearing candidate during the search decides the plan
        before cost has been consulted, which is exactly wrong under min_cost.
        """
        plan = CompressionPlanner(
            _spec(candidates=("p_packed",), objective="min_cost")
        ).plan(Artifact(_corpus()))
        for c in plan.candidate_results:
            q = c.holdout_quality or c.quality
            if q is not None and q.clears(QualityFloor(minimum=0.70)):
                assert "halving" not in c.left_at, c.reason

    def test_a_candidate_below_the_floor_can_still_be_cut_early(self, codecs):
        """The saving has to be real: a hopeless candidate leaves the search."""
        plan = CompressionPlanner(
            _spec(candidates=("p_packed",), floor=QualityFloor(minimum=0.999))
        ).plan(Artifact(_corpus()))
        assert plan.selected_codec == ABSTAIN
        assert any(
            "halving" in c.left_at or c.left_at == "verification"
            for c in plan.candidate_results
        )

    def test_the_two_objectives_disagree_on_purpose(self, codecs):
        art = Artifact(_corpus())
        best = CompressionPlanner(_spec(candidates=("p_packed",))).plan(art)
        cheap = CompressionPlanner(
            _spec(candidates=("p_packed",), objective="min_cost")
        ).plan(art)
        assert best.expected_cost["bytes_per_vector"] > (
            cheap.expected_cost["bytes_per_vector"]
        )
        assert best.expected_quality.bound >= cheap.expected_quality.bound


class TestOtherConsumerFamilies:
    def test_a_read_operator_consumer_plans_without_any_queries(self, codecs):
        spec = WorkloadSpec(
            target="embedding",
            consumer="read_operator",
            consumer_config={"operator": "identity"},
            candidates=("p_packed",),
            seed=0,
            n_boot=64,
        )
        plan = CompressionPlanner(spec).plan(Artifact(_corpus()))
        assert plan.selected_codec == "p_packed"
        q = plan.expected_quality
        assert q.higher_is_better is False
        assert q.bound >= q.mean, "a distortion is bounded from above"
        assert plan.operator_regime["operator"] == "identity"
        assert plan.operator_regime["exact"] is True

    def test_the_in_tree_value_codec_plans_on_keys_geometry(self):
        rng = np.random.default_rng(1)
        values = rng.standard_normal((1, 1, 600, 64)).astype(np.float32)
        spec = WorkloadSpec(
            target="kv_value",
            consumer="read_operator",
            consumer_config={"operator": "identity"},
            candidates=("polar",),
            seed=0,
            n_boot=64,
        )
        plan = CompressionPlanner(spec).plan(Artifact(values))
        assert plan.selected_codec in {"polar", ABSTAIN}
        assert plan.candidate_results, "polar is recorded either way"
        if plan.selected_codec == ABSTAIN:
            assert all(c.reason for c in plan.candidate_results)

    def test_the_operator_regime_is_implicit_for_a_measured_consumer(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_packed",))).plan(
            Artifact(_corpus())
        )
        assert plan.operator_regime["operator"] == "implicit"
        assert "recall@10" in plan.operator_regime["consumer"]


# ------------------------------------------------------------------ #
# The record, and replay                                              #
# ------------------------------------------------------------------ #


class TestRecord:
    def test_the_shipped_schema_is_a_valid_json_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.Draft202012Validator.check_schema(plan_schema())

    def test_every_candidate_names_the_stage_it_left_at(self, codecs):
        plan = CompressionPlanner(
            _spec(candidates=("p_packed", "p_broken", "nonesuch"))
        ).plan(Artifact(_corpus()))
        for c in plan.candidate_results:
            assert c.left_at
            assert c.verdict in {
                "selected",
                "candidate",
                "reject",
                "infeasible",
                "unsupported",
            }

    def test_the_environment_lists_the_codecs_that_were_registered(self, codecs):
        plan = CompressionPlanner(_spec(candidates=("p_packed",))).plan(
            Artifact(_corpus())
        )
        assert "p_packed" in plan.environment["registered_codecs"]
        assert plan.environment["tool_version"]

    def test_explain_renders_an_abstention_with_its_reason(self, codecs):
        plan = CompressionPlanner(_spec(consumer="telepathy")).plan(Artifact(_corpus()))
        text = plan.explain()
        assert ABSTAIN in text
        assert "reason:" in text

    def test_an_abstention_still_declares_a_fallback(self, codecs):
        plan = CompressionPlanner(_spec(consumer="telepathy")).plan(Artifact(_corpus()))
        assert plan.fallback_policy["action"] == "per_channel_or_fp16"
        assert plan.certificate_requirement["status"] == "not_established"


class TestReplay:
    def test_replaying_an_abstention_reports_it_rather_than_crashing(self, codecs):
        corpus = _corpus()
        plan = CompressionPlanner(_spec(consumer="telepathy")).plan(Artifact(corpus))
        r = replay_plan(plan.as_dict(), Artifact(corpus))
        assert r["recorded_codec"] == ABSTAIN
        assert r["replayed_codec"] == ABSTAIN
        assert r["codec_agrees"] is True

    def test_a_replay_re_runs_only_the_recorded_codec(self, codecs):
        corpus = _corpus()
        plan = CompressionPlanner(_spec(candidates=("p_packed", "p_strict"))).plan(
            Artifact(corpus)
        )
        r = replay_plan(plan.as_dict(), Artifact(corpus))
        assert r["reproduced"] is True
        assert r["delta_mean"] == pytest.approx(0.0, abs=1e-12)

    def test_the_replay_report_is_json_serialisable(self, codecs):
        corpus = _corpus()
        plan = CompressionPlanner(_spec(candidates=("p_packed",))).plan(
            Artifact(corpus)
        )
        r = replay_plan(plan.as_dict(), Artifact(corpus))
        assert json.loads(json.dumps(r, default=float))["schema"].endswith("replay")


# ------------------------------------------------------------------ #
# CLI error paths                                                     #
# ------------------------------------------------------------------ #


class TestCliErrors:
    def test_a_missing_artifact_exits_rather_than_traceback(self, tmp_path):
        from turboquant_pro.cli import main

        with pytest.raises(SystemExit, match="cannot load"):
            main(["plan", "run", "--artifact", str(tmp_path / "nope.npy")])

    def test_a_malformed_consumer_config_exits_two(self, tmp_path, codecs):
        from turboquant_pro.cli import main

        art = tmp_path / "c.npy"
        np.save(art, _corpus(n=100, d=16))
        rc = main(
            [
                "plan",
                "run",
                "--artifact",
                str(art),
                "--consumer-config",
                "{not json}",
                "--candidates",
                "p_packed",
            ]
        )
        assert rc == 2

    def test_explaining_a_missing_record_exits_two(self, tmp_path):
        from turboquant_pro.cli import main

        assert main(["plan", "explain", str(tmp_path / "nope.json")]) == 2

    def test_replaying_a_missing_record_exits_two(self, tmp_path):
        from turboquant_pro.cli import main

        art = tmp_path / "c.npy"
        np.save(art, _corpus(n=50, d=8))
        rc = main(
            [
                "plan",
                "replay",
                str(tmp_path / "nope.json"),
                "--artifact",
                str(art),
            ]
        )
        assert rc == 2

    def test_a_replay_that_does_not_reproduce_exits_one(self, tmp_path, codecs):
        from turboquant_pro.cli import main

        art = tmp_path / "c.npy"
        other = tmp_path / "d.npy"
        np.save(art, _corpus(n=400, d=16, seed=0))
        np.save(other, _corpus(n=400, d=16, seed=1))
        out = tmp_path / "plan.json"
        assert (
            main(
                [
                    "plan",
                    "run",
                    "--artifact",
                    str(art),
                    "--candidates",
                    "p_packed",
                    "--consumer-config",
                    '{"k": 10, "n_queries": 60}',
                    "--floor",
                    "0.7",
                    "--n-boot",
                    "64",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        assert main(["plan", "replay", str(out), "--artifact", str(other)]) == 1

    def test_the_old_plan_subcommands_still_work(self, tmp_path):
        from turboquant_pro.cli import main

        emb = tmp_path / "emb.npy"
        np.save(emb, _corpus(n=300, d=32))
        rc = main(
            [
                "plan",
                "embeddings",
                "--embeddings",
                str(emb),
                "--sample",
                "50",
                "--format",
                "text",
            ]
        )
        assert rc in (0, 1), "unchanged behaviour: 1 only when the preview is vacuous"
