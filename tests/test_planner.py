"""The quantization control plane decides on the consumer's metric, or abstains.

This suite is the executable form of the acceptance criteria in issue #169.
Each test names the property it pins:

- three materially different codec families compete in one planning run, two of
  them supplied through the plugin mechanism rather than shipped in tree;
- the choice turns on a declared consumer metric, and a codec with the better
  reconstruction cosine loses to one with the better consumer behaviour;
- a third-party codec can beat the native one, and the record says so;
- the reason is machine-readable, not prose in a log line;
- a rejected codec reports its false-clear condition explicitly;
- the plan carries evidence with a declared kind and a runtime fallback;
- re-running the same plan on the same artifact reproduces it;
- an unmeasured regime abstains instead of inventing a recommendation.

The codecs below are deliberately simple and deliberately *registered the way
an out-of-tree package registers*, so the test exercises the same path a third
party would.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro import consumers as consumers_mod
from turboquant_pro import plugins
from turboquant_pro.planner import (
    ABSTAIN,
    Artifact,
    Budget,
    CompressionPlanner,
    QualityFloor,
    QuantizationControlPlane,
    WorkloadSpec,
    container_bytes,
    preflight,
    replay_plan,
)
from turboquant_pro.schemas import load_schema

SCHEMA_NAME = "compression_plan.schema.json"


# ------------------------------------------------------------------ #
# Three out-of-tree codecs                                            #
# ------------------------------------------------------------------ #


class _Uniform:
    """Per-row uniform scalar quantization. Honest, unremarkable, cheap."""

    def __init__(self, bits: int = 4, **_ignored):
        if bits not in (2, 4, 8):
            raise ValueError(f"unsupported bits {bits}")
        self.bits = int(bits)

    def capabilities(self):
        return {"bit_widths": (2, 4, 8), "requires_calibration": False}

    def compress(self, x, **_):
        a = np.asarray(x, dtype=np.float32)
        flat = a.reshape(-1, a.shape[-1])
        lo = flat.min(axis=1, keepdims=True)
        hi = flat.max(axis=1, keepdims=True)
        levels = (1 << self.bits) - 1
        scale = np.maximum((hi - lo) / levels, 1e-20).astype(np.float32)
        codes = np.rint((flat - lo) / scale).clip(0, levels).astype(np.uint8)
        return {
            "codes": codes,
            "lo": lo.astype(np.float32),
            "scale": scale,
            "shape": a.shape,
            "bits": self.bits,
        }

    def decompress(self, c):
        out = c["codes"].astype(np.float32) * c["scale"] + c["lo"]
        return out.reshape(c["shape"])


class _FineGrained(_Uniform):
    """The same family at a finer grid: better consumer behaviour, more bytes.

    Stands in for a third-party codec that genuinely wins on the metric the
    consumer reads. Nothing subtle: the point of the test is that the planner
    is willing to pick it over the in-tree codec when the measurement says so.
    """

    def __init__(self, bits: int = 8, **_ignored):
        super().__init__(bits=8)

    def capabilities(self):
        return {"bit_widths": (8,), "default_bits": 8}


class _CosineFlatterer(_Uniform):
    """High cosine, scrambled ranking. The failure this package is named for.

    Quantizes at 8 bits, then adds a deterministic perturbation whose size is
    a fixed fraction of each row's norm. Cosine stays near 0.97, which clears
    any conventional reconstruction gate, while near-ties in the consumer's
    ranking reorder.
    """

    def __init__(self, bits: int = 8, noise: float = 0.25, **_ignored):
        super().__init__(bits=8)
        self.noise = float(noise)

    def capabilities(self):
        return {"bit_widths": (8,), "default_bits": 8}

    def decompress(self, c):
        out = super().decompress(c)
        flat = out.reshape(-1, out.shape[-1])
        rng = np.random.default_rng(1234)
        pert = rng.standard_normal(flat.shape).astype(np.float32)
        pert /= np.maximum(np.linalg.norm(pert, axis=1, keepdims=True), 1e-20)
        pert *= np.linalg.norm(flat, axis=1, keepdims=True) * self.noise
        return (flat + pert).reshape(out.shape)


@pytest.fixture
def registry():
    """Register the out-of-tree codecs, then put the registry back."""
    before = dict(plugins._REGISTRY)
    all_targets = frozenset(
        {
            plugins.TARGET_EMBEDDING,
            plugins.TARGET_KV_KEY,
            plugins.TARGET_KV_VALUE,
            plugins.TARGET_WEIGHT,
        }
    )
    for name, factory, desc in (
        ("t_uniform", _Uniform, "uniform per-row scalar quantizer"),
        ("t_fine", _FineGrained, "finer grid, more bytes, better consumer metric"),
        ("t_flatterer", _CosineFlatterer, "high cosine, scrambled ranking"),
    ):
        plugins.register(
            plugins.PluginSpec(
                name=name,
                factory=(lambda f: lambda **cfg: f(**cfg))(factory),
                targets=all_targets,
                tier="experimental",
                description=desc,
            ),
            overwrite=True,
        )
    yield
    plugins._REGISTRY.clear()
    plugins._REGISTRY.update(before)


def _corpus(n=900, d=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _retrieval_spec(**over):
    base = dict(
        target="embedding",
        consumer="topk_inner_product",
        consumer_config={"k": 10, "n_queries": 120},
        budget=Budget(),
        floor=QualityFloor(minimum=0.80, confidence=0.95),
        candidates=("t_uniform", "t_fine", "t_flatterer"),
        seed=0,
        n_boot=128,
    )
    base.update(over)
    return WorkloadSpec(**base)


# ------------------------------------------------------------------ #
# Criterion 1-5: families compete, the consumer decides, and it says why
# ------------------------------------------------------------------ #


def test_three_families_compete_and_the_consumer_metric_decides(registry):
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    doc = plan.as_dict()

    codecs = {c["codec"] for c in doc["candidate_results"]}
    assert len(codecs) >= 3, codecs
    sources = {c["codec"]: c["source"] for c in doc["candidate_results"]}
    assert sum(1 for v in sources.values() if v == "plugin") >= 2

    assert plan.selected_codec != ABSTAIN
    assert doc["selection"]["decided_on"] == doc["consumer"]["metric"]
    assert "recall@10" in doc["consumer"]["metric"]
    assert doc["selection"]["not_decided_on"].startswith("reconstruction cosine")
    assert doc["selection"]["rule"]
    assert doc["selection"]["reason"]


def test_the_codec_with_the_better_cosine_can_still_lose(registry):
    """The whole thesis, as a single assertion.

    The flatterer's reconstruction cosine beats the 2-bit uniform codec's by a
    wide margin. Its consumer metric does not, and the consumer metric is what
    the plan turns on.
    """
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    by_name = {}
    for c in plan.candidate_results:
        if c.nominal.get("mean") is not None:
            by_name.setdefault(c.codec, c)
    flat = by_name["t_flatterer"]
    coarse = [
        c
        for c in plan.candidate_results
        if c.codec == "t_uniform" and c.config.get("bits") == 2
    ][0]
    assert flat.nominal["mean"] > coarse.nominal["mean"]
    assert plan.selected_codec != "t_flatterer"
    assert flat.verdict != "selected"


def test_a_third_party_codec_can_beat_the_in_tree_one():
    """On KV keys, where the in-tree per-channel codec is a real competitor."""
    before = dict(plugins._REGISTRY)
    plugins.register(
        plugins.PluginSpec(
            name="t_fine_kv",
            factory=lambda **cfg: _FineGrained(**cfg),
            targets=frozenset({plugins.TARGET_KV_KEY}),
            tier="experimental",
            description="out-of-tree 8-bit codec for keys",
        ),
        overwrite=True,
    )
    try:
        rng = np.random.default_rng(3)
        keys = rng.standard_normal((1, 1, 800, 64)).astype(np.float32)
        queries = rng.standard_normal((64, 64)).astype(np.float32)
        spec = WorkloadSpec(
            target="kv_key",
            consumer="attention_softmax",
            consumer_config={"mode": "spearman", "n_queries": 64},
            floor=QualityFloor(minimum=0.90, confidence=0.95),
            candidates=("per_channel", "t_fine_kv"),
            seed=0,
            n_boot=128,
        )
        plan = CompressionPlanner(spec).plan(
            Artifact(keys, context={"queries": queries})
        )
        assert plan.selected_codec == "t_fine_kv"
        assert plan.selection["third_party"] is True
        assert plan.selection["beat"]["codec"] == "per_channel"
        assert plan.selection["beat"]["margin"] > 0
    finally:
        plugins._REGISTRY.clear()
        plugins._REGISTRY.update(before)


def test_the_reason_is_machine_readable(registry):
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    sel = plan.selection
    for key in ("rule", "objective", "reason", "decided_on", "selected_source"):
        assert key in sel
    doc = json.loads(json.dumps(plan.as_dict(), default=float))
    assert doc["selection"]["objective"] == "max_quality"
    for cand in doc["candidate_results"]:
        assert cand["verdict"] in {
            "selected",
            "candidate",
            "reject",
            "infeasible",
            "unsupported",
        }
        assert cand["left_at"]


# ------------------------------------------------------------------ #
# Criterion 6: a rejected codec reports its false clear                #
# ------------------------------------------------------------------ #


def test_a_rejected_codec_reports_its_false_clear(registry):
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    flat = [c for c in plan.candidate_results if c.codec == "t_flatterer"][0]
    assert flat.verdict != "selected"
    fc = flat.false_clear
    assert fc, "the false-clear diagnostic must be attached to every evaluation"
    assert fc["false_clear"] is True
    assert fc["false_clear_given_cleared"] > 0.0
    assert fc["verdict"] in {"warn", "fail"}
    assert flat.nominal["role"].startswith("diagnostic only")
    assert "cosine" in flat.nominal["metric"]


# ------------------------------------------------------------------ #
# Criterion 7-8: evidence and fallback                                 #
# ------------------------------------------------------------------ #


def test_the_plan_carries_evidence_and_a_declared_fallback(registry):
    from turboquant_pro import runtime_policy as rp

    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    kinds = {e["kind"] for e in plan.evidence}
    assert "statistical" in kinds
    assert "measured_cost" in kinds
    stat = [e for e in plan.evidence if e["kind"] == "statistical"][0]
    assert stat["population"].startswith("held-out")
    assert "bootstrap" in stat["method"]

    assert plan.certificate_requirement["status"] in {
        "certified",
        "exact_rerank_required",
        "not_established",
    }
    fb = plan.fallback_policy
    assert fb["action"] in {
        rp.PROCEED,
        rp.RERANK_MORE,
        rp.REQUIRE_EXACT_RERANK,
        rp.PER_CHANNEL_OR_FP16,
    }
    assert fb["triggers"] and fb["module"].endswith("TQPRuntimePolicy")


def test_quality_is_accepted_on_the_bound_not_the_mean(registry):
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    q = plan.expected_quality
    assert q is not None
    assert q.bound <= q.mean  # higher-is-better: the conservative end is lower
    assert q.ci_low <= q.mean <= q.ci_high
    assert q.clears(QualityFloor(minimum=0.80, confidence=0.95))
    assert q.as_dict()["bound_is"] == "lower"


# ------------------------------------------------------------------ #
# Criterion 9: reproducibility                                         #
# ------------------------------------------------------------------ #


def test_the_same_plan_on_the_same_artifact_reproduces(registry):
    corpus = _corpus()
    a = CompressionPlanner(_retrieval_spec()).plan(Artifact(corpus))
    b = CompressionPlanner(_retrieval_spec()).plan(Artifact(corpus))
    assert a.selected_codec == b.selected_codec
    assert a.selected_parameters == b.selected_parameters
    assert a.expected_quality.mean == pytest.approx(b.expected_quality.mean)

    report = replay_plan(a.as_dict(), Artifact(corpus))
    assert report["artifact_identity_matches"] is True
    assert report["codec_agrees"] is True
    assert report["reproduced"] is True


def test_a_replay_against_other_bytes_reports_the_mismatch(registry):
    corpus = _corpus()
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(corpus))
    report = replay_plan(plan.as_dict(), Artifact(_corpus(seed=99)))
    assert report["artifact_identity_matches"] is False
    assert report["reproduced"] is False


# ------------------------------------------------------------------ #
# Criterion 10: abstention                                             #
# ------------------------------------------------------------------ #


def test_an_unregistered_consumer_abstains(registry):
    spec = _retrieval_spec(consumer="telepathy")
    plan = CompressionPlanner(spec).plan(Artifact(_corpus()))
    assert plan.selected_codec == ABSTAIN
    assert plan.abstained
    assert "telepathy" in plan.selection["reason"]
    assert plan.fallback_policy["action"]


def test_a_consumer_defined_for_another_target_abstains(registry):
    spec = _retrieval_spec(target="kv_key", consumer="topk_inner_product")
    plan = CompressionPlanner(spec).plan(Artifact(_corpus()))
    assert plan.selected_codec == ABSTAIN
    assert "kv_key" in plan.selection["reason"]


def test_a_floor_nothing_reaches_abstains(registry):
    spec = _retrieval_spec(floor=QualityFloor(minimum=0.999999, confidence=0.99))
    plan = CompressionPlanner(spec).plan(Artifact(_corpus()))
    assert plan.selected_codec == ABSTAIN
    assert "floor" in plan.selection["reason"]
    assert plan.candidate_results, "the candidates that failed are still recorded"


def test_a_bit_budget_makes_a_codec_infeasible_before_it_runs(registry):
    spec = _retrieval_spec(budget=Budget(max_bits=4))
    plan = CompressionPlanner(spec).plan(Artifact(_corpus()))
    eights = [c for c in plan.candidate_results if c.declared_bits == 8]
    assert eights
    assert all(c.verdict == "infeasible" for c in eights)
    assert all(c.left_at == "prior" for c in eights)


def test_a_byte_budget_rejects_at_verification(registry):
    spec = _retrieval_spec(budget=Budget(max_bytes_per_vector=1.0))
    plan = CompressionPlanner(spec).plan(Artifact(_corpus()))
    assert plan.selected_codec == ABSTAIN
    assert any(
        c.verdict == "infeasible" and "stored bytes/vector" in c.reason
        for c in plan.candidate_results
    )


# ------------------------------------------------------------------ #
# The measurement primitives                                           #
# ------------------------------------------------------------------ #


def test_container_bytes_counts_every_buffer():
    q = _Uniform(bits=4)
    x = _corpus(n=100, d=32)
    c = q.compress(x)
    acct = container_bytes(c)
    assert acct["total_bytes"] == (
        c["codes"].nbytes + c["lo"].nbytes + c["scale"].nbytes
    )
    assert set(acct["breakdown"]) >= {
        "container.codes",
        "container.lo",
        "container.scale",
    }


def test_preflight_names_the_preconditions_that_change_the_answer():
    x = _corpus(n=300, d=16)
    x[5] = 0.0
    report = preflight(x)
    assert report["zero_rows"] == 1
    assert any("all-zero" in f for f in report["flags"])
    assert 0.0 < report["spectrum"]["variance_at_d_over_2"] <= 1.0


def test_the_alias_in_the_issue_is_the_same_object():
    assert QuantizationControlPlane is CompressionPlanner


# ------------------------------------------------------------------ #
# The record                                                          #
# ------------------------------------------------------------------ #


def test_the_plan_record_validates_against_the_shipped_schema(registry):
    jsonschema = pytest.importorskip("jsonschema")
    from turboquant_pro.cli import _json_safe

    schema = load_schema(SCHEMA_NAME)
    jsonschema.Draft202012Validator.check_schema(schema)
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    doc = json.loads(json.dumps(_json_safe(plan.as_dict()), allow_nan=False))
    jsonschema.validate(doc, schema)


def test_an_abstention_also_validates(registry):
    jsonschema = pytest.importorskip("jsonschema")
    from turboquant_pro.cli import _json_safe

    plan = CompressionPlanner(_retrieval_spec(consumer="telepathy")).plan(
        Artifact(_corpus())
    )
    doc = json.loads(json.dumps(_json_safe(plan.as_dict()), allow_nan=False))
    jsonschema.validate(doc, load_schema(SCHEMA_NAME))


def test_explain_reads_as_a_decision(registry):
    plan = CompressionPlanner(_retrieval_spec()).plan(Artifact(_corpus()))
    text = plan.explain()
    assert "compression plan" in text
    assert "selected:" in text
    assert plan.selected_codec in text
    assert "consumer (bound)" in text


# ------------------------------------------------------------------ #
# The consumer contract                                               #
# ------------------------------------------------------------------ #


def test_retrieval_consumer_scores_one_item_per_query():
    c = consumers_mod.create_consumer("topk_inner_product", k=10, n_queries=50)
    x = _corpus(n=400, d=32)
    scores = c.per_item(x, x)
    assert scores.shape == (50,)
    assert np.allclose(scores, 1.0), "an identity round-trip loses no neighbours"
    assert c.higher_is_better is True


def test_attention_consumer_sees_what_cosine_misses():
    rng = np.random.default_rng(7)
    keys = rng.standard_normal((300, 48)).astype(np.float32)
    queries = rng.standard_normal((32, 48)).astype(np.float32)
    q = _CosineFlatterer()
    recon = q.decompress(q.compress(keys))

    cos = consumers_mod.row_cosine(keys, recon).mean()
    consumer = consumers_mod.create_consumer("attention_softmax", mode="spearman")
    rho = consumer.per_item(keys, recon, queries=queries).mean()
    assert cos > 0.9, "cosine is reassuring"
    assert rho < cos, "the attention ranking is not"


def test_read_operator_consumer_is_a_distortion_not_a_similarity():
    c = consumers_mod.create_consumer("read_operator", operator="identity")
    x = _corpus(n=200, d=16)
    assert c.higher_is_better is False
    assert float(np.mean(c.per_item(x, x))) == pytest.approx(0.0, abs=1e-12)
    noisy = x + 0.1 * np.random.default_rng(0).standard_normal(x.shape)
    assert float(np.mean(c.per_item(x, noisy))) > 0.0


def test_an_unknown_consumer_raises_rather_than_substituting_one():
    with pytest.raises(KeyError, match="abstains"):
        consumers_mod.get_consumer("vibes")


def test_nominal_metric_is_aligned_to_the_consumers_items():
    c = consumers_mod.create_consumer("topk_inner_product", k=5, n_queries=40)
    x = _corpus(n=300, d=24)
    q = _Uniform(bits=4)
    recon = q.decompress(q.compress(x))
    per_item = c.per_item(x, recon)
    nominal = consumers_mod.nominal_per_item(c, x, recon)
    assert nominal is not None
    assert nominal.shape == per_item.shape


# ------------------------------------------------------------------ #
# The CLI                                                             #
# ------------------------------------------------------------------ #


def test_cli_plan_run_and_explain(tmp_path, registry, capsys):
    from turboquant_pro.cli import main

    art = tmp_path / "corpus.npy"
    np.save(art, _corpus(n=600, d=32))
    out = tmp_path / "plan.json"
    rc = main(
        [
            "plan",
            "run",
            "--artifact",
            str(art),
            "--target",
            "embedding",
            "--consumer",
            "topk_inner_product",
            "--consumer-config",
            '{"k": 10, "n_queries": 80}',
            "--candidates",
            "t_uniform,t_fine,t_flatterer",
            "--floor",
            "0.8",
            "--n-boot",
            "64",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "turboquant-pro/compression-plan"
    assert doc["selected_codec"] != ABSTAIN

    capsys.readouterr()
    assert main(["plan", "explain", str(out)]) == 0
    text = capsys.readouterr().out
    assert "selected:" in text


def test_cli_plan_run_exits_one_on_abstention(tmp_path, registry):
    from turboquant_pro.cli import main

    art = tmp_path / "corpus.npy"
    np.save(art, _corpus(n=400, d=32))
    rc = main(
        [
            "plan",
            "run",
            "--artifact",
            str(art),
            "--consumer",
            "telepathy",
            "--candidates",
            "t_uniform",
        ]
    )
    assert rc == 1


def test_cli_plan_consumers_lists_the_registry(registry, capsys):
    from turboquant_pro.cli import main

    assert main(["plan", "consumers", "--target", "embedding"]) == 0
    out = capsys.readouterr().out
    assert "topk_inner_product" in out
    assert "t_uniform" in out
