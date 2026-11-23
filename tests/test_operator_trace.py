# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Tests for operator-regime tracing (turboquant_pro.operator_trace)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from turboquant_pro.operator_trace import (  # noqa: E402
    OperatorRegime,
    QuantTarget,
    discipline_for,
    recommend_quantization,
    trace_operators,
)

# --------------------------------------------------------------------------- #
# Synthetic architectures                                                      #
# --------------------------------------------------------------------------- #


class TinyAttention(nn.Module):
    """Standard attention with conventional projection names."""

    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = F.softmax(q @ k.transpose(-1, -2), dim=-1)
        return self.o_proj(attn @ v)


class ObfuscatedAttention(nn.Module):
    """Same computation, names that reveal nothing (only fx can classify)."""

    def __init__(self, d=16):
        super().__init__()
        self.left = nn.Linear(d, d, bias=False)  # is really the query
        self.right = nn.Linear(d, d, bias=False)  # is really the key
        self.val = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(d, d, bias=False)

    def forward(self, x):
        attn = F.softmax(self.left(x) @ self.right(x).transpose(-1, -2), dim=-1)
        return self.mix(attn @ self.val(x))


class SwiGLU(nn.Module):
    """SwiGLU MLP: `gate_proj` is a linear activation gate, NOT a router."""

    def __init__(self, d=16, h=32):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyMoE(nn.Module):
    """A router that selects experts via top-k on the gate logits."""

    def __init__(self, d=16, n_experts=4, k=2):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(n_experts)]
        )
        self.k = k

    def forward(self, x):
        logits = self.router(x)
        _, idx = torch.topk(logits, self.k, dim=-1)
        return x + idx.sum() * 0.0  # keep idx in the graph


class TinySSM(nn.Module):
    """A minimal selective-scan block with Mamba-style parameter names."""

    def __init__(self, d=16):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(d))
        self.x_proj = nn.Linear(d, d, bias=False)
        self.dt_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        a = -torch.exp(self.A_log)
        dt = self.dt_proj(x)
        decayed = torch.cumsum(self.x_proj(x) * a * dt, dim=1)  # a scan
        return self.out_proj(decayed)


# --------------------------------------------------------------------------- #
# Structural classification                                                    #
# --------------------------------------------------------------------------- #


def _regime(plan, name):
    return plan.tensors[name].regime


def test_structural_standard_attention():
    plan = trace_operators(TinyAttention(), prefer="structural")
    assert _regime(plan, "q_proj.weight") is OperatorRegime.SOFTMAX_SCORE
    assert _regime(plan, "k_proj.weight") is OperatorRegime.SOFTMAX_SCORE
    assert _regime(plan, "v_proj.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "o_proj.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "norm.weight") is OperatorRegime.NORM


def test_swiglu_gate_proj_is_not_a_router():
    """The name-collision correctness test: gate_proj != router gate."""
    plan = trace_operators(SwiGLU(), prefer="structural")
    for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        assert _regime(plan, name) is OperatorRegime.LINEAR_RESIDUAL


def test_structural_moe_router():
    plan = trace_operators(TinyMoE(), prefer="structural")
    assert _regime(plan, "router.weight") is OperatorRegime.GATE_SELECTION
    # Experts are ordinary residual-writing linears.
    assert _regime(plan, "experts.0.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_structural_ssm():
    plan = trace_operators(TinySSM(), prefer="structural")
    assert _regime(plan, "A_log") is OperatorRegime.STATE_DECAY
    assert _regime(plan, "x_proj.weight") is OperatorRegime.STATE_DECAY
    assert _regime(plan, "dt_proj.weight") is OperatorRegime.STATE_DECAY
    assert _regime(plan, "out_proj.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_structural_misses_obfuscated_names():
    """Structural alone cannot see the score path when names hide it."""
    plan = trace_operators(ObfuscatedAttention(), prefer="structural")
    # `left`/`right` look like nothing -> not SOFTMAX_SCORE structurally.
    assert _regime(plan, "left.weight") is not OperatorRegime.SOFTMAX_SCORE
    assert _regime(plan, "right.weight") is not OperatorRegime.SOFTMAX_SCORE


# --------------------------------------------------------------------------- #
# Graph (fx) classification --- the "unseen architecture" capability           #
# --------------------------------------------------------------------------- #


def test_fx_recovers_obfuscated_score_path():
    """fx follows the tensors into softmax and tags the real Q/K by graph."""
    plan = trace_operators(ObfuscatedAttention(), prefer="auto")
    assert plan.traced is True
    assert _regime(plan, "left.weight") is OperatorRegime.SOFTMAX_SCORE
    assert _regime(plan, "right.weight") is OperatorRegime.SOFTMAX_SCORE
    assert plan.tensors["left.weight"].method == "fx"
    # `val`/`mix` are downstream of softmax -> stay residual.
    assert _regime(plan, "val.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "mix.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_fx_standard_attention_matches_structural():
    plan = trace_operators(TinyAttention(), prefer="auto")
    assert plan.traced is True
    assert _regime(plan, "q_proj.weight") is OperatorRegime.SOFTMAX_SCORE
    assert _regime(plan, "k_proj.weight") is OperatorRegime.SOFTMAX_SCORE
    # v/o feed the softmax *output*, not the softmax -> not score path.
    assert _regime(plan, "v_proj.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "o_proj.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_fx_moe_router_by_topk():
    plan = trace_operators(TinyMoE(), prefer="auto")
    assert _regime(plan, "router.weight") is OperatorRegime.GATE_SELECTION


def test_fx_ssm_by_cumsum():
    plan = trace_operators(TinySSM(), prefer="auto")
    # A_log feeds exp then the cumsum scan; x_proj feeds the scan.
    assert _regime(plan, "x_proj.weight") is OperatorRegime.STATE_DECAY


# --------------------------------------------------------------------------- #
# The (A2) discipline table                                                    #
# --------------------------------------------------------------------------- #


def test_discipline_flips_between_weight_and_kv_activation():
    """The core (A2) content: the sensitive coordinate flips by target."""
    # Score path: weights are robust+symmetric; cached keys are fragile+DC.
    w = discipline_for(OperatorRegime.SOFTMAX_SCORE, QuantTarget.WEIGHT)
    a = discipline_for(OperatorRegime.SOFTMAX_SCORE, QuantTarget.KV_ACTIVATION)
    assert w.family == "symmetric" and w.protect_dc is False and w.sensitivity == "low"
    assert (
        a.family == "per_channel" and a.protect_dc is True and a.sensitivity == "high"
    )

    # Residual path: weights are symmetric but HIGH sensitivity (V/O finding);
    # cached values are cheap polar.
    wr = discipline_for(OperatorRegime.LINEAR_RESIDUAL, QuantTarget.WEIGHT)
    ar = discipline_for(OperatorRegime.LINEAR_RESIDUAL, QuantTarget.KV_ACTIVATION)
    assert wr.family == "symmetric" and wr.sensitivity == "high"
    assert ar.family == "polar" and ar.sensitivity == "low"


def test_norm_and_unknown_are_conservative():
    n = discipline_for(OperatorRegime.NORM)
    assert n.family == "keep_fp"
    u = discipline_for(OperatorRegime.UNKNOWN)
    # Unknown must never discard scale.
    assert u.family == "per_channel" and u.protect_dc is True


def test_recommend_quantization_end_to_end():
    recs = recommend_quantization(TinyAttention(), target="weight", prefer="auto")
    assert recs["k_proj.weight"].family == "symmetric"  # robust score-path weight
    assert recs["o_proj.weight"].sensitivity == "high"  # sensitive residual weight
    # Same model, KV-activation target: keys become fragile per-channel.
    recs_kv = recommend_quantization(
        TinyAttention(), target=QuantTarget.KV_ACTIVATION, prefer="auto"
    )
    assert recs_kv["k_proj.weight"].family == "per_channel"
    assert recs_kv["k_proj.weight"].protect_dc is True


def test_plan_summary_and_coverage():
    plan = trace_operators(TinyAttention(), prefer="auto")
    summ = plan.summary()
    assert summ["n_tensors"] == len(plan.tensors)
    assert 0.0 <= summ["coverage"] <= 1.0
    assert summ["coverage"] == 1.0  # every tensor here is classifiable


# --------------------------------------------------------------------------- #
# Graph (autograd) classification: one real forward pass, no names read        #
# --------------------------------------------------------------------------- #


class ValueRouter(nn.Module):
    """A router whose top-k values weight two experts; names reveal nothing."""

    def __init__(self, d=16, n_experts=4):
        super().__init__()
        self.p0 = nn.Linear(d, n_experts, bias=False)  # is really the gate
        self.p1 = nn.Linear(d, d, bias=False)
        self.p2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        w = torch.topk(torch.softmax(self.p0(x), dim=-1), 2, dim=-1).values
        return x + w.sum(-1, keepdim=True) * (self.p1(x) + self.p2(x))


class LoopRecurrence(nn.Module):
    """h_t = a * h_{t-1} + b_t written as a Python loop, with no scan operator."""

    def __init__(self, d=16):
        super().__init__()
        self.p0 = nn.Parameter(torch.zeros(d))  # is really the log decay rate
        self.p1 = nn.Linear(d, d, bias=False)  # the input map
        self.p2 = nn.Linear(d, d, bias=False)  # the output map

    def forward(self, x):
        a = torch.exp(-torch.exp(self.p0))
        b = self.p1(x)
        h = torch.zeros_like(b[:, 0])
        hs = []
        for t in range(x.shape[1]):
            h = a * h + b[:, t]
            hs.append(h)
        return self.p2(torch.stack(hs, dim=1))


class RejectsInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        raise ValueError("no")


def _x(d=16):
    torch.manual_seed(0)
    return torch.randn(2, 6, d)


def test_graph_mode_recovers_obfuscated_score_path():
    plan = trace_operators(ObfuscatedAttention(), example_inputs=_x(), prefer="graph")
    assert plan.autograd_status == "ok" and plan.n_sinks == 1 and plan.traced
    assert _regime(plan, "left.weight") is OperatorRegime.SOFTMAX_SCORE
    assert _regime(plan, "right.weight") is OperatorRegime.SOFTMAX_SCORE
    assert plan.tensors["left.weight"].method == "autograd"
    assert _regime(plan, "val.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "mix.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_graph_mode_selection_outranks_the_softmax_before_it():
    plan = trace_operators(ValueRouter(), example_inputs=_x(), prefer="graph")
    assert _regime(plan, "p0.weight") is OperatorRegime.GATE_SELECTION
    assert _regime(plan, "p1.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "p2.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_graph_mode_finds_a_loop_recurrence_and_spares_its_input_map():
    plan = trace_operators(LoopRecurrence(), example_inputs=_x(), prefer="graph")
    assert _regime(plan, "p0") is OperatorRegime.STATE_DECAY
    assert _regime(plan, "p1.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "p2.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_graph_mode_scan_labels_the_multiplier_not_the_input_map():
    plan = trace_operators(TinySSM(), example_inputs=_x(), prefer="graph")
    assert _regime(plan, "A_log") is OperatorRegime.STATE_DECAY
    assert _regime(plan, "x_proj.weight") is OperatorRegime.LINEAR_RESIDUAL
    assert _regime(plan, "out_proj.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_graph_mode_norm_by_module_type_and_swiglu_is_not_a_router():
    plan = trace_operators(TinyAttention(), example_inputs=_x(), prefer="graph")
    assert _regime(plan, "norm.weight") is OperatorRegime.NORM
    plan = trace_operators(SwiGLU(), example_inputs=_x(), prefer="graph")
    assert plan.autograd_status == "ok" and plan.n_sinks == 0 and not plan.traced
    assert _regime(plan, "gate_proj.weight") is OperatorRegime.LINEAR_RESIDUAL


def test_a_failed_forward_is_reported_not_hidden():
    plan = trace_operators(RejectsInput(), example_inputs=_x(4), prefer="graph")
    assert plan.autograd_status.startswith("failed: ValueError")
    assert not plan.traced
    with pytest.raises(ValueError):
        trace_operators(TinyAttention(), prefer="graph")


def test_auto_mode_uses_the_forward_pass_when_given_inputs():
    plan = trace_operators(LoopRecurrence(), example_inputs=_x(), prefer="auto")
    assert plan.tensors["p0"].method == "autograd"
    assert _regime(plan, "p0") is OperatorRegime.STATE_DECAY
    # requires_grad flags are left as they were found
    model = ObfuscatedAttention()
    model.left.weight.requires_grad_(False)
    plan = trace_operators(model, example_inputs=_x(), prefer="graph")
    assert _regime(plan, "left.weight") is OperatorRegime.SOFTMAX_SCORE
    assert model.left.weight.requires_grad is False


class FusedAttention(nn.Module):
    """Attention through the fused scaled-dot-product operator."""

    def __init__(self, d=16):
        super().__init__()
        self.p0 = nn.Linear(d, d, bias=False)  # query
        self.p1 = nn.Linear(d, d, bias=False)  # key
        self.p2 = nn.Linear(d, d, bias=False)  # value
        self.p3 = nn.Linear(d, d, bias=False)  # output

    def forward(self, x):
        y = F.scaled_dot_product_attention(self.p0(x), self.p1(x), self.p2(x))
        return self.p3(y)


def test_graph_mode_reads_the_fused_attention_operator():
    # three- and four-dimensional inputs reach different autograd nodes
    for x in (_x(), torch.randn(2, 4, 6, 16)):
        plan = trace_operators(FusedAttention(), example_inputs=x, prefer="graph")
        assert plan.n_sinks == 1
        assert _regime(plan, "p0.weight") is OperatorRegime.SOFTMAX_SCORE
        assert _regime(plan, "p1.weight") is OperatorRegime.SOFTMAX_SCORE
        assert _regime(plan, "p2.weight") is OperatorRegime.LINEAR_RESIDUAL
        assert _regime(plan, "p3.weight") is OperatorRegime.LINEAR_RESIDUAL
