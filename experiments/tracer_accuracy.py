# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""How often does the operator tracer assign the right regime? (experiment E7)

The tracer (turboquant_pro/operator_trace.py) maps each parameter tensor to the
operator that consumes it. This measures it on models whose consuming operators
are known by construction.

Two groups of models:

  A. Library architectures, built from their configuration classes at a small
     size with random weights (no checkpoint is downloaded, since only the
     module graph and the parameter names matter): Llama, Mistral, Qwen2, Gemma,
     GPT-2, Mixtral, OLMoE, Mamba. Ground truth comes from each architecture's
     own parameter names (TRUTH below). These models may defeat torch.fx
     symbolic tracing; the script records whether the graph pass succeeded and
     so how often the tracer fell back to names.

  B. Small traceable models written here: an attention block, a top-k router,
     a diagonal recurrence, and a SwiGLU block, each in two namings: the usual
     names, and names that carry no signal (p0, p1, ...). With the uninformative
     names the structural pass has nothing to read, so the score is the graph
     pass alone.

A tensor is scored only if its consumer is declared. Biases, embeddings,
convolutions and the Mamba projections that feed the step size have no declared
truth here and are counted as "not scored". GPT-2's fused c_attn holds query,
key and value in one tensor, so its truth is mixed; it is reported separately.

Output: a JSON with, per model and pooled per group, the confusion counts
(true regime by predicted regime), accuracy by true regime, the number of
tensors not scored, and whether the graph pass succeeded. No GPU is needed.

Usage:
    python experiments/tracer_accuracy.py [out.json]
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tracer():
    """Import operator_trace.py by path, so the package need not be installed."""
    for cand in (
        os.path.join(HERE, "..", "turboquant_pro", "operator_trace.py"),
        os.path.join(HERE, "operator_trace.py"),
    ):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("operator_trace", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["operator_trace"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("operator_trace.py")


OT = load_tracer()
R = OT.OperatorRegime

# Ground truth for library architectures, by parameter name. First match wins.
# None means "no declared truth, not scored".
TRUTH = [
    (r"(^|\.)a_log$", R.STATE_DECAY),
    (r"\.gate\.weight$", R.GATE_SELECTION),  # Mixtral / OLMoE router
    (r"(q_proj|k_proj)\.weight$", R.SOFTMAX_SCORE),
    (r"c_attn\.weight$", "mixed"),  # GPT-2: Q, K and V fused
    (r"(v_proj|o_proj)\.weight$", R.LINEAR_RESIDUAL),
    (r"attn\.c_proj\.weight$", R.LINEAR_RESIDUAL),  # GPT-2 attention output
    (
        r"(gate_proj|up_proj|down_proj|w1|w2|w3|c_fc|mlp\.c_proj|fc1|fc2)\.weight$",
        R.LINEAR_RESIDUAL,
    ),
    (r"mixer\.out_proj\.weight$", R.LINEAR_RESIDUAL),  # Mamba block output
    (r"(norm|ln_\d|ln_f|layernorm|layer_norm)[^.]*\.weight$", R.NORM),
    (r".*", None),
]


def truth_of(name):
    low = name.lower()
    for pat, reg in TRUTH:
        if re.search(pat, low):
            return reg
    return None


# ---------------------------------------------------------------------------
# Group B: small traceable models with known consumers
# ---------------------------------------------------------------------------


class Attn(nn.Module):
    """One attention block. names = (query, key, value, output, norm)."""

    def __init__(self, names, d=16):
        super().__init__()
        self.n = names
        for nm in names[:4]:
            setattr(self, nm, nn.Linear(d, d, bias=False))
        setattr(self, names[4], nn.LayerNorm(d))

    def forward(self, x):
        q, k, v, o, norm = (getattr(self, nm) for nm in self.n)
        h = norm(x)
        s = torch.matmul(q(h), k(h).transpose(-1, -2))
        a = torch.softmax(s, dim=-1)
        return x + o(torch.matmul(a, v(h)))


class Router(nn.Module):
    """A top-k router over two experts' worth of linear maps. names = (gate, e0, e1)."""

    def __init__(self, names, d=16, n_exp=4):
        super().__init__()
        self.n = names
        setattr(self, names[0], nn.Linear(d, n_exp, bias=False))
        setattr(self, names[1], nn.Linear(d, d, bias=False))
        setattr(self, names[2], nn.Linear(d, d, bias=False))

    def forward(self, x):
        gate, e0, e1 = (getattr(self, nm) for nm in self.n)
        w = torch.topk(gate(x), 2, dim=-1).values.sum(-1, keepdim=True)
        return x + w * (e0(x) + e1(x))


class Recurrence(nn.Module):
    """A diagonal recurrence with a stored log-decay. names = (decay parameter, input map)."""

    def __init__(self, names, d=16):
        super().__init__()
        self.n = names
        setattr(self, names[0], nn.Parameter(torch.zeros(d)))
        setattr(self, names[1], nn.Linear(d, d, bias=False))

    def forward(self, x):
        a_log, inp = getattr(self, self.n[0]), getattr(self, self.n[1])
        a = torch.exp(-torch.exp(a_log))
        return torch.cumsum(inp(x) * a, dim=1)


class SwiGLU(nn.Module):
    """A SwiGLU feed-forward block: its gate projection is not a router. names = (gate, up, down)."""

    def __init__(self, names, d=16):
        super().__init__()
        self.n = names
        for nm in names:
            setattr(self, nm, nn.Linear(d, d, bias=False))

    def forward(self, x):
        g, u, dn = (getattr(self, nm) for nm in self.n)
        return x + dn(F.silu(g(x)) * u(x))


def group_b():
    """(label, model, truth dict by parameter name)."""
    out = []
    for tag, an, rn, cn, sn in [
        (
            "usual names",
            ("q_proj", "k_proj", "v_proj", "o_proj", "norm"),
            ("gate", "expert0", "expert1"),
            ("A_log", "in_proj"),
            ("gate_proj", "up_proj", "down_proj"),
        ),
        (
            "uninformative names",
            ("p0", "p1", "p2", "p3", "p4"),
            ("p0", "p1", "p2"),
            ("p0", "p1"),
            ("p0", "p1", "p2"),
        ),
    ]:
        out.append(
            (
                f"attention block, {tag}",
                Attn(an),
                {
                    f"{an[0]}.weight": R.SOFTMAX_SCORE,
                    f"{an[1]}.weight": R.SOFTMAX_SCORE,
                    f"{an[2]}.weight": R.LINEAR_RESIDUAL,
                    f"{an[3]}.weight": R.LINEAR_RESIDUAL,
                    f"{an[4]}.weight": R.NORM,
                },
            )
        )
        out.append(
            (
                f"top-k router, {tag}",
                Router(rn),
                {
                    f"{rn[0]}.weight": R.GATE_SELECTION,
                    f"{rn[1]}.weight": R.LINEAR_RESIDUAL,
                    f"{rn[2]}.weight": R.LINEAR_RESIDUAL,
                },
            )
        )
        out.append(
            (
                f"diagonal recurrence, {tag}",
                Recurrence(cn),
                {cn[0]: R.STATE_DECAY, f"{cn[1]}.weight": R.LINEAR_RESIDUAL},
            )
        )
        out.append(
            (
                f"SwiGLU block, {tag}",
                SwiGLU(sn),
                {
                    f"{sn[0]}.weight": R.LINEAR_RESIDUAL,
                    f"{sn[1]}.weight": R.LINEAR_RESIDUAL,
                    f"{sn[2]}.weight": R.LINEAR_RESIDUAL,
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Group A: library architectures at a small size, random weights
# ---------------------------------------------------------------------------


def group_a():
    import transformers as tf

    small = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=128,
    )
    specs = [
        ("Llama", "LlamaConfig", dict(small, num_key_value_heads=2)),
        ("Mistral", "MistralConfig", dict(small, num_key_value_heads=2)),
        ("Qwen2", "Qwen2Config", dict(small, num_key_value_heads=2)),
        ("Gemma", "GemmaConfig", dict(small, num_key_value_heads=1, head_dim=8)),
        ("GPT-2", "GPT2Config", dict(n_embd=32, n_layer=2, n_head=4, vocab_size=128)),
        (
            "Mixtral",
            "MixtralConfig",
            dict(
                small, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2
            ),
        ),
        (
            "OLMoE",
            "OlmoeConfig",
            dict(small, num_key_value_heads=4, num_experts=8, num_experts_per_tok=2),
        ),
        (
            "Mamba",
            "MambaConfig",
            dict(hidden_size=32, num_hidden_layers=2, vocab_size=128, state_size=8),
        ),
    ]
    out, skipped = [], []
    for label, cfg_name, kw in specs:
        try:
            cfg = getattr(tf, cfg_name)(**kw)
            model = tf.AutoModelForCausalLM.from_config(cfg).eval()
        except Exception as e:  # an architecture this transformers build lacks
            skipped.append(
                {"model": label, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            )
            continue
        truth = {n: truth_of(n) for n, _ in model.named_parameters()}
        out.append((label, model, truth))
    return out, skipped


# ---------------------------------------------------------------------------


def fx_status(model):
    """Does torch.fx symbolic tracing of the model succeed? The tracer's own `traced` flag
    is True only when the graph pass found a sink, so it cannot tell a failed trace from a
    graph with no softmax, top-k or scan in it."""
    import torch.fx as fx

    try:
        fx.symbolic_trace(model)
        return {"symbolic_trace": "ok"}
    except Exception as e:
        return {
            "symbolic_trace": "failed",
            "error": f"{type(e).__name__}: {str(e)[:160]}",
        }


def score(model, truth, prefer):
    plan = OT.trace_operators(model, prefer=prefer)
    conf, per_true, mixed, not_scored, by_method = (
        {},
        {},
        [],
        0,
        {"structural": 0, "fx": 0},
    )
    for name, t in plan.tensors.items():
        want = truth.get(name)
        if want is None:
            not_scored += 1
            continue
        if want == "mixed":
            mixed.append(
                {"tensor": name, "predicted": t.regime.value, "method": t.method}
            )
            continue
        by_method[t.method] += 1
        conf.setdefault(want.value, {}).setdefault(t.regime.value, 0)
        conf[want.value][t.regime.value] += 1
        a = per_true.setdefault(want.value, [0, 0])
        a[1] += 1
        a[0] += int(t.regime is want)
    right = sum(a[0] for a in per_true.values())
    total = sum(a[1] for a in per_true.values())
    return {
        "graph_pass_succeeded": bool(plan.traced),
        "fx": fx_status(model),
        "scored": total,
        "correct": right,
        "accuracy": (right / total) if total else None,
        "accuracy_by_true_regime": {
            k: {"correct": a[0], "n": a[1]} for k, a in per_true.items()
        },
        "confusion_true_by_predicted": conf,
        "not_scored": not_scored,
        "mixed_truth": mixed,
        "scored_by_method": by_method,
    }


def pool(rows):
    conf, per = {}, {}
    for r in rows:
        for tr, d in r["confusion_true_by_predicted"].items():
            for pr, n in d.items():
                conf.setdefault(tr, {}).setdefault(pr, 0)
                conf[tr][pr] += n
        for tr, a in r["accuracy_by_true_regime"].items():
            p = per.setdefault(tr, {"correct": 0, "n": 0})
            p["correct"] += a["correct"]
            p["n"] += a["n"]
    right, total = sum(p["correct"] for p in per.values()), sum(
        p["n"] for p in per.values()
    )
    return {
        "scored": total,
        "correct": right,
        "accuracy": right / total if total else None,
        "accuracy_by_true_regime": per,
        "confusion_true_by_predicted": conf,
        "models": len(rows),
        "models_where_graph_pass_succeeded": sum(
            r["graph_pass_succeeded"] for r in rows
        ),
        "models_where_symbolic_trace_ok": sum(
            r["fx"]["symbolic_trace"] == "ok" for r in rows
        ),
    }


def main():
    torch.manual_seed(0)
    import transformers

    out = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "truth_rules": [[p, (r.value if hasattr(r, "value") else r)] for p, r in TRUTH],
        "group_a": {},
        "group_b": {},
    }
    a_models, skipped = group_a()
    out["group_a_skipped"] = skipped
    for prefer in ("auto", "structural"):
        rows = {}
        for label, model, truth in a_models:
            rows[label] = score(model, truth, prefer)
        out["group_a"][prefer] = {
            "per_model": rows,
            "pooled": pool(list(rows.values())),
        }
    for prefer in ("auto", "structural"):
        rows = {}
        for label, model, truth in group_b():
            rows[label] = score(model, truth, prefer)
        usual = [v for k, v in rows.items() if k.endswith("usual names")]
        blind = [v for k, v in rows.items() if k.endswith("uninformative names")]
        out["group_b"][prefer] = {
            "per_model": rows,
            "pooled_usual_names": pool(usual),
            "pooled_uninformative_names": pool(blind),
        }
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "results_matched_bit", "tracer_accuracy.json")
    )
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for grp in ("group_a", "group_b"):
        for prefer, v in out[grp].items():
            for k, p in v.items():
                if k.startswith("pooled"):
                    print(
                        grp,
                        prefer,
                        k,
                        "accuracy",
                        p["accuracy"],
                        "scored",
                        p["scored"],
                        "graph pass ok on",
                        p["models_where_graph_pass_succeeded"],
                        "of",
                        p["models"],
                    )
    print("skipped:", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
