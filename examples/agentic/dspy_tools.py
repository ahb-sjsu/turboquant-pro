# TurboQuant Pro -- DSPy tool + module wrappers
# MIT License
"""Expose TurboQuant Pro as DSPy tools and a small ReAct module.

Delegates to :mod:`turboquant_pro.agent_tools`. The "goal is dynamic" principle
maps directly onto DSPy: the goal (recall ``k``/``min_recall``, or the
``consumer`` metric) is a signature input the program fills at runtime, so a
compiled DSPy program can pick the acceptance target per corpus/task -- and the
compression is always accepted on that goal's retrieval fidelity, never on
reconstruction cosine.

Run:
    pip install turboquant-pro dspy-ai numpy
    python dspy_tools.py            # prints the tool list

Use as tools in a DSPy agent:
    import dspy
    agent = dspy.ReAct("question -> answer", tools=TQP_TOOLS)
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import (
    best_compression_at_recall,
    certify_ranking,
    recommend_kv_key_quantizer,
)

try:
    import dspy
except ImportError:  # pragma: no cover - example runs without dspy installed
    dspy = None


def best_compression_at_recall_tool(
    corpus_npy: str, k: int = 10, min_recall: float = 0.99
) -> dict:
    """Best compression ratio for a corpus keeping recall@k >= min_recall.

    The goal (k, min_recall) is dynamic and chosen per task. Accepts on recall,
    not cosine. `corpus_npy` is the path to a 2-D .npy embedding matrix.
    """
    return best_compression_at_recall(np.load(corpus_npy), k=k, min_recall=min_recall)


def certify_ranking_tool(
    original_npy: str, reconstructed_npy: str, metric: str = "cosine"
) -> dict:
    """Distribution-free guaranteed floor on Spearman/Kendall rank agreement.

    Paths point to 2-D .npy matrices of the reference and reconstructed vectors.
    """
    return certify_ranking(
        np.load(original_npy), np.load(reconstructed_npy), metric=metric
    )


def recommend_kv_key_quantizer_tool(
    keys_npy: str, consumer: str = "attention_logits", bits: int = 4
) -> dict:
    """(A2) probe: recommend polar vs. per-channel quantization for attention keys.

    `consumer` is the dynamic goal metric to preserve. `keys_npy` is a .npy path.
    """
    return recommend_kv_key_quantizer(np.load(keys_npy), consumer=consumer, bits=bits)


_TOOL_FNS = [
    best_compression_at_recall_tool,
    certify_ranking_tool,
    recommend_kv_key_quantizer_tool,
]

# dspy.Tool wraps a plain function; docstring + signature become the tool spec.
TQP_TOOLS = [dspy.Tool(fn) for fn in _TOOL_FNS] if dspy is not None else _TOOL_FNS


if dspy is not None:

    class CompressionAdvisor(dspy.Module):
        """Answer corpus-compression questions using the TurboQuant Pro tools.

        A minimal ReAct agent whose only tools are the three above; the goal it
        optimizes (recall target, consumer metric) is whatever the question
        specifies, so acceptance stays task-relative.
        """

        def __init__(self) -> None:
            super().__init__()
            self.agent = dspy.ReAct("question -> answer", tools=TQP_TOOLS)

        def forward(self, question: str) -> dspy.Prediction:
            return self.agent(question=question)


if __name__ == "__main__":
    print("TurboQuant Pro DSPy tools:")
    for fn in _TOOL_FNS:
        print(f"  - {fn.__name__}: {fn.__doc__.splitlines()[0]}")
    if dspy is None:
        print("(install dspy-ai to get dspy.Tool wrappers and CompressionAdvisor)")
