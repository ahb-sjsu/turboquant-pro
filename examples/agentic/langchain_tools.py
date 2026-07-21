# TurboQuant Pro -- LangChain tool wrappers
# MIT License
"""Expose TurboQuant Pro as LangChain tools.

Thin ``@tool`` wrappers over :mod:`turboquant_pro.agent_tools`; the library
functions are the contract, LangChain is just transport. Every tool takes the
*goal* as an explicit argument (recall ``k``/``min_recall``, or the ``consumer``
metric) -- "keep the angle" means the goal is dynamic, chosen per call, so the
compression is accepted and certified against the goal the task actually cares
about, never against reconstruction cosine.

Run:
    pip install turboquant-pro langchain-core numpy
    python langchain_tools.py            # prints the tool list

Wire into an agent:
    from langchain.agents import create_react_agent   # or your framework
    agent = create_react_agent(llm, TQP_TOOLS, prompt)

Corpora are passed as paths to ``.npy`` files so a tool-calling model can hand
over an arbitrarily large array by reference instead of inlining it.
"""

from __future__ import annotations

import numpy as np
from langchain_core.tools import tool

from turboquant_pro import (
    best_compression_at_recall,
    certify_ranking,
    recommend_kv_key_quantizer,
)


def _load(path: str) -> np.ndarray:
    """Load an ``.npy`` matrix a model referenced by path."""
    return np.load(path)


@tool
def tqp_best_compression_at_recall(
    corpus_npy: str, k: int = 10, min_recall: float = 0.99
) -> dict:
    """Best compression ratio for a corpus that still keeps recall@k >= min_recall.

    Use when a user asks "how much can I shrink this corpus while keeping
    <recall>?". The GOAL is dynamic: you choose k and min_recall for the task.
    Acceptance is on recall@k (retrieval fidelity), never reconstruction cosine.

    Args:
        corpus_npy: Path to a .npy file with a 2-D embedding matrix (n, dim).
        k: Retrieval cut-off (top-k neighbours).
        min_recall: Recall threshold in [0, 1] to hold.

    Returns:
        A dict with compression_ratio, achieved_recall_at_k, meets_target,
        the chosen config, and a cosine_diagnostic (reported, not accepted on).
    """
    return best_compression_at_recall(_load(corpus_npy), k=k, min_recall=min_recall)


@tool
def tqp_certify_ranking(
    original_npy: str, reconstructed_npy: str, metric: str = "cosine"
) -> dict:
    """Guaranteed distribution-free floor on rank agreement (the certificate).

    Print this after choosing a config: a worst-case lower bound on Spearman and
    Kendall-tau between the exact and compressed rankings, holding without
    distributional assumptions. ``vacuous=true`` means exact reranking is
    required.

    Args:
        original_npy: Path to .npy of the reference vectors (n, dim).
        reconstructed_npy: Path to .npy of the decompressed vectors (n, dim).
        metric: Ranking metric to certify ("cosine" or "l2").
    """
    return certify_ranking(_load(original_npy), _load(reconstructed_npy), metric=metric)


@tool
def tqp_recommend_kv_key_quantizer(
    keys_npy: str, consumer: str = "attention_logits", bits: int = 4
) -> dict:
    """Recommend polar vs. per-channel quantization for LLM attention keys.

    Runs the (A2) probe against the declared consumer metric (the dynamic goal:
    "attention_logits" for keys, or "cosine"/"l2"). Catches the regime where the
    polar quotient's reconstruction stays high while its attention-logit ranking
    collapses.

    Args:
        keys_npy: Path to .npy of an attention-key sample (n, dim).
        consumer: Downstream metric to preserve.
        bits: Probe bit budget per dimension.
    """
    return recommend_kv_key_quantizer(_load(keys_npy), consumer=consumer, bits=bits)


TQP_TOOLS = [
    tqp_best_compression_at_recall,
    tqp_certify_ranking,
    tqp_recommend_kv_key_quantizer,
]


if __name__ == "__main__":
    print("TurboQuant Pro LangChain tools:")
    for t in TQP_TOOLS:
        print(f"  - {t.name}: {t.description.splitlines()[0]}")
