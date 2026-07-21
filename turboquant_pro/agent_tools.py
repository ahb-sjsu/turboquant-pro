# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""Agent-facing tool surface for TurboQuant Pro.

Small, JSON-in / JSON-out functions with docstrings written for *autonomous
agents* (LangChain, DSPy, MCP servers, custom-GPT Actions), not just humans.
Each returns a plain ``dict`` that serializes cleanly to JSON, so a
tool-calling model can consume the result directly without post-processing.

Framework wrappers (LangChain ``@tool``, a DSPy module, an MCP server, and an
OpenAPI/Actions manifest) live in ``examples/agentic/`` and all delegate to the
functions here -- this module is the single source of truth for the tool
contract.

The design principle: the goal is dynamic
------------------------------------------
"Keep the angle" says the coordinate worth preserving is whichever one carries
the *task's* geometry -- and which coordinate that is depends on the goal. So
the goal is a **runtime input, not a library constant**: the agent declares it
per call (which ``consumer`` metric, which ``k``, which ``min_recall``), and the
library reports the best compression *for that goal*, certified *against that
goal*. This is exactly why reconstruction cosine can never be the acceptance
gate -- cosine is a fixed, goal-agnostic reconstruction proxy, blind to whatever
ranking the current task actually consumes.

Concretely: **acceptance is the declared goal's retrieval fidelity, never
cosine.** The gate is ``recall@k`` (does the compressed index return the same
neighbours?), the consumer-metric (A2) probe (:func:`recommend_kv_key_quantizer`),
or the distribution-free *rank certificate* (:func:`certify_ranking`). Cosine is
reported only as a labelled diagnostic: a high round-trip cosine with a
collapsed ranking is the exact failure these tools exist to prevent, so a tool
must never accept a configuration on cosine alone. The flagship entry point,
:func:`best_compression_at_recall`, is the "best ratio at a target recall"
hook -- point an agent at it when a user asks "how much can I compress this
corpus while keeping 0.99 recall?".
"""

from __future__ import annotations

import numpy as np

from .a2_probe import recommend_key_quantizer
from .auto_compress import auto_compress
from .rank_certificate import certificate_from_embeddings

__all__ = [
    "best_compression_at_recall",
    "certify_ranking",
    "recommend_kv_key_quantizer",
    "list_tools",
]


def _f(x: object) -> float | None:
    """Coerce a numpy/python scalar to a JSON-safe float (or ``None``)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def best_compression_at_recall(
    embeddings: np.ndarray,
    k: int = 10,
    min_recall: float = 0.99,
    sample_size: int = 200,
    bit_widths: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """Find the highest compression ratio that keeps ``recall@k >= min_recall``.

    This is the tool to call for "how small can I make this corpus while still
    returning the same neighbours?". It sweeps PCA dimensions, bit widths, and
    uniform vs. eigenvalue-weighted quantization, and accepts a configuration
    **only on true recall@k** (exact vs. reconstructed nearest-neighbour
    rankings) -- not on reconstruction cosine, which can read high while the
    ranking collapses.

    Args:
        embeddings: 2-D float array ``(n, dim)`` -- the corpus to compress.
        k: Retrieval cut-off for recall (top-``k`` neighbours). Default 10.
        min_recall: Acceptance threshold in ``[0, 1]``. Default 0.99.
        sample_size: Random subsample used to measure quality (speed knob).
        bit_widths: Bit widths to try; ``None`` sweeps ``[2, 3, 4]``.
        seed: Determinism seed.

    Returns:
        JSON-serializable dict::

            {
              "tool": "best_compression_at_recall",
              "target": "recall@10 >= 0.99",
              "meets_target": true,
              "achieved_recall_at_k": 0.991,
              "recall_k": 10,
              "compression_ratio": 21.3,        # x smaller than float32
              "config": {"label": "...", "bits": 3, "pca_dim": 384,
                         "weighted": false},
              "cosine_diagnostic": 0.982,       # reported, NOT accepted on
              "reproduce_cli": "tqp plan embeddings --target 'recall@10>=0.99'",
              "note": "Accepted on recall@k; cosine is a diagnostic only."
            }

        When no configuration reaches ``min_recall``, ``meets_target`` is
        ``false`` and the fields describe the highest-recall candidate found --
        the honest signal that this corpus needs exact reranking, not a config
        silently accepted on cosine.

    Example:
        >>> import numpy as np
        >>> from turboquant_pro import best_compression_at_recall
        >>> corpus = np.random.default_rng(0).standard_normal((5000, 768))
        >>> out = best_compression_at_recall(corpus, k=10, min_recall=0.99)
        >>> out["compression_ratio"], out["meets_target"]  # doctest: +SKIP
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (n, dim); got shape {embeddings.shape}"
        )

    target = f"recall@{k} >= {min_recall:g}"
    result = auto_compress(
        embeddings,
        target=target,
        sample_size=sample_size,
        bit_widths=bit_widths,
        seed=seed,
        verbose=False,
    )
    cfg = result.config
    achieved = _f(cfg.get(f"recall@{k}"))
    meets = achieved is not None and achieved >= min_recall

    return {
        "tool": "best_compression_at_recall",
        "target": target,
        "meets_target": bool(meets),
        "achieved_recall_at_k": achieved,
        "recall_k": int(k),
        "compression_ratio": _f(result.ratio),
        "config": {
            "label": cfg.get("label"),
            "bits": cfg.get("bits"),
            "pca_dim": cfg.get("pca_dim"),
            "weighted": bool(cfg.get("weighted", False)),
        },
        "cosine_diagnostic": _f(result.mean_cosine),
        "reproduce_cli": f"tqp plan embeddings --target '{target}'",
        "note": (
            "Accepted on recall@k (retrieval fidelity); cosine is reported as a "
            "labelled diagnostic only, never the acceptance signal. If "
            "meets_target is false, the corpus needs exact reranking at this k."
        ),
    }


def certify_ranking(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metric: str = "cosine",
    n_anchors: int = 200,
    seed: int = 0,
) -> dict:
    """Emit a distribution-free floor on rank agreement (the "certificate").

    Given the original vectors and their compressed-then-reconstructed
    counterparts, returns a *guaranteed* lower bound on Kendall-tau and
    Spearman rank correlation between exact and compressed rankings -- a
    worst-case certificate that holds without distributional assumptions
    (robust-distortion + Daniels' inequality). Use this as the mathematical
    receipt an agent prints after choosing a configuration.

    Args:
        original: Reference vectors ``(n, dim)``.
        reconstructed: Decompressed vectors ``(n, dim)`` (same shape).
        metric: Ranking metric to certify ("cosine" or "l2").
        n_anchors: Anchor points used to estimate distortion.
        seed: Determinism seed.

    Returns:
        JSON-serializable dict::

            {
              "tool": "certify_ranking",
              "metric": "cosine",
              "kappa": 1.4,                 # measured robust distortion
              "spearman_floor": 0.97,       # GUARANTEED Spearman rho
              "tau_floor": 0.95,            # GUARANTEED Kendall tau
              "mu_hat": 0.01,
              "n_pairs": 199000,
              "max_certifiable_kappa": 2.0,
              "vacuous": false,             # true => exact reranking required
              "verdict": "certified: Spearman >= 0.97"
            }

    Example:
        >>> from turboquant_pro import certify_ranking
        >>> cert = certify_ranking(orig, recon, metric="cosine")  # doctest: +SKIP
        >>> cert["spearman_floor"]  # doctest: +SKIP
    """
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if original.shape != reconstructed.shape:
        raise ValueError(
            "original and reconstructed must have the same shape; got "
            f"{original.shape} vs {reconstructed.shape}"
        )

    cert = certificate_from_embeddings(
        original,
        reconstructed,
        n_anchors=n_anchors,
        metric=metric,
        seed=seed,
    )
    vacuous = bool(getattr(cert, "vacuous", False))
    spearman_floor = _f(cert.spearman_floor)
    if vacuous or spearman_floor is None:
        verdict = "vacuous: no finite distortion certifies rank -- rerank exactly"
    else:
        verdict = f"certified: Spearman >= {spearman_floor:.3f}"

    return {
        "tool": "certify_ranking",
        "metric": metric,
        "kappa": _f(cert.kappa),
        "spearman_floor": spearman_floor,
        "tau_floor": _f(cert.tau_floor),
        "mu_hat": _f(cert.mu_hat),
        "n_pairs": int(cert.n_pairs),
        "max_certifiable_kappa": _f(cert.max_certifiable_kappa),
        "vacuous": vacuous,
        "verdict": verdict,
    }


def recommend_kv_key_quantizer(
    keys: np.ndarray,
    queries: np.ndarray | None = None,
    consumer: str = "attention_logits",
    bits: int = 4,
    seed: int = 0,
) -> dict:
    """Recommend the KV-key quantizer family via the (A2) consumer probe.

    Attention keys often carry large shared per-channel offsets (post-RoPE),
    where the polar (norm + direction) quotient's *reconstruction* stays high
    while its *ranking* of attention logits collapses. This tool runs the (A2)
    probe -- comparing rank agreement of the declared consumer metric under the
    polar vs. per-channel quotients -- and recommends the family whose
    preserved component actually tracks the consumer. It is the calibration
    check that would have caught the v1.2.0 keys regression.

    Args:
        keys: Attention-key sample ``(n, dim)``.
        queries: Optional query sample ``(m, dim)``; defaults to key rows.
        consumer: Downstream metric -- "attention_logits" (keys), "cosine",
            or "l2".
        bits: Probe bit budget per dimension. Default 4.
        seed: Determinism seed.

    Returns:
        JSON-serializable dict with the probe verdict, e.g.::

            {
              "tool": "recommend_kv_key_quantizer",
              "recommendation": "per_channel",
              "consumer": "attention_logits",
              "spearman_polar": 0.61,
              "spearman_per_channel": 0.98,
              "median_tangential_fraction": 0.99,
              "margin": 0.37,
              "verdict": "use per_channel: polar ranking collapses on these keys"
            }

    Example:
        >>> from turboquant_pro import recommend_kv_key_quantizer
        >>> rec = recommend_kv_key_quantizer(keys)  # doctest: +SKIP
        >>> rec["recommendation"]  # doctest: +SKIP
    """
    keys = np.asarray(keys)
    res = recommend_key_quantizer(
        keys,
        queries=queries,
        bits=bits,
        seed=seed,
    )
    d = res.as_dict()
    d["consumer"] = consumer
    d["tool"] = "recommend_kv_key_quantizer"
    d["verdict"] = (
        f"use {res.recommendation}: preserves the {consumer} ranking best "
        f"(margin {res.margin:.3f} Spearman)"
    )
    return d


def list_tools() -> list[dict]:
    """Return machine-readable schemas for the agent tools (for discovery).

    Each entry is ``{"name", "description", "parameters"}`` with a JSON-Schema
    ``parameters`` block, suitable for registering with an LLM tool-calling API,
    building an MCP tool list, or generating a custom-GPT Actions manifest.
    Kept in sync with the functions above; ``examples/agentic/tool_manifest.json``
    is generated from this.
    """
    corpus = {
        "type": "array",
        "description": "2-D array of embeddings, shape (n, dim).",
    }
    return [
        {
            "name": "best_compression_at_recall",
            "description": (
                "Highest compression ratio for a corpus that still keeps "
                "recall@k above a threshold. Accepts on recall, not cosine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "embeddings": corpus,
                    "k": {"type": "integer", "default": 10},
                    "min_recall": {"type": "number", "default": 0.99},
                    "sample_size": {"type": "integer", "default": 200},
                },
                "required": ["embeddings"],
            },
        },
        {
            "name": "certify_ranking",
            "description": (
                "Distribution-free guaranteed floor on Spearman/Kendall rank "
                "agreement between exact and compressed rankings (the "
                "mathematical certificate)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "original": corpus,
                    "reconstructed": corpus,
                    "metric": {"type": "string", "default": "cosine"},
                    "n_anchors": {"type": "integer", "default": 200},
                },
                "required": ["original", "reconstructed"],
            },
        },
        {
            "name": "recommend_kv_key_quantizer",
            "description": (
                "Run the (A2) probe to recommend polar vs. per-channel "
                "quantization for LLM attention keys, by consumer metric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": corpus,
                    "consumer": {"type": "string", "default": "attention_logits"},
                    "bits": {"type": "integer", "default": 4},
                },
                "required": ["keys"],
            },
        },
    ]
