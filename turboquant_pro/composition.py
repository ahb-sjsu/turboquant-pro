# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Certify a pipeline, not a stage: composing rank certificates (issue #182).

The consumer of a retrieval system does not read the output of one codec. It
reads the end of a chain: an encoder, a compressed index, an approximate
search, a reranker. Each stage may carry its own rank certificate, and each
certificate is a statement about that stage alone. The application's question
is about the composition.

Two things compose, and this module is careful to distinguish them.

**The chain itself.** A certificate records the sha256 of the arrays it was
issued over. A chain is well formed when each stage's reconstructed hash is
the next stage's original hash. That check is exact and it is the one that
catches a pipeline assembled out of stages that were never connected, or a
stage silently re-run on different data. A chain that does not connect is
refused; nothing is composed over it.

**The distortion.** ``kappa`` is a ratio of distances, so bi-Lipschitz
constants multiply: a stage that moves every pairwise distance by at most a
factor ``k1`` followed by one that moves it by at most ``k2`` moves it by at
most ``k1 k2``. The chain's floor is then the corpus's own inversion at the
product, ``tau >= 1 - 2 mu_hat(product)``, evaluated on the *source*
distances, which is why a sample of the source is required rather than
optional.

**The caveat that travels with it.** `certify` measures kappa between the
2.5th and 97.5th percentiles by default, trimming the most distorted pairs.
Trimmed constants do not compose exactly: the product of two robust kappas
bounds the composition's robust kappa only under the assumption that the
trimmed pairs of the two stages are not adversarially disjoint. The strict
constants (``lo=0, hi=100``) do compose unconditionally. The report states
which kind it composed and refuses to call a robust composition
unconditional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ChainError", "CompositionReport", "compose"]


class ChainError(ValueError):
    """A chain that does not connect, or cannot be composed."""


def _hash(block: dict, which: str) -> str | None:
    return ((block.get("inputs") or {}).get(which) or {}).get("sha256")


@dataclass
class CompositionReport:
    stages: list
    metric: str
    product_kappa: float
    mu_hat: float
    tau_floor: float
    spearman_floor: float
    vacuous: bool
    unconditional: bool
    weakest: str
    n_pairs: int
    passed: bool
    interpretation: str
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema": "turboquant-pro/composition-certificate",
            "schema_version": 1,
            "stages": list(self.stages),
            "metric": self.metric,
            "product_kappa": self.product_kappa,
            "mu_hat": self.mu_hat,
            "tau_floor": self.tau_floor,
            "spearman_floor": self.spearman_floor,
            "vacuous": self.vacuous,
            "unconditional": self.unconditional,
            "weakest_stage": self.weakest,
            "n_pairs": self.n_pairs,
            "passed": self.passed,
            "interpretation": self.interpretation,
            **self.meta,
        }

    def explain(self) -> str:
        L = [
            "PIPELINE CERTIFICATE",
            f"  metric {self.metric}, {self.n_pairs} pairs",
            "",
        ]
        for i, st in enumerate(self.stages, 1):
            L.append(
                f"  {i}. {st['certificate']}  kappa {st['kappa']:.4f}"
                f"  tau floor {st['tau_floor']:.4f}"
            )
        L.append("")
        L.append("Chain connects: every stage output is the next stage input")
        L.append(
            f"Product kappa: {self.product_kappa:.4f}  (weakest stage: {self.weakest})"
        )
        L.append(
            f"Chain floor: Kendall tau >= {self.tau_floor:.4f}, "
            f"Spearman rho >= {self.spearman_floor:.4f}"
        )
        if not self.unconditional:
            L.append(
                "  conditional: the stages recorded robust (trimmed) kappas, which "
                "do not compose unconditionally"
            )
        L.append(f"=> {self.interpretation}")
        return "\n".join(L)


def compose(
    certificates: list,
    source: np.ndarray,
    *,
    metric: str | None = None,
    min_tau: float | None = None,
    unconditional: bool = False,
) -> CompositionReport:
    """Compose stage certificates into one statement about the chain.

    Args:
        certificates: ``(path, document)`` pairs, in pipeline order.
        source: a sample of the arrays the first stage was certified over;
            the chain's floor is the corpus's own inversion at the product
            kappa, so the source distances are needed, not optional.
        metric: the ranking metric; defaults to the stages' own, which must
            agree.
        min_tau: a floor to judge the chain against. Without one the chain
            passes when its floor is not vacuous.
        unconditional: declare that the stage kappas are strict (``lo=0,
            hi=100``) rather than the trimmed default, so the product is an
            unconditional bound.

    Raises:
        ChainError: the chain does not connect, a stage does not pass, the
            metrics disagree, or fewer than two stages are given.
    """
    from turboquant_pro.rank_certificate import mu_hat, pairwise_distances

    if len(certificates) < 2:
        raise ChainError("a composition needs at least two stages")

    metrics = set()
    stages = []
    for i, (path, doc) in enumerate(certificates):
        if doc.get("schema") != "turboquant-pro/rank-certificate":
            raise ChainError(f"stage {i + 1} ({path}) is not a rank certificate")
        if not doc.get("passed", False):
            raise ChainError(
                f"stage {i + 1} ({path}) does not pass on its own: "
                f"{doc.get('interpretation', 'no interpretation recorded')}"
            )
        cert = doc.get("certificate") or {}
        if "kappa" not in cert:
            raise ChainError(f"stage {i + 1} ({path}) records no kappa")
        metrics.add((doc.get("params") or {}).get("metric", "cosine"))
        stages.append(
            {
                "certificate": path,
                "kappa": float(cert["kappa"]),
                "tau_floor": float(cert.get("tau_floor", float("nan"))),
                # Certificates written before the tail penalty record none.
                "tail_fraction": float(cert.get("tail_fraction") or 0.0),
                "original_sha256": _hash(doc, "original"),
                "reconstructed_sha256": _hash(doc, "reconstructed"),
            }
        )

    if len(metrics) > 1:
        raise ChainError(
            "stages were certified under different metrics "
            f"({sorted(metrics)}); a composition over them is not defined"
        )
    chain_metric = metric or next(iter(metrics))

    for i in range(len(stages) - 1):
        out_, in_ = stages[i]["reconstructed_sha256"], stages[i + 1]["original_sha256"]
        if not out_ or not in_ or out_ != in_:
            raise ChainError(
                f"the chain does not connect at stage {i + 1} -> {i + 2}: "
                f"{stages[i]['certificate']} produced {out_}, "
                f"{stages[i + 1]['certificate']} was certified over {in_}"
            )

    src = np.asarray(source)
    first = stages[0]["original_sha256"]
    import hashlib

    if first and hashlib.sha256(np.ascontiguousarray(src)).hexdigest() != first:
        raise ChainError(
            "the source sample is not what the first stage was certified over "
            f"(it recorded {first})"
        )

    product = 1.0
    for st in stages:
        product *= st["kappa"]
    weakest = max(stages, key=lambda s: s["kappa"])["certificate"]

    exact = pairwise_distances(src, metric=chain_metric)
    mu = float(mu_hat(exact, product))
    # A pair of distances is controlled by the product kappa only if both
    # distances sit inside every stage's band; by the union bound the
    # uncontrolled fraction is at most the sum of the stages' tail fractions.
    tail = min(1.0, sum(st["tail_fraction"] for st in stages))
    tau = 1.0 - 2.0 * mu - 2.0 * tail
    rho = 1.0 - 3.0 * mu - 3.0 * tail if np.isfinite(mu) else float("nan")
    vacuous = not (tau > 0.0)

    if min_tau is not None:
        passed = bool(np.isfinite(tau) and tau >= min_tau)
        interp = (
            f"PASS: the chain certifies Kendall tau >= {tau:.4f}, at or above the "
            f"required {min_tau}"
            if passed
            else f"FAIL: the chain certifies only Kendall tau >= {tau:.4f}, below the "
            f"required {min_tau}; uncertified: validate the chain or rerank"
        )
    else:
        passed = not vacuous
        interp = (
            f"the chain certifies Kendall tau >= {tau:.4f}, Spearman rho >= {rho:.4f}"
            if passed
            else "VACUOUS: the composed distortion certifies no rank agreement on this "
            "corpus; uncertified: validate the chain or rerank"
        )
    if not unconditional:
        interp += " (conditional on the stages' trimmed kappas)"

    return CompositionReport(
        stages=stages,
        metric=chain_metric,
        product_kappa=product,
        mu_hat=mu,
        tau_floor=tau,
        spearman_floor=rho,
        vacuous=vacuous,
        unconditional=bool(unconditional),
        weakest=weakest,
        n_pairs=int(np.asarray(exact).size),
        passed=passed,
        interpretation=interp,
        meta={
            "min_tau": min_tau,
            "rule": (
                "bi-Lipschitz constants multiply, so the chain's distortion is at "
                "most the product of the stages'; the floor is the corpus's own "
                "inversion at that product"
            ),
        },
    )
