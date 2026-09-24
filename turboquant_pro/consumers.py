# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Consumer metrics: the quantity a compression decision is allowed to turn on.

:mod:`turboquant_pro.plugins` plugs in the codecs being judged.
:mod:`turboquant_pro.read_operators` plugs in the operator ``P_C`` they are
judged against. This module plugs in the third side of the same contract: the
**measurement** that stands in for the consumer, item by item, so that a
planner can rank codecs on what the downstream computation does with the
vectors rather than on how close the bytes look.

The coherence rule of this package applies here with no exception:
reconstruction cosine is never an acceptance metric. It is available through
:func:`nominal_per_item` for one purpose only, which is to measure how often
it would have cleared a result the consumer rejects
(:mod:`turboquant_pro.false_clear`).

Three consumer families ship in tree, each registered through the public
interface so that an out-of-tree package registers the same way:

``topk_inner_product`` / ``topk_cosine`` / ``topk_l2``
    Retrieval. Per item is per query: the fraction of that query's exact
    top-k that survives a search over the compressed corpus. Higher is
    better.

``attention_softmax`` / ``attention_topk``
    Attention keys. Per item is per query: rank agreement of the attention
    logits over the keys (Spearman), the softmax KL, or the top-k attended
    set. This is the consumer whose failure the package documents in
    ``docs/KV_KEYS_FINDING.md``, where cosine read 0.995 and perplexity went
    to about 1e4.

``read_operator``
    Any consumer with a known or recovered ``P_C``. Per item is the
    consumer-felt squared error ``delta_i^T P_C delta_i``, whose mean is
    ``tr(P_C Sigma_delta)``. Lower is better. This is the only family that
    needs no query sample, and the only one that is exact rather than
    sampled.

A caller-supplied metric enters through ``declared``.

**What a consumer metric is not.** It is not a guarantee. A metric measured on
a sample carries the sample's uncertainty, which is why the planner quotes a
bootstrap interval and verifies on held-out data rather than reporting the
mean alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .metrics import exact_scores
from .plugins import (
    TARGET_EMBEDDING,
    TARGET_KV_KEY,
    TARGET_KV_VALUE,
    TARGET_WEIGHT,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ENTRY_POINT_GROUP",
    "AttentionScoreConsumer",
    "ConsumerMetric",
    "ConsumerSpec",
    "DeclaredConsumer",
    "ReadOperatorConsumer",
    "TopKRetrievalConsumer",
    "available_consumers",
    "create_consumer",
    "get_consumer",
    "load_entry_point_consumers",
    "nominal_per_item",
    "register_consumer",
    "row_cosine",
]

ENTRY_POINT_GROUP = "turboquant_pro.consumers"


@runtime_checkable
class ConsumerMetric(Protocol):
    """Original and reconstructed in, one score per consumed item out.

    ``name`` identifies the metric in a plan record. ``higher_is_better``
    fixes the direction once, so no caller has to remember whether a
    particular number is a similarity or a distance.

    ``per_item`` returns a 1-D array with one entry per *item the consumer
    reads*, which is a query for retrieval and attention and a row for an
    operator-based metric. Per item rather than aggregate, because the
    interval and the false-clear rate both need the spread.

    A provider must ignore context keywords it does not understand rather
    than fail on them, so a caller can pass a superset.
    """

    name: str
    higher_is_better: bool

    def per_item(
        self, original: np.ndarray, reconstructed: np.ndarray, **context: Any
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ConsumerSpec:
    """A named consumer-metric factory, discoverable like a quantizer plugin.

    ``exact`` says whether the metric *is* the consumer's own computation
    (an attention softmax, an exact top-k) or a proxy standing in for it.
    ``evidence_kind`` is the strongest kind of claim a measurement of this
    metric can support, in the vocabulary of ``docs/DESIGN_planner.md``
    section 2.6: ``"certificate"``, ``"statistical"`` or ``"analytic"``.
    """

    name: str
    factory: Callable[..., Any]
    targets: frozenset
    description: str = ""
    exact: bool = False
    evidence_kind: str = "statistical"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError(f"invalid consumer name: {self.name!r}")
        valid = {TARGET_KV_KEY, TARGET_KV_VALUE, TARGET_WEIGHT, TARGET_EMBEDDING}
        bad = set(self.targets) - valid
        if bad:
            raise ValueError(f"unknown targets {sorted(bad)}; valid: {sorted(valid)}")
        if self.evidence_kind not in {"certificate", "statistical", "analytic"}:
            raise ValueError(f"unknown evidence_kind {self.evidence_kind!r}")


_REGISTRY: dict = {}
_ENTRY_POINTS_LOADED = False


def register_consumer(spec: ConsumerSpec, *, overwrite: bool = False) -> ConsumerSpec:
    """Register a consumer metric. Raises on collision unless ``overwrite``."""
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"consumer {spec.name!r} already registered "
            f"(pass overwrite=True to replace)"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get_consumer(name: str) -> ConsumerSpec:
    """Look up a consumer metric by name (loads entry points on first miss)."""
    if name not in _REGISTRY:
        load_entry_point_consumers()
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"no consumer metric {name!r}; available: {known}. A planner asked "
            "to judge an unregistered consumer abstains rather than guessing."
        ) from None


def create_consumer(name: str, /, **config: Any) -> Any:
    """Instantiate a registered consumer metric.

    ``name`` is positional-only: a consumer's own configuration may contain a
    key called ``name`` (``declared`` does, for the metric's label) and a
    keyword parameter here would swallow it.
    """
    c = get_consumer(name).factory(**config)
    for attr in ("name", "higher_is_better", "per_item"):
        if not hasattr(c, attr):
            raise TypeError(
                f"consumer {name!r} factory returned {type(c).__name__}, which "
                f"has no {attr!r}"
            )
    return c


def available_consumers(*, target: str | None = None) -> dict:
    """All registered consumer metrics, optionally filtered by target."""
    load_entry_point_consumers()
    specs = dict(sorted(_REGISTRY.items()))
    if target is not None:
        specs = {n: s for n, s in specs.items() if target in s.targets}
    return specs


def load_entry_point_consumers(*, force: bool = False) -> list:
    """Discover out-of-tree consumer metrics via the entry-point group.

    Same contract as :func:`turboquant_pro.plugins.load_entry_point_plugins`:
    an entry point may resolve to a spec, an iterable of specs, or a callable
    returning either, and one broken package is logged and skipped rather than
    taking the registry down.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED and not force:
        return []
    _ENTRY_POINTS_LOADED = True
    new = []
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - importlib.metadata quirks
        logger.exception("consumer entry-point discovery failed")
        return new
    for ep in eps:
        try:
            obj = ep.load()
            if callable(obj) and not isinstance(obj, ConsumerSpec):
                obj = obj()
            specs = [obj] if isinstance(obj, ConsumerSpec) else list(obj)
            for spec in specs:
                try:
                    register_consumer(spec, overwrite=False)
                    new.append(spec.name)
                except ValueError:
                    logger.debug(
                        "entry point %r: consumer %r already registered",
                        ep.name,
                        spec.name,
                    )
        except Exception:
            logger.exception("skipping broken consumer entry point %r", ep.name)
    return new


# ------------------------------------------------------------------ #
# Shared numerics                                                     #
# ------------------------------------------------------------------ #


def _as_2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a.reshape(-1, a.shape[-1])


def row_cosine(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity. A diagnostic, never an acceptance signal."""
    a = _as_2d(original)
    b = _as_2d(reconstructed)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-30)


_scores = exact_scores  # the one definition, in turboquant_pro.metrics


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def _spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Spearman rho between two score matrices of equal shape."""

    def rank(m):
        order = np.argsort(m, axis=1)
        r = np.empty_like(order, dtype=np.float64)
        n = m.shape[1]
        rows = np.arange(m.shape[0])[:, None]
        r[rows, order] = np.arange(n, dtype=np.float64)[None, :]
        return r

    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    return num / np.maximum(den, 1e-30)


def _split_queries(x: np.ndarray, n_queries: int, seed: int):
    """Hold out ``n_queries`` rows as queries, the rest as corpus.

    Used only when the caller gives no query sample. The planner records that
    the corpus stood in for the query distribution, because it is an
    assumption and the consumer-basis work suggests it can be wrong.
    """
    n = x.shape[0]
    n_q = int(min(max(n_queries, 1), max(n - 1, 1)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[:n_q], perm[n_q:]


# ------------------------------------------------------------------ #
# Retrieval                                                           #
# ------------------------------------------------------------------ #


class TopKRetrievalConsumer:
    """Recall@k of a search over the compressed corpus, one score per query.

    The queries are read in full precision and the corpus is compressed,
    which is the asymmetric arrangement every index in this package uses.
    The reference is exact search over the *original* corpus, so the score
    measures what compression cost, not what the index approximated.

    Context keywords: ``queries`` (an ``(n_q, D)`` array). Without them the
    consumer holds out ``n_queries`` corpus rows and says so. ``searcher``
    (supplied by the planner when a codec has its own search) returns the
    codec's top-``n`` candidate ids for a query batch; without it candidates
    are ranked by the inner product with the decompressed rows.

    ``rerank`` is the rescoring depth: with ``rerank=r`` the top ``k * r``
    candidates are rescored exactly against the original rows and the top
    ``k`` of those are scored, the protocol of the RaBitQ campaign's ``rr5``
    (``r = 5``). ``rerank=1`` scores the compressed ranking alone.
    """

    def __init__(
        self,
        k: int = 10,
        metric: str = "inner_product",
        n_queries: int = 200,
        seed: int = 0,
        rerank: int = 1,
    ):
        self.k = int(k)
        self.metric = str(metric)
        self.n_queries = int(n_queries)
        self.seed = int(seed)
        self.rerank = int(rerank)
        if self.rerank < 1:
            raise ValueError("rerank must be at least 1")
        suffix = f"+rerank{self.rerank}" if self.rerank > 1 else ""
        self.name = f"recall@{self.k}/{self.metric}{suffix}"
        self.higher_is_better = True
        self.corpus_as_queries = False

    def _prepare(self, original, reconstructed, context):
        corpus = _as_2d(original)
        recon = _as_2d(reconstructed)
        if corpus.shape != recon.shape:
            raise ValueError(f"shape mismatch: {corpus.shape} vs {recon.shape}")
        q = context.get("queries")
        if q is None:
            q_idx, c_idx = _split_queries(corpus, self.n_queries, self.seed)
            self.corpus_as_queries = True
            return corpus[q_idx], corpus[c_idx], recon[c_idx]
        self.corpus_as_queries = False
        queries = _as_2d(q)
        if queries.shape[1] != corpus.shape[1]:
            raise ValueError(
                f"queries are {queries.shape[1]}-dimensional but the corpus is "
                f"{corpus.shape[1]}-dimensional"
            )
        return queries[: self.n_queries], corpus, recon

    def per_item(self, original, reconstructed, **context) -> np.ndarray:
        queries, corpus, recon = self._prepare(original, reconstructed, context)
        k = min(self.k, corpus.shape[0])
        exact = _topk(_scores(queries, corpus, self.metric), k)
        depth = min(k * self.rerank, corpus.shape[0])
        searcher = context.get("searcher")
        if searcher is not None and not self.corpus_as_queries:
            cand = np.asarray(searcher(queries, depth))
        else:
            cand = _topk(_scores(queries, recon, self.metric), depth)
        if self.rerank > 1:
            rescored = np.take_along_axis(
                _scores(queries, corpus, self.metric), cand, axis=1
            )
            order = np.argsort(-rescored, axis=1, kind="stable")[:, :k]
            approx = np.take_along_axis(cand, order, axis=1)
        else:
            approx = cand[:, :k]
        hits = np.empty(queries.shape[0], dtype=np.float64)
        for i in range(queries.shape[0]):
            hits[i] = len(set(exact[i].tolist()) & set(approx[i].tolist())) / float(k)
        self._last_exact = exact
        return hits

    def nominal_per_item(self, original, reconstructed, **context) -> np.ndarray:
        """Cosine of the rows this query actually reads, aligned per query.

        The cheap metric has to be scored on the same items as the consumer
        or the false-clear rate compares two different populations.
        """
        queries, corpus, recon = self._prepare(original, reconstructed, context)
        k = min(self.k, corpus.shape[0])
        exact = _topk(_scores(queries, corpus, self.metric), k)
        cos = row_cosine(corpus, recon)
        return cos[exact].mean(axis=1)


# ------------------------------------------------------------------ #
# Attention keys                                                      #
# ------------------------------------------------------------------ #


class AttentionScoreConsumer:
    """What a softmax attention head does with compressed keys, per query.

    ``mode`` selects the reading:

    ``spearman``
        Rank agreement of the logits ``q . k / sqrt(d)`` over all keys.
        Higher is better. The default, because attention is a soft ranking
        and this is the reading that collapsed while cosine held.
    ``kl``
        ``KL(exact softmax || compressed softmax)`` per query. Lower is
        better. The reading closest to what the next layer receives.
    ``topk``
        Fraction of the exact top-k attended keys still in the compressed
        top-k. Higher is better, and comparable to a retrieval recall.

    Context keywords: ``queries`` (required; a head's queries are not
    recoverable from its keys), ``scale`` (defaults to ``1/sqrt(d)``).
    """

    def __init__(self, mode: str = "spearman", k: int = 8, n_queries: int = 256):
        if mode not in {"spearman", "kl", "topk"}:
            raise ValueError(f"unknown attention mode {mode!r}")
        self.mode = mode
        self.k = int(k)
        self.n_queries = int(n_queries)
        self.name = {
            "spearman": "attention_logit_spearman",
            "kl": "attention_softmax_kl",
            "topk": f"attention_top{k}_recall",
        }[mode]
        self.higher_is_better = mode != "kl"

    def per_item(self, original, reconstructed, **context) -> np.ndarray:
        keys = _as_2d(original)
        recon = _as_2d(reconstructed)
        if keys.shape != recon.shape:
            raise ValueError(f"shape mismatch: {keys.shape} vs {recon.shape}")
        q = context.get("queries")
        if q is None:
            raise ValueError(
                "attention consumers need the head's queries; pass "
                "queries=<(n_q, D) array>. A head's read direction is not "
                "recoverable from its keys."
            )
        queries = _as_2d(q)[: self.n_queries]
        if queries.shape[1] != keys.shape[1]:
            raise ValueError(
                f"queries are {queries.shape[1]}-dimensional but keys are "
                f"{keys.shape[1]}-dimensional"
            )
        scale = float(context.get("scale") or 1.0 / np.sqrt(keys.shape[1]))
        z_exact = (queries @ keys.T) * scale
        z_approx = (queries @ recon.T) * scale
        if self.mode == "spearman":
            return _spearman_rows(z_exact, z_approx)
        if self.mode == "topk":
            k = min(self.k, keys.shape[0])
            ex = _topk(z_exact, k)
            ap = _topk(z_approx, k)
            return np.array(
                [
                    len(set(ex[i].tolist()) & set(ap[i].tolist())) / float(k)
                    for i in range(ex.shape[0])
                ],
                dtype=np.float64,
            )
        p = _softmax(z_exact)
        q_ = _softmax(z_approx)
        return (p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(q_, 1e-30)))).sum(
            axis=1
        )

    def nominal_per_item(self, original, reconstructed, **context) -> np.ndarray:
        """Cosine of the keys this query attends to most, aligned per query."""
        keys = _as_2d(original)
        recon = _as_2d(reconstructed)
        q = context.get("queries")
        if q is None:
            raise ValueError("attention consumers need queries=")
        queries = _as_2d(q)[: self.n_queries]
        scale = float(context.get("scale") or 1.0 / np.sqrt(keys.shape[1]))
        k = min(self.k, keys.shape[0])
        attended = _topk((queries @ keys.T) * scale, k)
        cos = row_cosine(keys, recon)
        return cos[attended].mean(axis=1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-30)


# ------------------------------------------------------------------ #
# Read-operator distortion                                            #
# ------------------------------------------------------------------ #


class ReadOperatorConsumer:
    """``delta_i^T P_C delta_i`` per item; its mean is ``tr(P_C Sigma_delta)``.

    The only family here that needs no query sample and the only one whose
    value is exact for a given ``P_C`` rather than sampled from a query
    distribution. It is therefore the right consumer when the read operator
    is known analytically, and the wrong one when ``P_C`` itself had to be
    guessed.

    ``operator`` names a provider in
    :mod:`turboquant_pro.read_operators`; extra keywords configure it.
    Errors are mean-centred first, matching
    :func:`turboquant_pro.read_operators.error_covariance`, because a shared
    offset is a different failure from a spread.
    """

    def __init__(self, operator: str = "identity", **operator_config: Any):
        from .read_operators import create_read_operator, get_read_operator

        self.operator_name = str(operator)
        self._spec = get_read_operator(self.operator_name)
        self._provider = create_read_operator(self.operator_name, **operator_config)
        self.name = f"consumer_distortion/{self.operator_name}"
        self.higher_is_better = False
        self.exact = bool(getattr(self._spec, "exact", False))

    def per_item(self, original, reconstructed, **context) -> np.ndarray:
        a = _as_2d(original)
        b = _as_2d(reconstructed)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
        P = np.asarray(self._provider.operator(a, **context), dtype=np.float64)
        delta = b - a
        delta = delta - delta.mean(axis=0, keepdims=True)
        return np.einsum("ij,jk,ik->i", delta, P, delta)

    def nominal_per_item(self, original, reconstructed, **context) -> np.ndarray:
        return row_cosine(original, reconstructed)


class DeclaredConsumer:
    """A caller-supplied per-item metric, given a name and a direction.

    The escape hatch that keeps the contract honest: a consumer this package
    has never seen is declared rather than approximated by one it has.
    """

    def __init__(
        self,
        fn: Callable[..., np.ndarray],
        name: str = "declared",
        higher_is_better: bool = True,
    ):
        if not callable(fn):
            raise TypeError("declared consumer needs a callable fn(original, recon)")
        self._fn = fn
        self.name = str(name)
        self.higher_is_better = bool(higher_is_better)

    def per_item(self, original, reconstructed, **context) -> np.ndarray:
        out = np.asarray(self._fn(original, reconstructed, **context), dtype=np.float64)
        return out.ravel()


def nominal_per_item(consumer: Any, original, reconstructed, **context):
    """The cheap metric, scored on the same items as ``consumer``.

    Returns ``None`` when the two populations cannot be aligned, which is the
    conservative answer: an unaligned false-clear rate compares a cosine over
    rows against a consumer outcome over queries and means nothing.
    """
    fn = getattr(consumer, "nominal_per_item", None)
    if fn is not None:
        try:
            return np.asarray(fn(original, reconstructed, **context), dtype=np.float64)
        except Exception:  # noqa: BLE001 - a diagnostic must never break a plan
            logger.debug("nominal metric unavailable for %s", consumer, exc_info=True)
            return None
    try:
        return row_cosine(original, reconstructed)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ #
# In-tree registrations, through the public interface (dogfood)       #
# ------------------------------------------------------------------ #

_RETRIEVAL_TARGETS = frozenset({TARGET_EMBEDDING})
_KEY_TARGETS = frozenset({TARGET_KV_KEY})
_ALL_TARGETS = frozenset(
    {TARGET_EMBEDDING, TARGET_KV_KEY, TARGET_KV_VALUE, TARGET_WEIGHT}
)

for _metric in ("inner_product", "cosine", "l2"):
    register_consumer(
        ConsumerSpec(
            name=f"topk_{_metric}",
            factory=(lambda m: lambda **cfg: TopKRetrievalConsumer(metric=m, **cfg))(
                _metric
            ),
            targets=_RETRIEVAL_TARGETS,
            description=f"recall@k of a search over compressed vectors ({_metric})",
            exact=True,
            evidence_kind="statistical",
            metadata={"consumer": "retrieval", "needs": "queries (optional)"},
        )
    )

register_consumer(
    ConsumerSpec(
        name="attention_softmax",
        factory=lambda **cfg: AttentionScoreConsumer(
            mode=cfg.pop("mode", "spearman"), **cfg
        ),
        targets=_KEY_TARGETS,
        description="rank agreement or KL of a softmax head's logits over keys",
        exact=True,
        evidence_kind="statistical",
        metadata={"consumer": "attention_softmax", "needs": "queries"},
    )
)

register_consumer(
    ConsumerSpec(
        name="attention_topk",
        factory=lambda **cfg: AttentionScoreConsumer(mode="topk", **cfg),
        targets=_KEY_TARGETS,
        description="fraction of the exact top-k attended keys still attended",
        exact=True,
        evidence_kind="statistical",
        metadata={"consumer": "attention_softmax", "needs": "queries"},
    )
)

register_consumer(
    ConsumerSpec(
        name="read_operator",
        factory=lambda **cfg: ReadOperatorConsumer(**cfg),
        targets=_ALL_TARGETS,
        description="tr(P_C Sigma_delta) per item, for a known read operator",
        exact=True,
        evidence_kind="analytic",
        metadata={"consumer": "declared operator", "needs": "operator="},
    )
)

register_consumer(
    ConsumerSpec(
        name="declared",
        factory=lambda **cfg: DeclaredConsumer(**cfg),
        targets=_ALL_TARGETS,
        description="a caller-supplied per-item metric",
        exact=False,
        evidence_kind="statistical",
        metadata={"consumer": "caller-defined", "needs": "fn="},
    )
)
