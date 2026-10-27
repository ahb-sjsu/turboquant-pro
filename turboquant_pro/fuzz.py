# TurboQuant Pro: geometry-aware retrieval fuzzer, MVP (issue #162)
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Coverage-guided search for queries where compression changes what a consumer sees.

The fuzzed object is a **query**; the corpus and the compressed index are fixed.
A campaign starts from the given queries, mutates them in the corpus's whitened
space (:mod:`turboquant_pro.geometry`), and for every candidate compares exact
retrieval with TurboQuant-Pro's compressed retrieval:

* **compressed path**: the real one, an :class:`~turboquant_pro.adc_index.ADCIndex`
  over PCA-Matryoshka + TurboQuant codes (``TQEmbeddingCodec``), searched with
  the numpy reference scorer by default so a case replays bit for bit on any
  machine (``scorer="fast"`` uses the AVX2 kernel where present; the bundle
  records which ran);
* **exact path**: :func:`turboquant_pro.metrics.exact_scores` over the fp32
  corpus, ties broken by row index.

Mutators (MVP): ``radial`` scales the whitened query, z' = alpha z, moving it
toward or away from the corpus centre with its direction kept; ``shell`` turns
it by a random angle at fixed whitened norm, so Mahalanobis centrality is held
exactly and only the angular position changes.

Coverage cell of a candidate: (Mahalanobis-radius quantile of the query, N_k
quantile of its exact nearest neighbour, exact top-k margin quantile, recall
band). A candidate is retained when it enters a new cell, or is more severe
(lower recall@k) than the cell's best so far, or brings a failure class the cell
has not had.

Classification, kept distinct on purpose (a retrieval discrepancy is not a
correctness failure):

* ``certified_violation``: the strict, distribution-free rank certificate
  (:mod:`turboquant_pro.rank_certificate`, ``lo=0, hi=100``) computed on this
  case's own candidate distances certifies a Kendall-tau floor, and the observed
  tau is below it. The floor is a theorem for those distances, so this class
  means a bug (in the certificate, the codec, or the metric plumbing), never a
  hard query;
* ``consumer_regression``: recall@k below the declared ``recall_floor``;
* ``stress_discovery``: retained for coverage or severity, neither of the above.

Every retained case is written as a checksummed bundle that
:func:`replay_bundle` re-evaluates; a bundle with a missing or altered file, an
unknown schema, or a corpus whose hash differs fails closed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .geometry import Whitener, _sha256_rows, geometry_profile

__all__ = [
    "CLASSES",
    "MUTATORS",
    "Target",
    "evaluate",
    "run_campaign",
    "replay_bundle",
    "BundleError",
]

CASE_SCHEMA = "turboquant-pro/fuzz-case"
CASE_SCHEMA_VERSION = 1
RUN_SCHEMA = "turboquant-pro/fuzz-run"
REPLAY_SCHEMA = "turboquant-pro/fuzz-replay"
CLASSES = ("certified_violation", "consumer_regression", "stress_discovery")
#: Replay tolerances, recorded in every case: ids and classes must match exactly.
TOLERANCES = {"ids": "exact", "classification": "exact", "float_abs": 1e-6}
#: Corpora up to this size are copied into each bundle so it replays on its own.
EMBED_CORPUS_MAX_BYTES = 4 * 1024 * 1024
REQUIRED_FILES = (
    "case.json",
    "query.npy",
    "expected_exact.json",
    "observed_tqp.json",
    "environment.json",
)


class BundleError(ValueError):
    """A bundle that cannot be trusted: missing, altered, or for another corpus."""


# --------------------------------------------------------------------------- target


class Target:
    """The fixed side of a campaign: corpus, compressed index, and both searches."""

    def __init__(
        self,
        corpus: np.ndarray,
        *,
        bits: int = 4,
        out_dim: int | None = None,
        codec_seed: int = 0,
        train_rows: int = 20000,
        metric: str = "cosine",
        scorer: str = "exact",
    ):
        from .adc_index import ADCIndex
        from .embedding_codecs import TQEmbeddingCodec
        from .metrics import check_metric

        check_metric(metric)
        if scorer not in ("exact", "fast"):
            raise ValueError(f"scorer must be 'exact' or 'fast'; got {scorer!r}")
        self.corpus = np.ascontiguousarray(corpus, dtype=np.float32)
        self.metric, self.scorer = metric, scorer
        codec = TQEmbeddingCodec(
            bits=bits, out_dim=out_dim, seed=codec_seed, train_rows=train_rows
        )
        cont = codec.compress(self.corpus)
        self.recon = np.ascontiguousarray(codec.decompress(cont), dtype=np.float32)
        self.index = ADCIndex(cont.shared["pipeline"], metric=metric).add(self.corpus)
        self.config = {
            "codec": "tq_embedding",
            "bits": int(bits),
            "out_dim": int(cont.shared["pipeline"].output_dim),
            "out_dim_requested": None if out_dim is None else int(out_dim),
            "codec_seed": int(codec_seed),
            "train_rows": int(train_rows),
            "metric": metric,
            "scorer": scorer,
        }
        self.corpus_sha256 = _sha256_rows(self.corpus)

    def _scores(self, q: np.ndarray, rows: np.ndarray) -> np.ndarray:
        from .metrics import exact_scores

        return exact_scores(q[None, :].astype(np.float32), rows, self.metric)[0].astype(
            np.float64
        )

    def exact_order(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self._scores(q, self.corpus)
        order = np.argsort(-s, kind="stable")  # ties by row index
        return order, s

    def recon_order(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self._scores(q, self.recon)
        return np.argsort(-s, kind="stable"), s

    def compressed_topk(
        self, q: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        idx, sc = self.index.search(
            q[None, :].astype(np.float32), k=k, mode=self.scorer
        )
        prov = getattr(self.index, "last_scorer", None) or {}
        return np.asarray(idx)[0], np.asarray(sc, dtype=np.float64)[0], dict(prov)

    def distances(self, scores: np.ndarray) -> np.ndarray | None:
        """Scores (larger is nearer) as certificate distances; None for IP."""
        if self.metric == "cosine":
            return 1.0 - scores
        if self.metric == "l2":
            return np.sqrt(np.maximum(-scores, 0.0))
        return None


# --------------------------------------------------------------------------- oracle


def _kendall_tau_strict(a: np.ndarray, b: np.ndarray) -> float:
    """1 - 2 * (strictly discordant pairs / all pairs), what the certificate floors."""
    n = len(a)
    if n < 2:
        return float("nan")
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(n, 1)
    disc = int(((da * db)[iu] < 0).sum())
    return 1.0 - 2.0 * disc / (n * (n - 1) / 2)


def evaluate(
    target: Target,
    q: np.ndarray,
    k: int,
    *,
    recall_floor: float = 0.9,
    pool: int | None = None,
) -> dict:
    """Exact versus compressed retrieval for one query: the case's oracle record."""
    from .rank_certificate import certify

    n = len(target.corpus)
    if not 0 < k < n:
        raise ValueError(f"k={k} must be in (0, {n})")
    pool = min(n, pool or max(5 * k, 50))
    q = np.asarray(q, dtype=np.float32)
    e_order, e_s = target.exact_order(q)
    r_order, r_s = target.recon_order(q)
    c_ids, c_scores, scorer = target.compressed_topk(q, k)
    e_k = e_order[:k]
    recall = len(set(e_k.tolist()) & set(c_ids.tolist())) / k
    first_bad = next((i + 1 for i in range(k) if c_ids[i] != e_k[i]), None)
    pos = np.empty(n, dtype=np.int64)
    pos[r_order] = np.arange(1, n + 1)
    rerank_depth = int(
        pos[e_k].max()
    )  # compressed-ranking depth that holds the whole exact top-k
    margin = float(e_s[e_order[k - 1]] - e_s[e_order[k]])
    cand = np.unique(np.concatenate([e_order[:pool], r_order[:pool]]))
    tau = _kendall_tau_strict(-e_s[cand], -r_s[cand])
    cert = {"applicable": False}
    de, dc = target.distances(e_s[cand]), target.distances(r_s[cand])
    if de is not None:
        c = certify(de, dc, lo=0.0, hi=100.0)
        floor = float(c.tau_floor)
        cert = {
            "applicable": True,
            "regime": "strict",
            "kappa": float(c.kappa),
            "mu_hat": float(c.mu_hat),
            "tau_floor": floor,
            "strength": (
                "vacuous" if c.vacuous else ("weak" if floor < 0.5 else "informative")
            ),
            "violated": bool(np.isfinite(floor) and floor > 0 and tau < floor - 1e-9),
        }
    if cert.get("violated"):
        cls = "certified_violation"
    elif recall < recall_floor:
        cls = "consumer_regression"
    else:
        cls = "stress_discovery"
    return {
        "k": int(k),
        "recall_floor": float(recall_floor),
        "candidate_pool": int(len(cand)),
        "recall_at_k": float(recall),
        "first_incorrect_rank": first_bad,
        "rerank_depth": rerank_depth,
        "kendall_tau_candidates": float(tau),
        "exact_margin": margin,
        "adc_matches_reconstruction": bool(np.array_equal(c_ids, r_order[:k])),
        "scorer": scorer,
        "certificate": cert,
        "classification": cls,
        "severity": float(1.0 - recall + 1e-3 * (1.0 - tau)),
        "exact_topk": [int(i) for i in e_k],
        "exact_topk_scores": [float(e_s[i]) for i in e_k],
        "compressed_topk": [int(i) for i in c_ids],
        "compressed_topk_scores": [float(s) for s in c_scores],
    }


# --------------------------------------------------------------------------- mutators


def _radial(
    z: np.ndarray, rng: np.random.Generator, alpha_range=(0.25, 4.0)
) -> tuple[np.ndarray, dict]:
    lo, hi = np.log(alpha_range[0]), np.log(alpha_range[1])
    alpha = float(np.exp(rng.uniform(lo, hi)))
    return alpha * z, {"alpha": alpha}


def _shell(
    z: np.ndarray, rng: np.random.Generator, max_angle=np.pi / 4
) -> tuple[np.ndarray, dict]:
    r = float(np.linalg.norm(z))
    g = rng.standard_normal(z.shape)
    if r == 0.0:
        return z.copy(), {"angle_rad": 0.0, "note": "zero radius: nothing to turn"}
    u = z / r
    g -= (g @ u) * u  # the tangent direction at u
    g /= np.linalg.norm(g)
    angle = float(rng.uniform(0.0, max_angle))
    return r * (np.cos(angle) * u + np.sin(angle) * g), {"angle_rad": angle}


MUTATORS = {"radial": _radial, "shell": _shell}


# --------------------------------------------------------------------------- coverage


def _band(x: float, edges) -> int:
    return int(np.searchsorted(np.asarray(edges), x, side="right"))


class Coverage:
    """Bins a candidate into its coverage cell against the corpus's distributions."""

    def __init__(
        self,
        whitener: Whitener,
        corpus: np.ndarray,
        counts: np.ndarray,
        ref_margins: np.ndarray,
    ):
        self.w = whitener
        self.radii = np.sort(whitener.radius(corpus))
        self.counts = counts
        self.counts_sorted = np.sort(counts)
        self.margins = np.sort(ref_margins)

    def inputs(self, q: np.ndarray, oracle: dict) -> dict:
        r = float(self.w.radius(q)[0])
        nn = oracle["exact_topk"][0]
        return {
            "mahalanobis_radius": r,
            "mahalanobis_quantile": float(
                np.searchsorted(self.radii, r, side="right") / len(self.radii)
            ),
            "nn_row": int(nn),
            "nn_n_k": float(self.counts[nn]),
            "hubness_quantile": float(
                np.searchsorted(self.counts_sorted, self.counts[nn], side="right")
                / len(self.counts)
            ),
            "exact_margin_quantile": float(
                np.searchsorted(self.margins, oracle["exact_margin"], side="right")
                / len(self.margins)
            ),
            "recall_at_k": oracle["recall_at_k"],
        }

    @staticmethod
    def signature(inp: dict) -> tuple:
        # Radius band 5 means beyond every corpus row (quantile 1.0).
        m = (
            5
            if inp["mahalanobis_quantile"] >= 1.0
            else _band(inp["mahalanobis_quantile"], [0.2, 0.4, 0.6, 0.8])
        )
        h = _band(inp["hubness_quantile"], [0.2, 0.4, 0.6, 0.8])
        g = _band(inp["exact_margin_quantile"], [0.2, 0.4, 0.6, 0.8])
        rc = inp["recall_at_k"]
        e = (
            0
            if rc >= 1.0
            else 1 if rc >= 0.9 else 2 if rc >= 0.7 else 3 if rc >= 0.5 else 4
        )
        return (m, h, g, e)


# --------------------------------------------------------------------------- bundles


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _finite(x):
    """JSON-safe copy: non-finite floats become null; every artifact is valid JSON."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _write_json(p: Path, doc: dict) -> None:
    text = json.dumps(_finite(doc), indent=1, sort_keys=True, allow_nan=False)
    p.write_text(text + "\n", encoding="utf-8")


def _environment(target: Target) -> dict:
    from turboquant_pro import __version__

    return {
        "tool_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "adc_kernel_available": bool(target.index.uses_kernel),
    }


def _write_bundle(
    case_dir: Path, case: dict, q: np.ndarray, oracle: dict, target: Target
) -> None:
    case_dir.mkdir(parents=True, exist_ok=False)
    np.save(case_dir / "query.npy", np.asarray(q, dtype=np.float32))
    if target.corpus.nbytes <= EMBED_CORPUS_MAX_BYTES:
        np.save(case_dir / "corpus.npy", target.corpus)
        case["corpus"]["embedded"] = True
    _write_json(case_dir / "case.json", case)
    _write_json(
        case_dir / "expected_exact.json",
        {"ids": oracle["exact_topk"], "scores": oracle["exact_topk_scores"]},
    )
    _write_json(
        case_dir / "observed_tqp.json",
        {"ids": oracle["compressed_topk"], "scores": oracle["compressed_topk_scores"]},
    )
    _write_json(case_dir / "environment.json", _environment(target))
    (case_dir / "replay.sh").write_text(
        "#!/bin/sh\n# Re-evaluate this case. Without corpus.npy here, pass\n"
        "# --embeddings <corpus.npy>.\n"
        'exec tqp fuzz replay "$(dirname "$0")" "$@"\n',
        encoding="utf-8",
    )
    names = sorted(p.name for p in case_dir.iterdir() if p.name != "checksums.txt")
    (case_dir / "checksums.txt").write_text(
        "".join(f"{_sha256_file(case_dir / n)}  {n}\n" for n in names), encoding="utf-8"
    )


def _read_bundle(case_dir: Path) -> dict:
    """Verify and load a bundle, or raise BundleError: it fails closed."""
    case_dir = Path(case_dir)
    ck = case_dir / "checksums.txt"
    if not ck.is_file():
        raise BundleError(f"{case_dir}: no checksums.txt")
    listed = {}
    for line in ck.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            raise BundleError(f"{ck}: malformed line {line!r}")
        listed[parts[1]] = parts[0]
    missing = [f for f in REQUIRED_FILES if f not in listed]
    if missing:
        raise BundleError(f"{case_dir}: checksums.txt does not cover {missing}")
    for name, digest in listed.items():
        p = case_dir / name
        if not p.is_file():
            raise BundleError(f"{case_dir}: {name} is missing")
        if _sha256_file(p) != digest:
            raise BundleError(f"{case_dir}: {name} does not match its checksum")
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if (
        case.get("schema") != CASE_SCHEMA
        or case.get("schema_version") != CASE_SCHEMA_VERSION
    ):
        raise BundleError(
            f"{case_dir}: unknown schema {case.get('schema')!r} "
            f"v{case.get('schema_version')!r}"
        )
    return {
        "case": case,
        "query": np.load(case_dir / "query.npy"),
        "corpus": np.load(case_dir / "corpus.npy") if "corpus.npy" in listed else None,
    }


# --------------------------------------------------------------------------- campaign


def run_campaign(
    corpus: np.ndarray,
    queries: np.ndarray,
    out_dir: str | Path,
    *,
    k: int = 10,
    budget: int = 1000,
    seed: int = 42,
    mutators=("radial", "shell"),
    recall_floor: float = 0.9,
    bits: int = 4,
    out_dim: int | None = None,
    codec_seed: int = 0,
    train_rows: int = 20000,
    metric: str = "cosine",
    scorer: str = "exact",
    shrinkage: float | str = "ledoit-wolf",
    eig_floor: float = 1e-6,
    profile_sample: int | None = None,
    corpus_path: str | None = None,
    log=None,
) -> dict:
    """Run one campaign; write ``run.json``, ``geometry.json`` and ``cases/``."""
    from turboquant_pro import __version__

    unknown = [m for m in mutators if m not in MUTATORS]
    if unknown or not mutators:
        raise ValueError(
            f"unknown or empty mutators {list(mutators)}; "
            f"registered: {sorted(MUTATORS)}"
        )
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} is not empty; a campaign never overwrites another's cases"
        )
    (out / "cases").mkdir(parents=True, exist_ok=True)
    corpus = np.ascontiguousarray(corpus, dtype=np.float32)
    queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
    if queries.shape[1] != corpus.shape[1]:
        raise ValueError(
            f"queries have dim {queries.shape[1]}, corpus {corpus.shape[1]}"
        )

    profile, w, counts = geometry_profile(
        corpus,
        k=k,
        sample=profile_sample,
        shrinkage=shrinkage,
        eig_floor=eig_floor,
        seed=seed,
    )
    _write_json(out / "geometry.json", profile)
    geometry_sha = _sha256_file(out / "geometry.json")
    target = Target(
        corpus,
        bits=bits,
        out_dim=out_dim,
        codec_seed=codec_seed,
        train_rows=train_rows,
        metric=metric,
        scorer=scorer,
    )

    seed_oracles = [evaluate(target, q, k, recall_floor=recall_floor) for q in queries]
    cov = Coverage(
        w, corpus, counts, np.array([o["exact_margin"] for o in seed_oracles])
    )
    rng = np.random.default_rng(seed)

    pool: list = []  # (query vector, case_id or None, chain)
    best: dict = {}  # cell -> (severity, set of classes)
    retained: list = []
    stats = {c: 0 for c in CLASSES}
    by_mutator = {m: {"tried": 0, "retained": 0} for m in mutators}
    worst_recall = [float("nan")]  # lowest recall@k over every evaluated candidate

    def consider(q, oracle, parent, chain, mseed):
        r = float(oracle["recall_at_k"])
        if not r >= worst_recall[0]:  # also replaces the initial NaN
            worst_recall[0] = r
        inp = cov.inputs(q, oracle)
        cell = Coverage.signature(inp)
        sev, cls = oracle["severity"], oracle["classification"]
        prev = best.get(cell)
        reason = None
        if prev is None:
            reason = "new_cell"
        elif sev > prev[0] + 1e-12:
            reason = "more_severe"
        elif cls not in prev[1]:
            reason = "new_class_in_cell"
        if prev is None:
            best[cell] = (sev, {cls})
        else:
            best[cell] = (max(prev[0], sev), prev[1] | {cls})
        if reason is None:
            return None
        case_id = f"case-{len(retained):06d}"
        case = {
            "schema": CASE_SCHEMA,
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": case_id,
            "retained_because": reason,
            "parent": parent,
            "campaign_seed": int(seed),
            "mutation_seed": mseed,
            "mutator_chain": chain,
            "coverage_signature": list(cell),
            "coverage_inputs": inp,
            "oracle": {
                kk: v
                for kk, v in oracle.items()
                if not kk.endswith(("topk", "topk_scores"))
            },
            "classification": cls,
            "k": int(k),
            "codec": target.config,
            "corpus": {
                "sha256": target.corpus_sha256,
                "shape": list(corpus.shape),
                "path": corpus_path,
                "embedded": False,
            },
            "whitener": w.metadata(),
            "geometry_sha256": geometry_sha,
            "tolerances": TOLERANCES,
            "tool_version": __version__,
        }
        _write_bundle(out / "cases" / case_id, case, q, oracle, target)
        retained.append(case_id)
        stats[cls] += 1
        return case_id

    for i, (q, o) in enumerate(zip(queries, seed_oracles)):
        cid = consider(q, o, None, [], None)
        pool.append((q, cid if cid else f"seed-query-{i}", []))

    for it in range(int(budget)):
        q0, parent, chain0 = pool[int(rng.integers(len(pool)))]
        name = mutators[int(rng.integers(len(mutators)))]
        mseed = [int(seed), int(it)]
        mrng = np.random.default_rng(mseed)
        z, params = MUTATORS[name](w.whiten(q0)[0], mrng)
        q = w.unwhiten(z)[0].astype(np.float32)
        chain = chain0 + [{"mutator": name, "params": params, "mutation_seed": mseed}]
        o = evaluate(target, q, k, recall_floor=recall_floor)
        by_mutator[name]["tried"] += 1
        cid = consider(q, o, parent, chain, mseed)
        if cid:
            by_mutator[name]["retained"] += 1
            pool.append((q, cid, chain))
        if log and (it + 1) % max(1, budget // 10) == 0:
            log(
                f"fuzz: {it + 1}/{budget} candidates, {len(retained)} retained, "
                f"{len(best)} cells"
            )

    run = {
        "schema": RUN_SCHEMA,
        "schema_version": 1,
        "tool_version": __version__,
        "seed": int(seed),
        "budget": int(budget),
        "k": int(k),
        "mutators": list(mutators),
        "recall_floor": float(recall_floor),
        "codec": target.config,
        "corpus": {
            "sha256": target.corpus_sha256,
            "shape": list(corpus.shape),
            "path": corpus_path,
        },
        "n_seed_queries": int(len(queries)),
        "geometry_sha256": geometry_sha,
        "whitener": w.metadata(),
        "cells_covered": len(best),
        "retained": len(retained),
        "by_class": stats,
        "by_mutator": by_mutator,
        "worst_recall_at_k": worst_recall[0],
        "environment": _environment(target),
    }
    _write_json(out / "run.json", run)
    return run


# --------------------------------------------------------------------------- replay


def replay_bundle(case_dir: str | Path, corpus: np.ndarray | None = None) -> dict:
    """Re-evaluate a bundle. Raises :class:`BundleError` when it cannot be trusted;
    otherwise returns ``{"reproduced": bool, "differences": [...], ...}``."""
    b = _read_bundle(Path(case_dir))
    case = b["case"]
    if b["corpus"] is not None:
        corpus = b["corpus"]
    if corpus is None:
        raise BundleError(
            f"{case_dir}: the corpus is not embedded; "
            "pass the corpus it was recorded on"
        )
    corpus = np.ascontiguousarray(corpus, dtype=np.float32)
    sha = _sha256_rows(corpus)
    if sha != case["corpus"]["sha256"]:
        raise BundleError(
            f"{case_dir}: corpus sha256 {sha[:12]}... is not the recorded "
            f"{case['corpus']['sha256'][:12]}..."
        )
    c = case["codec"]
    target = Target(
        corpus,
        bits=c["bits"],
        out_dim=c["out_dim_requested"],
        codec_seed=c["codec_seed"],
        train_rows=c["train_rows"],
        metric=c["metric"],
        scorer=c["scorer"],
    )
    rec = case["oracle"]
    now = evaluate(target, b["query"], case["k"], recall_floor=rec["recall_floor"])
    tol = float(case["tolerances"]["float_abs"])
    exp_e = json.loads(
        (Path(case_dir) / "expected_exact.json").read_text(encoding="utf-8")
    )
    exp_c = json.loads(
        (Path(case_dir) / "observed_tqp.json").read_text(encoding="utf-8")
    )
    diffs = []
    if now["exact_topk"] != exp_e["ids"]:
        diffs.append(
            {
                "field": "exact_topk",
                "recorded": exp_e["ids"],
                "replayed": now["exact_topk"],
            }
        )
    if now["compressed_topk"] != exp_c["ids"]:
        diffs.append(
            {
                "field": "compressed_topk",
                "recorded": exp_c["ids"],
                "replayed": now["compressed_topk"],
            }
        )
    if now["classification"] != case["classification"]:
        diffs.append(
            {
                "field": "classification",
                "recorded": case["classification"],
                "replayed": now["classification"],
            }
        )
    for f in (
        "recall_at_k",
        "kendall_tau_candidates",
        "exact_margin",
        "rerank_depth",
        "first_incorrect_rank",
    ):
        a, b2 = rec.get(f), now.get(f)
        if (a is None) != (b2 is None) or (
            a is not None and abs(float(a) - float(b2)) > tol
        ):
            diffs.append({"field": f, "recorded": a, "replayed": b2})
    return {
        "schema": REPLAY_SCHEMA,
        "schema_version": 1,
        "case_id": case["case_id"],
        "reproduced": not diffs,
        "differences": diffs,
        "tolerances": case["tolerances"],
        "recorded": {
            "recall_at_k": rec["recall_at_k"],
            "classification": case["classification"],
        },
        "replayed": {
            "recall_at_k": now["recall_at_k"],
            "classification": now["classification"],
        },
        "environment": _environment(target),
    }
