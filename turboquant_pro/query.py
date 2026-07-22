"""`tqp query` — a deliberately tiny SQL-ish interface over TQE indexes.

Three verbs, one lifecycle (the PostgreSQL analogy made literal):

  ANALYZE INDEX 'path' [USING QUERIES 'q.npy'] [WITH (SAMPLE=2000, ...)]
      -> writes the *workload statistics catalog* (<index>.stats.json):
         geometry descriptors (intrinsic dimension, spectrum, hubness + its
         anatomy) AND a measured recall/latency calibration sweep over the
         index's own rerank knob.

  EXPLAIN SELECT id, score FROM 'path' ORDER BY COSINE(:q) LIMIT 10
      WITH (RECALL >= 0.95)
      -> the planner reads the catalog and picks the cheapest operating point
         whose *measured* recall meets the target; prints plan + prediction.

  SELECT id, score FROM 'path' ORDER BY COSINE(:q) LIMIT 10
      WITH (RECALL >= 0.95[, CERTIFY])
      -> executes the plan against the index (queries bound via --queries);
         CERTIFY attaches the index's rank certificate to the result doc.

Honesty boundary (adopted from OpenVector Bench WP5-lite, results/WP5LITE_RESULT.md
in that repository): geometry descriptors do NOT yet predict recall-cost behaviour,
so the planner is strictly *calibration-based* — every prediction traces to a
measured sweep on this index; the geometry block is advisory context only.

The dialect is intentionally minimal: one statement shape, no WHERE (TQE rows
carry no payloads), no joins, no general SQL.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Parser                                                                       #
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<str>'[^']*'|"[^"]*")            # quoted string
      | (?P<num>\d+\.\d+|\.\d+|\d+)         # number
      | (?P<op>>=|<=|=|\(|\)|,)             # operators / punctuation
      | (?P<word>[A-Za-z_:][A-Za-z0-9_.:/\\-]*)  # keyword / ident / path
    )""",
    re.VERBOSE,
)

_SIMFNS = ("COSINE", "L2", "DOT")


class QuerySyntaxError(ValueError):
    """Raised with a caret-position message on any parse failure."""


def _tokenize(text: str) -> list[tuple[str, str, int]]:
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip():
                raise QuerySyntaxError(
                    f"cannot tokenize at position {pos}: {text[pos:pos+20]!r}"
                )
            break
        for kind in ("str", "num", "op", "word"):
            if m.group(kind) is not None:
                out.append((kind, m.group(kind), m.start(kind)))
                break
        pos = m.end()
    return out


@dataclass
class SelectStmt:
    index: str
    columns: list[str]
    simfn: str = "COSINE"
    limit: int = 10
    options: dict = field(default_factory=dict)  # RECALL/RERANK/CERTIFY/...
    explain: bool = False


@dataclass
class AnalyzeStmt:
    index: str
    queries: str | None = None
    options: dict = field(default_factory=dict)


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0

    def _peek(self):
        return (
            self.toks[self.i]
            if self.i < len(self.toks)
            else ("eof", "", len(self.text))
        )

    def _next(self):
        t = self._peek()
        self.i += 1
        return t

    def _expect_word(self, *words: str) -> str:
        kind, val, pos = self._next()
        if kind != "word" or val.upper() not in words:
            raise QuerySyntaxError(
                f"expected {' | '.join(words)} at position {pos}, got {val!r}"
            )
        return val.upper()

    def _expect_op(self, op: str):
        kind, val, pos = self._next()
        if kind != "op" or val != op:
            raise QuerySyntaxError(f"expected {op!r} at position {pos}, got {val!r}")

    def _path(self) -> str:
        kind, val, pos = self._next()
        if kind == "str":
            return val[1:-1]
        if kind == "word":
            return val
        raise QuerySyntaxError(f"expected an index path at position {pos}, got {val!r}")

    def _with_options(self) -> dict:
        """WITH ( NAME [>=|<=|= value] [, ...] ) — bare NAME means a flag."""
        opts: dict = {}
        if self._peek()[1].upper() != "WITH":
            return opts
        self._next()
        self._expect_op("(")
        while True:
            kind, name, pos = self._next()
            if kind != "word":
                raise QuerySyntaxError(
                    f"expected option name at position {pos}, got {name!r}"
                )
            key = name.upper()
            kind, val, _ = self._peek()
            if kind == "op" and val in (">=", "<=", "="):
                self._next()
                vkind, vval, vpos = self._next()
                if vkind == "num":
                    opts[key] = float(vval) if "." in vval else int(vval)
                elif vkind == "str":
                    opts[key] = vval[1:-1]
                elif vkind == "word":
                    opts[key] = vval
                else:
                    raise QuerySyntaxError(f"expected option value at position {vpos}")
            else:
                opts[key] = True
            kind, val, pos = self._next()
            if kind == "op" and val == ")":
                return opts
            if not (kind == "op" and val == ","):
                raise QuerySyntaxError(
                    f"expected ',' or ')' at position {pos}, got {val!r}"
                )

    def parse(self):
        head = self._expect_word("ANALYZE", "EXPLAIN", "SELECT")
        if head == "ANALYZE":
            self._expect_word("INDEX")
            index = self._path()
            queries = None
            if self._peek()[1].upper() == "USING":
                self._next()
                self._expect_word("QUERIES")
                queries = self._path()
            stmt = AnalyzeStmt(
                index=index, queries=queries, options=self._with_options()
            )
        else:
            explain = head == "EXPLAIN"
            if explain:
                self._expect_word("SELECT")
            stmt = self._select(explain)
        kind, val, pos = self._peek()
        if kind != "eof":
            raise QuerySyntaxError(
                f"unexpected trailing input at position {pos}: {val!r}"
            )
        return stmt

    def _select(self, explain: bool) -> SelectStmt:
        cols = []
        while True:
            kind, val, pos = self._next()
            if kind != "word" or val.upper() not in ("ID", "SCORE"):
                raise QuerySyntaxError(
                    f"selectable columns are id, score — got {val!r} at position {pos}"
                )
            cols.append(val.lower())
            kind, val, _ = self._peek()
            if kind == "op" and val == ",":
                self._next()
                continue
            break
        self._expect_word("FROM")
        index = self._path()
        simfn, limit = "COSINE", 10
        if self._peek()[1].upper() == "WHERE":
            raise QuerySyntaxError(
                "WHERE is not supported: TQE indexes carry no row payloads to filter on"
            )
        if self._peek()[1].upper() == "ORDER":
            self._next()
            self._expect_word("BY")
            simfn = self._expect_word(*_SIMFNS)
            self._expect_op("(")
            kind, val, pos = self._next()
            if val != ":q":
                raise QuerySyntaxError(
                    "the only query placeholder is :q (bound via --queries), "
                    f"got {val!r}"
                )
            self._expect_op(")")
        if self._peek()[1].upper() == "LIMIT":
            self._next()
            kind, val, pos = self._next()
            if kind != "num" or "." in val:
                raise QuerySyntaxError(f"LIMIT expects an integer at position {pos}")
            limit = int(val)
        return SelectStmt(
            index=index,
            columns=cols,
            simfn=simfn,
            limit=limit,
            options=self._with_options(),
            explain=explain,
        )


def parse(text: str):
    """Parse one statement -> AnalyzeStmt | SelectStmt. Raises QuerySyntaxError."""
    return _Parser(text).parse()


# --------------------------------------------------------------------------- #
# Statistics catalog (ANALYZE)                                                 #
# --------------------------------------------------------------------------- #


def catalog_path(index_path: str) -> str:
    return index_path + ".stats.json"


def _knn_self(x, k):
    """Exact top-(k+1) cosine neighbours of sample rows vs the full set; drops self."""
    import numpy as np

    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    q = xn[: min(len(xn), 1024)]
    sims = q @ xn.T
    idx = np.argsort(-sims, axis=1)[:, 1 : k + 2]
    d = np.sqrt(np.maximum(2.0 - 2.0 * np.take_along_axis(sims, idx, 1), 0))
    return d[:, : k + 1], idx[:, : k + 1]


def _geometry_block(x, k=10):
    """Compact descriptor block — mirrors the OpenVector Bench battery's estimators
    (two-NN id: Facco 2017; local-ID: Levina-Bickel 2004). Advisory only."""
    import numpy as np

    n, dim = x.shape
    d, idx = _knn_self(x, k)
    r1, r2 = d[:, 0], d[:, 1]
    mu = r2[r1 > 0] / np.maximum(r1[r1 > 0], 1e-12)
    mu = mu[mu > 1.0]
    if len(mu) >= 50:
        mu_t = mu[mu <= np.quantile(mu, 0.9)]
        id_twonn = float(len(mu_t) / np.sum(np.log(mu_t)))
    else:
        id_twonn = None
    tk, tj = d[:, k - 1 : k], d[:, : k - 1]
    good = (tj > 0).all(1) & (tk[:, 0] > 0)
    lid = 1.0 / np.maximum(np.log(tk[good] / tj[good]).mean(1), 1e-12)
    xc = x[: min(n, 20000)] - x[: min(n, 20000)].mean(0, keepdims=True)
    lam = np.linalg.svd(xc, compute_uv=False) ** 2
    lam = lam[lam > 0]
    frac = np.cumsum(lam) / lam.sum()
    counts = np.bincount(idx[:, :k].ravel(), minlength=n).astype(float)
    s = counts.std()
    skew = float(((counts - counts.mean()) ** 3).mean() / max(s**3, 1e-12))
    d10 = d[:, k - 1]
    nk_sample = counts[: len(d10)]
    dcorr = float(np.corrcoef(nk_sample, -d10)[0, 1]) if len(d10) > 2 else None
    return {
        "n": int(n),
        "dim": int(dim),
        "id_twonn": id_twonn,
        "local_id_iqr": float(np.subtract(*np.percentile(lid, [75, 25]))),
        "eff_rank": float(lam.sum() ** 2 / (lam**2).sum()),
        "dims90": int(np.searchsorted(frac, 0.90) + 1),
        "hubness": {
            "bb_skew": skew,
            "max_reverse_count": int(counts.max()),
            "corr_Nk_neg_d10": dcorr,
        },
    }


def _calibration_block(idx, queries, k, rerank_grid):
    """Measured recall/latency sweep — the planner's ONLY basis (WP5-lite rule)."""
    import numpy as np

    originals = getattr(idx, "_originals", None)
    if originals is None:
        return None
    q = np.asarray(queries, dtype=np.float32)
    live = np.flatnonzero(np.asarray(idx._tomb) == 0)
    base = np.asarray(originals)[live]
    bn = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    gt_pos = np.argsort(-(qn @ bn.T), axis=1)[:, :k]
    gt_ids = np.asarray(idx._ids)[live][gt_pos]
    sweep = []
    for rr in rerank_grid:
        t0 = time.perf_counter()
        ids, _ = idx.search(q, k=k, rerank=rr)
        lat = (time.perf_counter() - t0) / len(q)
        rec = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, gt_ids)]))
        sweep.append({"rerank": int(rr), "recall": rec, "latency_us": 1e6 * lat})
    return {"k": int(k), "n_queries": int(len(q)), "sweep": sweep}


def analyze(stmt: AnalyzeStmt) -> dict:
    """Build the statistics catalog for one single-file TQE index."""
    import json

    import numpy as np

    from .index import TQEIndex

    idx = TQEIndex.open(stmt.index)
    originals = getattr(idx, "_originals", None)
    live = np.flatnonzero(np.asarray(idx._tomb) == 0)
    if originals is not None:
        x = np.asarray(originals)[live]
        source = "originals"
    else:
        x = np.asarray(idx._reconstruct_rows(live))
        source = "reconstructed (keep_originals=False; geometry approximate)"
    sample = int(stmt.options.get("SAMPLE", 20000))
    if len(x) > sample:
        rng = np.random.default_rng(int(stmt.options.get("SEED", 0)))
        x = x[np.sort(rng.choice(len(x), size=sample, replace=False))]

    if stmt.queries:
        q = np.asarray(np.load(stmt.queries), dtype=np.float32)[:1000]
        q_source = stmt.queries
    else:
        rng = np.random.default_rng(int(stmt.options.get("SEED", 0)) + 1)
        q = x[rng.choice(len(x), size=min(500, len(x)), replace=False)]
        q_source = "sampled corpus rows (no held-out queries supplied)"

    calib = _calibration_block(
        idx, q, k=int(stmt.options.get("K", 10)), rerank_grid=(0, 2, 4, 8, 16, 32)
    )
    doc = {
        "schema": "turboquant-pro/query-catalog",
        "schema_version": 1,
        "index": stmt.index,
        "metric": idx.stats()["metric"],
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": {**_geometry_block(x), "source": source, "advisory": True},
        "queries": q_source,
        "calibration": calib,
        "planner_basis": (
            "calibration sweep only — geometry descriptors do not yet predict "
            "recall-cost (OpenVector Bench WP5-lite falsification, 2026-07-22)"
        ),
    }
    with open(catalog_path(stmt.index), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc


# --------------------------------------------------------------------------- #
# Planner (EXPLAIN) + executor (SELECT)                                        #
# --------------------------------------------------------------------------- #


def plan_from_catalog(catalog: dict, stmt: SelectStmt) -> dict:
    """Cheapest measured operating point meeting WITH (RECALL >= x); conservative."""
    calib = catalog.get("calibration")
    target = stmt.options.get("RECALL")
    if "RERANK" in stmt.options:
        rr = int(stmt.options["RERANK"])
        pred = None
        if calib:
            pred = next((p for p in calib["sweep"] if p["rerank"] == rr), None)
        return {
            "rerank": rr,
            "chosen_by": "explicit WITH (RERANK=...)",
            "predicted": pred,
        }
    if target is None:
        return {
            "rerank": 0,
            "chosen_by": "default (no RECALL target)",
            "predicted": None,
        }
    if not calib:
        raise RuntimeError(
            "WITH (RECALL >= ...) needs a calibration sweep, but the catalog has none "
            "(index built without originals). Re-create with keep_originals or use "
            "WITH (RERANK=n) explicitly."
        )
    sweep = sorted(calib["sweep"], key=lambda p: p["rerank"])
    for p in sweep:
        if p["recall"] >= float(target):
            return {
                "rerank": p["rerank"],
                "chosen_by": f"calibration (recall target {target})",
                "predicted": p,
            }
    best = max(sweep, key=lambda p: p["recall"])
    return {
        "rerank": best["rerank"],
        "chosen_by": f"calibration (recall target {target})",
        "predicted": best,
        "target_unreachable": True,
    }


def explain(stmt: SelectStmt, catalog: dict) -> dict:
    plan = plan_from_catalog(catalog, stmt)
    doc = {
        "schema": "turboquant-pro/query-plan",
        "schema_version": 1,
        "statement": {
            "index": stmt.index,
            "columns": stmt.columns,
            "simfn": stmt.simfn,
            "limit": stmt.limit,
            "options": stmt.options,
        },
        "plan": plan,
        "basis": {
            "analyzed_at": catalog.get("analyzed_at"),
            "calibration_k": (catalog.get("calibration") or {}).get("k"),
            "calibration_queries": catalog.get("queries"),
        },
        "advisory_geometry": catalog.get("geometry"),
        "honesty": catalog.get("planner_basis"),
    }
    ck = (catalog.get("calibration") or {}).get("k")
    if ck is not None and ck != stmt.limit:
        doc["warnings"] = [
            f"calibration measured at k={ck}, statement LIMIT={stmt.limit}: "
            "prediction is approximate; re-ANALYZE WITH (K={0}) to match".format(
                stmt.limit
            )
        ]
    metric = catalog.get("metric")
    want = {"COSINE": "cosine", "DOT": "cosine", "L2": "l2"}[stmt.simfn]
    if metric and metric != want:
        doc.setdefault("warnings", []).append(
            f"ORDER BY {stmt.simfn} but the index metric is {metric!r}: "
            "scores are computed in the index metric"
        )
    return doc


def execute(stmt: SelectStmt, queries, catalog: dict | None) -> dict:
    """Run SELECT: plan (if catalog + RECALL), search, optionally certify."""
    import numpy as np

    from .index import TQEIndex

    if stmt.options.get("RECALL") is not None and catalog is None:
        raise RuntimeError(
            f"WITH (RECALL >= ...) needs the statistics catalog; run: "
            f"tqp query \"ANALYZE INDEX '{stmt.index}'\" first"
        )
    plan = (
        plan_from_catalog(catalog, stmt)
        if catalog
        else {
            "rerank": int(stmt.options.get("RERANK", 0)),
            "chosen_by": "no catalog (un-analyzed index)",
            "predicted": None,
        }
    )
    idx = TQEIndex.open(stmt.index)
    q = np.asarray(queries, dtype=np.float32)
    t0 = time.perf_counter()
    ids, scores = idx.search(q, k=stmt.limit, rerank=plan["rerank"])
    elapsed = time.perf_counter() - t0
    results = []
    for i, (row, srow) in enumerate(zip(ids, scores)):
        entry: dict = {"query": i}
        if "id" in stmt.columns:
            entry["ids"] = [int(v) for v in row if v >= 0]
        if "score" in stmt.columns:
            entry["scores"] = [float(s) for s, v in zip(srow, row) if v >= 0]
        results.append(entry)
    doc = {
        "schema": "turboquant-pro/query-result",
        "schema_version": 1,
        "statement": {"index": stmt.index, "limit": stmt.limit, "simfn": stmt.simfn},
        "plan": plan,
        "n_queries": int(len(q)),
        "mean_latency_us": 1e6 * elapsed / max(len(q), 1),
        "results": results,
    }
    if stmt.options.get("CERTIFY"):
        cert = idx.certify()
        doc["certificate"] = cert.as_dict()
    return doc
