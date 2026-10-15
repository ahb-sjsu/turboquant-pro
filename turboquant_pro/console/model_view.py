# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The model page: six panels from :mod:`console.model_source` snapshots, in the one
PanelView shape the machine page uses (see :mod:`console.machine_view`).

What a live model's attention reads, per decoder layer and key head, and, when the
watcher scores a plan, whether the plan still fits: status counts, the heads that
are not VALID, an overlap map and a read-rank map (one cell per key head, one strip
per layer), the key codec's distortion as each head's consumer feels it, and the
source. A value the window could not support is "-" with its reason, never 0.
"""

from __future__ import annotations

TITLES = {
    "1": "1 plan status  per key head, by the certificate-expiry rules",
    "2": "2 heads not VALID  worst overlap first",
    "3": "3 overlap with the plan  one strip per layer, a cell per key head",
    "4": "4 read rank  effective rank over head width, per layer",
    "5": "5 key codec  tr(P Sigma_delta), as each head's consumer feels it",
    "6": "6 source",
}
PANELS = [1, 2, 3, 4, 5, 6]
STATUS_ROLE = {"VALID": "green", "STALE": "red", "INCONCLUSIVE": "amber"}
STATUS_ORDER = ("STALE", "INCONCLUSIVE", "VALID", "UNCHECKED")
MAX_STRIPS = 64  # layers drawn as strips (a 64-layer model fits; more are cut)


def _empty(message: str) -> dict:
    return {"state": "none", "message": message}


def _f(v, fmt: str = ".3f") -> str:
    return "-" if v is None else format(v, fmt)


def panels(st: dict) -> dict:
    """The model page's panels, keyed "1" to "6"."""
    doc = st.get("model")
    if doc is None:
        msg = "not attached: start with --model-readings FILE"
        return {str(n): _empty(msg) for n in PANELS}
    w = doc.get("window")
    if w is None:
        msg = (
            f"waiting for a complete window in {doc['path']}"
            if doc.get("exists")
            else f"no file yet: {doc['path']} (tqp model watch --readings writes it)"
        )
        out = {str(n): _empty(msg) for n in PANELS[:-1]}
        out["6"] = source(doc)
        return out
    heads = w["heads"]
    return {
        "1": status(w, heads),
        "2": not_valid(heads),
        "3": overlap_map(heads),
        "4": rank_map(heads),
        "5": key_codec(heads),
        "6": source(doc),
    }


def _by_layer(heads):
    layers: dict = {}
    for h in heads:
        layers.setdefault(h["layer"], []).append(h)
    return [
        (li, sorted(v, key=lambda h: h["kv_head"])) for li, v in sorted(layers.items())
    ]


def status(w: dict, heads: list) -> dict:
    scored = [h for h in heads if h.get("status")]
    head = f"window {w['index']}"
    if w.get("tokens"):
        head += f", {w['tokens']} tokens"
    if not scored:
        return {
            "state": "ok",
            "summary": [f"{head}, {len(heads)} key heads; no plan scored", "cyan"],
            "rows": [],
            "note": "start the watcher with --plan to score a plan",
        }
    counts = {s: 0 for s in STATUS_ORDER}
    for h in scored:
        counts[h["status"]] = counts.get(h["status"], 0) + 1
    rows = [
        [s, str(counts[s]), "deri", None, f"{counts[s] / len(scored):.0%}",
         STATUS_ROLE.get(s, "dim")]
        for s in STATUS_ORDER
    ]  # fmt: skip
    worst = (
        "red" if counts["STALE"] else ("amber" if counts["INCONCLUSIVE"] else "green")
    )
    return {
        "state": "ok",
        "summary": [f"{head}, {len(scored)} key heads scored", worst],
        "rows": rows,
        "note": "STALE: the head reads outside the plan's subspace; INCONCLUSIVE: the "
        "window cannot tell; UNCHECKED: the plan's operator is too flat",
    }


NV_COLS = [
    ["layer", 5, True],
    ["kv", 3, True],
    ["status", 12, False],
    ["overlap", 8, True],
    ["kappa", 7, True],
    ["regret", 7, True],
    ["why", 40, False],
]


def not_valid(heads: list) -> dict:
    bad = [h for h in heads if h.get("status") and h["status"] != "VALID"]
    if not any(h.get("status") for h in heads):
        return _empty("no plan scored")
    if not bad:
        return {"state": "ok", "summary": ["every scored head is VALID", "green"]}

    def ov(h):
        v = h["values"].get("model.plan.overlap")
        return 2.0 if v is None else v

    bad.sort(key=lambda h: (STATUS_ORDER.index(h["status"]), ov(h)))
    rows, roles = [], []
    for h in bad:
        v = h["values"]
        rows.append(
            [
                str(h["layer"]),
                str(h["kv_head"]),
                h["status"],
                _f(v.get("model.plan.overlap")),
                _f(v.get("model.plan.kappa"), ".3g"),
                _f(v.get("model.plan.regret"), ".3g"),
                h.get("reason", ""),
            ]
        )
        roles.append(STATUS_ROLE.get(h["status"], "dim"))
    return {
        "state": "ok",
        "title_extra": f"{len(bad)} heads",
        "table": {"cols": NV_COLS, "rows": rows, "roles": roles},
        "note": "overlap is relative to the plan's own (1 = unchanged); '-' = the "
        "window cannot support it",
    }


def _strips(heads, value):
    strips = []
    for li, hs in _by_layer(heads)[:MAX_STRIPS]:
        strips.append([f"L{li}", [value(h) for h in hs]])
    return strips


def overlap_map(heads: list) -> dict:
    if not any("model.plan.overlap" in h["values"] for h in heads):
        return _empty("no plan scored")

    def cell(h):
        v = h["values"].get("model.plan.overlap")
        return None if v is None else max(0.0, min(1.0, v))

    return {
        "state": "ok",
        "strips": _strips(heads, cell),
        "note": "cell height = overlap (full = the plan's own); a dot = not available",
    }


def rank_map(heads: list) -> dict:
    vals = [h["values"].get("model.read.effective_rank") for h in heads]
    if not any(v is not None for v in vals):
        return _empty("no read-rank readings in this window")
    # the head width is not in the readings; the widest effective rank seen bounds it
    top = max(v for v in vals if v is not None) or 1.0

    def cell(h):
        v = h["values"].get("model.read.effective_rank")
        return None if v is None else v / top

    return {
        "state": "ok",
        "title_extra": f"full cell = {top:.1f} directions",
        "strips": _strips(heads, cell),
        "note": "participation ratio of each head's read spectrum",
    }


def key_codec(heads: list) -> dict:
    rows = []
    for li, hs in _by_layer(heads):
        vals = [h["values"].get("model.kv.key_distortion") for h in hs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        rows.append(
            [f"L{li}", f"{sum(vals) / len(vals):.3g}", "meas", None,
             f"max {max(vals):.3g} over {len(vals)} heads", ""]
        )  # fmt: skip
    if not rows:
        return _empty("no key-codec readings: the watcher needs --kv-cache")
    return {
        "state": "ok",
        "rows": rows,
        "note": "mean over key heads: the codec's error, weighted by what each head "
        "reads",
    }


def source(doc: dict) -> dict:
    w = doc.get("window") or {}
    rows = [
        ["file", doc["path"], "meas", None, "" if doc.get("exists") else "absent", ""],
        ["model", str(w.get("model") or "-"), "meas", None, "", ""],
        ["run", str(w.get("run") or "-"), "meas", None, "", ""],
        ["consumer", str(w.get("consumer") or "-"), "meas", None, "", ""],
        ["windows", str(doc.get("windows_seen", 0)), "meas", None, "kept in view", ""],
    ]
    if doc.get("bad_lines"):
        rows.append(
            ["bad lines", str(doc["bad_lines"]), "meas", None, "skipped", "amber"]
        )
    return {
        "state": "ok",
        "rows": rows,
        "note": "read-only; the console loads no model",
    }
