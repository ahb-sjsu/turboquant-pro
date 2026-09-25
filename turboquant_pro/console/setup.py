"""Save and recall an instrument setup (the scope's and the analyzer's front panel).

A setup (``*.tqs``, JSON, schema ``console_setup.schema.json``) is pure data: channel
signals, scales and positions; the time base; the trigger; acquisition and persistence;
masks; the analyzer's traces, reference level, span, detector and limit; the view. It
records the observer contract's sha256 it was made under, so recalling it under another
observer says so. Loading validates every field and applies nothing until all of it is
valid; nothing in a setup is ever executed. ``apply(to_dict(x))`` reproduces ``x``.
"""

from __future__ import annotations

import json
import math

from .scope import SIGNALS, Channel, Scope, Trigger
from .spectrum import SOURCES, Analyzer

SCHEMA = "turboquant-pro/console-setup"
SCHEMA_VERSION = 1
VIEWS = ("scope", "spectrum", "overview")
_KINDS, _SLOPES = ("edge", "pulse", "logic"), ("rising", "falling", "either")
_MODES, _ACQ = ("auto", "normal", "single"), ("sample", "peak", "average")
_TMODES = ("write", "maxhold", "minhold", "average", "delta", "blank")
_OPS = ("==", "!=", ">", ">=", "<", "<=")


def to_dict(
    scope: Scope,
    analyzer: Analyzer,
    view: str = "scope",
    observer_sha256: str | None = None,
) -> dict:
    tg = scope.trigger
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "observer_sha256": observer_sha256,
        "view": view,
        "scope": {
            "channels": [
                {
                    "signal": c.signal,
                    "scale": c.scale,
                    "position": c.position,
                    "on": c.on,
                }
                for c in scope.channels
            ],
            "s_per_div": scope.s_per_div,
            "h_position": scope.h_position,
            "trigger": {
                "kind": tg.kind,
                "source": tg.source,
                "level": tg.level,
                "slope": tg.slope,
                "width": tg.width,
                "conditions": [list(c) for c in tg.conditions],
                "mode": tg.mode,
                "holdoff_s": tg.holdoff_s,
                "position": tg.position,
            },
            "acquire": scope.acquire,
            "decay": scope.decay,
            "masks": {k: list(v) for k, v in scope.masks.items()},
            "stop_on_fail": scope.stop_on_fail,
        },
        "analyzer": {
            "traces": [
                {"source": t.source, "mode": t.mode, "avg_n": t.avg_n}
                for t in analyzer.traces
            ],
            "ref_db": analyzer.ref_db,
            "db_div": analyzer.db_div,
            "start": analyzer.start,
            "stop": analyzer.stop,
            "detector": analyzer.detector,
            "limit": analyzer.limit,
            "limit_on": analyzer.limit_on,
            "limit_trace": analyzer.limit_trace,
            "waterfall_source": analyzer.waterfall_source,
        },
    }


class SetupError(ValueError):
    """A setup that cannot be applied; the message names the offending field."""


def _need(ok: bool, where: str, what: str) -> None:
    if not ok:
        raise SetupError(f"{where}: {what}")


def _num(v, where, lo=-math.inf, hi=math.inf, open_lo=False):
    _need(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
        where,
        "must be a finite number",
    )
    _need(
        v > lo if open_lo else v >= lo,
        where,
        f"must be {'>' if open_lo else '>='} {lo}",
    )
    _need(v <= hi, where, f"must be <= {hi}")
    return float(v)


def _int(v) -> bool:
    """A JSON integer: Python's bool is an int subclass, and true is not a width."""
    return isinstance(v, int) and not isinstance(v, bool)


def _one(v, where, allowed):
    _need(v in allowed, where, f"must be one of {', '.join(map(str, allowed))}")
    return v


def validate(doc: dict) -> None:
    """Every field checked; raises :class:`SetupError` naming the first bad one."""
    _need(isinstance(doc, dict), "setup", "must be an object")
    _need(doc.get("schema") == SCHEMA, "schema", f"must be {SCHEMA!r}")
    _need(
        doc.get("schema_version") == SCHEMA_VERSION,
        "schema_version",
        f"must be {SCHEMA_VERSION}",
    )
    sha = doc.get("observer_sha256")
    _need(
        sha is None
        or (
            isinstance(sha, str)
            and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha)
        ),
        "observer_sha256",
        "must be null or a 64-hex sha256",
    )
    _one(doc.get("view"), "view", VIEWS)
    sc = doc.get("scope")
    _need(isinstance(sc, dict), "scope", "must be an object")
    chans = sc.get("channels")
    _need(
        isinstance(chans, list) and len(chans) == 4,
        "scope.channels",
        "must list 4 channels",
    )
    for i, c in enumerate(chans):
        w = f"scope.channels[{i}]"
        _need(isinstance(c, dict), w, "must be an object")
        _one(c.get("signal"), f"{w}.signal", tuple(SIGNALS))
        _num(c.get("scale"), f"{w}.scale", 0, open_lo=True)
        _num(c.get("position"), f"{w}.position", -100, 100)
        _need(isinstance(c.get("on"), bool), f"{w}.on", "must be true or false")
    _num(sc.get("s_per_div"), "scope.s_per_div", 0, 86400, open_lo=True)
    _num(sc.get("h_position"), "scope.h_position", -1e6, 1e6)
    tg = sc.get("trigger")
    _need(isinstance(tg, dict), "scope.trigger", "must be an object")
    _one(tg.get("kind"), "scope.trigger.kind", _KINDS)
    _one(tg.get("source"), "scope.trigger.source", tuple(SIGNALS))
    _num(tg.get("level"), "scope.trigger.level")
    _one(tg.get("slope"), "scope.trigger.slope", _SLOPES)
    _need(
        _int(tg.get("width")) and 1 <= tg["width"] <= 10_000,
        "scope.trigger.width",
        "must be an integer in 1..10000",
    )
    conds = tg.get("conditions")
    _need(
        isinstance(conds, list) and len(conds) <= 16,
        "scope.trigger.conditions",
        "must be a list of at most 16 [field, op, value]",
    )
    for i, c in enumerate(conds):
        w = f"scope.trigger.conditions[{i}]"
        _need(
            isinstance(c, list)
            and len(c) == 3
            and isinstance(c[0], str)
            and len(c[0]) <= 64,
            w,
            "must be [field, op, value]",
        )
        _one(c[1], f"{w}[1]", _OPS)
        _need(
            isinstance(c[2], (str, int, float, bool)) or c[2] is None,
            f"{w}[2]",
            "must be a string, number, boolean or null",
        )
    _one(tg.get("mode"), "scope.trigger.mode", _MODES)
    _num(tg.get("holdoff_s"), "scope.trigger.holdoff_s", 0, 86400)
    _num(tg.get("position"), "scope.trigger.position", 0, 1)
    _one(sc.get("acquire"), "scope.acquire", _ACQ)
    _num(sc.get("decay"), "scope.decay", 0, 1)
    masks = sc.get("masks")
    _need(isinstance(masks, dict), "scope.masks", "must be an object")
    for k, v in masks.items():
        _one(k, "scope.masks key", tuple(SIGNALS))
        _need(
            isinstance(v, list) and len(v) == 2,
            f"scope.masks.{k}",
            "must be [low, high]",
        )
        for j, x in enumerate(v):
            if x is not None:
                _num(x, f"scope.masks.{k}[{j}]")
    _need(
        isinstance(sc.get("stop_on_fail"), bool),
        "scope.stop_on_fail",
        "must be true or false",
    )
    an = doc.get("analyzer")
    _need(isinstance(an, dict), "analyzer", "must be an object")
    trs = an.get("traces")
    _need(
        isinstance(trs, list) and len(trs) == 4, "analyzer.traces", "must list 4 traces"
    )
    for i, t in enumerate(trs):
        w = f"analyzer.traces[{i}]"
        _need(isinstance(t, dict), w, "must be an object")
        _one(t.get("source"), f"{w}.source", SOURCES)
        _one(t.get("mode"), f"{w}.mode", _TMODES)
        _need(
            _int(t.get("avg_n")) and 1 <= t["avg_n"] <= 10_000,
            f"{w}.avg_n",
            "must be an integer in 1..10000",
        )
    _num(an.get("ref_db"), "analyzer.ref_db", -1000, 1000)
    _num(an.get("db_div"), "analyzer.db_div", 0, 100, open_lo=True)
    _need(
        _int(an.get("start")) and an["start"] >= 0,
        "analyzer.start",
        "must be a non-negative integer",
    )
    _need(
        an.get("stop") is None or (_int(an["stop"]) and an["stop"] > an["start"]),
        "analyzer.stop",
        "must be null or an integer above start",
    )
    _one(an.get("detector"), "analyzer.detector", ("peak", "average"))
    if an.get("limit") is not None:
        _num(an["limit"], "analyzer.limit", -1000, 1000)
    _need(
        isinstance(an.get("limit_on"), bool),
        "analyzer.limit_on",
        "must be true or false",
    )
    _need(
        _int(an.get("limit_trace")) and an.get("limit_trace") in (0, 1, 2, 3),
        "analyzer.limit_trace",
        "must be 0..3",
    )
    _one(an.get("waterfall_source"), "analyzer.waterfall_source", SOURCES)


def apply(
    doc: dict, scope: Scope, analyzer: Analyzer, observer_sha256: str | None = None
) -> list:
    """Validate ``doc`` completely, then apply it. Returns warnings (for example a
    setup made under another observer). Nothing is changed if validation fails."""
    validate(doc)
    warnings = []
    made = doc.get("observer_sha256")
    if made and made != observer_sha256:
        warnings.append(
            f"this setup was made under observer {made[:12]}, not "
            f"{(observer_sha256 or 'none')[:12]}: its triggers and masks may not mean "
            "the same thing here"
        )
    sc, an = doc["scope"], doc["analyzer"]
    scope.channels = [
        Channel(c["signal"], float(c["scale"]), float(c["position"]), c["on"])
        for c in sc["channels"]
    ]
    scope.s_per_div = float(sc["s_per_div"])
    scope.h_position = float(sc["h_position"])
    t = sc["trigger"]
    scope.trigger = Trigger(
        kind=t["kind"],
        source=t["source"],
        level=float(t["level"]),
        slope=t["slope"],
        width=int(t["width"]),
        conditions=[tuple(c) for c in t["conditions"]],
        mode=t["mode"],
        holdoff_s=float(t["holdoff_s"]),
        position=float(t["position"]),
    )
    scope.acquire, scope.decay = sc["acquire"], float(sc["decay"])
    scope.masks = {k: tuple(v) for k, v in sc["masks"].items()}
    scope.stop_on_fail = sc["stop_on_fail"]
    scope.persist = {}
    for tr, d in zip(analyzer.traces, an["traces"]):
        tr.source, tr.mode, tr.avg_n = d["source"], d["mode"], int(d["avg_n"])
        tr.clear()
    analyzer.ref_db, analyzer.db_div = float(an["ref_db"]), float(an["db_div"])
    analyzer.start, analyzer.stop = an["start"], an["stop"]
    analyzer.detector = an["detector"]
    analyzer.limit = None if an["limit"] is None else float(an["limit"])
    analyzer.limit_on, analyzer.limit_trace = an["limit_on"], an["limit_trace"]
    analyzer.waterfall_source = an["waterfall_source"]
    return warnings


def save(path: str, doc: dict) -> None:
    validate(doc)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)


def load(path: str, max_bytes: int = 1 << 20) -> dict:
    """Read a setup file (at most 1 MiB: a setup is small, a larger file is refused)."""
    with open(path, "rb") as f:
        raw = f.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise SetupError(f"{path}: larger than {max_bytes} bytes; not a setup")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SetupError(f"{path}: not JSON ({e})") from None
    validate(doc)
    return doc
