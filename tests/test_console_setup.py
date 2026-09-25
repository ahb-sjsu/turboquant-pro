"""Save/recall of console setups (turboquant_pro.console.setup): round trip, strict
validation, all-or-nothing application, and agreement with the shipped JSON Schema."""

from __future__ import annotations

import copy
import json

import pytest

from turboquant_pro.console import setup as SU
from turboquant_pro.console.scope import Scope, Trigger
from turboquant_pro.console.spectrum import Analyzer
from turboquant_pro.schemas import load_schema

jsonschema = pytest.importorskip("jsonschema")
SHA = "ab" * 32


def _configured():
    sc, an = Scope(), Analyzer()
    sc.channels[1].on, sc.channels[1].signal, sc.channels[1].scale = True, "agree", 0.1
    sc.s_per_div, sc.h_position, sc.acquire, sc.decay = 0.2, -1.5, "average", 1.0
    sc.trigger = Trigger(
        kind="logic",
        source="scan",
        level=3.5,
        slope="falling",
        width=5,
        conditions=[("scan_path", "==", "numpy"), ("agree", "<", 0.5)],
        mode="single",
        holdoff_s=0.25,
        position=0.4,
    )
    sc.masks, sc.stop_on_fail = {"latency": (None, 20.0)}, True
    an.traces[2].mode, an.traces[3].source = "maxhold", "var"
    an.ref_db, an.db_div, an.start, an.stop = -10.0, 5.0, 4, 40
    an.detector, an.limit, an.limit_on, an.waterfall_source = (
        "average",
        -30.0,
        False,
        "noise",
    )
    return sc, an


def test_round_trip_reproduces_the_instrument():
    sc, an = _configured()
    doc = SU.to_dict(sc, an, view="spectrum", observer_sha256=SHA)
    sc2, an2 = Scope(), Analyzer()
    assert SU.apply(json.loads(json.dumps(doc)), sc2, an2, SHA) == []
    assert SU.to_dict(sc2, an2, "spectrum", SHA) == doc


def test_file_save_and_load(tmp_path):
    sc, an = _configured()
    p = tmp_path / "bench.tqs"
    SU.save(str(p), SU.to_dict(sc, an, observer_sha256=SHA))
    doc = SU.load(str(p))
    jsonschema.validate(doc, load_schema("console_setup.schema.json"))


def test_a_setup_from_another_observer_warns():
    sc, an = _configured()
    doc = SU.to_dict(sc, an, observer_sha256=SHA)
    warn = SU.apply(doc, Scope(), Analyzer(), observer_sha256="cd" * 32)
    assert len(warn) == 1 and SHA[:12] in warn[0] and "observer" in warn[0]


def test_nothing_is_applied_unless_everything_is_valid():
    sc, an = _configured()
    doc = SU.to_dict(sc, an)
    doc["analyzer"]["traces"][3]["mode"] = "explode"
    target, before = Scope(), SU.to_dict(Scope(), Analyzer())
    with pytest.raises(SU.SetupError, match=r"analyzer.traces\[3\].mode"):
        SU.apply(doc, target, Analyzer())
    assert SU.to_dict(target, Analyzer()) == before  # untouched


def test_oversized_non_json_and_non_finite_files_are_refused(tmp_path):
    big = tmp_path / "big.tqs"
    big.write_bytes(b" " * ((1 << 20) + 1))
    with pytest.raises(SU.SetupError, match="larger than"):
        SU.load(str(big))
    bad = tmp_path / "bad.tqs"
    bad.write_text("{not json")
    with pytest.raises(SU.SetupError, match="not JSON"):
        SU.load(str(bad))
    sc, an = _configured()
    doc = SU.to_dict(sc, an)
    nan = tmp_path / "nan.tqs"
    nan.write_text(json.dumps(doc).replace('"s_per_div": 0.2', '"s_per_div": NaN'))
    with pytest.raises(SU.SetupError, match="s_per_div"):
        SU.load(str(nan))  # Python's json accepts NaN; the validator must not


MUTATIONS = [
    ("schema", lambda d: d.update(schema="other")),
    ("version", lambda d: d.update(schema_version=2)),
    ("view", lambda d: d.update(view="radar")),
    ("sha", lambda d: d.update(observer_sha256="xyz")),
    ("signal", lambda d: d["scope"]["channels"][0].update(signal="rm -rf")),
    ("scale0", lambda d: d["scope"]["channels"][0].update(scale=0)),
    ("scale-str", lambda d: d["scope"]["channels"][0].update(scale="1")),
    ("on", lambda d: d["scope"]["channels"][0].update(on=1)),
    ("3 channels", lambda d: d["scope"]["channels"].pop()),
    ("s/div", lambda d: d["scope"].update(s_per_div=-1)),
    ("kind", lambda d: d["scope"]["trigger"].update(kind="magic")),
    ("width", lambda d: d["scope"]["trigger"].update(width=0)),
    ("op", lambda d: d["scope"]["trigger"]["conditions"][0].__setitem__(1, "~=")),
    ("cond len", lambda d: d["scope"]["trigger"]["conditions"][0].append(1)),
    ("position", lambda d: d["scope"]["trigger"].update(position=1.5)),
    ("acquire", lambda d: d["scope"].update(acquire="hires")),
    ("mask key", lambda d: d["scope"]["masks"].update(bogus=[None, 1])),
    ("trace src", lambda d: d["analyzer"]["traces"][0].update(source="psd")),
    ("db/div", lambda d: d["analyzer"].update(db_div=0)),
    ("limit_trace", lambda d: d["analyzer"].update(limit_trace=4)),
    ("width-bool", lambda d: d["scope"]["trigger"].update(width=True)),
    ("limit_trace-bool", lambda d: d["analyzer"].update(limit_trace=True)),
    ("avg_n-float", lambda d: d["analyzer"]["traces"][0].update(avg_n=2.5)),
]


@pytest.mark.parametrize("name,mut", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_the_python_validator_and_the_json_schema_agree(name, mut):
    sc, an = _configured()
    good = SU.to_dict(sc, an, observer_sha256=SHA)
    schema = load_schema("console_setup.schema.json")
    SU.validate(good)
    jsonschema.validate(good, schema)
    bad = copy.deepcopy(good)
    mut(bad)
    with pytest.raises(SU.SetupError):
        SU.validate(bad)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_cli_recalls_a_setup_at_start_in_a_real_terminal(tmp_path):
    """End to end: a setup saved with the spectrum view comes up as the analyzer."""
    import os
    import subprocess
    import sys
    import time

    from turboquant_pro.cli import build_parser

    assert build_parser().parse_args(["console", "--demo", "--setup", "x"]).setup == "x"
    pty = pytest.importorskip("pty")
    if sys.platform.startswith("win"):
        pytest.skip("no pty on Windows")
    p = tmp_path / "view.tqs"
    SU.save(str(p), SU.to_dict(Scope(), Analyzer(), view="spectrum"))
    master, slave = pty.openpty()
    env = dict(
        os.environ,
        TERM="xterm-256color",
        LINES="40",
        COLUMNS="120",
        LANG="C.UTF-8",
        LC_ALL="C.UTF-8",
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from turboquant_pro.cli import main; raise "
            f"SystemExit(main(['console', '--demo', '--setup', {str(p)!r}]))",
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    import select

    out, deadline = b"", time.time() + 60
    while time.time() < deadline and b"SPECTRUM" not in out:
        r, _, _ = select.select([master], [], [], 0.5)
        if r:
            try:
                out += os.read(master, 65536)
            except OSError:
                break
    assert b"SPECTRUM" in out, out[-400:]
    os.write(master, b"q")
    assert proc.wait(timeout=20) == 0
    os.close(master)
