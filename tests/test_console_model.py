"""The console's model page: the readings-file source and its panels."""

from __future__ import annotations

import json
import os

from turboquant_pro.console import model_view
from turboquant_pro.console.model_source import ModelReadingsMonitor
from turboquant_pro.console.viewmodel import pages
from turboquant_pro.telemetry.metrics import reading


def _lines(window, heads=((0, 0), (0, 1), (1, 0), (1, 1)), stale=(1, 1)):
    out = []
    for li, h in heads:
        lab = {"model": "m", "run": "r", "window": window, "layer": li, "kv_head": h,
               "consumer": "softmax", "tokens": 512}  # fmt: skip
        r = reading("model.read.effective_rank", 3.0 + h, as_of=100.0 + window)
        r["labels"] = lab
        out.append(r)
        st = "STALE" if (li, h) == stale else "VALID"
        r = reading("model.plan.overlap", 0.6 if st == "STALE" else 0.97, as_of=100.0)
        r.update(labels=lab, status=st, status_reason="why " + st)
        out.append(r)
        r = reading("model.plan.regret", None, reason="512 key rows < 1024 channels")
        r.update(labels=lab, status=st, status_reason="why " + st)
        out.append(r)
    return [json.dumps(r) for r in out]


def test_a_window_shows_once_complete_and_the_newest_wins(tmp_path):
    p = tmp_path / "r.jsonl"
    mon = ModelReadingsMonitor(str(p), settle_s=3600)
    assert mon.poll()["window"] is None and not mon.poll()["exists"]
    w0 = _lines(0)
    p.write_text("\n".join(w0) + "\n")
    assert mon.poll()["window"] is None  # still growing: not settled
    with open(p, "a") as f:
        f.write("\n".join(_lines(1)) + "\n" + _lines(2)[0][:20])  # a partial line
    doc = mon.poll()
    assert doc["window"]["index"] == 0  # 1 is still open, the partial line waits
    mon.settle_s = 0.0
    doc = mon.poll()
    assert doc["window"]["index"] == 1 and doc["bad_lines"] == 0
    heads = {(h["layer"], h["kv_head"]): h for h in doc["window"]["heads"]}
    assert heads[(1, 1)]["status"] == "STALE"
    assert "channels" in heads[(0, 0)]["why"]["model.plan.regret"]


def test_a_truncated_file_is_read_again(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(_lines(0) + _lines(1)) + "\n")
    mon = ModelReadingsMonitor(str(p), settle_s=0.0)
    assert mon.poll()["window"]["index"] == 1
    p.write_text("\n".join(_lines(0)) + "\n")  # shorter: a new run
    assert mon.poll()["window"]["index"] == 0
    with open(p, "a") as f:
        f.write("not json\n")
    assert mon.poll()["bad_lines"] == 1


def test_the_panels_say_what_they_show_and_why_not(tmp_path):
    assert model_view.panels({})["1"]["state"] == "none"
    p = tmp_path / "r.jsonl"
    st = {"model": ModelReadingsMonitor(str(p), settle_s=0.0).poll()}
    out = model_view.panels(st)
    assert "no file yet" in out["1"]["message"] and out["6"]["state"] == "ok"
    p.write_text("\n".join(_lines(0)) + "\n")
    st = {"model": ModelReadingsMonitor(str(p), settle_s=0.0).poll()}
    out = model_view.panels(st)
    assert set(out) == {"1", "2", "3", "4", "5", "6"}
    assert out["1"]["summary"][1] == "red"  # a STALE head colours the summary
    assert [r[0] for r in out["1"]["rows"]][:1] == ["STALE"]
    table = out["2"]["table"]
    assert table["rows"][0][:3] == ["1", "1", "STALE"] and table["roles"] == ["red"]
    assert table["rows"][0][4] == "-"  # an unsupported regret is '-', never 0
    strips = out["3"]["strips"]
    assert [s[0] for s in strips] == ["L0", "L1"] and strips[1][1][1] == 0.6
    assert out["5"]["state"] == "none" and "--kv-cache" in out["5"]["message"]
    for panel in out.values():
        json.dumps(panel)  # the client reads JSON


def test_the_page_exists_only_with_its_source():
    names = [p["name"] for p in pages({"model": True})]
    assert names == ["model"]
    assert "model" not in [p["name"] for p in pages({"index": True})]


def test_the_cli_refuses_the_web_view_for_the_model_page(tmp_path, capsys):
    from turboquant_pro.cli import main

    f = tmp_path / "r.jsonl"
    f.write_text("")
    rc = main(["console", "--model-readings", str(f), "--web"])
    assert rc == 2 and "terminal console only" in capsys.readouterr().err
    assert os.path.exists(f)
