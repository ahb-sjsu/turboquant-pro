"""The vector (matplotlib, ATC-style) console: the same grid and numbers as the
character display, drawn headless here with the Agg backend."""

from __future__ import annotations

import io
import time

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402

from turboquant_pro.console import fabric_view, tui, vector_view  # noqa: E402
from turboquant_pro.console.server import ConsoleServer, demo_index  # noqa: E402

from .test_console_grid import _fabric_doc  # noqa: E402


@pytest.fixture(scope="module")
def live():
    index, Q, X, source, codec = demo_index(n=1500, dim=64, out_dim=32)
    s = ConsoleServer(
        index, Q, qps=50, k=5, rerank=3, originals=X, source=source, http=False,
        codec=codec,
    ).start()  # fmt: skip
    st = tui.new_state(s)
    deadline = time.time() + 10
    while len(s.tracer.traces()) < 30 and time.time() < deadline:
        time.sleep(0.05)
    for i in range(3):  # three snapshots and sweeps, as the loop would take them
        now = time.time() + 2.0 * i
        st["_clock"]["started"] -= 5
        tui.update(st, s, now)
    st["fabric"] = _fabric_doc()
    st["fabric_hist"].add(st["fabric"])
    yield s, st
    s.stop()


def _texts(fig) -> str:
    return "\n".join(t.get_text() for t in fig.findobj(Text))


def _draw(st, **kw):
    fig = Figure(figsize=(16, 10))
    vector_view.draw(fig, dict(st, **kw))
    return fig


def test_the_vector_grid_has_the_same_eight_panels(live):
    _, st = live
    text = _texts(_draw(st))
    for title in vector_view.TITLES.values():
        assert title in text
    assert "NATS 1 leaf, 1 clients" in text  # panel 1 set from the same panel code
    assert "ADCIndex" in text and "q quit" in text


def test_traces_are_named_by_data_blocks_that_notes_can_hide(live):
    _, st = live
    fig = _draw(st)
    blocks = [a.get_text() for a in fig.findobj(Annotation) if a.arrowprops]
    assert any(b.startswith("1 latency") for b in blocks)  # a scope channel
    assert any(b.startswith("T1 ") for b in blocks)  # a spectrum trace
    assert any(b.startswith("QPS ") for b in blocks)
    hidden = _draw(st, annotate=False)
    assert not [a for a in hidden.findobj(Annotation) if a.arrowprops]


def test_zoom_draws_one_instrument_and_the_fabric_in_type(live):
    _, st = live
    scope = _draw(st, zoom="scope")
    assert "7 scope" in _texts(scope) and "1 system" not in _texts(scope)
    fab = _draw(st, zoom="fabric")
    assert "2 leaf links (1)" in _texts(fab)
    help_ = _draw(st, overlay="help")
    assert "zoom: the focused panel full screen" in _texts(help_)


def test_it_saves_as_svg(live):
    _, st = live
    buf = io.StringIO()
    _draw(st).savefig(buf, format="svg", facecolor=vector_view.BG)
    assert buf.getvalue().lstrip().startswith("<?xml") and "<svg" in buf.getvalue()


def test_a_session_without_an_index_draws_what_it_has():
    st = tui.new_state(ConsoleServer(None, None, http=False))
    st["fabric"] = _fabric_doc()
    st["fabric_hist"] = fabric_view.History()
    text = _texts(_draw(st))
    assert "no index attached" in text and "NATS 1 leaf" in text
