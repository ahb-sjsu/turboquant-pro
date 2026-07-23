"""Schema freeze for tqp-strata-report/1: recompute vs committed goldens.

The golden files pin field names, verdicts, classes, causes, and values
(floats to 3 decimals). Changing the writer so these differ is a FORMAT
BREAK: either fix the regression or, for an intentional schema revision,
regenerate via tests/golden/strata/generate.py and treat it as an API
change (closed registry — additions only, never renames).
"""

import importlib.util
import json
import sys
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden" / "strata"


def _build():
    spec = importlib.util.spec_from_file_location(
        "strata_golden_generate", GOLDEN / "generate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strata_golden_generate"] = mod
    spec.loader.exec_module(mod)
    return mod.build()


def _diff(path, got, want, out):
    if isinstance(want, dict):
        if set(got) != set(want):
            out.append(f"{path}: keys {sorted(set(got) ^ set(want))} differ")
            return
        for k in want:
            _diff(f"{path}.{k}", got[k], want[k], out)
    elif isinstance(want, list):
        if len(got) != len(want):
            out.append(f"{path}: length {len(got)} != {len(want)}")
            return
        for i, (g, w) in enumerate(zip(got, want)):
            _diff(f"{path}[{i}]", g, w, out)
    elif got != want:
        out.append(f"{path}: {got!r} != {want!r}")


def test_strata_reports_match_goldens():
    anatomy, hubdiff = _build()
    problems: list[str] = []
    for name, got in (("anatomy_report", anatomy), ("hubdiff_report", hubdiff)):
        want = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
        _diff(name, got, want, problems)
    assert not problems, "format break vs golden strata reports:\n" + "\n".join(
        problems[:40]
    )
