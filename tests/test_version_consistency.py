# TurboQuant Pro: version string must agree across package, CHANGELOG, README.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Guard against version drift.

``__version__`` is single-sourced from ``turboquant_pro/__init__.py`` (hatch
reads it), but the prose caught up late more than once: the README said
2.0.0a2 for a week after the bump to a3, and the CHANGELOG had two
``## Unreleased`` sections and no a3 header. These checks make a bump that
forgets the prose fail CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import turboquant_pro

REPO = Path(__file__).resolve().parents[1]
VERSION = turboquant_pro.__version__


def test_changelog_has_a_header_naming_the_current_version():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    headers = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert any(VERSION in h for h in headers), (
        f"No '## ' header in CHANGELOG.md mentions {VERSION}; "
        f"top headers: {headers[:3]}"
    )


def test_changelog_has_at_most_one_unreleased_section():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    n = len(re.findall(r"^## Unreleased", text, flags=re.M))
    assert n <= 1, f"CHANGELOG.md has {n} '## Unreleased' sections"


def test_readme_names_the_current_version():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert VERSION in text, f"README.md does not mention {VERSION}"
