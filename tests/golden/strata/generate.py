"""Regenerate the tqp-strata-report/1 golden fixtures.

Run from the repo root:  python tests/golden/strata/generate.py

The goldens freeze the report SCHEMA and semantics: field names, verdicts,
classes, causes exactly; floats rounded to 3 decimals (cross-BLAS safety).
``software_version`` is masked at generation time — a version bump is not a
format change. A writer change that alters these files is a format break
and fails CI (docs/STRATA_RFC.md §2.2 phase gate).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_strata import _map_of, _strata_corpus  # noqa: E402

from turboquant_pro.anatomy import knn_exact  # noqa: E402
from turboquant_pro.strata import (  # noqa: E402
    stratified_anatomy,
    stratified_hub_differential,
)


def normalize(obj):
    if isinstance(obj, dict):
        return {
            k: ("<masked>" if k == "software_version" else normalize(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    return obj


def build():
    x, labels = _strata_corpus(seed=0)
    amap = _map_of(x, labels)
    anatomy = stratified_anatomy(x, amap, k=10, n_min=50, q_min=50, seed=0)

    rng = np.random.default_rng(3)
    _, exact = knn_exact(x, x, 10, exclude_self=True)
    counts = np.bincount(exact.ravel(), minlength=len(x))
    anti = counts <= np.quantile(counts, 0.10)
    approx = exact.copy()
    qlab = np.asarray(labels)
    for i in range(len(x)):
        if qlab[i] == "D" and anti[exact[i, 0]]:
            approx[i] = rng.integers(0, len(x), size=10)
    hubdiff = stratified_hub_differential(
        exact,
        approx,
        len(x),
        amap,
        labels,
        k=10,
        min_anti_recall=0.9,
        n_min=50,
        q_min=50,
    )
    return normalize(anatomy), normalize(hubdiff)


if __name__ == "__main__":
    here = Path(__file__).parent
    anatomy, hubdiff = build()
    (here / "anatomy_report.json").write_text(
        json.dumps(anatomy, indent=1, sort_keys=True), encoding="utf-8"
    )
    (here / "hubdiff_report.json").write_text(
        json.dumps(hubdiff, indent=1, sort_keys=True), encoding="utf-8"
    )
    print("wrote", here / "anatomy_report.json")
    print("wrote", here / "hubdiff_report.json")
