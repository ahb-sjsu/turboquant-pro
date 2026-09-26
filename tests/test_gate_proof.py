"""Proof, by exhaustive enumeration of finite domains, that Part II's gate machinery
(``benchmarks/kvquant_matrix/score_keys.py``) meets its specification.

The machinery is split so that proof is possible:

- ``classify`` reads a disposition's content and returns one of four KINDS. It holds
  the only floating-point comparison (``pins``); that part is tested below, not proven.
- ``decide(passed, kinds)`` sees only the gate's outcome and WHICH kinds exist. Its
  domain is {None, True, False} x the 16 subsets of KINDS, 48 points, and every
  requirement is checked at every point: a proof, not a sample.
- ``verdict_status`` is a maximum over a rank, so it depends only on the set of
  statuses present; every nonempty set of the five statuses (31) is checked, and
  every placement of them over one and two gates of three models.

The requirements are stated as properties, not as a second copy of the code.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix")
)
import score_keys as SK  # noqa: E402

STATUSES = (
    SK.PASS,
    SK.PENDING,
    SK.FAIL_UNEXPLAINED,
    SK.FAIL_EXPLAINED,
    SK.FAIL_EXPLAINED_POSTHOC,
)
EXPLAINED = {SK.FAIL_EXPLAINED, SK.FAIL_EXPLAINED_POSTHOC}
EXPLAINING = {SK.EXPLAINS, SK.EXPLAINS_POSTHOC}
INERT = {SK.IRRELEVANT, SK.OTHER_NUMBERS}


def subsets(xs):
    xs = list(xs)
    return [
        frozenset(c) for r in range(len(xs) + 1) for c in itertools.combinations(xs, r)
    ]


KIND_SETS = subsets(SK.KINDS)
DOMAIN = [(p, k) for p in (None, True, False) for k in KIND_SETS]


def test_the_domain_is_complete():
    assert len(KIND_SETS) == 16 and len(DOMAIN) == 48


@pytest.mark.parametrize("p,kinds", DOMAIN)
def test_decide_meets_every_requirement_at_every_point(p, kinds):
    st = SK.decide(p, kinds)
    # R0 total: every point has exactly one of the five statuses
    assert st in STATUSES
    # R1 PENDING exactly when the gate could not be evaluated
    assert (st == SK.PENDING) == (p is None)
    # R2 PASS exactly when the gate passed; a disposition never changes a pass
    assert (st == SK.PASS) == (p is True)
    if p is False:
        # R3 a failure is explained exactly when some disposition pins these numbers
        assert (st in EXPLAINED) == bool(kinds & EXPLAINING)
        # R4 a post-hoc explanation is never reported as a prior one
        assert (st == SK.FAIL_EXPLAINED_POSTHOC) == (SK.EXPLAINS_POSTHOC in kinds)
        # R5 inert dispositions (other gate or model, other numbers) change nothing
        for extra in subsets(INERT):
            assert SK.decide(p, kinds | extra) == st
        # R6 adding a disposition never un-explains a failure
        for more in KIND_SETS:
            if st in EXPLAINED:
                assert SK.decide(p, kinds | more) in EXPLAINED


def test_decide_refuses_what_is_outside_its_domain():
    with pytest.raises(ValueError, match="passed"):
        SK.decide(0, frozenset())
    with pytest.raises(ValueError, match="kinds"):
        SK.decide(False, frozenset({"EXPLAINS_SOMEHOW"}))


# One concrete disposition of each kind, for a fixed gate result.
GATE, MODEL = "G1", "mistral-7b-instruct"
OBS = {"qasper": 29.8, "ppl": 5.95}
REP = {
    SK.IRRELEVANT: {
        "gate": "G1",
        "model": "llama2-7b-chat-4k",
        "observed": dict(OBS),
        "amendment": "A-irrelevant",
        "after_verdicts": True,
    },
    SK.OTHER_NUMBERS: {
        "gate": GATE,
        "model": MODEL,
        "observed": {"qasper": 28.0, "ppl": 5.95},
        "amendment": "A-other",
        "after_verdicts": True,
    },
    SK.EXPLAINS: {
        "gate": GATE,
        "model": MODEL,
        "observed": dict(OBS),
        "amendment": "A-prior",
        "after_verdicts": False,
    },
    SK.EXPLAINS_POSTHOC: {
        "gate": GATE,
        "model": MODEL,
        "observed": dict(OBS),
        "amendment": "A-posthoc",
        "after_verdicts": True,
    },
}


def test_classify_puts_each_representative_in_its_kind():
    for kind, d in REP.items():
        assert SK.classify(d, GATE, MODEL, OBS) == kind


@pytest.mark.parametrize("p", (None, True, False))
def test_gate_status_is_decide_over_every_list_of_dispositions(p):
    """Every list of up to four dispositions drawn from the representatives (341
    lists, repeats and every order included): the status is ``decide`` over the set
    of kinds, so order and multiplicity cannot matter."""
    n = 0
    for r in range(5):
        for seq in itertools.product(SK.KINDS, repeat=r):
            got = SK.gate_status(GATE, MODEL, p, OBS, [REP[k] for k in seq])
            assert got["status"] == SK.decide(p, frozenset(seq))
            n += 1
    assert n == 341


def test_pins_is_exact_on_keys_and_missing_values():
    assert SK.pins({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})
    assert SK.pins({"a": 1.0}, {"a": 1.0 + 5e-10})
    assert not SK.pins({"a": 1.0}, {"a": 1.0 + 2e-9})
    assert not SK.pins({"a": 1.0}, {"a": 1.0, "b": 2.0})  # observed has more
    assert not SK.pins({"a": 1.0, "b": 2.0}, {"a": 1.0})  # pinned has more
    assert not SK.pins({"a": None}, {"a": None})
    assert not SK.pins({"a": 1.0}, {"a": None})


STATUS_SETS = [s for s in subsets(STATUSES) if s]


@pytest.mark.parametrize("present", STATUS_SETS)
def test_verdict_status_meets_every_requirement_for_every_set(present):
    gates = {"G": {f"m{i}": {"status": s} for i, s in enumerate(present)}}
    v = SK.verdict_status(gates)
    # V1..V4: each outcome exactly when its condition holds
    assert (v == "WITHHELD") == (SK.FAIL_UNEXPLAINED in present)
    assert (v == "PROVISIONAL") == (
        SK.FAIL_UNEXPLAINED not in present and SK.PENDING in present
    )
    assert (v == "FINAL_WITH_POSTHOC_EXPLANATION") == (
        not present & {SK.FAIL_UNEXPLAINED, SK.PENDING}
        and SK.FAIL_EXPLAINED_POSTHOC in present
    )
    assert (v == "FINAL") == (present <= {SK.PASS, SK.FAIL_EXPLAINED})
    # V5 monotone: adding any gate result never makes the verdicts more final
    for more in STATUS_SETS:
        w = SK.verdict_status(
            {"G": {f"m{i}": {"status": s} for i, s in enumerate(present | more)}}
        )
        assert SK.VERDICTS.index(w) >= SK.VERDICTS.index(v)


def test_verdict_status_depends_only_on_the_set_present():
    """Every placement of the five statuses over two gates of three models (15,625):
    the verdict status equals that of the set of statuses present."""
    by_set = {
        s: SK.verdict_status({"G": {f"m{i}": {"status": x} for i, x in enumerate(s)}})
        for s in STATUS_SETS
    }
    for a in itertools.product(STATUSES, repeat=6):
        gates = {
            "G1": {f"m{i}": {"status": a[i]} for i in range(3)},
            "G2": {f"m{i}": {"status": a[3 + i]} for i in range(3)},
        }
        assert SK.verdict_status(gates) == by_set[frozenset(a)]


def test_a_report_that_checked_nothing_has_no_status():
    with pytest.raises(ValueError, match="checked nothing"):
        SK.verdict_status({})
    with pytest.raises(ValueError, match="checked nothing"):
        SK.verdict_status({"G1": {}})
