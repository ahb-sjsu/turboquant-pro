"""Exact multiple-choice knapsack for weight bit allocation (weight_plan)."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from turboquant_pro import weight_plan as W
from turboquant_pro.cli import main as cli_main
from turboquant_pro.schemas import load_schema

COST_TABLE_SCHEMA = "weight_cost_table.schema.json"
PLAN_SCHEMA = "weight_plan.schema.json"


def _validator(jsonschema, name: str):
    """The shipped schema, checked as a schema before it is used to check data."""
    schema = load_schema(name)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _table(seed: int, n: int = 5, levels=(2, 3, 4, 8)) -> W.CostTable:
    rng = np.random.default_rng(seed)
    mats, costs = {}, {}
    for i in range(n):
        numel = 128 * int(rng.integers(1, 12))
        mats[f"m{i}"] = {"numel": numel, "group": 128}
        scale = float(rng.lognormal(0, 1.5))
        # decreasing in bits, like real damage, plus noise so no structure is assumed
        costs[f"m{i}"] = {
            b: scale * 4.0 ** (-b) * float(rng.uniform(0.5, 2)) for b in levels
        }
    return W.CostTable("toy", "random", mats, costs, {"seed": seed})


def _brute(table: W.CostTable, budget: int):
    names = list(table.costs)
    best = None
    for combo in itertools.product(*(sorted(table.costs[n]) for n in names)):
        size = sum(table.size(n, b) for n, b in zip(names, combo))
        if size > budget:
            continue
        c = sum(table.costs[n][b] for n, b in zip(names, combo))
        if best is None or c < best:
            best = c
    return best


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force_and_the_dual_bounds_it(seed):
    t = _table(seed)
    lo = sum(min(t.size(n, b) for b in t.costs[n]) for n in t.costs)
    hi = sum(max(t.size(n, b) for b in t.costs[n]) for n in t.costs)
    for budget in np.linspace(lo, hi, 7).astype(int):
        p = W.solve(t, int(budget))
        assert p.cost == pytest.approx(_brute(t, int(budget)), rel=1e-12)
        assert p.stored_bits <= budget
        assert p.stored_bits == sum(t.size(n, b) for n, b in p.bits.items())
        assert p.dual_bound <= p.cost * (1 + 1e-9) + 1e-15
        assert p.gap >= -1e-12


def test_more_budget_never_costs_more_and_the_top_buys_the_cheapest_everywhere():
    t = _table(3, n=8)
    hi = sum(max(t.size(n, b) for b in t.costs[n]) for n in t.costs)
    costs = [W.solve(t, int(b)).cost for b in np.linspace(hi * 0.4, hi, 9)]
    assert all(a >= b - 1e-15 for a, b in zip(costs, costs[1:]))
    full = W.solve(t, hi)
    assert full.cost == pytest.approx(sum(min(d.values()) for d in t.costs.values()))


def test_uniform_rate_budget_admits_the_uniform_plan_exactly():
    t = _table(5, n=6, levels=(3, 4, 5, 6, 8))
    budget = W.budget_for_rate(t, 4)
    assert budget == sum(t.size(n, 4) for n in t.costs)
    p = W.solve(t, budget)
    assert p.cost <= sum(t.costs[n][4] for n in t.costs) + 1e-15


def test_refusals_infeasible_oversized_lattice_and_bad_tables():
    t = _table(1)
    with pytest.raises(ValueError, match="infeasible"):
        W.solve(t, 10)
    hi = sum(max(t.size(n, b) for b in t.costs[n]) for n in t.costs)
    with pytest.raises(ValueError, match="refusing to approximate"):
        W.solve(t, hi, max_states=3)
    with pytest.raises(ValueError, match="negative or not finite"):
        W.CostTable("x", "p", {"a": {"numel": 128}}, {"a": {4: float("nan")}})
    with pytest.raises(ValueError, match="different sets"):
        W.CostTable("x", "p", {"a": {"numel": 128}}, {"b": {4: 1.0}})


def test_provenance_hash_travels_and_foreign_costs_are_refused():
    t = _table(2)
    p = W.solve(t, W.budget_for_rate(t, 4)).as_dict()
    assert p["cost_table_hash"] == t.content_hash()
    W.check_matrices(p, t)
    doc = t.as_dict()
    doc["costs"]["m0"]["4"] *= 2
    other = W.CostTable.from_dict(json.loads(json.dumps(doc)))
    assert other.content_hash() != t.content_hash()
    with pytest.raises(ValueError, match="different cost table"):
        W.check_matrices(p, other)


@pytest.mark.parametrize("seed", range(4))
def test_every_cost_table_this_file_builds_matches_its_schema(seed):
    jsonschema = pytest.importorskip("jsonschema")
    v = _validator(jsonschema, COST_TABLE_SCHEMA)
    v.validate(_table(seed).as_dict())
    v.validate(_table(seed, n=6, levels=(3, 4, 5, 6, 8)).as_dict())


def test_cost_table_schema_rejects_missing_required_field():
    jsonschema = pytest.importorskip("jsonschema")
    v = _validator(jsonschema, COST_TABLE_SCHEMA)
    doc = _table(0).as_dict()
    v.validate(doc)

    del doc["predictor"]
    assert not v.is_valid(doc)


def test_cost_table_schema_rejects_malformed_shapes():
    """A bad file must fail at the door, not deep inside the solver."""
    jsonschema = pytest.importorskip("jsonschema")
    v = _validator(jsonschema, COST_TABLE_SCHEMA)

    doc = _table(0).as_dict()
    doc["costs"]["m0"]["not-a-bit-width"] = 1.0
    assert not v.is_valid(doc)

    doc = _table(0).as_dict()
    doc["costs"]["m0"]["4"] = -1.0
    assert not v.is_valid(doc)

    doc = _table(0).as_dict()
    doc["costs"]["m0"] = {}
    assert not v.is_valid(doc)

    doc = _table(0).as_dict()
    doc["matrices"]["m0"]["numel"] = 0
    assert not v.is_valid(doc)

    doc = _table(0).as_dict()
    doc["schema"] = "tqp.weight_cost_table/2"
    assert not v.is_valid(doc)


def test_cli_plan_weights_round_trip(tmp_path, capsys):
    t = _table(4, n=6, levels=(3, 4, 5, 6, 8))
    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(t.as_dict()))
    out = tmp_path / "plan.json"

    rc = cli_main(
        ["plan", "weights", "--costs", str(cp), "--bits-per-weight", "4.5"]
        + ["--out", str(out)]
    )

    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == W.PLAN_SCHEMA
    assert doc["cost_table_hash"] == t.content_hash()
    assert doc["stored_bits"] <= doc["budget_bits"] == W.budget_for_rate(t, 4.5)
    assert "dual bound" in capsys.readouterr().out
    assert cli_main(["plan", "weights", "--costs", str(cp)]) == 2


def test_pin_restricts_one_matrix():
    table = _table(0)

    name = "m0"
    bits = 8

    pinned = W.pin(table, name, bits)

    assert pinned.costs[name] == {bits: table.costs[name][bits]}

    for other_name in table.costs:
        if other_name != name:
            assert pinned.costs[other_name] == table.costs[other_name]

    assert pinned.content_hash() != table.content_hash()


def test_pinned_matrix_gets_exact_width_and_rest_stays_optimal():
    table = _table(1, n=4)
    pinned = W.pin(table, "m0", 8)

    budget = sum(pinned.size(n, min(pinned.costs[n])) for n in pinned.costs)

    plan = W.solve(pinned, budget)

    assert plan.bits["m0"] == 8
    assert plan.cost == pytest.approx(_brute(pinned, budget), rel=1e-12)
    assert plan.stored_bits <= budget


def test_cli_pin_records_pins_and_pinned_hash(tmp_path):
    table = _table(4, n=6, levels=(3, 4, 5, 6, 8))

    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(table.as_dict()))

    out = tmp_path / "plan.json"

    rc = cli_main(
        [
            "plan",
            "weights",
            "--costs",
            str(cp),
            "--bits-per-weight",
            "4.5",
            "--pin",
            "m0=8",
            "--out",
            str(out),
        ]
    )

    assert rc == 0

    doc = json.loads(out.read_text())

    assert doc["pins"] == {"m0": 8}
    assert doc["bits"]["m0"] == 8

    pinned = W.pin(table, "m0", 8)
    assert doc["cost_table_hash"] == pinned.content_hash()

    assert doc["stored_bits"] <= doc["budget_bits"]


def test_cli_pin_supports_fnmatch(tmp_path):
    table = _table(5, n=4, levels=(3, 4, 5, 6, 8))

    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(table.as_dict()))

    out = tmp_path / "plan.json"

    rc = cli_main(
        [
            "plan",
            "weights",
            "--costs",
            str(cp),
            "--bits-per-weight",
            "8",
            "--pin",
            "m[01]=8",
            "--out",
            str(out),
        ]
    )

    assert rc == 0

    doc = json.loads(out.read_text())

    assert doc["pins"] == {"m0": 8, "m1": 8}
    assert doc["bits"]["m0"] == 8
    assert doc["bits"]["m1"] == 8


def test_cli_bad_pin_exits_2(tmp_path, capsys):
    table = _table(6, n=4, levels=(3, 4, 5, 6, 8))

    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(table.as_dict()))

    rc = cli_main(
        [
            "plan",
            "weights",
            "--costs",
            str(cp),
            "--bits-per-weight",
            "4.5",
            "--pin",
            "does_not_exist=8",
        ]
    )

    assert rc == 2
    assert "matched no matrix" in capsys.readouterr().err


def test_cli_pin_rejects_unoffered_bits(tmp_path, capsys):
    table = _table(7, n=4, levels=(3, 4, 5, 6, 8))

    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(table.as_dict()))

    rc = cli_main(
        [
            "plan",
            "weights",
            "--costs",
            str(cp),
            "--bits-per-weight",
            "4.5",
            "--pin",
            "m0=7",
        ]
    )

    assert rc == 2

    output = capsys.readouterr().err
    assert "does not offer 7 bits" in output
    assert "available" in output


def test_cli_pin_conflict_names_both_pins_not_a_false_availability(tmp_path, capsys):
    """#237: a narrower first pin must not make a later pin look unoffered."""
    table = _table(9, n=3, levels=(3, 4, 6, 8))

    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(table.as_dict()))

    rc = cli_main(
        [
            "plan",
            "weights",
            "--costs",
            str(cp),
            "--bits-per-weight",
            "4.5",
            "--pin",
            "m0=8",
            "--pin",
            "m*=4",
        ]
    )

    assert rc == 2

    err = capsys.readouterr().err
    assert "m0=8" in err
    assert "m*=4" in err
    assert "conflict" in err
    # 4 bits *is* offered by m0, so the availability wording would be a lie here
    assert "does not offer" not in err


def test_cli_pin_same_width_twice_over_an_overlap_still_succeeds(tmp_path):
    """#237: two pins agreeing on a width are not a conflict, even via fnmatch."""
    table = _table(9, n=3, levels=(3, 4, 6, 8))

    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(table.as_dict()))

    out = tmp_path / "plan.json"

    rc = cli_main(
        [
            "plan",
            "weights",
            "--costs",
            str(cp),
            "--bytes",
            "1000000",
            "--pin",
            "m0=8",
            "--pin",
            "m*=8",
            "--out",
            str(out),
        ]
    )

    assert rc == 0

    doc = json.loads(out.read_text())
    assert doc["pins"] == {"m0": 8, "m1": 8, "m2": 8}
    assert doc["bits"] == {"m0": 8, "m1": 8, "m2": 8}


def test_every_plan_the_cli_writes_matches_its_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    v = _validator(jsonschema, PLAN_SCHEMA)
    t = _table(4, n=6, levels=(3, 4, 5, 6, 8))
    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(t.as_dict()))
    for argv in (
        ["--bits-per-weight", "4.5"],
        ["--bits-per-weight", "3.2"],
        ["--bytes", str(W.budget_for_rate(t, 4) // 8)],
    ):
        out = tmp_path / "plan.json"
        assert (
            cli_main(
                ["plan", "weights", "--costs", str(cp)] + argv + ["--out", str(out)]
            )
            == 0
        )
        doc = json.loads(out.read_text())
        v.validate(doc)
        # The budget invariant the schema cannot state (it compares two fields).
        assert doc["stored_bits"] <= doc["budget_bits"]
        assert doc["duality_gap"] >= -1e-9


def test_plan_schema_rejects_missing_required_field(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    v = _validator(jsonschema, PLAN_SCHEMA)
    t = _table(4, n=6, levels=(3, 4, 5, 6, 8))
    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(t.as_dict()))
    out = tmp_path / "plan.json"
    assert (
        cli_main(
            [
                "plan",
                "weights",
                "--costs",
                str(cp),
                "--bits-per-weight",
                "4",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    doc = json.loads(out.read_text())
    v.validate(doc)

    del doc["dual_bound"]
    assert not v.is_valid(doc)


def test_plan_schema_rejects_malformed_fields(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    v = _validator(jsonschema, PLAN_SCHEMA)
    t = _table(4, n=6, levels=(3, 4, 5, 6, 8))
    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(t.as_dict()))
    out = tmp_path / "plan.json"
    assert (
        cli_main(
            [
                "plan",
                "weights",
                "--costs",
                str(cp),
                "--bits-per-weight",
                "4",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    good = json.loads(out.read_text())
    v.validate(good)

    doc = json.loads(out.read_text())
    doc["cost_table_hash"] = "not-a-sha256"
    assert not v.is_valid(doc)

    doc = json.loads(out.read_text())
    doc["bits"]["m0"] = 0
    assert not v.is_valid(doc)

    doc = json.loads(out.read_text())
    doc["schema"] = "tqp.weight_plan/2"
    assert not v.is_valid(doc)

    doc = json.loads(out.read_text())
    doc["solver"] = "greedy"
    assert not v.is_valid(doc)


def test_a_pinned_plan_matches_the_plan_schema(tmp_path):
    """#233 (--pin) adds `pins` to the plan; #232's schema must accept it."""
    jsonschema = pytest.importorskip("jsonschema")
    table = _table(7, n=5, levels=(3, 4, 5, 6, 8))
    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(table.as_dict()))
    out = tmp_path / "plan.json"
    rc = cli_main(
        [
            "plan",
            "weights",
            "--costs",
            str(cp),
            "--bits-per-weight",
            "5",
            "--pin",
            "m0=8",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    doc = json.loads(out.read_text())
    jsonschema.validate(doc, load_schema("weight_plan.schema.json"))
    doc["pins"]["m0"] = "eight"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, load_schema("weight_plan.schema.json"))
