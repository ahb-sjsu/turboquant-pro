"""Exact mixed-precision planning for weight matrices: a multiple-choice knapsack.

Each weight matrix ``m`` is one item group, and its quantization choices
``b in B_m`` are the items. Choosing exactly one per matrix,

    minimize    sum_m C_m(b_m)
    subject to  sum_m S_m(b_m) <= B,      b_m in B_m,

where ``C_m(b)`` is a measured cost table (a behaviour-damage estimate, such as
the diagonal weight Fisher along the quantization error or a measured
single-matrix KL) and ``S_m(b)`` is the stored size in bits: codes plus the
per-group scale and offset.

The problem is solved EXACTLY. Stored sizes are integers, so dividing every
size by their greatest common divisor puts the budget on a small integer
lattice (tens of thousands of points for a 1.5B-parameter decoder, because a
transformer has only a handful of matrix shapes), and dynamic programming over
that lattice is exact. An instance whose lattice exceeds ``max_states`` is
refused, never approximated. Any deficit of an exactly planned model therefore
belongs to the cost model, not to the optimizer.

Every plan also carries the Lagrangian dual bound

    max_{l >= 0} [ sum_m min_b (C_m(b) + l S_m(b)) - l B ],

a lower bound on the optimum computed independently of the dynamic program.
The reported gap (optimum minus bound) is the duality gap of the discrete
problem and is never negative for a correct solve.

The cost table carries its provenance (model, predictor, calibration) and a
content hash. A plan records the hash, so it cannot silently be applied with
costs from a different model or predictor (``check_matrices``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import reduce

import numpy as np

GROUP = 128
SCALE_BITS = 32  # two fp16 per group: scale and offset
SCHEMA = "tqp.weight_cost_table/1"
PLAN_SCHEMA = "tqp.weight_plan/1"


def stored_bits(numel: int, bits: int, group: int = GROUP) -> int:
    """Codes plus two fp16 per group of ``group`` weights: the stored size."""
    return numel * bits + (numel // group) * SCALE_BITS


@dataclass
class CostTable:
    """``costs[name][bits]`` per matrix, with the sizes and origin needed."""

    model: str
    predictor: str
    matrices: dict  # name -> {"numel": int, "group": int}
    costs: dict  # name -> {bits(int): float}
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.costs = {
            n: {int(b): float(c) for b, c in d.items()} for n, d in self.costs.items()
        }
        if set(self.costs) != set(self.matrices):
            raise ValueError("costs and matrices name different sets of matrices")
        for n, d in self.costs.items():
            if not d:
                raise ValueError(f"{n}: no choices")
            bad = [b for b, c in d.items() if not math.isfinite(c) or c < 0]
            if bad:
                raise ValueError(f"{n}: cost at bits {bad} is negative or not finite")

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "model": self.model,
            "predictor": self.predictor,
            "matrices": self.matrices,
            "costs": {
                n: {str(b): c for b, c in sorted(d.items())}
                for n, d in self.costs.items()
            },
            "provenance": self.provenance,
        }

    def content_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> CostTable:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} document")
        return cls(
            d["model"],
            d["predictor"],
            d["matrices"],
            d["costs"],
            d.get("provenance", {}),
        )

    def size(self, name: str, bits: int) -> int:
        m = self.matrices[name]
        return stored_bits(int(m["numel"]), bits, int(m.get("group", GROUP)))


def pin(table: CostTable, name: str, bits: int) -> CostTable:
    """Return a copy of ``table`` with one matrix restricted to ``bits``."""
    costs = {n: dict(choices) for n, choices in table.costs.items()}
    costs[name] = {bits: costs[name][bits]}

    return CostTable(
        table.model,
        table.predictor,
        dict(table.matrices),
        costs,
        dict(table.provenance),
    )


@dataclass
class WeightPlan:
    bits: dict  # name -> chosen bits
    cost: float
    stored_bits: int
    budget_bits: int
    dual_bound: float
    lattice_unit: int
    lattice_states: int
    cost_table_hash: str
    model: str
    predictor: str

    @property
    def gap(self) -> float:
        return self.cost - self.dual_bound

    def as_dict(self) -> dict:
        return {
            "schema": PLAN_SCHEMA,
            "solver": "exact multiple-choice knapsack (DP on the gcd lattice)",
            "model": self.model,
            "predictor": self.predictor,
            "cost_table_hash": self.cost_table_hash,
            "budget_bits": self.budget_bits,
            "stored_bits": self.stored_bits,
            "cost": self.cost,
            "dual_bound": self.dual_bound,
            "duality_gap": self.gap,
            "lattice_unit_bits": self.lattice_unit,
            "lattice_states": self.lattice_states,
            "matrices": len(self.bits),
            "bits": dict(sorted(self.bits.items())),
        }


def dual_bound(table: CostTable, budget_bits: int) -> float:
    """Lagrangian lower bound on the optimum.

    The dual ``g(l) = sum_m min_b (C_m(b) + l S_m(b)) - l B`` is concave and
    piecewise linear in ``l``. Its breakpoints are where a matrix's cheapest
    choice changes, at the pairwise slopes ``(C_j - C_i) / (S_i - S_j)``, so its
    maximum over ``l >= 0`` sits at ``l = 0`` or at one of them: evaluating
    those points is exact.
    """
    names = list(table.costs)
    C = [np.array(list(table.costs[n].values())) for n in names]
    S = [np.array([table.size(n, b) for b in table.costs[n]], float) for n in names]

    def g(lam):
        return sum(float((c + lam * s).min()) for c, s in zip(C, S)) - lam * budget_bits

    points = {0.0}
    for c, s in zip(C, S):
        ds = s[:, None] - s[None, :]
        dc = c[None, :] - c[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = dc / ds
        points.update(float(x) for x in sl[(ds != 0) & (sl > 0)].ravel())
    return max(g(lam) for lam in points)


def solve(
    table: CostTable, budget_bits: int, max_states: int = 5_000_000
) -> WeightPlan:
    """The exact minimum-cost choice per matrix within ``budget_bits`` stored."""
    names = list(table.costs)
    choices = [sorted(table.costs[n]) for n in names]
    sizes = [[table.size(n, b) for b in ch] for n, ch in zip(names, choices)]
    unit = reduce(math.gcd, (s for row in sizes for s in row))
    units = [np.array([s // unit for s in row], dtype=np.int64) for row in sizes]
    floor = sum(int(u.min()) for u in units)
    cap = budget_bits // unit
    if cap < floor:
        need = floor * unit
        raise ValueError(
            f"infeasible: the smallest choice for every matrix needs {need} bits"
        )
    top = sum(int(u.max()) for u in units)
    cap = min(cap, top)
    if cap + 1 > max_states:
        raise ValueError(
            f"lattice of {cap + 1} states exceeds max_states={max_states}; "
            "refusing to approximate"
        )
    inf = np.inf
    dp = np.full(cap + 1, inf)
    dp[0] = 0.0
    arg = np.zeros((len(names), cap + 1), dtype=np.int16)
    for m, (u, ch) in enumerate(zip(units, choices)):
        cost = np.array([table.costs[names[m]][b] for b in ch])
        new = np.full(cap + 1, inf)
        for k in range(len(ch)):
            if u[k] > cap:
                continue
            cand = np.full(cap + 1, inf)
            cand[u[k] :] = dp[: cap + 1 - u[k]] + cost[k]
            better = cand < new
            new[better] = cand[better]
            arg[m, better] = k
        dp = new
    end = int(np.argmin(dp))
    if not math.isfinite(dp[end]):
        raise ValueError("infeasible budget")
    bits, pos = {}, end
    for m in range(len(names) - 1, -1, -1):
        k = int(arg[m, pos])
        bits[names[m]] = choices[m][k]
        pos -= int(units[m][k])
    assert pos == 0
    total = sum(table.size(n, b) for n, b in bits.items())
    cost = float(sum(table.costs[n][b] for n, b in bits.items()))
    return WeightPlan(
        bits=bits,
        cost=cost,
        stored_bits=int(total),
        budget_bits=int(budget_bits),
        dual_bound=dual_bound(table, budget_bits),
        lattice_unit=int(unit),
        lattice_states=int(cap + 1),
        cost_table_hash=table.content_hash(),
        model=table.model,
        predictor=table.predictor,
    )


def budget_for_rate(table: CostTable, bits_per_weight: float) -> int:
    """The stored-bit budget at ``bits_per_weight`` code bits per weight, plus
    the per-group overhead every choice pays, so rates compare across plans."""
    total = 0
    for m in table.matrices.values():
        n, g = int(m["numel"]), int(m.get("group", GROUP))
        total += n * bits_per_weight + (n // g) * SCALE_BITS
    return int(math.floor(total))


def check_matrices(plan: dict, table: CostTable) -> None:
    """Refuse a plan whose cost table is not this one (a foreign origin)."""
    if plan.get("cost_table_hash") != table.content_hash():
        raise ValueError("plan was made from a different cost table")
