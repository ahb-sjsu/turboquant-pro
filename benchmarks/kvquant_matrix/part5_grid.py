"""The grid of observer-advantage Part V (docs/PREREG_full_consumer_kv.md).

One source of truth for Part V's runner and scorer, built on Part II's registered grid
(``keys_grid``), which it imports and never changes: the shipped recipe, the models, the
tasks and the arm encoding are Part II's. Every arm is an env line for
``tq_paper_lb_shard.py`` / ``wikitext_ppl.py``; every comparison an (arm, reference,
floor arm) triple, the floor being |mean(floor) - mean(reference)|. Nothing here is fitted.

The baselines of section 3 (``aatc``, ``aatc_native``, ``ea``, ``ea_u3``) are not env lines
of this harness; they are named in BASELINES and run by their own runners.
"""

from __future__ import annotations

from keys_grid import (  # noqa: F401
    G0_TASKS,
    MODELS,
    PPL,
    SHIPPED,
    TASKS,
    TIER_A,
    TIER_B,
    _arm,
)

ARMS: dict[str, str] = {"fp16": "NOQUANT=1"}
for b in (3, 2, 4):
    ARMS.update(
        {
            f"u{b}": _arm("uniform", b),
            f"u{b}_jit": _arm("uniform", b, KEY_JITTER=1),
            f"u{b}_key": _arm("uniform", b, KEY_ALLOC="key"),
            f"u{b}_read": _arm("uniform", b, KEY_ALLOC="read"),
            f"u{b}_head": _arm("uniform", b, KEY_ALLOC="head"),
            f"u{b}_head_cal": _arm("uniform", b, KEY_ALLOC="head", BASIS_FIT="calib"),
            f"u{b}_fisher_cal": _arm(
                "uniform", b, KEY_ALLOC="fisher", BASIS_FIT="calib"
            ),
            f"u{b}_read_cal": _arm("uniform", b, KEY_ALLOC="read", BASIS_FIT="calib"),
        }
    )

# G1: u3 and u3_jit run twice (section 2); a repeat is its own cell with the same env
# line, so the runner's resume never mistakes it for the first run.
G1_REPEATS = {"u3_rep": "u3", "u3_jit_rep": "u3_jit"}
ARMS.update({rep: ARMS[first] for rep, first in G1_REPEATS.items()})

# G0: the identity codebook through each allocation path must reproduce g0_native.
G0_ARMS = {
    "g0_native": _arm("identity", 3),
    "g0_read": _arm("identity", 3, KEY_ALLOC="read"),
    "g0_head": _arm("identity", 3, KEY_ALLOC="head"),
    "g0_head_cal": _arm("identity", 3, KEY_ALLOC="head", BASIS_FIT="calib"),
    "g0_fisher_cal": _arm("identity", 3, KEY_ALLOC="fisher", BASIS_FIT="calib"),
    "g0_key": _arm("identity", 3, KEY_ALLOC="key"),  # reported arms' paths
    "g0_read_cal": _arm("identity", 3, KEY_ALLOC="read", BASIS_FIT="calib"),
}


def g0_gate(arm: str) -> str:
    """The G0 arm that gates ``arm``: its allocation and fit through the identity
    codebook (uniform arms, the jitter floor included, by g0_native)."""
    env = dict(kv.split("=", 1) for kv in {**ARMS, **G0_ARMS}[arm].split())
    alloc = env.get("KEY_ALLOC", "uniform")
    if alloc == "uniform":
        return "g0_native"
    return f"g0_{alloc}" + ("_cal" if env.get("BASIS_FIT") == "calib" else "")


# Registered calibration (section 1); the scorer checks every calibrated cell's record.
CALIB_SEQ, CALIB_N, FISHER_SEED = 1024, 16, 20261013

# G1's recorded numbers: Part II's scored u3 cells, read from Part II's scorer output,
# pinned by hash so a changed file cannot move the gate. The hash is of the parsed JSON
# (json.dumps, sort_keys=True), so a checkout's line endings do not change it.
G1_SOURCE = "results/results_keys_20260927.json"
G1_SOURCE_SHA256 = "7f987bfebd6634be9f8f275f5c2d3c9337a307f81112e98f4b40c59934b0bdcf"
# Cross-GPU difference Part II measured on one cell (its Amendment 8): qasper points,
# and perplexity as a relative difference.
G1_CROSS_GPU = {"qasper": 0.09, "ppl": 0.001}
G1_FACTOR = 3.0

BASELINES = ("aatc", "aatc_native", "ea", "ea_u3")

COMPARISONS = {
    "V1": [("u3_head", "u3_read", "u3_jit")],
    "V2": [("u3_head", "u3", "u3_jit")],
    "V3": [("u3_fisher_cal", "u3_head_cal", "u3_jit")],
    "V4": [("u3_head", "u3_head_cal", "u3_jit")],
    "V5": [("u3_head", "aatc", "u3_jit"), ("u3_head", "ea", "u3_jit")],
}
REPORTED = [
    ("u3_read_cal", "u3_read", "u3_jit"),
    ("u3_key", "u3", "u3_jit"),
    ("aatc_native", "u3_head", "u3_jit"),
    ("ea_u3", "u3", "u3_jit"),
] + [
    (f"u{b}_{x}", f"u{b}", f"u{b}_jit")
    for b in (2, 4)
    for x in ("key", "read", "head", "head_cal", "fisher_cal", "read_cal")
]

# Pod waves (docs/PART5_RUN_DESIGN.md): each wave is one pod's arm list on one model, run
# in this order. "gate" runs first on every Tier A model: it is G1, and its full-length
# cells measure the GPU utilization every later wave is sized from. Calibration tables
# are fitted by the first cell that needs one (g0_head_cal, g0_fisher_cal in "g0").
WAVES: dict[str, list[str]] = {
    "gate": ["u3", "u3_jit", "u3_rep", "u3_jit_rep"],
    "g0": ["g0_native", "g0_read", "g0_head", "g0_head_cal", "g0_fisher_cal", "fp16"],
    "verdict": ["u3_read", "u3_head_cal", "u3_fisher_cal"],
    "head": ["u3_head"],
    "reported3": ["g0_key", "g0_read_cal", "u3_key", "u3_read_cal"],
}
for b in (2, 4):
    WAVES[f"reported{b}"] = [
        f"u{b}", f"u{b}_jit", f"u{b}_key", f"u{b}_read", f"u{b}_read_cal",
        f"u{b}_head_cal", f"u{b}_fisher_cal",
    ]
    WAVES[f"head{b}"] = [f"u{b}_head"]
VERDICT_WAVES = ("gate", "g0", "verdict", "head")
