"""The grid of observer-advantage Part V (docs/PREREG_full_consumer_kv.md).

One source of truth for Part V's runner and scorer, built on Part II's registered grid
(``keys_grid``), which it imports and never changes: the shipped recipe, the models, the
tasks and the arm encoding are Part II's. Every arm is an env line for
``tq_paper_lb_shard.py`` / ``wikitext_ppl.py``; every comparison an (arm, reference,
floor arm) triple, the floor being |mean(floor) - mean(reference)|. Nothing here is fitted.

The baselines of section 3 are named in BASELINES. AATC (``aatc``, ``aatc_native``) runs in
this harness (aatc.py) and has env lines in BASELINE_ARMS; Expected Attention (``ea``,
``ea_u3``) runs through kvpress. FIDELITY_ARMS are AATC's fidelity check (section 3): AATC
against KIVI at 2 bits on Qwen2.5-7B, the published comparison it must reproduce.
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

# AATC at u3's stored key bytes: AATC_KEY_BITS=match makes the runner set the average code
# bits to u3's measured mean stored key bits per element on the same model's qasper
# documents minus AATC's per-token scale (16 bits per 512-channel group), and record both
# in the cell's baseline.json; values at Part II's 4 bits, as in every arm.
BASELINE_ARMS = {
    "aatc": _arm("uniform", 3, AATC="k", AATC_KEY_BITS="match", AATC_TRANSFORM=1),
    "aatc_native": _arm("uniform", 3, AATC="k", AATC_KEY_BITS="match", AATC_TRANSFORM=0),
    # Expected Attention (ea.py): u3's line, so its matched bytes use u3's HOT and VAL_BITS;
    # EA=ea turns u3's coding off and keeps fp16 entries, EA=ea_u3 keeps it on.
    "ea": _arm("uniform", 3, EA="ea"),
    "ea_u3": _arm("uniform", 3, EA="ea_u3"),
}
# ea_u3 stores fewer bytes than u3 by construction (eviction on top of u3's coding): it is
# reported with its own bytes, so only these are held to u3's within 1%.
MATCHED_BASELINES = ("aatc", "aatc_native", "ea")
# V5's endpoints per baseline (owner, 2026-10-14). Expected Attention evicts, and its
# perplexity under the stored-cache convention (every query reads the evicted cache;
# ea.py, two passes) was 15-25x fp16 at the matched ratio on a 0.5B check, which would
# settle the perplexity endpoint in u3_head's favour by construction. It is judged on
# qasper alone; its perplexity is run and reported.
V5_ENDPOINTS = {"aatc": ("qasper", "ppl"), "ea": ("qasper",)}
AATC_SCALE_BITS = 16 / 512
FIDELITY_MODEL = "qwen2.5-7b-instruct"
FIDELITY_ARMS = {
    # KIVI as the paper ran it: keys per channel in groups of 32 tokens, values per token
    # in groups of 32 channels, 2 bits each, 4 sink and 128 recent tokens in fp16.
    "kivi2": ("NOQUANT=0 CODEBOOK=uniform KEY_BITS=2 VAL_BITS=2 GROUP=32 HOT=128 SINK=4 "
              "OUTLIER_FRAC=0 PREROPE=0 VAL_GROUP=32"),
    # AATC at an average of 2 bits on keys and values (the paper's 2-bit operating point).
    "aatc2": _arm("uniform", 3, AATC="kv", AATC_KEY_BITS=2, AATC_VAL_BITS=2, AATC_TRANSFORM=1),
}


def all_lines() -> dict:
    """Every env line the runner can run, by arm name."""
    return {**ARMS, **G0_ARMS, **BASELINE_ARMS, **FIDELITY_ARMS}

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
WAVES["baselines"] = ["aatc", "aatc_native", "ea", "ea_u3"]  # after the gate wave (u3)
WAVES["fidelity"] = ["kivi2", "aatc2"]  # FIDELITY_MODEL only, before registration
VERDICT_WAVES = ("gate", "g0", "verdict", "head")
