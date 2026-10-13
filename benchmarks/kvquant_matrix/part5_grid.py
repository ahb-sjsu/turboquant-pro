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

# G0: the identity codebook through each new allocation must reproduce g0_native.
G0_ARMS = {
    "g0_native": _arm("identity", 3),
    "g0_head": _arm("identity", 3, KEY_ALLOC="head"),
    "g0_head_cal": _arm("identity", 3, KEY_ALLOC="head", BASIS_FIT="calib"),
    "g0_fisher_cal": _arm("identity", 3, KEY_ALLOC="fisher", BASIS_FIT="calib"),
}

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
