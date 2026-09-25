"""Predictor-blind, fixed-rate quantized variants of one model (Part III).

A variant assigns every decoder linear matrix a bit width from ``GEN_LEVELS``. Variants
are drawn at four registered rates (parameter-weighted mean code bits over the matrices),
fifty per rate, by a seeded random process that never sees a predictor, a statistic or a
measurement: a random start, then random single-matrix moves toward the target rate. Two
variants at one rate therefore store the same bits (to the tolerance) and differ only in
where the bits went, which is the "fixed rate" of the hypothesis.

Two bits is not a generator level: the Qwen2.5-0.5B pilot showed that random allocations
reaching 2 bits leave the model broken (KL 4-8 nats per token even at a mean of 4 bits), where
a ranking says nothing about usable quantization. Uniform controls ``u3 .. u8`` are measured
beside the strata and reported, never scored (the 8-bit control verifies the pipeline).
Pilots override the rates and count with WO_RATES / WO_PER_RATE; a registered run uses the
defaults.
"""

from __future__ import annotations

import os

import numpy as np

GEN_LEVELS = (3, 4, 5, 6, 8)
CONTROLS = (3, 4, 5, 6, 8)
RATES = tuple(
    float(r) for r in os.environ.get("WO_RATES", "3.5,4.0,4.5,5.0").split(",")
)
PER_RATE = int(os.environ.get("WO_PER_RATE", "50"))
SEED = 20261001
TOL = 0.02  # bits per parameter


def variant_id(rate: float, i: int) -> str:
    return f"r{rate:.1f}-v{i:02d}"


def mean_bits(bits: np.ndarray, sizes: np.ndarray) -> float:
    return float((bits * sizes).sum() / sizes.sum())


def draw(sizes: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """One allocation of GEN_LEVELS over matrices of ``sizes`` parameters at ``rate``."""
    levels = np.asarray(GEN_LEVELS)
    idx = rng.integers(0, len(levels), size=len(sizes))
    for _ in range(100_000):
        mb = mean_bits(levels[idx], sizes)
        if abs(mb - rate) <= TOL:
            return levels[idx].astype(int)
        m = rng.integers(0, len(sizes))
        if mb > rate and idx[m] > 0:
            idx[m] -= 1
        elif mb < rate and idx[m] < len(levels) - 1:
            idx[m] += 1
    raise RuntimeError(f"no allocation at rate {rate} within {TOL}")


def generate(names: list, sizes: list) -> dict:
    """{variant_id: {matrix name: bits}}: the uniform controls, then every rate."""
    sizes = np.asarray(sizes, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    out = {f"u{b}": {n: b for n in names} for b in CONTROLS}
    for rate in RATES:
        for i in range(PER_RATE):
            b = draw(sizes, rate, rng)
            out[variant_id(rate, i)] = {n: int(x) for n, x in zip(names, b)}
    return out
