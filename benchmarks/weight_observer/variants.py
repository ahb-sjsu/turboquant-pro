"""Predictor-blind, fixed-rate quantized variants of one model (Part III).

A variant assigns every decoder linear matrix a bit width from ``quant.LEVELS``. Variants
are drawn at four registered rates (parameter-weighted mean code bits over the matrices),
fifty per rate, by a seeded random process that never sees a predictor, a statistic or a
measurement: a random start, then random single-matrix moves toward the target rate. Two
variants at one rate therefore store the same bits (to the tolerance) and differ only in
where the bits went, which is the "fixed rate" of the hypothesis.
"""

from __future__ import annotations

import numpy as np

from .quant import LEVELS

RATES = (2.5, 3.0, 3.5, 4.0)
PER_RATE = 50
SEED = 20261001
TOL = 0.02  # bits per parameter


def variant_id(rate: float, i: int) -> str:
    return f"r{rate:.1f}-v{i:02d}"


def mean_bits(bits: np.ndarray, sizes: np.ndarray) -> float:
    return float((bits * sizes).sum() / sizes.sum())


def draw(sizes: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """One allocation of LEVELS over matrices of ``sizes`` parameters at mean ``rate``."""
    levels = np.asarray(LEVELS)
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
    """{variant_id: {matrix name: bits}} for every registered rate."""
    sizes = np.asarray(sizes, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    out = {}
    for rate in RATES:
        for i in range(PER_RATE):
            b = draw(sizes, rate, rng)
            out[variant_id(rate, i)] = {n: int(x) for n, x in zip(names, b)}
    return out
