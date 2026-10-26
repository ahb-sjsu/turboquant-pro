# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Scalar codebooks for a unit-Gaussian coordinate.

A stored code is a list of indices, and what an index means is the codebook, so
every table here is part of the persisted format. A table is never edited in
place. A better table is added under a new name, and a record that uses it says
so (TQE1 version 3, the ``codebook`` field of a persisted index).

Two tables exist:

- ``"legacy"`` is the table every release before this one wrote, and it stays
  the default. It was described as the Lloyd-Max quantizer for N(0, 1), but at 3
  and 4 bits it is not: its mean-square error on N(0, 1) is 0.045556 at 3 bits
  against 0.034548 for the Lloyd-Max quantizer (32% higher), and 0.010803 at 4
  bits against 0.009501 (14% higher). At 2 bits it agrees with Lloyd-Max to the
  three decimals it carries. Data written with it decodes exactly as before.
- ``"lloyd-max"`` is the Lloyd-Max (MSE-optimal) quantizer for N(0, 1), a fixed
  point of Lloyd's iteration computed in 30-digit arithmetic and rounded to six
  decimals. The rounding moves the MSE by less than 1e-11.

Both are scaled by ``1/sqrt(dim)`` at use, where ``dim`` is the rotated
dimension, so a rotated unit vector's coordinates (variance ``1/dim``) see the
table at unit scale.
"""

from __future__ import annotations

import numpy as np

LEGACY = "legacy"
LLOYD_MAX = "lloyd-max"
CODEBOOK_NAMES = (LEGACY, LLOYD_MAX)


def _symmetric(positive: list[float]) -> np.ndarray:
    pos = np.asarray(positive, dtype=np.float64)
    return np.concatenate([-pos[::-1], pos])


# Frozen: the pre-1.x tables, byte-for-byte. Do not "fix" these values; that
# would silently change what every stored legacy code decodes to.
_LEGACY: dict[int, np.ndarray] = {
    1: _symmetric([0.7979]),  # sqrt(2/pi): the sign at the half-normal mean
    2: np.array([-1.510, -0.453, 0.453, 1.510]),
    3: np.array([-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]),
    4: np.array(
        [
            -2.401,
            -1.844,
            -1.437,
            -1.099,
            -0.800,
            -0.524,
            -0.262,
            -0.066,
            0.066,
            0.262,
            0.524,
            0.800,
            1.099,
            1.437,
            1.844,
            2.401,
        ]
    ),
}

# Frozen once released: the Lloyd-Max quantizer for N(0, 1).
_LLOYD_MAX: dict[int, np.ndarray] = {
    1: _symmetric([0.797885]),
    2: _symmetric([0.452780, 1.510418]),
    3: _symmetric([0.245094, 0.756005, 1.343909, 2.151946]),
    4: _symmetric(
        [
            0.128395,
            0.388048,
            0.656759,
            0.942340,
            1.256231,
            1.618046,
            2.069017,
            2.732590,
        ]
    ),
}

_TABLES = {LEGACY: _LEGACY, LLOYD_MAX: _LLOYD_MAX}


def check_codebook(name: str) -> str:
    """Return ``name`` if it is a known codebook, else raise ``ValueError``."""
    if name not in _TABLES:
        raise ValueError(
            f"Unsupported codebook={name!r}; choose from {list(CODEBOOK_NAMES)}"
        )
    return name


def codebook(bits: int, name: str = LEGACY) -> np.ndarray:
    """The unit-scale centroids of codebook ``name`` at ``bits`` bits (a copy)."""
    table = _TABLES[check_codebook(name)]
    if bits not in table:
        raise ValueError(f"Unsupported bits={bits}; choose from {sorted(table)}")
    return table[bits].copy()
