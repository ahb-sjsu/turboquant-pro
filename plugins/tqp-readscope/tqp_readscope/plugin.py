# tqp-readscope: readscope-backed read operators for turboquant-pro
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Blind read-operator recovery, for consumers with no closed form.

The first external provider of turboquant-pro's read-operator contract,
living out of tree and discovered through the
``turboquant_pro.read_operators`` entry point. Neither package imports the
other: turboquant-pro declares the protocol, readscope measures the operator,
and this adapter is the only thing that knows about both.

That separation is deliberate. readscope is a numpy-only instrument meant to
be usable by people with no interest in compression, and folding it into a
production quantization stack would cost it that. A small adapter — the
providers plus a zero-copy ingestion contract — is the whole price of
keeping them apart.

**Two providers, and the second is usually the one you want.**

``readscope_blind``
    For a scalar-margin consumer: a logit, a ranking score, a single
    attention weight. Recovers ``E[g g^T]`` from output differences.

``readscope_jacobian``
    For a vector-valued consumer, such as the full attention distribution
    over a key set. Recovers the Jacobian Gram ``E[J^T J]``, which carries
    ``m`` numbers per probe direction instead of one and is what the
    published attention measurements used.

**The budget is not a tuning knob.** readscope's own calibration found that
recovery against the direction budget ``k/d`` is a cliff at ``k = d``, and
that the cliff is rank independent: asking for one direction costs the same
as asking for sixteen, because below full dimension the estimate is a
projection onto a random subspace and a projected operator's leading
eigenvector is not the operator's. So these providers default to ``k = d``
and warn when asked for less, rather than quietly returning a one-direction
reading that looks like a measurement. That is the instrument's
specification propagating into its integration, which is what a
specification is for.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def _channels(activations, force_numpy: bool = False):
    """Ingest activations with as few copies as physics allows.

    Accepts numpy arrays, and any DLPack producer (torch, CuPy, JAX):

    - **numpy** — unchanged legacy behavior (float64 view/copy).
    - **CPU DLPack tensors** (e.g. torch on CPU) — ``np.from_dlpack``:
      zero-copy, native float dtype preserved.
    - **CUDA DLPack tensors** — zero-copy into **CuPy** when CuPy is
      installed; the probe's linear algebra then runs where the data
      lives (readscope's core is backend-generic and draws its random
      directions in numpy either way, so readings are seed-identical
      across backends). Without CuPy — or with ``force_numpy=True`` —
      **exactly one** explicit device-to-host copy is made, with a
      warning naming it. GPU-to-numpy zero-copy does not exist: numpy
      cannot address CUDA memory, and this adapter will not pretend
      otherwise by hiding per-call copies.

    Note for the CuPy path: the consumer callable will receive CuPy
    vectors. A torch-based consumer can accept them zero-copy via
    ``torch.from_dlpack``.

    The last axis is channels; every leading axis (batch, sequence,
    image patches, audio frames — the consumer contract is modality-
    blind) is flattened into operating points.
    """
    if isinstance(activations, np.ndarray):
        a = np.asarray(activations, dtype=np.float64)
        d = a.shape[-1]
        return a.reshape(-1, d), d

    dev = getattr(activations, "__dlpack_device__", None)
    if dev is not None:
        dev_type = int(dev()[0])
        if dev_type == 1:  # kDLCPU
            a = np.from_dlpack(activations)          # zero-copy
            d = a.shape[-1]
            return a.reshape(-1, d), d
        # CUDA / ROCm / managed
        if not force_numpy:
            try:
                import cupy

                a = cupy.from_dlpack(activations)    # zero-copy, on-GPU
                d = int(a.shape[-1])
                return a.reshape(-1, d), d
            except ImportError:
                pass
        warnings.warn(
            "activations live on a GPU and CuPy is "
            + ("not installed" if not force_numpy else "bypassed")
            + "; making ONE explicit device-to-host copy. Install cupy "
            "(pip install turboquant-pro[gpu]) to keep the measurement "
            "on-device, zero-copy.",
            RuntimeWarning,
            stacklevel=3,
        )
        host = activations
        for meth in ("detach", "cpu"):
            f = getattr(host, meth, None)
            if callable(f):
                host = f()
        a = np.from_dlpack(host) if hasattr(host, "__dlpack__")             else np.asarray(host)
        a = np.asarray(a, dtype=np.float64)
        d = a.shape[-1]
        return a.reshape(-1, d), d

    a = np.asarray(activations, dtype=np.float64)
    d = a.shape[-1]
    return a.reshape(-1, d), d


def _resolve_budget(n_directions: int | None, d: int, label: str) -> int:
    if n_directions is None:
        return d
    k = int(n_directions)
    if k < d:
        warnings.warn(
            f"{label} was given n_directions={k} in {d} dimensions. "
            f"readscope's budget law is a cliff at k = d and is rank "
            f"independent, so a sub-dimensional budget resolves one or two "
            f"directions whatever rank you intend to use. Pass "
            f"n_directions>={d} or accept a dominant-direction reading.",
            RuntimeWarning,
            stacklevel=3,
        )
    return k


class ReadscopeBlindOperator:
    """Blind recovery for a scalar-margin consumer."""

    def __init__(
        self,
        consumer=None,
        n_directions: int | None = None,
        eps: float = 1e-3,
        seed: int = 0,
        max_points: int | None = 256,
    ):
        self.consumer = consumer
        self.n_directions = n_directions
        self.eps = float(eps)
        self.seed = int(seed)
        self.max_points = max_points

    def operator(self, activations: np.ndarray, **context: Any) -> np.ndarray:
        from readscope import blind_probe

        consumer = context.get("consumer", self.consumer)
        if consumer is None:
            raise ValueError(
                "readscope_blind needs the consumer itself; pass "
                "consumer=<callable from a vector to a scalar> either at "
                "creation or in the call context"
            )
        pts, d = _channels(activations,
                           force_numpy=context.get("force_numpy", False))
        if self.max_points is not None and pts.shape[0] > self.max_points:
            pts = pts[: self.max_points]
        k = _resolve_budget(self.n_directions, d, "readscope_blind")

        res = blind_probe(
            consumer,
            pts,
            mode="exact" if k >= d else "lstsq",
            sketch_dim=None if k >= d else k,
            eps=self.eps,
            rng=np.random.default_rng(self.seed),
        )
        return res.S


class ReadscopeJacobianOperator:
    """Blind recovery for a vector-valued consumer."""

    def __init__(
        self,
        consumer=None,
        n_directions: int | None = None,
        eps: float = 1e-3,
        seed: int = 0,
        max_points: int | None = 64,
    ):
        self.consumer = consumer
        self.n_directions = n_directions
        self.eps = float(eps)
        self.seed = int(seed)
        self.max_points = max_points

    def operator(self, activations: np.ndarray, **context: Any) -> np.ndarray:
        from readscope import jacobian_probe

        consumer = context.get("consumer", self.consumer)
        if consumer is None:
            raise ValueError(
                "readscope_jacobian needs the consumer itself; pass "
                "consumer=<callable from a vector to a vector> either at "
                "creation or in the call context"
            )
        pts, d = _channels(activations,
                           force_numpy=context.get("force_numpy", False))
        if self.max_points is not None and pts.shape[0] > self.max_points:
            pts = pts[: self.max_points]
        k = _resolve_budget(self.n_directions, d, "readscope_jacobian")

        res = jacobian_probe(
            consumer,
            pts,
            n_directions=k,
            eps=self.eps,
            rng=np.random.default_rng(self.seed),
        )
        return res.S


def _spec(name, factory, description):
    from turboquant_pro.read_operators import ReadOperatorSpec

    return ReadOperatorSpec(
        name=name,
        factory=factory,
        description=description,
        exact=False,
        requires=("readscope",),
        metadata={
            "budget_law": "cliff at k = d, rank independent",
            "source": "readscope",
        },
    )


SPEC_BLIND = _spec(
    "readscope_blind",
    lambda **cfg: ReadscopeBlindOperator(**cfg),
    "blind E[g g^T] recovery for a scalar-margin consumer",
)

SPEC_JACOBIAN = _spec(
    "readscope_jacobian",
    lambda **cfg: ReadscopeJacobianOperator(**cfg),
    "blind E[J^T J] recovery for a vector-valued consumer",
)
