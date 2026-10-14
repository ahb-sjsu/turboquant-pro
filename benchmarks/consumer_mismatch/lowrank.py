"""Consumer-relative low-rank approximation of linear maps (docs/THEORY_consumer_mismatch.md).

For ``y = W x`` with consumer moments ``Σ = E[x xᵀ]`` and ``P = E[g gᵀ]`` the distortion of a
weight error is ``D(Δ) = tr(P Δ Σ Δᵀ)``. ``decompose`` returns every component of the
optimal solution (Proposition 1), ``allocate`` chooses which to keep under a global or a
uniform budget (Proposition 5), and ``rescore`` scores calibrated components under another
consumer from projections alone (section 5).

Everything here is float64 torch and works on CPU or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def psd_power(m: torch.Tensor, damp: float) -> tuple:
    """(M^{1/2}, M^{-1/2}) of a symmetric PSD matrix after relative damping.

    The eigenvalues are floored at ``damp`` times their mean, so a damping of 0 means the
    matrix must already be positive definite."""
    m = m.double()
    m = (m + m.T) / 2
    ev, q = torch.linalg.eigh(m)
    floor = damp * float(ev.clamp_min(0).mean())
    ev = ev.clamp_min(max(floor, 1e-300))
    return (q * ev.sqrt()) @ q.T, (q * ev.rsqrt()) @ q.T


@dataclass
class Components:
    """``W ≈ Σ_k a_k b_kᵀ`` with ``a = A[:, k]``, ``b = B[k, :]``; ``sig2[k]`` is the
    calibrating consumer's cost of dropping component ``k`` (non-increasing in ``k``).
    """

    A: torch.Tensor  # m x R
    B: torch.Tensor  # R x n
    sig2: torch.Tensor  # R

    def weight(self, keep: torch.Tensor) -> torch.Tensor:
        """The approximation built from the components selected by a boolean mask."""
        return self.A[:, keep] @ self.B[keep, :]


def decompose(
    w: torch.Tensor, p: torch.Tensor, s: torch.Tensor, damp: float
) -> Components:
    """All components of the consumer-optimal low-rank solution (Proposition 1).

    ``p = None`` gives the one-sided (activation-whitened, SVD-LLM) solution and
    ``p = s = None`` the plain SVD."""
    w = w.double()
    m, n = w.shape
    dev = w.device
    if p is None:
        ph = phi = torch.eye(m, dtype=torch.float64, device=dev)
    else:
        ph, phi = psd_power(p, damp)
    if s is None:
        sh = shi = torch.eye(n, dtype=torch.float64, device=dev)
    else:
        sh, shi = psd_power(s, damp)
    u, sv, vt = torch.linalg.svd(ph @ w @ sh, full_matrices=False)
    return Components(A=phi @ (u * sv), B=vt @ shi, sig2=sv**2)


def distortion(delta: torch.Tensor, p: torch.Tensor, s: torch.Tensor) -> float:
    """``tr(P Δ Σ Δᵀ)``."""
    d = delta.double()
    return float(((p.double() @ d) * (d @ s.double())).sum())


def rescore(c: Components, p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``γ_k = (a_kᵀ P a_k)(b_kᵀ Σ b_k)`` for every component, under another consumer.

    With ``p`` and ``s`` given as matrices this is the exact diagonal of the new consumer in the
    component basis. ``rescore_from_projections`` computes the same from samples of ``g`` and
    ``x`` without forming either matrix."""
    pa = ((p.double() @ c.A) * c.A).sum(0)
    sb = ((c.B @ s.double()) * c.B).sum(1)
    return pa * sb


def rescore_from_projections(
    c: Components, ga2: torch.Tensor, xb2: torch.Tensor
) -> torch.Tensor:
    """Same as ``rescore`` from accumulated ``E[(a_kᵀ g)²]`` and ``E[(b_kᵀ x)²]``."""
    return ga2.double() * xb2.double()


@dataclass
class Allocation:
    keep: dict  # name -> bool mask over that map's components
    cost: float  # parameters kept
    budget: float
    skipped_first: float  # gain of the first component that did not fit (Prop 5 bound)


def allocate_global(gains: dict, widths: dict, budget: float) -> Allocation:
    """Water filling (Proposition 5): keep components in decreasing gain / width until the
    budget is spent. ``gains[name]`` are per-component costs of dropping, ``widths[name]`` the
    parameters one component of that map stores (m + n)."""
    names, ratio, gain, width, index = [], [], [], [], []
    for i, (nm, g) in enumerate(gains.items()):
        g = g.double().cpu()
        k = g.numel()
        names.append(nm)
        ratio.append(g / widths[nm])
        gain.append(g)
        width.append(torch.full((k,), float(widths[nm]), dtype=torch.float64))
        index.append(torch.stack([torch.full((k,), i), torch.arange(k)], 1))
    ratio, gain, width, index = (torch.cat(t) for t in (ratio, gain, width, index))
    order = torch.argsort(ratio, descending=True, stable=True)
    csum = torch.cumsum(width[order], 0)
    n_fit = int((csum <= budget).sum())
    chosen = order[:n_fit]
    keep = {nm: torch.zeros(gains[nm].numel(), dtype=torch.bool) for nm in names}
    for gi, ki in index[chosen].tolist():
        keep[names[gi]][ki] = True
    skipped = float(gain[order[n_fit]]) if n_fit < len(order) else 0.0
    return Allocation(
        keep=keep,
        cost=float(csum[n_fit - 1]) if n_fit else 0.0,
        budget=budget,
        skipped_first=skipped,
    )


def allocate_uniform(gains: dict, shapes: dict, fraction: float) -> Allocation:
    """Each map keeps ``fraction`` of its break-even rank ``m n / (m + n)``, choosing its own
    top components by gain."""
    keep, cost = {}, 0.0
    for nm, g in gains.items():
        m, n = shapes[nm]
        r = max(1, min(g.numel(), int(round(fraction * m * n / (m + n)))))
        mask = torch.zeros(g.numel(), dtype=torch.bool)
        mask[torch.argsort(g.double().cpu(), descending=True)[:r]] = True
        keep[nm] = mask
        cost += r * (m + n)
    return Allocation(keep=keep, cost=cost, budget=cost, skipped_first=0.0)


def budget_for_fraction(shapes: dict, fraction: float) -> float:
    """The parameter budget of a uniform allocation at ``fraction``, so both allocations are
    compared at the same stored size."""
    return float(
        sum(
            max(1, int(round(fraction * m * n / (m + n)))) * (m + n)
            for m, n in shapes.values()
        )
    )
