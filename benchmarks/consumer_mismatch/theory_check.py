"""Numerical check of docs/THEORY_consumer_mismatch.md on synthetic consumers.

Here the consumer-relative distortion D(Δ) = tr(P Δ Σ Δᵀ) is the exact loss (a quadratic
consumer reading y = W x), so every proposition must hold to rounding or as an inequality on
every draw. Exits non-zero on any failure.

    python -m consumer_mismatch.theory_check            (from benchmarks/)
"""

from __future__ import annotations

import sys

import torch

from .lowrank import (
    allocate_global,
    decompose,
    distortion,
    psd_power,
    rescore,
    rescore_from_projections,
)

torch.set_default_dtype(torch.float64)
G = torch.Generator().manual_seed(20261012)
FAIL = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}  {detail}")
    if not ok:
        FAIL.append(label)


def spd(k: int, spread: float = 3.0) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(k, k, generator=G))
    ev = torch.exp(spread * torch.rand(k, generator=G) - spread / 2)
    return (q * ev) @ q.T


def near(base: torch.Tensor, angle: float, scale_spread: float) -> torch.Tensor:
    """A consumer whose directions are rotated by about ``angle`` and whose eigenvalues are
    rescaled by up to ``exp(±scale_spread)`` relative to ``base``."""
    k = base.shape[0]
    a = torch.randn(k, k, generator=G)
    rot = torch.linalg.matrix_exp(angle * (a - a.T) / 2)
    ev, q = torch.linalg.eigh(base)
    ev = ev * torch.exp(scale_spread * (2 * torch.rand(k, generator=G) - 1))
    q = rot @ q
    return (q * ev) @ q.T


def cond_rel(pc: torch.Tensor, pd: torch.Tensor) -> float:
    _, hi = psd_power(pc, 0.0)
    ev = torch.linalg.eigvalsh(hi @ pd @ hi)
    return float(ev.max() / ev.min())


def tail(sig2: torch.Tensor, r: int) -> float:
    return float(sig2[r:].sum())


# ---------------------------------------------------------------- Proposition 1
def prop1() -> None:
    m, n = 12, 9
    w, p, s = torch.randn(m, n, generator=G), spd(m), spd(n)
    c = decompose(w, p, s, 0.0)
    worst_eq, worst_alt = 0.0, float("inf")
    for r in range(1, min(m, n)):
        keep = torch.arange(c.sig2.numel()) < r
        d = distortion(w - c.weight(keep), p, s)
        worst_eq = max(
            worst_eq, abs(d - tail(c.sig2, r)) / max(tail(c.sig2, r), 1e-300)
        )
        for other in (decompose(w, None, s, 0.0), decompose(w, None, None, 0.0)):
            worst_alt = min(worst_alt, distortion(w - other.weight(keep), p, s) - d)
        for _ in range(50):  # perturb the factors, so the alternative keeps rank r
            pa = c.A[:, keep] + 1e-2 * torch.randn(m, r, generator=G)
            pb = c.B[keep, :] + 1e-2 * torch.randn(r, n, generator=G)
            worst_alt = min(worst_alt, distortion(w - pa @ pb, p, s) - d)
    check(
        worst_eq < 1e-9,
        "P1 optimal loss equals the whitened tail",
        f"rel err {worst_eq:.1e}",
    )
    check(
        worst_alt > -1e-9,
        "P1 no rank-r alternative does better",
        f"min excess {worst_alt:.2e}",
    )


# ---------------------------------------------------------------- Corollaries 3 and 4
def cor3_cor4() -> None:
    m, n, worst, ratios = 10, 8, 0.0, []
    for i in range(400):
        w = torch.randn(m, n, generator=G)
        pc, sc = spd(m), spd(n)
        pd, sd = near(pc, 0.6 * (i % 4) / 3, 1.0), near(sc, 0.6 * (i % 5) / 4, 1.0)
        kappa = cond_rel(pc, pd) * cond_rel(sc, sd)
        cc, cd = decompose(w, pc, sc, 0.0), decompose(w, pd, sd, 0.0)
        for r in (1, 3, 6):
            keep = torch.arange(cc.sig2.numel()) < r
            got = distortion(w - cc.weight(keep), pd, sd)
            best = tail(cd.sig2, r)
            ratios.append(got / best / kappa)
            worst = max(worst, got / best / kappa)
    check(
        worst <= 1 + 1e-9,
        "C3 mismatch loss <= kappa_P kappa_S x optimum on every draw",
        f"max (loss/opt)/kappa = {worst:.3f} over {len(ratios)} cases",
    )
    w, pc, sc = torch.randn(m, n, generator=G), spd(m), spd(n)
    a, b = 3.7, 0.21
    cc, cd = decompose(w, pc, sc, 0.0), decompose(w, a * pc, b * sc, 0.0)
    err, ratio = 0.0, []
    for r in range(1, n):
        keep = torch.arange(n) < r
        err = max(err, float((cc.weight(keep) - cd.weight(keep)).norm() / w.norm()))
        dc = distortion(w - cc.weight(keep), pc, sc)
        ratio.append(distortion(w - cc.weight(keep), a * pc, b * sc) / dc)
    check(
        err < 1e-9, "C4 a change of scale leaves the solution unchanged", f"{err:.1e}"
    )
    check(
        max(abs(x - a * b) for x in ratio) < 1e-9, "C4 and multiplies the loss by a*b"
    )


# ---------------------------------------------------------------- exact allocation by DP
def dp_alloc(losses: list, widths: list, budget: int) -> tuple:
    """Exact integer allocation: losses[l][r] for r = 0..R_l, integer widths, total <= budget."""
    inf = float("inf")
    choice, table = [], [
        [0.0] * (budget + 1)
    ]  # table[l][b]: best loss of maps < l, cost <= b
    for li, (ls, wd) in enumerate(zip(losses, widths)):
        prev = table[-1] if li else None
        row, pick = [inf] * (budget + 1), [0] * (budget + 1)
        for bud in range(budget + 1):
            for r, val in enumerate(ls):
                cost = r * wd
                if cost > bud:
                    break
                base = prev[bud - cost] if prev is not None else 0.0
                if base + val < row[bud]:
                    row[bud], pick[bud] = base + val, r
        table.append(row)
        choice.append(pick)
    ranks, bud = [], budget
    for li in range(len(losses) - 1, -1, -1):
        r = choice[li][bud]
        ranks.append(r)
        bud -= r * widths[li]
    return table[-1][budget], ranks[::-1]


def instance(scales=None, angle=0.0):
    maps = []
    for i, (m, n) in enumerate(((6, 5), (7, 4), (5, 5), (8, 3))):
        w = torch.randn(m, n, generator=G) * (0.5 + i)
        pc, sc = spd(m), spd(n)
        if scales is None:
            pd, sd = near(pc, angle, 1.0), near(sc, angle, 1.0)
        else:
            pd, sd = scales[i] * pc, sc
        maps.append(dict(w=w, pc=pc, sc=sc, pd=pd, sd=sd, width=m + n))
    return maps


def greedy_prefix(sig2s: list, widths: list, budget: int) -> list:
    gains = {i: s for i, s in enumerate(sig2s)}
    al = allocate_global(gains, dict(enumerate(widths)), budget)
    return [int(al.keep[i].sum()) for i in range(len(sig2s))], al


# ---------------------------------------------------------------- Propositions 5 to 7
def props5to7() -> None:
    worst5, ok6, worst7, sub7 = 0.0, True, 0.0, 0.0
    for trial in range(60):
        scaled = trial % 2 == 0
        sc = [float(torch.exp(torch.randn(1, generator=G))) for _ in range(4)]
        maps = instance(scales=sc if scaled else None, angle=0.5)
        widths = [mp["width"] for mp in maps]
        budget = int(0.5 * sum(min(mp["w"].shape) * mp["width"] for mp in maps))
        comp_c = [decompose(mp["w"], mp["pc"], mp["sc"], 0.0) for mp in maps]
        comp_d = [decompose(mp["w"], mp["pd"], mp["sd"], 0.0) for mp in maps]
        # P5: greedy vs exact on the calibrating consumer
        lc = [[tail(c.sig2, r) for r in range(c.sig2.numel() + 1)] for c in comp_c]
        opt_c, _ = dp_alloc(lc, widths, budget)
        rc, al = greedy_prefix([c.sig2 for c in comp_c], widths, budget)
        greedy_loss = sum(lc[i][r] for i, r in enumerate(rc))
        worst5 = max(worst5, (greedy_loss - opt_c) - al.skipped_first)
        # P6: decomposition on the deploying consumer
        ld = [[tail(c.sig2, r) for r in range(c.sig2.numel() + 1)] for c in comp_d]
        opt_d, _ = dp_alloc(ld, widths, budget)
        dep = 0.0
        for i, (mp, c) in enumerate(zip(maps, comp_c)):
            keep = torch.arange(c.sig2.numel()) < rc[i]
            dep += distortion(mp["w"] - c.weight(keep), mp["pd"], mp["sd"])
        sub = dep - sum(ld[i][r] for i, r in enumerate(rc))
        alloc = sum(ld[i][r] for i, r in enumerate(rc)) - opt_d
        ok6 &= (
            sub >= -1e-9
            and alloc >= -1e-9
            and abs((sub + alloc) - (dep - opt_d)) < 1e-9
        )
        if scaled:  # P7
            pred = (
                sum(sc[i] * lc[i][r] for i, r in enumerate(rc))
                - dp_alloc(
                    [[sc[i] * v for v in lc[i]] for i in range(4)], widths, budget
                )[0]
            )
            worst7 = max(worst7, abs(pred - alloc) / max(dep, 1e-12))
            if abs(pred - alloc) > 1e-9 * max(dep, 1e-12):
                print(
                    f"      P7 debug trial {trial}: pred {pred:.6e} alloc {alloc:.6e} "
                    f"dep {dep:.6e} sub {sub:.3e}"
                )
            sub7 = max(sub7, abs(sub) / max(dep, 1e-12))
    check(
        worst5 <= 1e-9,
        "P5 greedy loss - exact optimum <= first skipped gain",
        f"max excess over bound {worst5:.2e}",
    )
    check(ok6, "P6 SUB >= 0, ALLOC >= 0 and SUB + ALLOC = total, on every draw")
    check(sub7 < 1e-9, "P7 per-map scaling gives SUB = 0", f"max |SUB|/loss {sub7:.1e}")
    check(
        worst7 < 1e-9,
        "P7 ALLOC from calibration spectra and scales matches exact",
        f"rel err {worst7:.1e}",
    )


# ---------------------------------------------------------------- section 5 rescoring
def rescoring() -> None:
    m, n = 9, 7
    w, pc, sc = torch.randn(m, n, generator=G), spd(m), spd(n)
    c = decompose(w, pc, sc, 0.0)
    g = rescore(c, 2.5 * pc, 0.3 * sc)
    check(
        float((g - 0.75 * c.sig2).abs().max() / c.sig2.max()) < 1e-9,
        "S5 rescoring under per-map scaling gives a*b*sig2 exactly",
    )
    # the projection estimator converges to the matrix form
    pd, sd = near(pc, 0.4, 0.5), near(sc, 0.4, 0.5)
    ph, _ = psd_power(pd, 0.0)
    sh, _ = psd_power(sd, 0.0)
    N = 400000
    gs = torch.randn(N, m, generator=G) @ ph
    xs = torch.randn(N, n, generator=G) @ sh
    est = rescore_from_projections(
        c, ((gs @ c.A) ** 2).mean(0), ((xs @ c.B.T) ** 2).mean(0)
    )
    rel = float(((est - rescore(c, pd, sd)).abs() / rescore(c, pd, sd)).max())
    check(
        rel < 0.02,
        "S5 projection estimator matches the matrix form",
        f"max rel {rel:.3f}",
    )
    # reported: how rescoring's ranking error grows with the change of direction
    for ang in (0.0, 0.3, 0.6, 1.2):
        pd, sd = near(pc, ang, 0.8), near(sc, ang, 0.8)
        kap = cond_rel(pc, pd) * cond_rel(sc, sd)
        cd = decompose(w, pd, sd, 0.0)
        gam = rescore(c, pd, sd)
        r = 3
        keep = torch.zeros(n, dtype=torch.bool)
        keep[torch.argsort(gam, descending=True)[:r]] = True
        got = distortion(w - c.weight(keep), pd, sd)
        print(
            f"      rescoring at angle {ang:.1f}: kappa {kap:7.1f}  "
            f"loss / deploying optimum {got / tail(cd.sig2, r):.3f}"
        )


if __name__ == "__main__":
    prop1()
    cor3_cor4()
    props5to7()
    rescoring()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
