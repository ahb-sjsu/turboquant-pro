"""Statistics, predictor tables and the atlas of Part III, one layer group at a time.

For a decoder linear map ``y = W x`` quantized with error ``D = Q_b(W) - W``, the model's
second-order behavioural damage is ``(1/2) E[(D x)^T R (D x)]`` with ``R = J^T F J`` the read
operator of everything downstream of ``y``. Treating inputs and read-outs as independent
gives the observer predictor

    observer   tr(P_y D Sigma_x D^T)        P_y = E[g g^T], Sigma_x = E[x x^T]

where ``g`` is the gradient of the log-likelihood of a label SAMPLED from the model at the
layer's output (so ``E[g g^T]`` is the true Fisher at that output, not an empirical one).
This is K-FAC's Kronecker factorization (Martens and Grosse 2015); its diagonal-in-P_y form
is BRECQ's output-Fisher weighting (Li et al. 2021). The competitors are its reductions:

    outdiag    tr(diag(P_y) D Sigma_x D^T)   output Fisher, diagonal (reported)
    act        tr(D Sigma_x D^T)             P_y = I: GPTQ / AWQ's layer objective
    fisher     sum F_ij D_ij^2               diagonal weight Fisher (SqueezeLLM), from
                                             per-sequence gradients G^T X with the same
                                             sampled labels
    raw        ||D||_F^2                     parameter-space distortion

A matrix's error depends only on its own bit width, so each predictor is a table over
(matrix, bits); a variant's prediction is a sum of table entries (the block-diagonal
assumption every one of these predictors makes). The measured KL carries the interactions.

The atlas (reported, never scored) summarizes the same statistics: effective ranks and top
spectra of Sigma_x and P_y, massive-activation channels in Sigma_x, and k-occurrence hubness
of the weight rows.
"""

from __future__ import annotations

import math

import torch

from .quant import LEVELS, rtn

PREDICTORS = ("raw", "act", "fisher", "outdiag", "observer")
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def linear_modules(model) -> dict:
    """{name: nn.Linear} for every decoder projection, in layer order."""
    out = {}
    for li, layer in enumerate(model.model.layers):
        for part, holder in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for n in LINEAR:
                m = getattr(holder, n, None)
                if m is not None:
                    out[f"layers.{li}.{part}.{n}"] = m
    return out


class Accumulator:
    """Hooks one group of Linear modules; accumulates Sigma_x, P_y and the weight Fisher."""

    def __init__(self, modules: dict):
        self.modules = modules
        self.stats = {}
        self._x = {}
        self.handles = []
        for name, m in modules.items():
            dev = m.weight.device
            o, i = m.weight.shape
            self.stats[name] = {
                "S": torch.zeros(i, i, device=dev, dtype=torch.float64),
                "P": torch.zeros(o, o, device=dev, dtype=torch.float64),
                "F": torch.zeros(o, i, device=dev, dtype=torch.float32),
                "tokens": 0,
                "seqs": 0,
            }
            self.handles.append(m.register_forward_hook(self._fwd(name)))
            self.handles.append(m.register_full_backward_hook(self._bwd(name)))

    def _fwd(self, name):
        def hook(mod, inp, out):
            self._x[name] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()

        return hook

    def _bwd(self, name):
        def hook(mod, grad_in, grad_out):
            g = grad_out[0].detach().reshape(-1, grad_out[0].shape[-1]).float()
            x = self._x.pop(name)
            st = self.stats[name]
            st["S"] += (x.T @ x).double()
            st["P"] += (g.T @ g).double()
            st["F"] += (
                g.T @ x
            ) ** 2  # per-sequence gradient of the log-likelihood, squared
            st["tokens"] += x.shape[0]
            st["seqs"] += 1

        return hook

    def close(self):
        for h in self.handles:
            h.remove()

    def finalized(self) -> dict:
        out = {}
        for name, st in self.stats.items():
            t, s = max(st["tokens"], 1), max(st["seqs"], 1)
            out[name] = {
                "S": (st["S"] / t).float(),
                "P": (st["P"] / t).float(),
                "F": st["F"] / s,
            }
        return out


def sampled_nll(model, input_ids: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Sum over positions of -log p(y) for labels y SAMPLED from the model (true Fisher)."""
    emb = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=emb, use_cache=False).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    with torch.no_grad():
        y = torch.multinomial(logp.exp().reshape(-1, logp.shape[-1]), 1, generator=gen)
    return -logp.reshape(-1, logp.shape[-1]).gather(1, y).sum()


def predictor_row(w: torch.Tensor, st: dict) -> dict:
    """{bits: {predictor: value}} for one matrix."""
    S, P, F = st["S"], st["P"], st["F"]
    pdiag = torch.diagonal(P)[:, None]
    row = {}
    for b in LEVELS:
        d = rtn(w, b) - w.float()
        ds = d @ S
        row[b] = {
            "raw": float((d * d).sum()),
            "act": float((ds * d).sum()),
            "fisher": float((F * d * d).sum()),
            "outdiag": float((pdiag * ds * d).sum()),
            "observer": float(((P @ d) * ds).sum()),
        }
    return row


def _spectrum(m: torch.Tensor, top: int = 32) -> dict:
    ev = torch.linalg.eigvalsh(m.double()).clamp_min(0).flip(0)
    tot = float(ev.sum())
    if tot <= 0:
        return {"effective_rank": 0.0, "participation": 0.0, "top": []}
    p = ev / tot
    ent = float(-(p[p > 0] * p[p > 0].log()).sum())
    return {
        "effective_rank": math.exp(ent),
        "participation": float(tot**2 / (ev**2).sum()),
        "top_share": [float(x) for x in (ev[:top] / tot)],
    }


def _hubness(w: torch.Tensor, k: int = 10, block: int = 2048) -> dict:
    """k-occurrence of weight rows under cosine kNN: skewness and the top hubs."""
    x = torch.nn.functional.normalize(w.float(), dim=1)
    n = x.shape[0]
    counts = torch.zeros(n, device=x.device)
    for s in range(0, n, block):
        sim = x[s : s + block] @ x.T
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -2.0
        nn = sim.topk(k, dim=1).indices
        counts += torch.bincount(nn.reshape(-1), minlength=n).float()
    c = counts - counts.mean()
    skew = float((c**3).mean() / (c**2).mean().clamp_min(1e-12) ** 1.5)
    top = counts.topk(min(8, n))
    return {
        "k": k,
        "skew": skew,
        "max": float(top.values[0]),
        "top_rows": [int(i) for i in top.indices],
    }


def atlas_row(w: torch.Tensor, st: dict) -> dict:
    d = torch.diagonal(st["S"])
    med = float(d.median().clamp_min(1e-30))
    massive = torch.nonzero(d > 20 * med).flatten()
    return {
        "sigma_x": _spectrum(st["S"]),
        "p_y": _spectrum(st["P"]),
        "massive_channels": [int(i) for i in massive[:32]],
        "massive_ratio": float(d.max() / med),
        "row_hubness": _hubness(w),
    }


def layer_groups(n_layers: int, size: int) -> list:
    return [list(range(s, min(n_layers, s + size))) for s in range(0, n_layers, size)]


def build(model, calib: list, group_size: int, seed: int, log=print) -> tuple:
    """(tables, atlas): {matrix: {bits: {predictor: value}}}, {matrix: atlas row}."""
    mods = linear_modules(model)
    n_layers = len(model.model.layers)
    tables, atlas = {}, {}
    for grp in layer_groups(n_layers, group_size):
        sel = {n: m for n, m in mods.items() if int(n.split(".")[1]) in grp}
        acc = Accumulator(sel)
        gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
        try:
            for ids in calib:
                model.zero_grad(set_to_none=True)
                sampled_nll(model, ids, gen).backward()
        finally:
            acc.close()
        stats = acc.finalized()
        for name, st in stats.items():
            w = sel[name].weight.detach()
            tables[name] = predictor_row(w, st)
            atlas[name] = atlas_row(w, st)
        del stats, acc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f"[tables] layers {grp[0]}-{grp[-1]} done ({len(sel)} matrices)")
    return tables, atlas
