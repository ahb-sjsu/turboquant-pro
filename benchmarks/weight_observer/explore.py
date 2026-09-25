"""EXPLORATORY predictors for Part III (not registered; nothing here is a verdict).

Part III's registered observer form, K-FAC's ``tr(P_y D Sigma_x D^T)``, lost to the diagonal
weight Fisher (``RESULTS_observer_advantage_weights.md``). K-FAC replaces the per-sequence
coupling of a layer's inputs and output gradients by the product of their separate averages;
the diagonal Fisher keeps the coupling but drops every cross term between weights. This pass
records what neither approximation drops. For calibration sequence ``s``, matrix ``m`` and
bit width ``b``, with ``D_{m,b} = Q_b(W_m) - W_m`` and ``G_s^T X_s`` the gradient of the
sequence's sampled log-likelihood with respect to ``W_m``:

    c[s, m, b] = <D_{m,b}, G_s^T X_s>            (Frobenius inner product)

from which, for any variant (a bit width ``b_m`` per matrix),

    exact_block  = sum_m mean_s c[s, m, b_m]^2           exact Fisher along D, per matrix
    exact_model  = mean_s (sum_m c[s, m, b_m])^2         ... and between matrices

``exact_model`` is the second-order KL along the variant's whole perturbation under the
sampled (true) Fisher: no diagonal, no Kronecker factorization, no block-diagonal assumption.
Two per-token forms are recorded beside them: ``tok_exact`` = sum_m mean_t (g_t^T D x_t)^2
(no cross-token terms) and ``fisher_tok`` = the diagonal Fisher from per-token gradients.

Calibration text, sequence count and label seed are the registered ones, so the registered
``fisher`` table can be recomputed here as a wiring check.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from . import tables as T
from .quant import rtn
from .variants import GEN_LEVELS

LEVELS = GEN_LEVELS


class ExactAccumulator:
    """Per-sequence inner products with the quantization error, per matrix and level."""

    def __init__(self, modules: dict, names: list, n_seq: int):
        self.modules = modules
        self.index = {n: names.index(n) for n in modules}
        self.D = {}
        self.c = torch.zeros(n_seq, len(names), len(LEVELS), dtype=torch.float64)
        self.tok = {n: torch.zeros(len(LEVELS), dtype=torch.float64) for n in modules}
        self.ftok = {}
        self.fseq = {}
        self.tokens = {n: 0 for n in modules}
        self.seq = 0
        self._x = {}
        self.handles = []
        for name, m in modules.items():
            w = m.weight.detach()
            self.D[name] = torch.stack([(rtn(w, b) - w.float()) for b in LEVELS])
            self.ftok[name] = torch.zeros_like(w, dtype=torch.float32)
            self.fseq[name] = torch.zeros_like(w, dtype=torch.float32)
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
            grad = g.T @ x  # d log p_s / dW, up to sign
            D = self.D[name]
            self.c[self.seq, self.index[name]] = (
                (D * grad[None]).sum(dim=(1, 2)).double().cpu()
            )
            gd = torch.einsum("to,lof->ltf", g, D)  # (levels, T, in)
            self.tok[name] += ((gd * x[None]).sum(-1) ** 2).sum(-1).double().cpu()
            self.ftok[name] += (g * g).T @ (x * x)
            self.fseq[name] += grad**2
            self.tokens[name] += x.shape[0]

        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def build(model, calib: list, group_size: int, seed: int, log=print) -> dict:
    mods = T.linear_modules(model)
    names = list(mods)
    n = len(calib)
    out = {
        "names": names,
        "numel": [mods[k].weight.numel() for k in names],
        "levels": list(LEVELS),
        "c": np.zeros((n, len(names), len(LEVELS))),
        "tok_exact": {},
        "fisher_tok": {},
        "fisher_check": {},
    }
    for grp in T.layer_groups(len(model.model.layers), group_size):
        sel = {k: m for k, m in mods.items() if int(k.split(".")[1]) in grp}
        acc = ExactAccumulator(sel, names, n)
        gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
        try:
            for i, ids in enumerate(calib):
                acc.seq = i
                model.zero_grad(set_to_none=True)
                T.sampled_nll(model, ids, gen).backward()
        finally:
            acc.close()
        for k in sel:
            j = names.index(k)
            out["c"][:, j, :] = acc.c[:, j, :].numpy()
            t = max(acc.tokens[k], 1)
            out["tok_exact"][k] = (acc.tok[k] / t).tolist()
            D = acc.D[k]
            ft = acc.ftok[k] / t
            fs = acc.fseq[k] / n
            out["fisher_tok"][k] = [
                float((ft * D[i] ** 2).sum()) for i in range(len(LEVELS))
            ]
            out["fisher_check"][k] = [
                float((fs * D[i] ** 2).sum()) for i in range(len(LEVELS))
            ]
        del acc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[explore] layers {grp[0]}-{grp[-1]} done")
    return out


def main(argv=None) -> int:
    import argparse

    from . import run as R

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group", type=int, default=1)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = R.load(a.model_path, a.device)
    text = open(f"{a.text}/train.txt", encoding="utf-8").read()
    calib = [c[None].to(a.device) for c in R.chunks(tok, text, R.N_CALIB)]
    t0 = time.time()
    res = build(model, calib, a.group, R.CALIB_SEED, log=lambda s: print(s, flush=True))
    np.save(os.path.join(a.out, "c.npy"), res.pop("c"))
    res["env"] = R.environment()
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, "explore.json"), "w"))
    print("EXPLORE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
