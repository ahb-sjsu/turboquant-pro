"""Part III-c (``docs/PREREG_weights_codec_allocation.md``): allocation and codec.

    python -m weight_observer.codec_run tables --model-path M --text T --out O
    python -m weight_observer.codec_run plans --out O
    python -m weight_observer.codec_run arms --model-path M --text T --out O

``tables`` records the identities of the scored and calibration samples (``hashes.json``:
the sha256 of each text file and, for this model's tokenizer, of the token ids of the 48
evaluation and 128 calibration windows), then, one group of layers at a time, the cost of
every (matrix, codec, width): the diagonal Fisher weighting of the error that codec makes,
``sum F * D^2`` (``codec_costs.jsonl``, resumable per matrix).

``plans`` turns those into the arms: uniform widths, and the exact knapsack plan
(``turboquant_pro.weight_plan``) of each codec's own table, at the logical stored bytes of
uniform 3- and 4-bit; a planned arm pays one byte per matrix for its width map. ``gptq_frtn``
encodes RTN's plan with GPTQ (C3). Every arm's stored bits are checked against its budget
(gate G1) before it is written.

``arms`` encodes each arm and measures its KL from the full-precision model on the 48
evaluation windows (``arms_results.jsonl``, resumable per arm).

GPTQ and AWQ read the input second moment ``S`` and the per-channel mean ``|x|`` of each
matrix. Both come from ``input_stats``, a forward-only pass over the calibration windows,
used identically by ``tables`` and ``arms``, so the codec output a cost was computed on is
the codec output that is measured. One-shot arms take the statistics from the
full-precision model; the reported sequential arm (``gptq_seq``) from the arm's own model,
whose earlier layers are already quantized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from . import quant as Q
from . import tables as T
from .run import CALIB_SEED, GROUP_LAYERS, N_CALIB, N_EVAL, chunks, environment, load

CODECS = ("rtn", "gptq", "awq")
LEVELS = Q.LEVELS
BUDGETS = (3, 4)
DAMP = 0.01
MAP_BITS = 8  # one byte per matrix: a planned arm's width map


# ----------------------------------------------------------------------------- samples


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def windows(tok, text_dir: str) -> tuple:
    """(calibration windows, evaluation windows, hashes): the sample identities."""
    raw = {n: open(f"{text_dir}/{n}", "rb").read() for n in ("train.txt", "test.txt")}
    calib = chunks(tok, raw["train.txt"].decode("utf-8"), N_CALIB)
    evalq = chunks(tok, raw["test.txt"].decode("utf-8"), N_EVAL)

    def ids_sha(ws):
        return _sha(b"".join(np.asarray(w, dtype=np.int64).tobytes() for w in ws))

    hashes = {
        "train.txt": _sha(raw["train.txt"]),
        "test.txt": _sha(raw["test.txt"]),
        "calibration_windows": {"n": len(calib), "sha256": ids_sha(calib)},
        "evaluation_windows": {"n": len(evalq), "sha256": ids_sha(evalq)},
    }
    return calib, evalq, hashes


# ----------------------------------------------------------------------------- statistics


class _InputStats:
    """Forward hooks summing x^T x and |x| of each module's input (float64)."""

    def __init__(self, modules: dict):
        self.sums, self.handles = {}, []
        for name, m in modules.items():
            i = m.weight.shape[1]
            dev = m.weight.device
            self.sums[name] = [
                torch.zeros(i, i, device=dev, dtype=torch.float64),
                torch.zeros(i, device=dev, dtype=torch.float64),
                0,
            ]
            self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            s = self.sums[name]
            s[0] += x.T @ x
            s[1] += x.abs().sum(0)
            s[2] += x.shape[0]

        return hook

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def input_stats(model, modules: dict, calib: list) -> dict:
    """{matrix: {"S": E[x x^T], "A": E|x|}} over the calibration windows, float32."""
    acc = _InputStats(modules)
    try:
        for ids in calib:
            model(ids, use_cache=False)
    finally:
        acc.close()
    return {
        n: {"S": (s[0] / s[2]).float(), "A": (s[1] / s[2]).float()}
        for n, s in acc.sums.items()
    }


def encode(codec: str, w: torch.Tensor, st: dict | None, bits: int):
    """(weights, awq alpha or None) of ``codec`` at ``bits``; float32."""
    if codec == "rtn":
        return Q.rtn(w, bits), None
    if codec == "gptq":
        return Q.gptq(w, st["S"], bits, damp=DAMP), None
    if codec == "awq":
        return Q.awq(w, st["S"], st["A"], bits)
    raise ValueError(f"unknown codec {codec!r}")


# ----------------------------------------------------------------------------- phases


def _setup(a):
    from transformers import AutoTokenizer

    os.makedirs(a.out, exist_ok=True)
    env = environment()
    ep = os.path.join(a.out, "env.json")
    if os.path.exists(ep) and json.load(open(ep)) != env:
        raise SystemExit(f"started on {json.load(open(ep))}, now {env}")
    json.dump(env, open(ep, "w"))
    tok = AutoTokenizer.from_pretrained(a.model_path)
    calib, evalq, hashes = windows(tok, a.text)
    hp = os.path.join(a.out, "hashes.json")
    if os.path.exists(hp) and json.load(open(hp)) != hashes:
        raise SystemExit("the sample identities differ from this run's hashes.json")
    json.dump(hashes, open(hp, "w"), indent=1)
    dev = a.device
    return [c[None].to(dev) for c in calib], [c[None].to(dev) for c in evalq], hashes


def tables(a) -> int:
    calib, _, _ = _setup(a)
    ref = load(a.model_path, a.device)
    mods = T.linear_modules(ref)
    cp = os.path.join(a.out, "codec_costs.jsonl")
    done = set()
    if os.path.exists(cp):
        for line in open(cp, encoding="utf-8"):
            done.add(json.loads(line)["matrix"])
    n_layers = len(ref.model.layers)
    with open(cp, "a", encoding="utf-8") as fo:
        for grp in T.layer_groups(n_layers, GROUP_LAYERS):
            sel = {n: m for n, m in mods.items() if int(n.split(".")[1]) in grp}
            if all(n in done for n in sel):
                continue
            t0 = time.time()
            ist = input_stats(ref, sel, calib)
            acc = T.Accumulator(sel)
            gen = torch.Generator(device=a.device).manual_seed(CALIB_SEED)
            try:
                for ids in calib:
                    ref.zero_grad(set_to_none=True)
                    T.sampled_nll(ref, ids, gen).backward()
            finally:
                acc.close()
            fish = acc.finalized()
            ref.zero_grad(set_to_none=True)
            for name in sel:
                if name in done:
                    continue
                w = sel[name].weight.detach().float()
                F = fish[name]["F"]
                row = {"matrix": name, "numel": int(w.numel()), "cost": {}, "alpha": {}}
                for codec in CODECS:
                    row["cost"][codec], row["alpha"][codec] = {}, {}
                    for b in LEVELS:
                        wq, alpha = encode(codec, w, ist[name], b)
                        d = wq - w
                        row["cost"][codec][str(b)] = float((F * d * d).sum())
                        if alpha is not None:
                            row["alpha"][codec][str(b)] = alpha
                fo.write(json.dumps(row) + "\n")
                fo.flush()
            del ist, fish, acc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[tables] layers {grp[0]}-{grp[-1]} {time.time() - t0:.0f}s",
                flush=True,
            )
    print("TABLES_DONE", flush=True)
    return 0


def cost_table(rows: list, codec: str, model: str) -> dict:
    return {
        "schema": "tqp.weight_cost_table/1",
        "model": model,
        "predictor": f"fisher/{codec}",
        "matrices": {
            r["matrix"]: {"numel": r["numel"], "group": Q.GROUP} for r in rows
        },
        "costs": {r["matrix"]: r["cost"][codec] for r in rows},
        "provenance": {"codec": codec, "levels": list(LEVELS)},
    }


def make_plans(rows: list, model: str = "") -> dict:
    """{arm: {"codec", "budget", "bits", "stored_bits", "budget_bits"}} at every budget."""
    from turboquant_pro import weight_plan as W

    tabs = {c: W.CostTable.from_dict(cost_table(rows, c, model)) for c in CODECS}
    names = [r["matrix"] for r in rows]
    t0 = tabs["rtn"]
    arms = {}
    for b in BUDGETS:
        budget = W.budget_for_rate(t0, b)
        planned_budget = budget - MAP_BITS * len(names)
        plan = {c: W.solve(tabs[c], planned_budget) for c in CODECS}
        spec = {f"{c}_u{b}": (c, {n: b for n in names}, budget, 0) for c in CODECS}
        spec.update({f"{c}_f{b}": (c, plan[c].bits, budget, MAP_BITS) for c in CODECS})
        spec[f"gptq_frtn{b}"] = ("gptq", plan["rtn"].bits, budget, MAP_BITS)
        spec[f"gptq_seq_u{b}"] = ("gptq_seq", {n: b for n in names}, budget, 0)
        for arm, (codec, bits, bud, per_matrix) in spec.items():
            stored = sum(t0.size(n, bits[n]) for n in names) + per_matrix * len(names)
            if stored > bud:  # gate G1: never over the budget
                raise ValueError(f"{arm}: {stored} stored bits exceed the budget {bud}")
            arms[arm] = {
                "codec": codec,
                "budget": b,
                "bits": dict(bits),
                "stored_bits": int(stored),
                "budget_bits": int(bud),
                "map_bits": per_matrix,
            }
    return arms


def plans(a) -> int:
    rows = [json.loads(line) for line in open(os.path.join(a.out, "codec_costs.jsonl"))]
    arms = make_plans(rows, a.model_key)
    json.dump(
        arms, open(os.path.join(a.out, "arms.json"), "w"), indent=1, sort_keys=True
    )
    for arm, v in sorted(arms.items()):
        print(f"[plans] {arm:14s} {v['stored_bits'] / v['budget_bits']:.4f} of budget")
    return 0


def _reset(ref_mods: dict, var_mods: dict) -> None:
    for n, m in var_mods.items():
        m.weight.copy_(ref_mods[n].weight)


@torch.no_grad()
def arms(a) -> int:
    from .measure import kl_per_sequence

    calib, evalq, _ = _setup(a)
    arms_spec = json.load(open(a.arms_file or os.path.join(a.out, "arms.json")))
    ref = load(a.model_path, a.device)
    var = load(a.model_path, a.device)
    rm, vm = T.linear_modules(ref), T.linear_modules(var)
    n_layers = len(ref.model.layers)
    rp = os.path.join(a.out, "arms_results.jsonl")
    done = set()
    if os.path.exists(rp):
        for line in open(rp, encoding="utf-8"):
            done.add(json.loads(line)["arm"])
    only = set(a.only.split(",")) if a.only else None
    with open(rp, "a", encoding="utf-8") as fo:
        for arm, spec in sorted(arms_spec.items()):
            if arm in done or (only and arm not in only):
                continue
            t0 = time.time()
            _reset(rm, vm)
            codec = spec["codec"]
            base = "gptq" if codec == "gptq_seq" else codec
            for grp in T.layer_groups(n_layers, GROUP_LAYERS):
                names = [n for n in rm if int(n.split(".")[1]) in grp]
                st = None
                if base != "rtn":
                    src = var if codec == "gptq_seq" else ref
                    st = input_stats(
                        src, {n: (vm if src is var else rm)[n] for n in names}, calib
                    )
                for n in names:
                    wq, _ = encode(
                        base, rm[n].weight.float(), st and st[n], spec["bits"][n]
                    )
                    vm[n].weight.copy_(wq.to(vm[n].weight.dtype))
                del st
            per = kl_per_sequence(ref, var, evalq)
            fo.write(json.dumps({"arm": arm, "seqs": per}) + "\n")
            fo.flush()
            kl = sum(s["kl_sum"] for s in per) / sum(s["tokens"] for s in per)
            print(f"[arm] {arm} kl/token {kl:.4f} {time.time() - t0:.0f}s", flush=True)
    print("ARMS_DONE", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tables", "plans", "arms"))
    ap.add_argument("--model-path")
    ap.add_argument("--model-key", default="")
    ap.add_argument("--text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="", help="arms: comma list of arms to run")
    ap.add_argument(
        "--arms-file",
        default="",
        help="arms: the plans to encode (default <out>/arms.json)",
    )
    a = ap.parse_args(argv)
    return {"tables": tables, "plans": plans, "arms": arms}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
