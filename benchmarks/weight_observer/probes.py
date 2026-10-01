"""Part III-c, the registered prereg's reported (not scored) probes for GPTQ (section 2).

    python -m weight_observer.probes make   --plans P --costs COSTS --out SPECS          (CPU)
    python -m weight_observer.probes cache  --model-path M --windows W --specs SPECS --cache C
    python -m weight_observer.probes verify --model-path M --windows W --cache C --plans P --registered R --out O
    python -m weight_observer.probes eval   --model-path M --windows W --cache C --specs SPECS --out O
    python -m weight_observer.probes search --model-path M --windows W --cache C --plans P --costs COSTS --out O
    python -m weight_observer.probes score  --registered R --plans P --costs COSTS --probes O --out J  (CPU)

**The code cache.** One-shot GPTQ's codes for a matrix at a width depend on the matrix, its
calibration statistics (from the full-precision model) and the width, never on the plan. So
``cache`` runs the harness's own per-group path once (``codec_run.input_stats``, then
``encode_unit("gptq")``, which encodes a shared-input set at every width as one stack, as the
cost tables and the arms do) and stores each matrix's codes at every width in the harness
dtype. Any plan is then assembled from the cache and measured with ``kl_per_sequence``,
exactly as ``codec_run arms`` measures an arm. ``verify`` is the proof that the two paths are
one: the cache-assembled ``gptq_f3``, ``gptq_f4`` and ``gptq_u4`` must reproduce the registered
arms' per-sequence KL bit for bit, and ``eval`` and ``search`` refuse to run without it.

The probes (section 2, reported, not scored):

- **additivity under GPTQ**: every (matrix, width) a planned GPTQ arm uses (``gptq_f`` and
  ``gptq_frtn`` at both budgets), quantized alone, the rest at full precision; each planned
  arm's measured KL against the sum of its matrices' single KLs;
- **the flatness curve around gptq_f**: ``flatness.perturb`` (same-type swaps, budget exact)
  with Part III's ``KS``, ``DRAWS`` and ``SEED``, at both budgets, the anchor re-measured in
  the same run;
- **the per-matrix bound**: from ``gptq_f4``, 64 seeded single swaps (same type, budget
  exact), each kept only if the mean KL over the 48 scored sequences falls. Resumes by
  replaying its log, and refuses a log the replay does not reproduce.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from . import codec_run as CR
from . import flatness as FL
from . import tables as T
from .measure import kl_per_sequence
from .run import GROUP_LAYERS, load

SEARCH_SEED = 20261001
SEARCH_STEPS = 64
PLANNED = ("gptq_f3", "gptq_f4", "gptq_frtn3", "gptq_frtn4")
VERIFY = ("gptq_f3", "gptq_f4", "gptq_u4")


def _cache_file(cache: str, name: str, bits: int) -> str:
    return os.path.join(cache, f"{name}@{bits}.pt")


def cache(a) -> int:
    """The GPTQ codes of every (matrix, width) the probes can reach (``specs["needed"]``), by
    the harness's own per-group path; one file per pair, resumable per group."""
    need = {}
    for m, b in json.load(open(a.specs))["needed"]:
        need.setdefault(m, set()).add(b)
    calib, _, _ = CR.load_windows(a.windows)
    calib = [c[None].to(a.device) for c in calib]
    ref = load(a.model_path, a.device)
    rm = T.linear_modules(ref)
    os.makedirs(a.cache, exist_ok=True)
    for grp in T.layer_groups(len(ref.model.layers), GROUP_LAYERS):
        names = [n for n in rm if int(n.split(".")[1]) in grp]
        if all(
            os.path.exists(_cache_file(a.cache, n, b))
            for n in names
            for b in need.get(n, ())
        ):
            continue
        t0 = time.time()
        st = CR.input_stats(ref, {n: rm[n] for n in names}, calib)
        for unit in CR.units(names):
            ws = {n: rm[n].weight.float() for n in unit}
            enc = CR.encode_unit("gptq", ws, st)
            for n in unit:
                for b in sorted(need.get(n, ())):
                    tmp = _cache_file(a.cache, n, b) + ".tmp"
                    torch.save(enc[n][b][0].to(rm[n].weight.dtype).cpu(), tmp)
                    os.replace(tmp, _cache_file(a.cache, n, b))
            del enc
        del st
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[cache] layers {grp[0]}-{grp[-1]} {time.time() - t0:.0f}s", flush=True)
    print("CACHE_DONE", flush=True)
    return 0


class Assembler:
    """The reference model, a variant and the code cache: ``measure(bits)`` sets exactly the
    matrices in ``bits`` from the cache, every other matrix to the reference's weights.
    """

    def __init__(self, a):
        _, evalq, _ = CR.load_windows(a.windows)
        self.evalq = [e[None].to(a.device) for e in evalq]
        self.ref = load(a.model_path, a.device)
        self.var = load(a.model_path, a.device)
        self.rm, self.vm = T.linear_modules(self.ref), T.linear_modules(self.var)
        self.cache = a.cache
        self.budget = int(float(getattr(a, "lru_gib", 0) or 24) * 2**30)
        self._codes, self._bytes = OrderedDict(), 0

    def codes(self, name: str, bits: int) -> torch.Tensor:
        """From a byte-bounded LRU: consecutive plans share most (matrix, width) pairs."""
        key = (name, bits)
        if key in self._codes:
            self._codes.move_to_end(key)
            return self._codes[key]
        t = torch.load(_cache_file(self.cache, name, bits), weights_only=True)
        self._codes[key] = t
        self._bytes += t.numel() * t.element_size()
        while self._bytes > self.budget and len(self._codes) > 1:
            _, old = self._codes.popitem(last=False)
            self._bytes -= old.numel() * old.element_size()
        return t

    @torch.no_grad()
    def measure(self, bits: dict) -> list:
        for n, m in self.vm.items():
            src = self.codes(n, bits[n]) if n in bits else self.rm[n].weight
            m.weight.copy_(src.to(m.weight.device))
        return kl_per_sequence(self.ref, self.var, self.evalq)


def _per_token(seqs: list) -> float:
    return sum(s["kl_sum"] for s in seqs) / sum(s["tokens"] for s in seqs)


def _jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def _registered(path: str) -> dict:
    return {r["arm"]: r["seqs"] for r in _jsonl(path)}


def verify(a) -> int:
    """The cache path IS the arms path: the cache-assembled arms reproduce the registered
    per-sequence KL bit for bit."""
    plans = json.load(open(a.plans))
    reg = _registered(a.registered)
    asm = Assembler(a)
    out = {}
    for arm in VERIFY:
        got = asm.measure(plans[arm]["bits"])
        same = [x["kl_sum"] == y["kl_sum"] for x, y in zip(got, reg[arm])]
        diff = max(
            abs(x["kl_sum"] - y["kl_sum"]) / x["tokens"] for x, y in zip(got, reg[arm])
        )
        out[arm] = {
            "identical": int(sum(same)),
            "of": len(same),
            "max_abs_diff_per_token": diff,
        }
        print(
            f"[verify] {arm}: {sum(same)}/{len(same)} identical, max |diff| {diff:.3e}",
            flush=True,
        )
    ok = all(v["identical"] == v["of"] == len(asm.evalq) for v in out.values())
    json.dump(
        {"arms": out, "identical": ok},
        open(os.path.join(a.out, "verify.json"), "w"),
        indent=1,
    )
    if not ok:
        raise SystemExit("the cache path does not reproduce the registered arms: stop")
    print("VERIFY_PASS", flush=True)
    return 0


def _verified(out: str) -> None:
    p = os.path.join(out, "verify.json")
    if not (os.path.exists(p) and json.load(open(p))["identical"]):
        raise SystemExit(
            f"{p}: run verify first (the cache must reproduce the registered arms)"
        )


def _numel(costs: str) -> dict:
    return {r["matrix"]: r["numel"] for r in _jsonl(costs)}


def make_specs(plans: dict, numel: dict) -> dict:
    """The flatness plans around gptq_f3 and gptq_f4, and the single-matrix sweep."""
    rng = np.random.default_rng(FL.SEED)
    flat, moved = {}, {}
    for b in (3, 4):
        anchor = plans[f"gptq_f{b}"]["bits"]
        flat[f"f{b}-anchor"] = anchor
        for k in FL.KS:
            for d in range(FL.DRAWS):
                q = f"f{b}-k{k:02d}d{d}"
                flat[q], moved[q] = FL.perturb(anchor, numel, k, rng)
    single = sorted({(m, b) for arm in PLANNED for m, b in plans[arm]["bits"].items()})
    # Swaps permute widths within a matrix type, so from an anchor every matrix can reach
    # exactly the widths its type holds there (search and flatness start from gptq_f3/f4).
    needed = {
        (m, b) for arm in (*PLANNED, *VERIFY) for m, b in plans[arm]["bits"].items()
    }
    for anchor in ("gptq_f3", "gptq_f4"):
        bits = plans[anchor]["bits"]
        by_type = {}
        for m, b in bits.items():
            by_type.setdefault(m.split(".")[-1], set()).add(b)
        needed |= {(m, b) for m in bits for b in by_type[m.split(".")[-1]]}
    return {
        "needed": sorted(needed),
        "flatness": flat,
        "moved": moved,
        "single": [{"matrix": m, "bits": b} for m, b in single],
        "seed": FL.SEED,
        "ks": list(FL.KS),
        "draws": FL.DRAWS,
    }


def make(a) -> int:
    specs = make_specs(json.load(open(a.plans)), _numel(a.costs))
    json.dump(specs, open(a.out, "w"), indent=1, sort_keys=True)
    print(
        f"[make] flatness {len(specs['flatness'])} plans, single {len(specs['single'])}, "
        f"cache {len(specs['needed'])} (matrix, width)"
    )
    return 0


def evaluate(a) -> int:
    """The flatness plans, then the single-matrix sweep; one line each, resumable."""
    _verified(a.out)
    specs = json.load(open(a.specs))
    asm = Assembler(a)
    for kind, items in (
        ("flatness", [(k, v) for k, v in sorted(specs["flatness"].items())]),
        (
            "single",
            [
                (f"{s['matrix']}@{s['bits']}", {s["matrix"]: s["bits"]})
                for s in specs["single"]
            ],
        ),
    ):
        path = os.path.join(a.out, f"{kind}.jsonl")
        done = {r["id"] for r in _jsonl(path)}
        with open(path, "a", encoding="utf-8") as fo:
            for i, (pid, bits) in enumerate(items):
                if pid in done:
                    continue
                t0 = time.time()
                per = asm.measure(bits)
                fo.write(json.dumps({"id": pid, "seqs": per}) + "\n")
                fo.flush()
                print(
                    f"[{kind}] {i + 1}/{len(items)} {pid} {_per_token(per):.5f} {time.time() - t0:.0f}s",
                    flush=True,
                )
    print("EVAL_DONE", flush=True)
    return 0


def search(a) -> int:
    """64 seeded single same-type swaps from gptq_f4, each kept only if the mean KL falls.
    On resume the log is replayed: the same rng must propose the same swaps."""
    _verified(a.out)
    plans = json.load(open(a.plans))
    numel = _numel(a.costs)
    path = os.path.join(a.out, "search.jsonl")
    log = _jsonl(path)
    rng = np.random.default_rng(SEARCH_SEED)
    cur = dict(plans["gptq_f4"]["bits"])
    asm = None
    if not log:
        asm = Assembler(a)
        per = asm.measure(cur)
        log.append({"step": 0, "swap": None, "seqs": per, "accepted": True})
        open(path, "a", encoding="utf-8").write(json.dumps(log[0]) + "\n")
    cur_kl = _per_token(log[0]["seqs"])
    with open(path, "a", encoding="utf-8") as fo:
        for step in range(1, SEARCH_STEPS + 1):
            prop, _ = FL.perturb(cur, numel, 1, rng)
            swap = sorted(m for m in cur if prop[m] != cur[m])
            if step < len(log):
                rec = log[step]
                if rec["swap"] != swap:
                    raise SystemExit(
                        f"step {step}: the log's swap {rec['swap']} is not the replayed {swap}"
                    )
            else:
                asm = asm or Assembler(a)
                t0 = time.time()
                per = asm.measure(prop)
                rec = {
                    "step": step,
                    "swap": swap,
                    "seqs": per,
                    "accepted": _per_token(per) < cur_kl,
                }
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                print(
                    f"[search] {step}/{SEARCH_STEPS} {_per_token(per):.5f} vs {cur_kl:.5f} "
                    f"{'kept' if rec['accepted'] else 'rejected'} {time.time() - t0:.0f}s",
                    flush=True,
                )
            if rec["accepted"]:
                cur, cur_kl = prop, _per_token(rec["seqs"])
    json.dump(
        {"bits": cur, "kl_per_token": cur_kl},
        open(os.path.join(a.out, "search_best.json"), "w"),
        indent=1,
    )
    print("SEARCH_DONE", flush=True)
    return 0


def _seq(seqs: list) -> np.ndarray:
    return np.array([s["kl_sum"] / s["tokens"] for s in seqs])


def _rel_ci(x: np.ndarray, y: np.ndarray, rng, n: int = 10_000) -> tuple:
    """mean(x)/mean(y) - 1 and its 95% paired percentile bootstrap (sequences resampled)."""
    idx = rng.integers(0, len(x), (n, len(x)))
    r = x[idx].mean(axis=1) / y[idx].mean(axis=1) - 1
    return (
        float(x.mean() / y.mean() - 1),
        float(np.percentile(r, 2.5)),
        float(np.percentile(r, 97.5)),
    )


def score_dir(
    registered: str, plans: dict, numel: dict, probes: str, specs: dict
) -> dict:
    rng = np.random.default_rng(0)
    reg = {k: _seq(v) for k, v in _registered(registered).items()}
    flat = {
        r["id"]: _seq(r["seqs"]) for r in _jsonl(os.path.join(probes, "flatness.jsonl"))
    }
    single = {
        r["id"]: float(_seq(r["seqs"]).mean())
        for r in _jsonl(os.path.join(probes, "single.jsonl"))
    }
    out = {"additivity": {}, "flatness": {}, "search": {}}

    def additive(bits):
        return sum(single[f"{m}@{b}"] for m, b in bits.items())

    for arm in PLANNED:
        meas = float(reg[arm].mean())
        add = additive(plans[arm]["bits"])
        out["additivity"][arm] = {
            "measured": meas,
            "sum_single": add,
            "ratio": meas / add,
        }
    for b in (3, 4):
        anchor = flat[f"f{b}-anchor"]
        rows = {
            "anchor_vs_registered_identical": bool(
                np.array_equal(anchor, reg[f"gptq_f{b}"])
            ),
            "k": {},
        }
        for k in specs["ks"]:
            ids = [f"f{b}-k{k:02d}d{d}" for d in range(specs["draws"])]
            x = np.mean([flat[i] for i in ids], axis=0)
            rel, lo, hi = _rel_ci(x, anchor, rng)
            rows["k"][k] = {
                "kl_rise_rel": rel,
                "ci": [lo, hi],
                "kl_rise_rel_draws": [
                    float(flat[i].mean() / anchor.mean() - 1) for i in ids
                ],
                "bits_moved": float(np.mean([specs["moved"][i] for i in ids])),
                "additive_rise_rel": float(
                    np.mean(
                        [
                            additive(specs["flatness"][i])
                            / additive(specs["flatness"][f"f{b}-anchor"])
                            - 1
                            for i in ids
                        ]
                    )
                ),
            }
        out["flatness"][f"{b}-bit"] = rows
    log = _jsonl(os.path.join(probes, "search.jsonl"))
    if log:
        start = _seq(log[0]["seqs"])
        best = start
        for r in log[1:]:
            if r["accepted"]:
                best = _seq(r["seqs"])
        rel, lo, hi = _rel_ci(best, start, rng)
        out["search"] = {
            "steps": len(log) - 1,
            "accepted": sum(1 for r in log[1:] if r["accepted"]),
            "start_identical_to_registered": bool(
                np.array_equal(start, reg["gptq_f4"])
            ),
            "start_kl": float(start.mean()),
            "best_kl": float(best.mean()),
            "best_vs_start": {"rel": rel, "ci": [lo, hi]},
            "trajectory": [round(float(_seq(r["seqs"]).mean()), 6) for r in log],
        }
    return out


def score(a) -> int:
    got = score_dir(
        a.registered,
        json.load(open(a.plans)),
        _numel(a.costs),
        a.probes,
        json.load(open(a.specs)),
    )
    json.dump(got, open(a.out, "w"), indent=1)
    print(
        json.dumps(
            {
                k: (v if k != "search" else {x: v[x] for x in v if x != "trajectory"})
                for k, v in got.items()
            },
            indent=1,
        )[:4000]
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd", choices=("cache", "verify", "make", "eval", "search", "score")
    )
    ap.add_argument("--model-path", default="")
    ap.add_argument("--windows", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--plans", default="", help="the registered plans (arms.json)")
    ap.add_argument(
        "--registered", default="", help="the registered arms_results.jsonl"
    )
    ap.add_argument("--costs", default="", help="the registered codec_costs.jsonl")
    ap.add_argument("--specs", default="")
    ap.add_argument("--probes", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lru-gib", default="24", help="host memory for cached codes")
    a = ap.parse_args(argv)
    if a.cmd in ("verify", "eval", "search"):
        os.makedirs(a.out, exist_ok=True)
    return {
        "cache": cache,
        "verify": verify,
        "make": make,
        "eval": evaluate,
        "search": search,
        "score": score,
    }[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
