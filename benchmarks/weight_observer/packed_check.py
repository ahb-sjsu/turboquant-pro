"""Part III-c follow-up: the KL of the stored (TQPW) weights, ``docs/CHECK_packed_weights_kl.md``.

    python -m weight_observer.packed_check run   --model-path M --windows W --arms-file P \\
        --registered R --out O [--only A,B]
    python -m weight_observer.packed_check score --results DIR --out J      (CPU)

``run`` encodes each GPTQ arm the product ships with the product encoder
(``turboquant_pro.weight_codec.encode_model``), measures the per-sequence KL with the
harness's ``kl_per_sequence`` (the codec measurement, compared with the registered arm),
then writes the stored form to a ``.tqpw`` file, reads it back, decodes it into the model
(``apply_packed``) and measures again (the decoded measurement). One line per arm in
``<out>/packed_results.jsonl``; a rerun skips finished arms.

``score`` judges criteria E, S and V of the check. It replays the registered scorer's
comparisons in their resampling order and must reproduce ``results_codec.json`` before it
substitutes the decoded arms.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ARMS = ("gptq_u3", "gptq_u4", "gptq_f3", "gptq_f4", "gptq_frtn3", "gptq_frtn4")
E_TOL = 1e-6  # nats per token: the registered G2 tolerance
S_BAND = 0.01  # the decoded mean KL within 1% of the codec's


def _jsonl(path: str) -> dict:
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["arm"]] = r
    return out


def run(a) -> int:
    import torch

    from turboquant_pro import packed_weights as PW
    from turboquant_pro import weight_codec as WC

    from . import codec_run as CR
    from . import tables as T
    from .measure import kl_per_sequence
    from .run import load

    calib, evalq, _ = CR._setup(a)
    arms = json.load(open(a.arms_file))
    registered = {k: v["seqs"] for k, v in _jsonl(a.registered).items()}
    rp = os.path.join(a.out, "packed_results.jsonl")
    done = set(_jsonl(rp))
    todo = [x for x in (a.only.split(",") if a.only else ARMS) if x not in done]
    if not todo:
        print("PACKED_DONE (nothing to do)", flush=True)
        return 0
    ref = load(a.model_path, a.device)
    var = load(a.model_path, a.device)
    rm, vm = T.linear_modules(ref), T.linear_modules(var)
    with open(rp, "a", encoding="utf-8") as fo:
        for arm in todo:
            spec = arms[arm]
            if spec["codec"] != "gptq":
                raise SystemExit(f"{arm} is not a GPTQ arm")
            t0 = time.time()
            with torch.no_grad():
                CR._reset(rm, vm)
            summary = WC.encode_model(var, {"bits": spec["bits"]}, calib, "gptq")
            with torch.no_grad():
                codec = json.loads(json.dumps(kl_per_sequence(ref, var, evalq)))
            path = os.path.join(a.out, f"{arm}.tqpw")
            size = PW.write(path, summary["packed"], {"codec": "gptq", "arm": arm})
            _, back = PW.read(path)
            os.remove(path)
            WC.apply_packed(var, back)
            with torch.no_grad():
                decoded = json.loads(json.dumps(kl_per_sequence(ref, var, evalq)))
            payload = sum(m.payload_bits for m in back)
            planned = spec["stored_bits"] - spec["map_bits"] * len(spec["bits"])
            if payload != planned:
                raise SystemExit(f"{arm}: payload {payload} bits, plan {planned}")
            row = {
                "arm": arm,
                "codec_seqs": codec,
                "decoded_seqs": decoded,
                "bit_identical_to_registered": codec == registered.get(arm),
                "payload_bits": payload,
                "tqpw_bytes": size,
                "grid_rounding": summary["grid_rounding"],
                "seconds": round(time.time() - t0, 1),
            }
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            mc = sum(s["kl_sum"] for s in codec) / sum(s["tokens"] for s in codec)
            md = sum(s["kl_sum"] for s in decoded) / sum(s["tokens"] for s in decoded)
            print(
                f"[packed] {arm} codec {mc:.5f} decoded {md:.5f} "
                f"identical {row['bit_identical_to_registered']} "
                f"{row['seconds']:.0f}s",
                flush=True,
            )
    print("PACKED_DONE", flush=True)
    return 0


# ----------------------------------------------------------------------------- score


def _comparisons(kl: dict, models) -> tuple:
    """The registered scorer's comparison loop (``score_codec.score``), same order, same
    generator: {key: compare}, {hypothesis: verdict}."""
    from . import score_codec as SC

    rng = np.random.default_rng(SC.SEED)
    comps, verdicts = {}, {}
    for h, (x, y) in SC.HYPOTHESES.items():
        per_model = {}
        for m in SC.MODELS:
            for b in SC.BUDGETS:
                ax, ay = f"{x}{b}", f"{y}{b}"
                if ax in kl.get(m, {}) and ay in kl.get(m, {}):
                    c = SC.compare(kl[m][ax], kl[m][ay], rng)
                    comps[f"{m}|{ax}|{ay}"] = c
                    per_model.setdefault(m, {})[b] = c["judgement"]
        verdicts[h] = SC.verdict(per_model, models)
    return comps, verdicts


def score(results_dir: str) -> dict:
    from . import score_codec as SC

    kl, packed = {}, {}
    for m in SC.MODELS:
        d = os.path.join(results_dir, m)
        kl[m] = {
            k: SC.per_seq(v["seqs"])
            for k, v in _jsonl(os.path.join(d, "arms_results.jsonl")).items()
        }
        packed[m] = _jsonl(os.path.join(d, "packed", "packed_results.jsonl"))
    reg = json.load(open(os.path.join(results_dir, "results_codec.json")))
    comps, verdicts = _comparisons(kl, SC.MODELS)
    stated = {h: v for h, v in reg["verdicts"].items() if v != "WITHHELD"}
    if comps != reg["comparisons"] or any(verdicts[h] != v for h, v in stated.items()):
        raise SystemExit("the replay does not reproduce results_codec.json")

    out = {"criteria": {"E_tol": E_TOL, "S_band": S_BAND}, "arms": {}}
    rng = np.random.default_rng(SC.SEED)
    e_ok = s_ok = complete = True
    sub = {m: dict(v) for m, v in kl.items()}
    for m in SC.MODELS:
        for arm in ARMS:
            r = packed[m].get(arm)
            if r is None:
                complete = False
                continue
            codec, dec = SC.per_seq(r["codec_seqs"]), SC.per_seq(r["decoded_seqs"])
            e_dev = float(np.abs(codec - kl[m][arm]).max())
            s = SC.compare(dec, codec, rng)
            e = e_dev <= E_TOL
            si = -S_BAND <= s["lo"] and s["hi"] <= S_BAND
            e_ok &= e
            s_ok &= si
            sub[m][arm] = dec
            out["arms"][f"{m}|{arm}"] = {
                "E": e,
                "E_max_dev": e_dev,
                "bit_identical_to_registered": r["bit_identical_to_registered"],
                "S": si,
                "decoded_vs_codec": {k: s[k] for k in ("rel", "lo", "hi")},
                "grid_rounding_max_steps": r["grid_rounding"]["max_steps"],
                "payload_bits": r["payload_bits"],
                "tqpw_bytes": r["tqpw_bytes"],
            }
    comps_d, verdicts_d = _comparisons(sub, SC.MODELS)
    cells = {
        k: {"registered": reg["comparisons"][k]["judgement"], "decoded": c["judgement"]}
        for k, c in comps_d.items()
    }
    v_ok = verdicts_d == verdicts and all(
        c["registered"] == c["decoded"] for c in cells.values()
    )
    out.update(
        {
            "comparisons_decoded": comps_d,
            "cells": cells,
            "verdicts": {"registered": verdicts, "decoded": verdicts_d},
            "E": e_ok,
            "S": s_ok,
            "V": v_ok,
            "complete": complete,
            "result": (
                "INCOMPLETE"
                if not complete
                else ("PASS" if e_ok and s_ok and v_ok else "FAIL")
            ),
        }
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "score"))
    ap.add_argument("--model-path")
    ap.add_argument("--windows", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--arms-file")
    ap.add_argument("--registered", help="run: the registered arms_results.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="")
    ap.add_argument("--results", help="score: results/codec")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        doc = score(a.results)
        json.dump(doc, open(a.out, "w"), indent=1, sort_keys=True)
        print(f"E {doc['E']} S {doc['S']} V {doc['V']} -> {doc['result']}", flush=True)
        return 0
    os.makedirs(a.out, exist_ok=True)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
