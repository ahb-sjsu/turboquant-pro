"""Part III-c, before registration: what a registered model's jobs hold on the GPU, and
whether the harness's reference precision is sound for it (docs/PREREG_weights_codec_allocation.md).

    python -m weight_observer.sizecheck memprobe --model-path M --out O
    python -m weight_observer.sizecheck refcheck --model-path M --text T --out O

``memprobe`` builds the model's architecture (its ``config.json``) with random weights,
made on the device in the harness's dtype, and runs the harness's own per-group code on
one group of layers: ``codec_run.table_rows`` (the body of ``tables``), then, on two
copies, ``codec_run.encode_group`` and ``measure.kl_per_sequence`` (the body of ``arms``)
for GPTQ and AWQ, the codecs with working memory. GPU memory depends on shapes, not
values, every group has the same shapes, and a window's working memory does not depend on
how many windows there are, so the peaks are the real jobs' peaks. No registered weights
are read, and nothing registered is quantized before registration.

``refcheck`` runs the full-precision model only, on the 48 scored windows. The harness
scores in fp16 with sdpa attention (``run.load``); the checkpoint's own reference is bf16
with eager attention. Per sequence it records KL(reference || harness), KL(reference ||
bf16 with sdpa) to separate the attention kernel from the dtype, and non-finite logits;
per decoder layer, the harness's largest finite |hidden state| and its non-finite count.

The rule for keeping fp16 + sdpa for a model was fixed before any refcheck ran
(``RULE``): (a) no non-finite logit or hidden state; (b) every layer's largest |hidden|
at most fp16's largest finite value over 8 (three bits of range to spare); (c) the mean
KL(reference || harness) at most 1.5e-4 nats per token, a tenth of the prereg's 5% bar
on the smallest KL an arm is expected to reach (about 0.03 nats per token: Part III's
Fisher plans at 4 bits measured 0.066 and 0.102, and GPTQ should be lower).

2026-09-29, after the first refchecks (Gemma-2-2B, Qwen2.5-3B): no overflow (4.5-4.8% of
the fp16 range), but (c) failed for both (2.9e-3, 1.2e-3). The rule's design was wrong:
bf16 (8-bit mantissa) is coarser than fp16 (11-bit), so KL(bf16 || fp16) mixes the
reference's own rounding with the harness's and cannot say which precision is off. Two
checks against fp32, the ground truth, were fixed here before they ran:
- ``refcheck --reference float32``: the same RULE, with an fp32 eager reference.
- ``invariance``: the pilot's arms measured in fp32 (``codec_run arms --dtype float32``)
  against the same arms in fp16. The fp16 harness is kept only if every one of the
  prereg's eight scored ratios (C1a, C1b, C2, C3 at both budgets) is within
  ``PRECISION_INVARIANCE_TOL`` (1%) of its fp32 value: a fifth of the prereg's 5% bar, and
  the size of Part III's run-to-run intervals.

    python -m weight_observer.sizecheck invariance --a FP16_DIR --b FP32_DIR --out O

Gemma-2-2B against fp32 (same day): KL(fp32 eager || fp16 sdpa) 7.4e-4, of which KL(fp32
eager || fp32 sdpa) is 7.2e-4: the gap is the attention kernel (sdpa drops Gemma-2's
logit soft-cap), fp16 adds at most ~3e-5. ``run.load`` now takes each architecture's
own kernel (``run.attention``: eager for gemma2); the same RULE is applied again to the
fp16 eager harness, and ``memprobe`` builds with the same kernel (eager holds more).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from . import codec_run as CR
from . import run as R
from . import tables as T
from .hostmem import HostMem
from .measure import kl_per_sequence

N_PROBE = 2  # windows per memprobe phase: working memory is per window
FP16_MAX = 65504.0
RULE = {"headroom_factor": 8, "max_kl_per_token": 1.5e-4}
PRECISION_INVARIANCE_TOL = 0.01
# The prereg's scored comparisons (section 2): (id, X, Y) at each budget.
COMPARISONS = (
    ("C1a", "gptq_f", "gptq_u"),
    ("C1b", "gptq_f", "awq_u"),
    ("C2", "gptq_f", "rtn_f"),
    ("C3", "gptq_f", "gptq_frtn"),
)


def _kl_mean(seqs: list) -> float:
    return sum(s["kl_sum"] for s in seqs) / sum(s["tokens"] for s in seqs)


def precision_invariance(a: dict, b: dict, budgets=(3, 4)) -> dict:
    """``a``, ``b``: {arm: per-sequence records} of the same arms in two precisions. Each
    scored ratio mean(KL_X) / mean(KL_Y) in ``a`` against ``b``; kept only if every one
    is within PRECISION_INVARIANCE_TOL."""
    rows = {}
    for cid, x, y in COMPARISONS:
        for bud in budgets:
            ra = _kl_mean(a[f"{x}{bud}"]) / _kl_mean(a[f"{y}{bud}"])
            rb = _kl_mean(b[f"{x}{bud}"]) / _kl_mean(b[f"{y}{bud}"])
            rows[f"{cid}@{bud}"] = {"a": ra, "b": rb, "rel_diff": ra / rb - 1}
    worst = max(abs(r["rel_diff"]) for r in rows.values())
    return {
        "comparisons": rows,
        "worst_rel_diff": worst,
        "tolerance": PRECISION_INVARIANCE_TOL,
        "keep_a": worst <= PRECISION_INVARIANCE_TOL,
    }


def _arms(d: str) -> dict:
    out = {}
    for line in open(os.path.join(d, "arms_results.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        out[r["arm"]] = r["seqs"]
    return out


def invariance(a) -> int:
    got = precision_invariance(_arms(a.a), _arms(a.b))
    got["a"], got["b"] = a.a, a.b
    json.dump(got, open(os.path.join(a.out, "invariance.json"), "w"), indent=1)
    for k, r in got["comparisons"].items():
        print(
            f"[invariance] {k:7s} {r['a']:.4f} vs {r['b']:.4f} ({r['rel_diff']:+.2%})"
        )
    print(
        "[invariance] keep fp16:", got["keep_a"], f"(worst {got['worst_rel_diff']:.2%})"
    )
    return 0


def _gib(x: float) -> float:
    return round(x / 2**30, 3)


def _cuda(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _fresh(device: str) -> None:
    if _cuda(device):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peaks(device: str) -> dict:
    """Peak allocated and reserved since the last ``_fresh``, and the CUDA context: what the
    driver holds beyond the allocator (the difference nvidia-smi shows)."""
    if not _cuda(device):
        return {"allocated_gib": None, "reserved_gib": None, "context_gib": None}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": _gib(torch.cuda.max_memory_allocated()),
        "reserved_gib": _gib(torch.cuda.max_memory_reserved()),
        "context_gib": _gib(total - free - torch.cuda.memory_reserved()),
    }


def _random_model(cfg, device: str):
    """The architecture in the harness's dtype and attention, made on the device (the
    host never holds it), frozen as ``run.load`` leaves a model."""
    from transformers import AutoModelForCausalLM

    with torch.device(device):
        m = AutoModelForCausalLM.from_config(
            cfg,
            torch_dtype=torch.float16,
            attn_implementation=R.attention(cfg.model_type),
        )
    m.eval()
    m.requires_grad_(False)
    return m


def memprobe(a) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(a.model_path)
    dev = a.device
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    def window():
        return torch.randint(0, cfg.vocab_size, (1, R.SEQ), generator=g).to(dev)

    calib = [window() for _ in range(N_PROBE)]
    evalq = [window() for _ in range(N_PROBE)]
    out = {
        "environment": R.environment(),
        "total_gib": _gib(torch.cuda.mem_get_info()[1]) if _cuda(dev) else None,
        "windows": N_PROBE,
        "seq": R.SEQ,
    }
    _fresh(dev)
    ref = _random_model(cfg, dev)
    out["weights_gib"] = _gib(
        sum(p.numel() * p.element_size() for p in ref.parameters())
    )
    out["after_load"] = _peaks(dev)
    rm = T.linear_modules(ref)
    grp = T.layer_groups(len(ref.model.layers), R.GROUP_LAYERS)[0]
    names = [n for n in rm if int(n.split(".")[1]) in grp]
    out["group"] = {"layers": list(grp), "matrices": len(names)}

    _fresh(dev)
    t0 = time.time()
    rows = list(CR.table_rows(ref, {n: rm[n] for n in names}, calib, dev))
    out["tables"] = {**_peaks(dev), "rows": len(rows), "seconds": time.time() - t0}
    print("[memprobe] tables", json.dumps(out["tables"]), flush=True)

    var = _random_model(cfg, dev)
    vm = T.linear_modules(var)
    out["arms"] = {}
    for codec in ("gptq", "awq"):
        _fresh(dev)
        t0 = time.time()
        spec = {"codec": codec, "bits": {n: 4 for n in names}}
        CR.encode_group(ref, var, rm, vm, names, spec, calib)
        per = kl_per_sequence(ref, var, evalq)
        out["arms"][codec] = {
            **_peaks(dev),
            "sequences": len(per),
            "seconds": time.time() - t0,
        }
        print(f"[memprobe] arms {codec}", json.dumps(out["arms"][codec]), flush=True)
    json.dump(out, open(os.path.join(a.out, "memprobe.json"), "w"), indent=1)
    print("MEMPROBE_DONE", flush=True)
    return 0


def _set_attention(model, impl: str) -> str:
    """Switch a loaded model's attention kernel; returns the one now in effect (recorded,
    so a switch that did not take is visible rather than read as a zero difference)."""
    try:
        model.set_attn_implementation(impl)
    except (AttributeError, ValueError, NotImplementedError):
        model.config._attn_implementation = impl
    return model.config._attn_implementation


def _kl(lr: torch.Tensor, lv: torch.Tensor) -> float:
    return float((lr.exp() * (lr - lv)).sum())


def precision_verdict(sequences: list, layers: list) -> dict:
    """The fixed rule (``RULE``) applied to refcheck's records."""
    tokens = sum(s["tokens"] for s in sequences)
    kl = sum(s["kl_harness"] for s in sequences) / max(tokens, 1)
    nonfinite = sum(s["nonfinite_logits"] for s in sequences) + sum(
        x["nonfinite"] for x in layers
    )
    amax = max((x["amax"] for x in layers), default=0.0)
    checks = {
        "a_finite": nonfinite == 0 and math.isfinite(kl),
        "b_headroom": amax <= FP16_MAX / RULE["headroom_factor"],
        "c_kl": math.isfinite(kl) and kl <= RULE["max_kl_per_token"],
    }
    return {
        "kl_harness_per_token": kl,
        "max_abs_hidden": amax,
        "fp16_range_used": amax / FP16_MAX,
        "nonfinite": nonfinite,
        "checks": checks,
        "keep_harness": all(checks.values()),  # fp16 with its own attention
        "rule": RULE,
    }


@torch.no_grad()
def refcheck(a) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model_path)
    test = open(os.path.join(a.text, "test.txt"), encoding="utf-8").read()
    evalq = R.chunks(tok, test, CR.N_EVAL)
    eval_sha = CR.ids_sha(evalq)
    evalq = [w[None].to(dev) for w in evalq]

    reference = getattr(a, "reference", "") or "bfloat16"
    refm = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": dev},
        low_cpu_mem_usage=True,
    ).eval()
    if reference != "bfloat16":  # the checkpoint is bf16: widening it is exact
        refm = refm.to(getattr(torch, reference))
    harness = R.load(a.model_path, dev)
    layers = [{"amax": 0.0, "nonfinite": 0} for _ in harness.model.layers]

    def hook(i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            fin = torch.isfinite(h)
            layers[i]["nonfinite"] += int((~fin).sum())
            if fin.any():
                layers[i]["amax"] = max(layers[i]["amax"], float(h[fin].abs().max()))

        return f

    handles = [
        ly.register_forward_hook(hook(i)) for i, ly in enumerate(harness.model.layers)
    ]
    seqs = []
    try:
        for ids in evalq:
            lr = torch.log_softmax(refm(ids, use_cache=False).logits.float(), -1)
            logits = harness(ids, use_cache=False).logits
            bad = int((~torch.isfinite(logits)).sum())
            lh = torch.log_softmax(logits.float(), -1)
            _set_attention(refm, "sdpa")
            ls = torch.log_softmax(refm(ids, use_cache=False).logits.float(), -1)
            _set_attention(refm, "eager")
            seqs.append(
                {
                    "kl_harness": _kl(lr, lh),
                    "kl_attention": _kl(lr, ls),
                    "tokens": int(lr.shape[1]),
                    "top1_agree": float(
                        (lr.argmax(-1) == lh.argmax(-1)).float().mean()
                    ),
                    "nonfinite_logits": bad,
                }
            )
            del lr, lh, ls, logits
    finally:
        for h in handles:
            h.remove()
    tokens = sum(s["tokens"] for s in seqs)
    out = {
        "environment": R.environment(),
        "evaluation_windows": {"n": len(seqs), "sha256": eval_sha},
        "attention": {
            "reference": _set_attention(refm, "eager"),
            "harness": harness.config._attn_implementation,
        },
        "dtype": {
            "reference": str(next(refm.parameters()).dtype).replace("torch.", ""),
            "harness": str(next(harness.parameters()).dtype),
        },
        "kl_attention_per_token": sum(s["kl_attention"] for s in seqs) / max(tokens, 1),
        "verdict": precision_verdict(seqs, layers),
        "layers": layers,
        "sequences": seqs,
    }
    name = "refcheck.json" if reference == "bfloat16" else f"refcheck_{reference}.json"
    json.dump(out, open(os.path.join(a.out, name), "w"), indent=1)
    print("[refcheck]", json.dumps(out["verdict"]), flush=True)
    print("REFCHECK_DONE", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("memprobe", "refcheck", "invariance"))
    ap.add_argument("--model-path", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reference", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--a", default="", help="invariance: fp16 arms directory")
    ap.add_argument("--b", default="", help="invariance: fp32 arms directory")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "invariance":
        return invariance(a)
    with HostMem(os.path.join(a.out, "host_mem.jsonl"), a.cmd):
        return {"memprobe": memprobe, "refcheck": refcheck}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
