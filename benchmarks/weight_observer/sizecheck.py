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
            cfg, torch_dtype=torch.float16, attn_implementation="sdpa"
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
        "keep_fp16_sdpa": all(checks.values()),
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

    refm = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": dev},
        low_cpu_mem_usage=True,
    ).eval()
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
            "reference": "bfloat16",
            "harness": str(next(harness.parameters()).dtype),
        },
        "kl_attention_per_token": sum(s["kl_attention"] for s in seqs) / max(tokens, 1),
        "verdict": precision_verdict(seqs, layers),
        "layers": layers,
        "sequences": seqs,
    }
    json.dump(out, open(os.path.join(a.out, "refcheck.json"), "w"), indent=1)
    print("[refcheck]", json.dumps(out["verdict"]), flush=True)
    print("REFCHECK_DONE", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("memprobe", "refcheck"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    with HostMem(os.path.join(a.out, "host_mem.jsonl"), a.cmd):
        return {"memprobe": memprobe, "refcheck": refcheck}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
