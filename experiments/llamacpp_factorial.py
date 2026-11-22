# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Why does the Q/K-vs-V/O ordering differ? One factor at a time (E4).

Rababah et al. rank Q/K above V/O by weight statistics under llama.cpp K-quant
presets; our matched-bit study ranks V/O above Q/K by output KL under equal-bit
per-output-channel absmax. Three things differ at once: the bit allocation, the
quantization scheme and the sensitivity metric. This crosses them.

At each llama.cpp level L (Q2_K, Q3_K_M by default), four arms:

  kquant_mix    the llama.cpp preset as shipped (its per-tensor type mixture)
  kquant_pure   the same K-quant type for every tensor (llama-quantize --pure)
  chan_mix      per-output-channel absmax, each projection at the bit width of
                its tensor type in kquant_mix (read from the GGUF itself)
  chan_equal    per-output-channel absmax, every projection at the level's bits

and for each arm and each projection P in {Q, K, V, O} (only P perturbed, in
every layer), both readouts:

  weight metrics  |d kurtosis|, |d skewness|, histogram KL, 1 - cosine and
                  relative Frobenius error of the weight entries, layer mean
  functional      mean KL(p_base || p_quant) of the next-token distribution and
                  the top-1 flip rate, on the matched-bit study's test set

The K-quant weights are dequantized from GGUF files made by llama-quantize from
an f16 conversion of the same checkpoint. llama.cpp's converter permutes the
rows of Q and K for its rotary layout; the inverse is applied and checked
against the f16 GGUF, which must reproduce the HF weights exactly.

Usage (GPU for the functional readout):
    NB_MODEL=unsloth/Llama-3.2-3B NB_GGUF_DIR=/path/with/ggufs \
        python llamacpp_factorial.py
expects <tag>-f16.gguf, <tag>-<LEVEL>.gguf and <tag>-<LEVEL>-pure.gguf.

The run can also be split in two, for a small CPU pod beside the GGUF files and a
GPU pod with no host memory to spare (NRP). Both phases compute what the single run
computes, one tensor at a time:

  NB_PHASE=weights     CPU. Base weights from NB_BASE (a glob of safetensors files,
                       cast to float16 as from_pretrained(dtype=float16) does), the
                       f16 round-trip check, every weight metric, the tensor types,
                       the K-quant arms' weights as float16 .npy files striped over
                       NB_ARM_DIRS by layer, and hashes of some per-channel arm
                       weights. Writes e4_weights_<tag>.json.
  NB_PHASE=functional  GPU. Reads that file (NB_WEIGHTS) and the .npy files,
                       quantizes the per-channel arms on the GPU (checked against the
                       hashes), and writes llamacpp_factorial_<tag>.json in the single
                       run's format.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import exp_io
import numpy as np
import torch

PROJ = {"Q": "attn_q", "K": "attn_k", "V": "attn_v", "O": "attn_output"}
HF = {"Q": "q_proj", "K": "k_proj", "V": "v_proj", "O": "o_proj"}
LEVEL_BITS = {"Q2_K": 2, "Q3_K_M": 3, "Q3_K": 3, "Q4_K_M": 4}
TYPE_BITS = {
    "Q2_K": 2,
    "Q3_K": 3,
    "Q4_K": 4,
    "Q5_K": 5,
    "Q6_K": 6,
    "Q4_0": 4,
    "Q5_0": 5,
    "Q8_0": 8,
}


def quantize_per_outchannel(W: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    amax = np.maximum(np.abs(W).max(axis=1, keepdims=True), 1e-8)
    scale = amax / qmax
    return np.clip(np.round(W / scale), -qmax - 1, qmax) * scale


def quantize_per_outchannel_t(W: torch.Tensor, bits: int) -> torch.Tensor:
    """The same quantizer on a float32 tensor, with the same IEEE operations. The step
    divides by a tensor: a CUDA tensor divided by a Python number is multiplied by the
    number's reciprocal, which differs from numpy's division in the last bit."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / torch.full_like(amax, qmax)
    return torch.clamp(torch.round(W / scale), -qmax - 1, qmax) * scale


def unpermute(w: np.ndarray, n_head: int) -> np.ndarray:
    """Inverse of convert_hf_to_gguf's LlamaModel.permute for one Q/K matrix.

    The converter reads the rows as (n_head, 2, half) and writes them as
    (n_head, half, 2); the inverse reads (n_head, half, 2) and swaps back."""
    rows = w.shape[0]
    return (
        w.reshape(n_head, rows // n_head // 2, 2, *w.shape[1:])
        .swapaxes(1, 2)
        .reshape(w.shape)
    )


def read_attn(path: str, n_layers: int, n_head: int, n_kv: int):
    """Dequantized attention projections and their ggml type names."""
    from gguf import GGUFReader
    from gguf.quants import dequantize

    rd = GGUFReader(path)
    by_name = {t.name: t for t in rd.tensors}
    out, types = {p: [] for p in PROJ}, {p: [] for p in PROJ}
    for i in range(n_layers):
        for p, g in PROJ.items():
            t = by_name[f"blk.{i}.{g}.weight"]
            w = dequantize(t.data, t.tensor_type).astype(np.float32)
            w = w.reshape(int(t.shape[1]), int(t.shape[0]))  # ne is reversed
            if p == "Q":
                w = unpermute(w, n_head)
            elif p == "K":
                w = unpermute(w, n_kv)
            out[p].append(w)
            types[p].append(t.tensor_type.name)
    return out, types


def read_one(by_name: dict, i: int, p: str, n_head: int, n_kv: int):
    """One dequantized attention projection (float32, HF row order) and its type."""
    from gguf.quants import dequantize

    t = by_name[f"blk.{i}.{PROJ[p]}.weight"]
    w = dequantize(t.data, t.tensor_type).astype(np.float32)
    w = w.reshape(int(t.shape[1]), int(t.shape[0]))
    if p == "Q":
        w = unpermute(w, n_head)
    elif p == "K":
        w = unpermute(w, n_kv)
    return w, t.tensor_type.name


def moments(x: np.ndarray):
    x = x.astype(np.float64).ravel()
    m, s = x.mean(), x.std() + 1e-30
    z = (x - m) / s
    return float((z**3).mean()), float((z**4).mean() - 3.0)


METRICS = ("d_kurtosis", "d_skewness", "hist_kl", "one_minus_cos", "rel_fro")
ARMS = ("kquant_mix", "kquant_pure", "chan_mix", "chan_equal")


def tensor_metrics(w0: np.ndarray, w1: np.ndarray) -> dict:
    """Weight metrics of one quantized tensor against its base."""
    s0, k0 = moments(w0)
    s1, k1 = moments(w1)
    lo, hi = float(w0.min()), float(w0.max())
    h0, _ = np.histogram(w0, bins=256, range=(lo, hi))
    h1, _ = np.histogram(np.clip(w1, lo, hi), bins=256, range=(lo, hi))
    p0 = (h0 + 1e-10) / (h0 + 1e-10).sum()
    p1 = (h1 + 1e-10) / (h1 + 1e-10).sum()
    a, b = w0.ravel().astype(np.float64), w1.ravel().astype(np.float64)
    return {
        "d_kurtosis": abs(k1 - k0),
        "d_skewness": abs(s1 - s0),
        "hist_kl": float((p0 * np.log(p0 / p1)).sum()),
        "one_minus_cos": 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
        "rel_fro": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
    }


def layer_mean(per_layer: list[dict]) -> dict:
    return {k: float(np.mean([m[k] for m in per_layer])) for k in METRICS}


def weight_metrics(base: list[np.ndarray], quant: list[np.ndarray]) -> dict:
    return layer_mean([tensor_metrics(w0, w1) for w0, w1 in zip(base, quant)])


def functional(model, attns, batch, base_lp, p, weights, base_w, device) -> dict:
    """Mean KL and top-1 flip rate with projection p of every layer set to `weights`,
    then p restored from `base_w`. Weights are float32 arrays or tensors."""
    mods = [getattr(a, HF[p]) for a in attns]

    def put(m, w):
        w = torch.from_numpy(w) if isinstance(w, np.ndarray) else w
        m.weight.data.copy_(w.to(m.weight.device, m.weight.dtype))

    for m, w in zip(mods, weights):
        put(m, w)
    kls, flips, ns = [], [], []
    with torch.no_grad():
        for ids, b_cpu in zip(batch, base_lp):
            v = torch.log_softmax(
                model(ids.to(device).unsqueeze(0)).logits[0].float(), -1
            )
            b = b_cpu.to(device)
            kl = (b.exp() * (b - v)).sum(-1)
            kls.append(float(kl.mean()))
            flips.append(float((b.argmax(-1) != v.argmax(-1)).float().mean()))
            ns.append(int(kl.numel()))
    for m, w in zip(mods, base_w):
        put(m, w)
    w8 = np.asarray(ns, dtype=np.float64)
    return {
        "kl": float(np.average(kls, weights=w8)),
        "flip": float(np.average(flips, weights=w8)),
    }


def ratios(rec: dict) -> None:
    """The Q/K over V/O ratios of one arm, added to `rec` in place."""
    for key in ("kl",):
        qk = (rec["Q"]["functional"][key] + rec["K"]["functional"][key]) / 2
        vo = (rec["V"]["functional"][key] + rec["O"]["functional"][key]) / 2
        rec["functional_qk_over_vo"] = qk / max(vo, 1e-30)
    for key in rec["Q"]["weight"]:
        qk = (rec["Q"]["weight"][key] + rec["K"]["weight"][key]) / 2
        vo = (rec["V"]["weight"][key] + rec["O"]["weight"][key]) / 2
        rec[f"weight_{key}_qk_over_vo"] = qk / max(vo, 1e-30)


def report(level: str, arm: str, rec: dict) -> None:
    print(
        f"  {level} {arm:12s}"
        f" functional QK/VO={rec['functional_qk_over_vo']:.3f}"
        f" relfro QK/VO={rec['weight_rel_fro_qk_over_vo']:.3f}"
        f" histKL QK/VO={rec['weight_hist_kl_qk_over_vo']:.3f}"
        f" dkurt QK/VO={rec['weight_d_kurtosis_qk_over_vo']:.3f}"
    )


def save(model_name, levels, n_seqs, n_seq, results, out_dir, tag, device) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"llamacpp_factorial_{tag}.json")
    rec = {
        "model": model_name,
        "levels": levels,
        "n_seqs": n_seqs,
        "window": n_seq,
        "ratios": (
            "mean over Q and K divided by mean over V and O; below 1 = V/O larger"
        ),
        "results": results,
    }
    if os.environ.get("NB_PHASE", "all") != "all":
        rec["phases"] = "weights on CPU (NB_PHASE=weights), functional on GPU"
        rec["device"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[saved] {path}")
    return path


def config_dims(model_dir: str) -> tuple[int, int, int]:
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    return (
        cfg["num_hidden_layers"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
    )


class BaseWeights:
    """HF attention weights one tensor at a time, cast to float16 as the model holds
    them, returned as float32 arrays."""

    def __init__(self, pattern: str):
        from safetensors import safe_open

        self.files = {}
        for fn in sorted(glob.glob(pattern)):
            with safe_open(fn, framework="pt") as f:
                for k in f.keys():
                    self.files[k] = fn
        if not self.files:
            raise SystemExit(f"no safetensors match {pattern}")

    def get(self, i: int, p: str) -> np.ndarray:
        from safetensors import safe_open

        k = f"model.layers.{i}.self_attn.{HF[p]}.weight"
        with safe_open(self.files[k], framework="pt") as f:
            return f.get_tensor(k).to(torch.float16).float().numpy()


def arm_path(arm_dirs: list[str], level: str, arm: str, p: str, i: int) -> str:
    return os.path.join(arm_dirs[i % len(arm_dirs)], level, arm, f"{p}_{i:02d}.npy")


def f16_sha(w: np.ndarray) -> str:
    b = np.ascontiguousarray(w.astype(np.float16)).tobytes()
    return hashlib.sha256(b).hexdigest()


def weights_phase(model_name, tag, gdir, levels, out_dir) -> str:
    """CPU: everything but the functional readout, one tensor at a time."""
    from gguf import GGUFReader

    n_layers, n_head, n_kv = config_dims(os.environ["NB_MODEL_DIR"])
    base = BaseWeights(os.environ["NB_BASE"])
    arm_dirs = [d for d in os.environ.get("NB_ARM_DIRS", "").split(",") if d]
    t0 = time.time()

    def names(path):
        return {t.name: t for t in GGUFReader(path).tensors}

    f16 = names(os.path.join(gdir, f"{tag}-f16.gguf"))
    rd = {
        (lv, kind): names(os.path.join(gdir, f"{tag}-{lv}{suf}.gguf"))
        for lv in levels
        for kind, suf in (("mix", ""), ("pure", "-pure"))
    }
    per = {lv: {a: {p: [] for p in PROJ} for a in ARMS} for lv in levels}
    types = {lv: {k: {p: [] for p in PROJ} for k in ("mix", "pure")} for lv in levels}
    chan_sha = {}
    for i in range(n_layers):
        for p in PROJ:
            w0 = base.get(i, p)
            wf, _ = read_one(f16, i, p, n_head, n_kv)
            err = float(np.abs(wf - w0).max())
            assert err == 0.0, f"f16 round trip failed for {p} layer {i}: {err}"
            for lv in levels:
                mix, tmix = read_one(rd[(lv, "mix")], i, p, n_head, n_kv)
                pure, tpure = read_one(rd[(lv, "pure")], i, p, n_head, n_kv)
                types[lv]["mix"][p].append(tmix)
                types[lv]["pure"][p].append(tpure)
                arms = {
                    "kquant_mix": mix,
                    "kquant_pure": pure,
                    "chan_mix": quantize_per_outchannel(w0, TYPE_BITS[tmix]),
                    "chan_equal": quantize_per_outchannel(w0, LEVEL_BITS[lv]),
                }
                for a, w1 in arms.items():
                    per[lv][a][p].append(tensor_metrics(w0, w1))
                    if a.startswith("kquant") and arm_dirs:
                        path = arm_path(arm_dirs, lv, a, p, i)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        np.save(path + ".part.npy", w1.astype(np.float16))
                        os.replace(path + ".part.npy", path)
                    if a.startswith("chan") and i in (0, n_layers - 1):
                        chan_sha[f"{lv}/{a}/{p}/{i}"] = f16_sha(w1)
        print(f"  layer {i} done at {time.time() - t0:.0f}s", flush=True)
    out = {
        "model": model_name,
        "levels": levels,
        "f16_check": "exact (every Q/K/V/O tensor of every layer)",
        "weight": {
            lv: {a: {p: layer_mean(per[lv][a][p]) for p in PROJ} for a in ARMS}
            for lv in levels
        },
        "types": types,
        "chan_sha256": chan_sha,
        "arm_dirs": arm_dirs,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"e4_weights_{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")
    return path


def functional_phase(model_name, tag, n_samples, n_seq, out_dir, device) -> str:
    """GPU: the functional readout, from the weights phase's file and arm weights."""
    W = json.load(open(os.environ["NB_WEIGHTS"]))
    levels = W["levels"]
    arm_dirs = [d for d in os.environ.get("NB_ARM_DIRS", "").split(",") if d]
    arm_dirs = arm_dirs or W["arm_dirs"]
    store = device if os.environ.get("NB_RESIDENT", "0") == "1" else "cpu"
    t0 = time.time()
    model, tok = exp_io.model_and_tokenizer(model_name, torch.float16, device)
    attns = [
        m for m in model.modules() if hasattr(m, "q_proj") and hasattr(m, "o_proj")
    ]
    n_layers = len(attns)
    if store == "cpu":
        base = {
            p: [getattr(a, HF[p]).weight.data.float().cpu().numpy() for a in attns]
            for p in PROJ
        }
    else:
        base = {p: [getattr(a, HF[p]).weight.data.clone() for a in attns] for p in PROJ}
    print(f"[load] {model_name} in {time.time() - t0:.1f}s store={store}")
    batch = exp_io.token_batch(tok, "test", n_samples, n_seq)
    base_lp = []
    with torch.no_grad():
        for ids in batch:
            lg = model(ids.to(device).unsqueeze(0)).logits[0]
            base_lp.append(torch.log_softmax(lg.float(), -1).to(store))

    def kquant(lv, a, p):
        def one(i):
            w = np.load(arm_path(arm_dirs, lv, a, p, i))
            return torch.from_numpy(w).to(device)

        with ThreadPoolExecutor(max(1, len(set(arm_dirs)))) as ex:
            return list(ex.map(one, range(n_layers)))

    def chan(lv, a, p):
        out = []
        for i in range(n_layers):
            if a == "chan_mix":
                bits = TYPE_BITS[W["types"][lv]["mix"][p][i]]
            else:
                bits = LEVEL_BITS[lv]
            w0 = torch.as_tensor(base[p][i]).to(device).float()
            w1 = quantize_per_outchannel_t(w0, bits)
            key = f"{lv}/{a}/{p}/{i}"
            if key in W["chan_sha256"]:
                got = f16_sha(w1.half().float().cpu().numpy())
                if got != W["chan_sha256"][key]:
                    raise SystemExit(f"{key}: GPU quantizer differs from the CPU one")
            out.append(w1.half())  # what the model holds; half the memory of float32
        return out

    results = {}
    for lv in levels:
        lev = {
            "types_mix": {p: sorted(set(W["types"][lv]["mix"][p])) for p in PROJ},
            "types_mix_by_layer": W["types"][lv]["mix"],
            "types_pure": {p: sorted(set(W["types"][lv]["pure"][p])) for p in PROJ},
            "arms": {},
        }
        for a in ARMS:
            rec = {}
            for p in PROJ:
                ws = kquant(lv, a, p) if a.startswith("kquant") else chan(lv, a, p)
                rec[p] = {
                    "weight": W["weight"][lv][a][p],
                    "functional": functional(
                        model, attns, batch, base_lp, p, ws, base[p], device
                    ),
                }
                del ws
            ratios(rec)
            lev["arms"][a] = rec
            report(lv, a, rec)
        results[lv] = lev
        print(f"[types] {lv} mix: {lev['types_mix']}  pure: {lev['types_pure']}")
    return save(model_name, levels, len(batch), n_seq, results, out_dir, tag, device)


def run_all(model_name, tag, levels, n_samples, n_seq, out_dir, device, thermal):
    """The single run (Atlas): every array in memory at once."""
    gdir = os.environ["NB_GGUF_DIR"]
    t0 = time.time()
    model, tok = exp_io.model_and_tokenizer(model_name, torch.float16, device)
    cfg = model.config
    n_layers, n_head, n_kv = (
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
    )
    attns = [
        m for m in model.modules() if hasattr(m, "q_proj") and hasattr(m, "o_proj")
    ]
    assert len(attns) == n_layers
    base = {
        p: [getattr(a, HF[p]).weight.data.float().cpu().numpy() for a in attns]
        for p in PROJ
    }
    print(f"[load] {model_name} in {time.time() - t0:.1f}s")

    f16, _ = read_attn(os.path.join(gdir, f"{tag}-f16.gguf"), n_layers, n_head, n_kv)
    for p in PROJ:
        err = max(float(np.abs(a - b).max()) for a, b in zip(f16[p], base[p]))
        assert err == 0.0, f"f16 round trip failed for {p}: max abs diff {err}"
    del f16
    print("[check] f16 GGUF reproduces the HF attention weights exactly")

    batch = exp_io.token_batch(tok, "test", n_samples, n_seq)
    base_lp = []
    with torch.no_grad():
        for ids in batch:
            lg = model(ids.to(device).unsqueeze(0)).logits[0]
            base_lp.append(torch.log_softmax(lg.float(), -1).cpu())

    results = {}
    for level in levels:
        mix, mix_types = read_attn(
            os.path.join(gdir, f"{tag}-{level}.gguf"), n_layers, n_head, n_kv
        )
        pure, pure_types = read_attn(
            os.path.join(gdir, f"{tag}-{level}-pure.gguf"), n_layers, n_head, n_kv
        )
        mix_bits = {p: [TYPE_BITS[t] for t in mix_types[p]] for p in PROJ}
        b_eq = LEVEL_BITS[level]
        arms = {
            "kquant_mix": mix,
            "kquant_pure": pure,
            "chan_mix": {
                p: [quantize_per_outchannel(w, b) for w, b in zip(base[p], mix_bits[p])]
                for p in PROJ
            },
            "chan_equal": {
                p: [quantize_per_outchannel(w, b_eq) for w in base[p]] for p in PROJ
            },
        }
        lev = {
            "types_mix": {p: sorted(set(mix_types[p])) for p in PROJ},
            "types_mix_by_layer": mix_types,
            "types_pure": {p: sorted(set(pure_types[p])) for p in PROJ},
            "arms": {},
        }
        for arm, ws in arms.items():
            rec = {}
            for p in PROJ:
                if thermal is not None:
                    torch.set_num_threads(thermal.get_threads())
                rec[p] = {
                    "weight": weight_metrics(base[p], ws[p]),
                    "functional": functional(
                        model, attns, batch, base_lp, p, ws[p], base[p], device
                    ),
                }
            ratios(rec)
            lev["arms"][arm] = rec
            report(level, arm, rec)
        results[level] = lev
        print(f"[types] {level} mix: {lev['types_mix']}  pure: {lev['types_pure']}")
    return save(model_name, levels, len(batch), n_seq, results, out_dir, tag, device)


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "unsloth/Llama-3.2-3B")
    phase = os.environ.get("NB_PHASE", "all")
    tag = os.environ.get("NB_TAG", model_name.split("/")[-1])
    levels = os.environ.get("NB_LEVELS", "Q2_K,Q3_K_M").split(",")
    n_samples = int(os.environ.get("NB_SAMPLES", "32"))
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    llama_cpp = os.environ.get("NB_LLAMA_CPP", "/home/claude/llama.cpp")
    sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(target_temp=80.0, max_threads=8, min_threads=2)
        thermal.start()
        torch.set_num_threads(thermal.get_threads())
    except Exception as e:
        n_thr = exp_io.fallback_threads()
        print(f"[thermal] controller unavailable ({e}); capping at {n_thr} threads")
        torch.set_num_threads(n_thr)

    if phase == "weights":
        weights_phase(model_name, tag, os.environ["NB_GGUF_DIR"], levels, out_dir)
    elif phase == "functional":
        functional_phase(model_name, tag, n_samples, n_seq, out_dir, device)
    elif phase == "all":
        run_all(model_name, tag, levels, n_samples, n_seq, out_dir, device, thermal)
    else:
        raise SystemExit(f"NB_PHASE={phase}: weights, functional or all")
    if thermal is not None:
        try:
            thermal.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
