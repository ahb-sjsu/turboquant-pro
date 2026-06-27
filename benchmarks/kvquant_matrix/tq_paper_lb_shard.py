#!/usr/bin/env python3
"""Paper experiment runner: KV-key quantization quality across models/tasks/methods.

Unlike ``tq_enh_lb_shard.py`` (a faithful but slow simulation that re-quantizes the
settled window every decode step), this implements the **deployable** method:
**quantize the prefill KV once** and freeze it.

  * On the first cache update of each layer (= the prefill, the long prompt), the
    settled region ``[0 : T-HOT]`` is quantized one time: group-wise per-channel keys
    (NF4 or uniform) with the first ``SINK`` tokens kept fp16 and the top-``OUT_FRAC``
    magnitude entries **per channel** (global over the prefill) kept fp16.
  * The most recent ``HOT`` prefill tokens and **all generated tokens** stay fp16.

For LongBench (max_new_tokens <= HOT) generated tokens never leave the hot window, so
the cache quantizes exactly once per layer -> ~fp16 speed, and quality matches the
slow per-step simulation (verified on Llama-2-7B). Values use per-token uniform.

Config via env; sharded by SHARD_ID / NUM_SHARDS. Model via MODEL (HF id) and
MODEL_KEY (LongBench config key, e.g. ``llama2-7b-chat-4k``) for max-length/prompts.
"""
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = os.environ.get("MODEL", "NousResearch/Llama-2-7b-chat-hf")
MODEL_KEY = os.environ.get("MODEL_KEY", "llama2-7b-chat-4k")
LBROOT = os.environ.get("LBROOT", "/root/LongBench/LongBench")
DATADIR = os.environ.get("DATADIR", "/root/lb_data/data")
DATASETS = os.environ.get("DATASETS", "trec,triviaqa,qasper").split(",")

KB = int(os.environ.get("KEY_BITS", "4"))
VB = int(os.environ.get("VAL_BITS", "4"))
G = int(os.environ.get("GROUP", "32"))
HOT = int(os.environ.get("HOT", "128"))  # fp16 recent window (== RESID)
SINK = int(os.environ.get("SINK", "0"))
OUT_FRAC = float(os.environ.get("OUTLIER_FRAC", "0.0"))
NUQ = int(os.environ.get("NUQ", "0"))
NOQUANT = int(os.environ.get("NOQUANT", "0"))
# Tier-1 levers:
#  CODEBOOK: uniform | nf4 | quantile | kmeans  (key codebook)
#    quantile/kmeans = ONLINE per-channel codebook fit to THIS prefill (calibration-free,
#    data-optimal -- the both-worlds answer to KVQuant's offline Fisher K-means).
#  PREROPE: quantize keys in pre-RoPE space (RoPE smears per-channel structure).
CODEBOOK = os.environ.get("CODEBOOK", "nf4" if NUQ else "uniform")
PREROPE = int(os.environ.get("PREROPE", "0"))
KMEANS_ITERS = int(os.environ.get("KMEANS_ITERS", "8"))
TAG = os.environ.get("TAG", "v0")
SHARD = int(os.environ["SHARD_ID"])
NSH = int(os.environ["NUM_SHARDS"])
CHAT = int(os.environ.get("CHAT", "1"))  # wrap prompt in the model chat template
OUT = f"/root/out_{TAG}"
os.makedirs(OUT, exist_ok=True)

NF4 = torch.tensor(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ]
)

# LongBench model max-context (truncation) and the chat-wrap style per model family.
MODEL_MAXLEN = {
    "llama2-7b-chat-4k": 3500,
    "llama2-13b-chat-4k": 3500,
    "mistral-7b-instruct": 31500,
    "qwen2.5-7b-instruct": 31500,
}
MAXLEN = int(os.environ.get("MAXLEN", str(MODEL_MAXLEN.get(MODEL_KEY, 3500))))


def _group_pad(x, g):
    n = x.shape[2]
    pad = (g - n % g) % g
    if pad:
        x = torch.cat([x, x[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
    return x, n


def _quant_uniform_group(x, bits, g):
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mn = xg.amin(3, keepdim=True)
    mx = xg.amax(3, keepdim=True)
    qm = 2**bits - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    xq = ((xg - mn) / sc).round().clamp(0, qm) * sc + mn
    return xq.reshape(B, H, Tg * g, D)[:, :, :n0, :]


def _quant_nf4_group(x, g, nf4):
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    amax = xg.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xg / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _quant_nf4a_group(x, g, nf4):
    """Asymmetric / zero-point NF4: subtract the per-group per-channel MEAN (the DC offset
    that wastes symmetric NF4's codes), NF4-quantize the centered residual, add the mean
    back. Keeps NF4's nonlinear level placement AND handles offset KV distributions -> aims
    to be robust across MHA (Llama) and high-GQA (Qwen) models with one codebook.
    Metadata: per-group mean + absmax (2 fp16 scalars/group/channel vs 1 for symmetric)."""
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mu = xg.mean(3, keepdim=True)
    xc = xg - mu
    amax = xc.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xc / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax + mu
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _kmeans_1d(x, k, iters):
    """Per-row 1-D k-means (Lloyd). x: (R, n) -> centroids (R, k). Init = quantiles."""
    qs = torch.linspace(0.0, 1.0, k, device=x.device, dtype=x.dtype)
    cent = torch.quantile(x, qs, dim=1).T.contiguous()  # (R, k)
    for _ in range(iters):
        idx = (x.unsqueeze(-1) - cent.unsqueeze(1)).abs().argmin(-1)  # (R, n)
        oh = torch.nn.functional.one_hot(idx, k).to(x.dtype)  # (R, n, k)
        cnt = oh.sum(1)  # (R, k)
        s = torch.einsum("rn,rnk->rk", x, oh)  # (R, k)
        new = torch.where(cnt > 0, s / cnt.clamp_min(1), cent)
        cent = new
    return cent


def _quant_perchannel_codebook(x, bits, mode):
    """Online per-channel non-uniform codebook fit to this block (quantile or kmeans).

    Fit on a token subsample (codebook is robust to it); assign via searchsorted on the
    sorted codebook -> O(n log nlev), not O(n*nlev).
    """
    B, H, n, D = x.shape
    nlev = 2**bits
    xt = x.permute(0, 1, 3, 2).reshape(-1, n).float()  # (B*H*D, n) per-channel rows
    sub = xt[:, :: max(1, n // 1024)][:, :1024] if n > 1536 else xt
    if mode == "kmeans":
        cent = _kmeans_1d(sub, nlev, KMEANS_ITERS)  # (R, nlev)
    else:  # quantile
        qs = torch.linspace(0.0, 1.0, nlev, device=x.device, dtype=torch.float32)
        cent = torch.quantile(sub, qs, dim=1).T  # (R, nlev)
    cent = cent.sort(dim=1).values.contiguous()
    pos = torch.searchsorted(cent, xt.contiguous()).clamp(1, nlev - 1)
    left = torch.gather(cent, 1, pos - 1)
    right = torch.gather(cent, 1, pos)
    idx = torch.where((xt - left).abs() <= (right - xt).abs(), pos - 1, pos)
    deq = torch.gather(cent, 1, idx)  # (R, n)
    return deq.reshape(B, H, D, n).permute(0, 1, 3, 2).to(x.dtype)


def qdq_key_block(x):
    """Quantize the full settled key block once (global per-channel outliers + sink)."""
    B, H, n, D = x.shape
    if CODEBOOK == "nf4" and KB == 4:
        hq = _quant_nf4_group(x, G, NF4)
    elif CODEBOOK == "nf4a" and KB == 4:
        hq = _quant_nf4a_group(x, G, NF4)
    elif CODEBOOK in ("quantile", "kmeans"):
        hq = _quant_perchannel_codebook(x, KB, CODEBOOK)
    else:
        hq = _quant_uniform_group(x, KB, G)
    keep = torch.zeros(B, H, n, D, dtype=torch.bool, device=x.device)
    if SINK > 0:
        keep[:, :, : min(SINK, n), :] = True
    if OUT_FRAC > 0:
        k = max(1, int(round(n * OUT_FRAC)))
        absn = x.abs()
        thr = absn.kthvalue(n - k + 1, dim=2, keepdim=True).values
        keep |= absn >= thr
    return torch.where(keep, x, hq).to(x.dtype)


def qdq_val_block(x):
    mn = x.amin(3, keepdim=True)
    mx = x.amax(3, keepdim=True)
    qm = 2**VB - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    return (((x - mn) / sc).round().clamp(0, qm) * sc + mn).to(x.dtype)


_orig_update = DynamicCache.update


def _patched_update(self, k, v, li, cache_kwargs=None):
    fk, fv = _orig_update(self, k, v, li, cache_kwargs)
    if NOQUANT:
        return fk, fv
    # Quantize the settled prefill region ONCE, IN PLACE in the stored cache tensor.
    # `fk` is `self.key_cache[li]`; later decode steps cat new fp16 tokens onto it, so
    # the quantized settled region persists with zero per-step overhead (~fp16 speed).
    if not hasattr(self, "_qdone"):
        self._qdone = set()
    if li not in self._qdone:
        T = fk.shape[2]
        n = max(0, T - HOT)
        if n > 0:
            if not PREROPE:  # post-RoPE keys quantized here; pre-RoPE done in the rope hook
                fk[:, :, :n, :] = qdq_key_block(fk[:, :, :n, :])
            fv[:, :, :n, :] = qdq_val_block(fv[:, :, :n, :])
        self._qdone.add(li)
    return fk, fv


DynamicCache.update = _patched_update


def _make_prerope_hook(orig):
    def _hook(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if not NOQUANT:
            T = k.shape[2]
            if T > HOT:  # prefill: quantize settled keys once, pre-RoPE
                n = T - HOT
                k = k.clone()
                k[:, :, :n, :] = qdq_key_block(k[:, :, :n, :])
        return orig(q, k, cos, sin, position_ids, unsqueeze_dim)

    return _hook


def _install_prerope():
    """Quantize keys in **pre-RoPE** space: hook ``apply_rotary_pos_emb`` so the settled
    prefill keys are quantized before the position rotation is applied (RoPE smears the
    per-channel structure that per-channel quantization relies on).

    HF defines ``apply_rotary_pos_emb`` per model family and the attention forward looks
    it up from the module globals at call time, so monkeypatching the module attribute is
    sufficient. Patch every family we benchmark that is importable in this transformers."""
    mods = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
    ]
    patched = []
    import importlib

    for mp in mods:
        try:
            m = importlib.import_module(mp)
        except Exception:
            continue
        if hasattr(m, "apply_rotary_pos_emb"):
            m.apply_rotary_pos_emb = _make_prerope_hook(m.apply_rotary_pos_emb)
            patched.append(mp.split(".")[-2])
    print(f"[prerope] patched: {patched}", flush=True)


if PREROPE:
    _install_prerope()


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def build_chat(tok, prompt):
    if not CHAT:
        return prompt
    if "llama2" in MODEL_KEY:
        return f"[INST]{prompt}[/INST]"
    try:  # Mistral / Qwen / others: use the tokenizer chat template
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


def main():
    print(f"[shard {SHARD}/{NSH}] TAG={TAG} MODEL={MODEL_KEY} KB={KB} HOT={HOT} "
          f"SINK={SINK} OUT={OUT_FRAC} NUQ={NUQ} NOQUANT={NOQUANT}", flush=True)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=config, torch_dtype=torch.float16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, trust_remote_code=True,
    ).cuda().eval()
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
    except Exception as e:  # some fast tokenizers need a newer tokenizers lib; fall back
        print(f"[tok] fast load failed ({repr(e)[:80]}); using slow tokenizer", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)
    device = torch.device("cuda:0")
    d2p = json.load(open(f"{LBROOT}/config/dataset2prompt.json"))
    d2m = json.load(open(f"{LBROOT}/config/dataset2maxlen.json"))
    no_chat = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    for dataset in DATASETS:
        data = load_jsonl(f"{DATADIR}/{dataset}.jsonl")
        pf = d2p[dataset]
        mg = int(d2m[dataset])
        fo = open(f"{OUT}/{dataset}.{SHARD}.jsonl", "w")
        for gi, o in enumerate(data):
            if gi % NSH != SHARD:
                continue
            prompt = pf.format(**o)
            tp = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tp) > MAXLEN:
                h = MAXLEN // 2
                prompt = tok.decode(tp[:h], skip_special_tokens=True) + tok.decode(
                    tp[-h:], skip_special_tokens=True
                )
            if dataset not in no_chat:
                prompt = build_chat(tok, prompt)
            inp = tok(prompt, truncation=False, return_tensors="pt").to(device)
            cl = inp.input_ids.shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=mg, num_beams=1, do_sample=False,
                    temperature=1.0, pad_token_id=tok.eos_token_id,
                )[0]
            pred = tok.decode(out[cl:], skip_special_tokens=True)
            fo.write(json.dumps({"idx": gi, "pred": pred, "answers": o["answers"],
                                 "all_classes": o["all_classes"]}) + "\n")
            fo.flush()
        fo.close()
        print(f"[shard {SHARD}] {dataset} done", flush=True)
    print(f"SHARD_{SHARD}_DONE", flush=True)


if __name__ == "__main__":
    main()
