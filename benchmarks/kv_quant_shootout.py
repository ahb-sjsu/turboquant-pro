# %% [markdown]
# # KV-Cache Quantization Shootout: scheme-level comparison
#
# A **single, portable notebook** comparing KV-cache quantization *schemes* head-to-head
# on the metric that actually matters: **end-task perplexity at matched bit-width**,
# alongside **memory** and **reconstruction fidelity**.
#
# ### Honest scope (read this first)
# * This compares the **quantization *algorithms***, reimplemented faithfully from each
#   paper: **KIVI** (2-bit asym, per-channel key / per-token value, group-wise, recent
#   fp16 window); **KVQuant** (pre-RoPE per-channel NUQ keys + 1% dense-sparse outliers,
#   per-token value); and two **TurboQuant-style** scalar schemes (uniform K4/V2 +
#   two-tier fp16 sink/hot, and a `+` variant adding NUQ keys + outliers).
# * These are **scheme reimplementations, not the authors' kernels.** In particular the
#   "TurboQuant" rows here are a **uniform-scalar approximation** of TurboQuant's
#   *configuration* (K4/V2 + two-tier) -- **NOT** the real library's rotation + Lloyd-Max
#   (PolarQuant) quantizer, which is a *post-RoPE KV-cache* method and is benchmarked
#   separately (see the "Real library" note below).
# * Quality is measured by **fake-quantization** (quantize->dequantize the KV during a
#   real forward pass) -- the standard way to ablate quantization quality without bespoke
#   kernels. Runs on **any GPU or CPU / Colab**.
# * **Throughput is out of scope** (needs vendor kernels; hardware-specific). Memory is
#   **analytical** (exact: bits x elements). Small default model + short context ->
#   results are **directional**.

# %%
# !pip install -q torch transformers pandas
# optional: !pip install -q datasets    # for real wikitext (else an embedded sample is used)

# %%
import math
import os

import pandas as pd
import torch

torch.manual_seed(0)
MODEL = os.environ.get("NB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SEQ_LEN = int(os.environ.get("NB_SEQ", "512"))
N_CHUNKS = int(os.environ.get("NB_CHUNKS", "4"))
print(f"model={MODEL} device={DEVICE} dtype={DTYPE} seq={SEQ_LEN} chunks={N_CHUNKS}")

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()

text = None
try:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if len(t) > 160)
except Exception as e:
    print("datasets unavailable -> embedded public-domain sample:", str(e)[:80])
    base = (
        "It is a truth universally acknowledged, that a single man in possession of a "
        "good fortune, must be in want of a wife. However little known the feelings or "
        "views of such a man may be on his first entering a neighbourhood, this truth is "
        "so well fixed in the minds of the surrounding families, that he is considered as "
        "the rightful property of some one or other of their daughters. "
    )
    text = base * 400
ids = tok(text, return_tensors="pt").input_ids[0]
chunks = [ids[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(N_CHUNKS)]
chunks = [c for c in chunks if len(c) == SEQ_LEN]
print("usable chunks:", len(chunks), "ctx:", SEQ_LEN)

# %% [markdown]
# ## Quantizers (fake quant-dequant).
# `axis` = the dim reduced to compute the scale: per-**channel** key quant reduces over
# tokens (axis=1); per-**token** value quant reduces over channels (axis=2). `group`
# enables block-wise scales (as KIVI/KVQuant actually do).


# %%
def qdq_uniform(x, bits, axis, asym=True, group=0):
    n = x.shape[axis]
    if group and n != group and n % group == 0:
        xt = x.transpose(axis, -1)
        b = xt.shape[:-1]
        xt = xt.reshape(*b, n // group, group)
        q = qdq_uniform(xt, bits, axis=-1, asym=asym, group=0)
        return q.reshape(*b, n).transpose(axis, -1).contiguous()
    qmax = 2**bits - 1
    if asym:
        mn = x.amin(dim=axis, keepdim=True)
        mx = x.amax(dim=axis, keepdim=True)
        scale = (mx - mn).clamp_min(1e-8) / qmax
        return ((x - mn) / scale).round().clamp(0, qmax) * scale + mn
    mx = x.abs().amax(dim=axis, keepdim=True).clamp_min(1e-8)
    lvl = 2 ** (bits - 1) - 1
    scale = mx / lvl
    return (x / scale).round().clamp(-lvl, lvl) * scale


def qdq_nuq(x, bits, axis=1):
    # Non-uniform (KVQuant-style): per-channel centroids = quantiles over the token dim.
    levels = 2**bits
    qs = torch.linspace(0, 1, levels, device=x.device, dtype=torch.float32)
    cent = (
        torch.quantile(x.float(), qs, dim=axis).permute(1, 2, 0).contiguous()
    )  # [B,C,levels]
    B, T, C = x.shape
    ce = cent.unsqueeze(1).expand(B, T, C, levels)
    idx = (x.unsqueeze(-1) - ce).abs().argmin(dim=-1, keepdim=True)
    return torch.gather(ce, 3, idx).squeeze(-1).to(x.dtype)


def outlier_mask(x, frac=0.01):
    n = x.numel()
    k = max(1, int(n * frac))
    if k >= n:
        return torch.ones_like(x, dtype=torch.bool)
    thr = x.abs().flatten().kthvalue(n - k + 1).values
    return x.abs() >= thr


# %% [markdown]
# ## Methods (applied at k_proj / v_proj outputs via forward hooks).

# %%
METHODS = [
    "fp16",
    "KIVI",
    "KVQuant",
    "TurboQuant(uniform-approx)",
    "TurboQuant+(uniform-approx)",
]
EFF_BITS = {}


def apply_method(method, x, is_key):
    B, T, C = x.shape
    G = 64
    if method == "fp16":
        EFF_BITS[method] = 16.0
        return x
    if method == "KIVI":
        R = min(32, T // 4)
        q = qdq_uniform(x, 2, axis=1 if is_key else 2, asym=True, group=G).clone()
        q[:, T - R :, :] = x[:, T - R :, :]
        EFF_BITS[method] = 2 * (1 - R / T) + 16 * (R / T)
        return q
    if method == "KVQuant":
        q = (
            qdq_nuq(x, 3, axis=1)
            if is_key
            else qdq_uniform(x, 3, axis=2, asym=True, group=G)
        )
        m = outlier_mask(x, 0.01)
        q = torch.where(m, x, q)
        EFF_BITS[method] = 3 * 0.99 + 16 * 0.01 + 0.16
        return q
    if (
        method == "TurboQuant(uniform-approx)"
    ):  # uniform K4/V2 group-wise + two-tier fp16 sink+hot
        S, H = 4, min(64, T // 4)
        q = qdq_uniform(
            x, 4 if is_key else 2, axis=1 if is_key else 2, asym=True, group=G
        ).clone()
        q[:, :S, :] = x[:, :S, :]
        q[:, T - H :, :] = x[:, T - H :, :]
        frac = (S + H) / T
        EFF_BITS[method] = 3.0 * (1 - frac) + 16 * frac
        return q
    if (
        method == "TurboQuant+(uniform-approx)"
    ):  # the two-tier idea + NUQ keys + dense-sparse outliers
        S, H = 4, min(64, T // 4)
        q = (
            qdq_nuq(x, 3, axis=1)
            if is_key
            else qdq_uniform(x, 3, axis=2, asym=True, group=G)
        )
        m = outlier_mask(x, 0.01)
        q = torch.where(m, x, q).clone()
        q[:, :S, :] = x[:, :S, :]
        q[:, T - H :, :] = x[:, T - H :, :]
        frac = (S + H) / T
        EFF_BITS[method] = (3 * 0.99 + 16 * 0.01 + 0.16) * (1 - frac) + 16 * frac
        return q
    return x


attn_projs = []
for name, mod in model.named_modules():
    if name.endswith(".self_attn.k_proj"):
        mod._kv = "key"
        attn_projs.append(mod)
    if name.endswith(".self_attn.v_proj"):
        mod._kv = "value"
        attn_projs.append(mod)
print("hooked KV projections:", len(attn_projs))

STATE = {"method": "fp16", "errs": []}


def hook(module, inp, out):
    q = apply_method(STATE["method"], out, module._kv == "key")
    if STATE["method"] != "fp16":
        STATE["errs"].append(((q - out).norm() / out.norm().clamp_min(1e-8)).item())
    return q


handles = [m.register_forward_hook(hook) for m in attn_projs]

# %% [markdown]
# ## Run: perplexity per method + reconstruction error (mean and max per-layer).


# %%
@torch.no_grad()
def perplexity(method):
    STATE["method"] = method
    STATE["errs"] = []
    losses = [
        model(
            input_ids=ch.unsqueeze(0).to(DEVICE), labels=ch.unsqueeze(0).to(DEVICE)
        ).loss.item()
        for ch in chunks
    ]
    e = STATE["errs"]
    return (
        math.exp(sum(losses) / len(losses)),
        (sum(e) / len(e) if e else 0.0),
        (max(e) if e else 0.0),
    )


rows = []
for m in METHODS:
    ppl, recon, mx = perplexity(m)
    rows.append(
        {
            "method": m,
            "eff_bits": round(EFF_BITS[m], 2),
            "mem_vs_fp16": round(16 / EFF_BITS[m], 1),
            "ppl": round(ppl, 3),
            "mean_recon_err": round(recon, 4),
            "max_recon_err": round(mx, 3),
        }
    )
    print(rows[-1])
for h in handles:
    h.remove()

# %%
df = pd.DataFrame(rows)
base = df.loc[df.method == "fp16", "ppl"].iloc[0]
df["dppl_vs_fp16"] = (df["ppl"] - base).round(3)
df = df[
    [
        "method",
        "eff_bits",
        "mem_vs_fp16",
        "ppl",
        "dppl_vs_fp16",
        "mean_recon_err",
        "max_recon_err",
    ]
]
print("\n================  KV-QUANT SHOOTOUT  ================")
print(df.to_string(index=False))

# %% [markdown]
# ## Reference result (Qwen2.5-1.5B-Instruct, 512 ctx, group-wise, CPU fp32)
# Your numbers vary with model/seed/context; the *ordering* is the point:
#
# | method | eff_bits | mem | ppl | dppl | recon |
# |---|---:|---:|---:|---:|---:|
# | fp16 | 16.0 | 1.0x | 12.24 | 0.00 | 0.000 |
# | KIVI | 2.88 | 5.6x | 26.86 | 14.63 | 0.359 |
# | **KVQuant** | **3.29** | **4.9x** | **15.36** | **3.13** | **0.183** |
# | TurboQuant(uniform-approx) | 4.73 | 3.4x | 17.51 | 5.28 | 0.266 |
# | TurboQuant+(uniform-approx) | 4.98 | 3.2x | 14.35 | 2.12 | 0.170 |
#
# **KVQuant beats a uniform-scalar scheme per bit** (3.29 bits / dPPL 3.1 vs 4.73 / 5.3);
# adding **two-tier + NUQ + outliers** (`+`) gets the lowest dPPL. `recon_err`
# *under-predicts* the PPL gap -- fidelity is necessary, not sufficient.

# %% [markdown]
# ## Real library (`turboquant_pro`) -- why it isn't in the table, and what we learned
# We *did* wire the actual `TurboQuantKV` (random rotation + Lloyd-Max **PolarQuant**)
# into this same pre-RoPE hook. Result, measured honestly:
#
# * **Values: near-lossless** (dPPL ~0.5) -- PolarQuant is excellent in its design domain.
# * **Keys: catastrophic** (PPL > 1e4) **despite faithful Euclidean reconstruction**
#   (max per-layer recon <= 0.23). Outlier-extraction and per-channel pre-scaling did
#   **not** fix it.
#
# The recon-vs-PPL paradox is a **domain mismatch, not a quantizer flaw**: TurboQuant is a
# *post-RoPE KV-cache* method (per-vector normalization tuned to post-RoPE statistics),
# but this hook feeds *pre-RoPE* keys (KVQuant's native domain). So the real library is
# **not comparable in this pre-RoPE harness** and is excluded rather than misrepresented.
# It is benchmarked properly in a dedicated **post-RoPE `TurboQuantKVCache` generate-loop
# harness** (`benchmark_kvcache_postrope.py`). **Takeaways that hold regardless:** value
# quantization is robust; **keys are the sensitive axis** (independently confirming the
# 1.1 "default keys to 4-bit" decision).

# %% [markdown]
# ## How to read this (honest)
# * **`dppl_vs_fp16`** read against **`eff_bits`** is the real comparison; winning dPPL
#   only by spending more bits is not a win.
# * **`max_recon_err`** exposes single-layer blowups the mean hides.
# * **Out of scope:** throughput (vendor kernels; hardware-specific). Memory is analytical.
# * **Caveats:** scheme reimplementations, not authors' kernels; all applied pre-RoPE
#   (KVQuant's native domain; see the real-library note); small model/seq -> directional.
