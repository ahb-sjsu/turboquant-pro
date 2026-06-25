# Post-RoPE KV-quant harness: is PolarQuant (per-vector) the specific culprit for KEYS,
# and does PER-CHANNEL key quantization fix it? Quantizes post-RoPE keys (B,H,S,D) via a
# pluggable key-quantizer; values via PolarQuant (known near-lossless). Reports PPL.
import math
import os

import numpy as np
import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant_pro import TurboQuantKV

MODEL = os.environ.get("NB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SEQ = int(os.environ.get("NB_SEQ", "512"))
NCH = int(os.environ.get("NB_CHUNKS", "3"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
HD = (
    getattr(model.config, "head_dim", None)
    or model.config.hidden_size // model.config.num_attention_heads
)
NKV = model.config.num_key_value_heads
_TQ = {
    b: TurboQuantKV(head_dim=HD, n_heads=NKV, bits=b, use_gpu=False, seed=0)
    for b in (2, 3, 4)
}

try:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    txt = "\n\n".join(t for t in ds["text"] if len(t) > 160)
except Exception as e:
    print("datasets unavailable -> varied embedded sample:", str(e)[:70])
    txt = (
        (
            "It is a truth universally acknowledged, that a single man in possession of a good "
            "fortune, must be in want of a wife. "
        )
        + (
            "In physics, the principle of least action states the path taken is the one for "
            "which the action is stationary. "
        )
        + (
            "The mitochondrion is a double-membrane-bound organelle generating most of the "
            "cell's supply of adenosine triphosphate. "
        )
    ) * 400
ids = tok(txt, return_tensors="pt").input_ids[0]
chunks = [ids[i * SEQ : (i + 1) * SEQ] for i in range(NCH)]
chunks = [c for c in chunks if len(c) == SEQ]


# ---- key quantizers on post-RoPE keys k:[B,H,T,D]; "channel"=D (reduce over tokens T=dim2) ----
def kq_polar(k, bits):  # TurboQuant PolarQuant (per-vector direction quant)
    arr = k.contiguous().float().cpu().numpy()
    deq = _TQ[bits].decompress(_TQ[bits].compress(arr))
    return torch.from_numpy(np.ascontiguousarray(deq)).to(k.dtype).to(k.device)


def kq_perchan_uni(k, bits):  # per-channel asymmetric uniform
    mn = k.amin(dim=2, keepdim=True)
    mx = k.amax(dim=2, keepdim=True)
    qmax = 2**bits - 1
    scale = (mx - mn).clamp_min(1e-8) / qmax
    return ((k - mn) / scale).round().clamp(0, qmax) * scale + mn


def kq_perchan_nuq(k, bits):  # per-channel non-uniform (KVQuant-style)
    levels = 2**bits
    qs = torch.linspace(0, 1, levels, device=k.device, dtype=torch.float32)
    cent = (
        torch.quantile(k.float(), qs, dim=2).permute(1, 2, 3, 0).contiguous()
    )  # [B,H,D,levels]
    B, H, T, D = k.shape
    ce = cent.unsqueeze(2).expand(B, H, T, D, levels)
    idx = (k.unsqueeze(-1) - ce).abs().argmin(dim=-1, keepdim=True)
    return torch.gather(ce, 4, idx).squeeze(-1).to(k.dtype)


try:  # the SHIPPED fix module
    from turboquant_pro import PerChannelKV

    _PCK = {b: PerChannelKV(head_dim=HD, n_heads=NKV, bits=b) for b in (3, 4)}
    _PCKN = {
        b: PerChannelKV(head_dim=HD, n_heads=NKV, bits=b, nuq=True) for b in (3, 4)
    }

    def kq_module(k, bits):  # real PerChannelKV (uniform)
        arr = k.contiguous().float().cpu().numpy()
        return (
            torch.from_numpy(_PCK[bits].decompress(_PCK[bits].compress(arr)))
            .to(k.dtype)
            .to(k.device)
        )

    def kq_module_nuq(k, bits):  # real PerChannelKV (nuq)
        arr = k.contiguous().float().cpu().numpy()
        return (
            torch.from_numpy(_PCKN[bits].decompress(_PCKN[bits].compress(arr)))
            .to(k.dtype)
            .to(k.device)
        )

    HAVE_PCK = True
except Exception as ex:
    HAVE_PCK = False
    print("PerChannelKV module unavailable:", str(ex)[:80])


def two_tier(out, orig, S=4, H=64):
    H = min(H, orig.shape[2] // 4)
    out = out.clone()
    out[:, :, :S, :] = orig[:, :, :S, :]
    out[:, :, orig.shape[2] - H :, :] = orig[:, :, orig.shape[2] - H :, :]
    return out


STATE = {"keyfn": None, "keybits": 0, "valbits": 0, "errs": []}
_orig_rope = qwen2.apply_rotary_pos_emb


def patched_rope(q, k, cos, sin, unsqueeze_dim=1):
    q2, k2 = _orig_rope(q, k, cos, sin, unsqueeze_dim)
    if STATE["keyfn"] is not None:
        kq = two_tier(STATE["keyfn"](k2, STATE["keybits"]), k2)
        STATE["errs"].append(((kq - k2).norm() / k2.norm().clamp_min(1e-8)).item())
        return q2, kq
    return q2, k2


qwen2.apply_rotary_pos_emb = patched_rope


def vhook(module, inp, out):  # values via PolarQuant (no RoPE)
    if not STATE["valbits"]:
        return out
    B, T, C = out.shape
    x = out.reshape(B, T, NKV, HD).permute(0, 2, 1, 3)
    return kq_polar(x, STATE["valbits"]).permute(0, 2, 1, 3).reshape(B, T, C)


vmods = [m for n, m in model.named_modules() if n.endswith(".self_attn.v_proj")]


@torch.no_grad()
def run(label, keyfn=None, keybits=0, valbits=0):
    STATE.update(keyfn=keyfn, keybits=keybits, valbits=valbits, errs=[])
    hs = [m.register_forward_hook(vhook) for m in vmods] if valbits else []
    losses = [
        model(
            input_ids=ch.unsqueeze(0).to(dev), labels=ch.unsqueeze(0).to(dev)
        ).loss.item()
        for ch in chunks
    ]
    for h in hs:
        h.remove()
    e = STATE["errs"]
    ppl = math.exp(sum(losses) / len(losses))
    rm = (sum(e) / len(e)) if e else 0.0
    mx = max(e) if e else 0.0
    print(
        f"{label:44s} ppl={ppl:9.3f}   key_recon mean={rm:.3f} max={mx:.3f}", flush=True
    )
    return ppl


print(
    f"model={MODEL} seq={SEQ} chunks={len(chunks)} head_dim={HD} n_kv={NKV} (GQA {model.config.num_attention_heads}q/{NKV}kv)"
)
run("fp16 (no quant)")
run("values-only PolarQuant V3", valbits=3)
run("KEYS PolarQuant K4 (+V2) [shipped]", kq_polar, 4, 2)
run("KEYS per-channel UNIFORM K4 (+V2)", kq_perchan_uni, 4, 2)
run("KEYS per-channel NUQ K3 (+V3)", kq_perchan_nuq, 3, 3)
if HAVE_PCK:
    run("KEYS PerChannelKV MODULE K4 (+V2)", kq_module, 4, 2)
    run("KEYS PerChannelKV MODULE nuq K3 (+V3)", kq_module_nuq, 3, 3)
print("POSTROPE_DONE", flush=True)
