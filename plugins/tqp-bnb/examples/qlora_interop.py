"""QLoRA interop demo (P2 exit criterion): a bitsandbytes-4bit (QLoRA-style)
model generating with REAL activations, its per-layer KV captured into
turboquant-pro's TurboQuantKVCache -- bnb quantizes the weights, tqp
quantizes the cache, and attention built from the compressed cache matches
attention on the exact KV."""

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquant_pro.core import TurboQuantKVCache

MODEL = "unsloth/Llama-3.2-3B"
rng = np.random.default_rng(0)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_cfg, device_map="cuda"
).eval()
print("[load] bnb-4bit NF4 double-quant (QLoRA config)", flush=True)

prompt = (
    "The theory of quantization error in attention mechanisms tells us "
    "that keys and values behave differently because "
) * 8
ids = tok(prompt, return_tensors="pt", truncation=True, max_length=384).input_ids.cuda()
with torch.no_grad():
    out = model(ids, use_cache=True)
pkv = out.past_key_values
print(f"[fwd] {ids.shape[1]} tokens, {len(pkv)} layers cached", flush=True)

layers = [0, 13, 27]
results = []
for li in layers:
    K = pkv[li][0][0].float().cpu().numpy()  # (kvH, S, dh) post-RoPE
    V = pkv[li][1][0].float().cpu().numpy()
    kvH, S, dh = K.shape
    cache = TurboQuantKVCache(
        head_dim=dh,
        n_heads=kvH,
        bits=4,
        use_gpu=False,
        seed=0,
        per_channel_keys=True,
        key_nf4_asym=True,
        key_outlier_frac=0.02,
        hot_window=64,
    )
    for s in range(S):
        cache.append(K[:, s, :].astype(np.float32), V[:, s, :].astype(np.float32))
    Kr = np.asarray(cache.get_keys(0, S))[0].transpose(0, 1, 2)
    q = rng.standard_normal((16, kvH, dh)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    kl_list, top1 = [], []
    for qq in q:
        se = np.einsum("hd,hsd->hs", qq, K.transpose(0, 1, 2)) / np.sqrt(dh)
        sr = np.einsum("hd,hsd->hs", qq, Kr) / np.sqrt(dh)
        pe = np.exp(se - se.max(1, keepdims=True))
        pe /= pe.sum(1, keepdims=True)
        pr = np.exp(sr - sr.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        kl_list.append(float((pe * np.log((pe + 1e-12) / (pr + 1e-12))).sum(1).mean()))
        top1.append(float((pe.argmax(1) == pr.argmax(1)).mean()))
    mem = cache.memory_stats()
    rec = {
        "layer": li,
        "attn_kl": float(np.mean(kl_list)),
        "top1_agree": float(np.mean(top1)),
        "kv_compression": round(mem["effective_ratio"], 2),
    }
    results.append(rec)
    print(
        f"  layer {li}: attn_KL={rec['attn_kl']:.5f} "
        f"top1_agree={rec['top1_agree']:.3f} ratio={rec['kv_compression']}x",
        flush=True,
    )

print(
    "[verdict] bnb-4bit weights + tqp KV cache compose: mean attn KL "
    f"{np.mean([r['attn_kl'] for r in results]):.5f}, mean top-1 agreement "
    f"{np.mean([r['top1_agree'] for r in results]):.3f}",
    flush=True,
)
print("=== JSON ===")
print(json.dumps({"model": MODEL, "seq": int(ids.shape[1]), "results": results}))
