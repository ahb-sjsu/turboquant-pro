"""Pre-registered keys comparison (P3): fp8 per-tensor / fp8 per-head /
nvfp4 block-16 / per-channel asym-NF4 as KEY formats on real DC-offset-family
keys. Prediction (registered before this run, design doc section 3): the
coarser the scale granularity, the worse the attention damage on
DC-offset keys -- fp8_tensor > fp8_head > nvfp4 ~ per_channel."""

import json

import numpy as np
import torch
from tqp_trtllm.plugin import FP8KVQuantizer, NVFP4KVQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant_pro.plugins import create

MODEL = "unsloth/Llama-3.2-3B"
rng = np.random.default_rng(0)


class FP8PerTensor(FP8KVQuantizer):
    def compress(self, x):
        c = super().compress(x)
        g = float(c.scale.max())
        c.scale[:] = g  # one scale for the whole tensor (TRT-LLM default mode)
        import tqp_trtllm.plugin as tp

        c.codes = tp._nearest(np.asarray(x, np.float32)[0] / g, tp._E4M3)[None]
        return c


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda"
).eval()
text = (
    "The scale granularity of a key quantizer determines whether the "
    "per-channel offsets that softmax reads survive compression. "
) * 20
ids = tok(text, return_tensors="pt", truncation=True, max_length=320).input_ids.cuda()
with torch.no_grad():
    pkv = model(ids, use_cache=True).past_key_values
print(f"[fwd] {ids.shape[1]} tokens", flush=True)

FORMATS = {
    "fp8_tensor": FP8PerTensor(),
    "fp8_head": FP8KVQuantizer(),
    "nvfp4_block16": NVFP4KVQuantizer(),
    "per_channel_nf4": create(
        "per_channel", head_dim=128, n_heads=8, nf4_asym=True, outlier_frac=0.02
    ),
}
out = {}
for li in (0, 13, 27):
    K = pkv[li][0][0].float().cpu().numpy()[None]  # (1, kvH, S, dh)
    q = rng.standard_normal((16, K.shape[1], K.shape[3])).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    se = np.einsum("qhd,hsd->qhs", q, K[0]) / np.sqrt(K.shape[3])
    pe = np.exp(se - se.max(-1, keepdims=True))
    pe /= pe.sum(-1, keepdims=True)
    for name, quant in FORMATS.items():
        Kr = np.asarray(quant.decompress(quant.compress(K)))[0]
        sr = np.einsum("qhd,hsd->qhs", q, Kr) / np.sqrt(K.shape[3])
        pr = np.exp(sr - sr.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        kl = float((pe * np.log((pe + 1e-12) / (pr + 1e-12))).sum(-1).mean())
        t1 = float((pe.argmax(-1) == pr.argmax(-1)).mean())
        out.setdefault(name, []).append({"layer": li, "kl": kl, "top1": t1})
        print(f"  L{li} {name}: KL={kl:.5f} top1={t1:.3f}", flush=True)

means = {n: float(np.mean([r["kl"] for r in rs])) for n, rs in out.items()}
order = sorted(means, key=means.get, reverse=True)
print(
    f"[verdict] damage order (worst first): {order}  means={ {k: round(v,5) for k,v in means.items()} }",
    flush=True,
)
print("=== JSON ===")
print(json.dumps({"model": MODEL, "results": out, "mean_kl": means}))
