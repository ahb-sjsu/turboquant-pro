#!/usr/bin/env python3
"""End-to-end on a real model + LongBench: does fused KV-decode preserve quality?

Loads a real HF model, feeds a real LongBench prompt, and (via a non-invasive
recording hook on the eager-attention function) captures the *true* post-RoPE
query/key/value activations per layer. For the decode position (the last query, which
attends to the whole context) it compares, per layer:

  fp16 standard attention   vs   fused 3-bit decode over the quantized K/V

This is the quality signal LongBench scores -- if the per-layer attention output is
preserved on real activations, generation is preserved -- measured here directly on
real long-context data without fragile generate()-loop surgery.
"""

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--task", default="qasper")
    ap.add_argument("--hot", type=int, default=512)  # fp16 hot window (recent tokens)
    ap.add_argument("--sink", type=int, default=4)  # fp16 attention-sink tokens (first)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant_pro import TurboQuantPGVector
    from turboquant_pro.kv_fused import fused_decode

    # --- real long-context prompt: LongBench parquet, else a real Gutenberg book ---
    prompt = None
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "THUDM/LongBench", a.task, split="test", trust_remote_code=True
        )
        prompt = ds[0]["context"][:200000]
        print(f"LongBench[{a.task}] sample, {len(prompt)} chars", flush=True)
    except Exception as e:
        print(
            f"LongBench script unavailable ({str(e)[:40]}); fetching real book text",
            flush=True,
        )
    if prompt is None:
        try:
            import urllib.request

            url = (
                "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice
            )
            prompt = (
                urllib.request.urlopen(url, timeout=30)
                .read()
                .decode("utf-8", "replace")
            )
            print(
                f"Gutenberg PG1342 (real natural text), {len(prompt)} chars", flush=True
            )
        except Exception as e:
            print(f"offline ({str(e)[:40]}); synthetic", flush=True)
            prompt = "The quick brown fox jumps over the lazy dog. " * 4000

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager",
    ).eval()

    # locate and monkeypatch the model's eager attention function to RECORD (q,k,v)
    import importlib

    attn_mod = importlib.import_module(type(model.model.layers[0].self_attn).__module__)
    orig = attn_mod.eager_attention_forward
    rec = []

    def recording(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kw
    ):
        rec.append(
            (
                query.detach().float().cpu().numpy(),
                key.detach().float().cpu().numpy(),
                value.detach().float().cpu().numpy(),
                float(scaling),
            )
        )
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)

    attn_mod.eager_attention_forward = recording
    ids = tok(
        prompt, return_tensors="pt", truncation=True, max_length=a.max_ctx
    ).input_ids.cuda()
    print(f"context tokens: {ids.shape[1]}", flush=True)
    with torch.no_grad():
        model(ids)
    attn_mod.eager_attention_forward = orig  # restore

    # per-layer: fp16 attention vs fused 3-bit decode at the last (decode) position
    def code(X, tq):  # X (H,S,d) -> codes (H,S,d) uint8, norms (H,S)
        n = np.linalg.norm(X, axis=2)
        r = tq._rotate(X / np.maximum(n[..., None], 1e-30))
        return np.searchsorted(tq.boundaries, r).astype(np.uint8), n.astype(np.float32)

    def relerr(a_, b_):
        return float(np.linalg.norm(a_ - b_) / (np.linalg.norm(b_) + 1e-9))

    errs_all, errs_tier = [], []
    for q, k, v, scaling in rec:
        # q (1,Hq,S,d); k,v (1,Hkv,S,d) post-RoPE. Expand KV to Hq (GQA), take last query.
        Hq, S, d = q.shape[1], q.shape[2], q.shape[3]
        rep = Hq // k.shape[1]
        K = np.repeat(k[0], rep, axis=0)  # (Hq, S, d)
        V = np.repeat(v[0], rep, axis=0)
        ql = q[0, :, -1, :]  # (Hq, d) decode query
        # fp16 reference (full attention of the last query over all keys)
        s = np.einsum("hd,hsd->hs", ql, K) * scaling
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        ref = np.einsum("hs,hsd->hd", p, V)
        tq = TurboQuantPGVector(dim=d, bits=a.bits)
        qf = ql * (
            scaling * np.sqrt(d)
        )  # fold HF scaling into fused_decode's 1/sqrt(d)
        # (1) all-cold 3-bit (worst case, no fp16 window)
        kc, nk = code(K, tq)
        vc, nv = code(V, tq)
        errs_all.append(relerr(fused_decode(qf, None, None, kc, vc, nk, nv, tq), ref))
        # (2) realistic two-tier: fp16 sink (first) + hot (recent), 3-bit cold middle
        sink = min(a.sink, S)
        hot = min(a.hot, max(S - sink, 0))
        lo, hi = sink, S - hot
        fk = np.concatenate([K[:, :sink], K[:, S - hot :]], axis=1)
        fv = np.concatenate([V[:, :sink], V[:, S - hot :]], axis=1)
        if hi > lo:
            ck, cnk = code(K[:, lo:hi], tq)
            cv, cnv = code(V[:, lo:hi], tq)
        else:
            ck = cv = cnk = cnv = None
        errs_tier.append(relerr(fused_decode(qf, fk, fv, ck, cv, cnk, cnv, tq), ref))

    ea, et = np.array(errs_all), np.array(errs_tier)
    print(
        f"\nlayers: {len(ea)}  attention-output rel err vs fp16 (real activations):",
        flush=True,
    )
    print(
        f"  all-cold {a.bits}-bit (worst case): mean {ea.mean():.4f} | median "
        f"{np.median(ea):.4f}",
        flush=True,
    )
    print(
        f"  two-tier (fp16 sink={a.sink}+hot={a.hot}, {a.bits}-bit cold): mean "
        f"{et.mean():.4f} | median {np.median(et):.4f} | max {et.max():.4f}",
        flush=True,
    )
    print(f"  cold KV compression: ~{16/a.bits:.1f}x", flush=True)


if __name__ == "__main__":
    main()
