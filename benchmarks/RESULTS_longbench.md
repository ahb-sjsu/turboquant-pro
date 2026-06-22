# Fused KV-decode on a real model: quality on real long-context activations

End-to-end check of the fused KV-decode on a real served model. The real LongBench
dataset loader is deprecated (HuggingFace `datasets` dropped script-based datasets),
so we use **real natural text** (Project Gutenberg PG-1342, *Pride and Prejudice*) on
**Qwen2.5-7B-Instruct** (eager attention), capturing the *true* post-RoPE
query/key/value via a non-invasive recording hook, and comparing per layer at the
decode position:

    fp16 standard attention   vs   fused decode over quantized K/V

(4096-token context, 28 layers; the kernel is exact vs decompress-then-attend, so this
isolates the **quantization** quality, not the kernel.)

| config | all-cold (worst case) | two-tier: fp16 sink=4 + hot=512, coded cold |
|---|---:|---:|
| 3-bit | mean 0.367 / median 0.32 | **mean 0.155 / median 0.119** (max 0.68) |
| 4-bit | mean 0.208 / median 0.17 | **mean 0.086 / median 0.048** (max 0.60) |

## Honest findings
1. **The kernel changes speed, not quality.** It reproduces decompress-then-attend
   exactly (<=4e-7, M0-M3); the errors above are pure quantization loss.
2. **3-bit KV is aggressive** for this model: ~12% median per-layer attention-output
   error even with a hot window. **4-bit halves it** to ~5% -> prefer 4-bit keys (or
   asymmetric `key_bits=4, value_bits=3`) for quality-sensitive decode.
3. **The two-tier scheme matters:** keeping the attention sink (first tokens) and the
   recent hot window in fp16 cuts error ~2.4x vs all-cold. This is what
   `TurboQuantKVCache` already does.
4. **A few layers stay high** (max ~0.6) -- the known attention-sink / outlier-channel
   problem in KV quantization; per-channel scaling or more fp16 sinks would help. Honest
   open item.

## Scope (honest)
This measures per-layer **attention-output fidelity** on real activations -- a direct
quality signal (preserved attention -> preserved generation) -- not a full LongBench
*task score*, which needs a generate()-loop attention monkeypatch (model-arch-specific,
fragile; scaffolded in `benchmark_longbench_parity.py`). The fidelity result already
says: ship 4-bit (or asymmetric) KV for quality; 3-bit is a memory-max setting.
