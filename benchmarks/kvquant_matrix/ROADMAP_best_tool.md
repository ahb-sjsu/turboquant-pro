# Roadmap: make TurboQuant-Pro the best KV-cache quantization tool

Honest scorecard of where we stand per competitive category, the gap to "best," and the
work to close it. Updated as we iterate. Baselines of record: KVQuant (calibrated NUQ),
KIVI (2-bit asym), fp16, plus HF/vLLM built-in KV quant for the production axes.

Legend: 🟢 leading / 🟡 competitive / 🔴 gap.

| # | Category | Status | Where we are | Gap to "best" |
|---|---|---|---|---|
| 1 | **Robustness across architectures** | 🟢 | asym-NF4 near-fp16 on MHA *and* high-GQA (Qwen2.5), where plain NF4 collapses | More families: Llama-3, Qwen2.5-3B/14B (GQA sweep), Phi, Gemma, an MoE |
| 2 | **Calibration-free** | 🟢 | Entire approach needs zero calibration data; KVQuant needs a Fisher pass | Keep it; prove no-calib ≈ calibrated broadly |
| 3 | **Quality at 4-bit** | 🟡 | asym-NF4 ties NF4, ~fp16; KVQuant (calibrated) still edges us ~0.25 qasper on Llama-7B | Close gap: per-group (vs per-channel) codebook, learned-free level tweaks |
| 4 | **Memory / compression ratio** | 🟡 | NF4/asym-NF4 store a *full expanded level table* per channel (16 floats) — wasteful | Store scalars (amax, +mean for asym): 16 floats → 1–2. Big ratio win |
| 5 | **Speed / decode throughput** | 🔴 | Research prototype; no fused dequant kernel for NF4/asym-NF4 | Triton/CUDA fused dequant + outlier-scatter; benchmark vs fp16 & vLLM |
| 6 | **Bit-depth range (2/3/8)** | 🟡 | uniform 2/3/4; NF4/asym-NF4 are 4-bit only | asym-NF at 3-bit (NF3 grid) + 8-bit path; 2-bit w/ residual (KIVI-style) |
| 7 | **Framework integration** | 🟡 | Custom cache + transformers monkeypatch | Drop-in HF `Cache` subclass; vLLM/SGLang plugin; one-line enable |
| 8 | **Breadth of validation** | 🟡 | 4 models × {3 core + 7 expanded} LongBench tasks + WikiText (running) | Full LongBench-E, more models, latency/mem numbers in-repo |
| 9 | **Same-harness baselines** | 🔴 | KVQuant NOW implemented in-harness (offline Fisher NUQ, pre-RoPE) + runs end-to-end, but scores only **8.16** qasper on Llama-7B (vs published ~21) — pipeline fixed, NOT yet faithful. KIVI code only. | Make KVQuant reproduce ~21 (calibration fidelity) before it's a credible baseline; then all 4 models |
| 10 | **Docs / reproducibility** | 🟢 | Guide + paper outline + results JSON + notebook (in progress) | Finish notebook (#14); ship guide as the README KV section |

## Prioritized iteration plan

**P0 — correctness & free wins (now)**
- [x] Ship asym-NF4 (`nf4_asym`) into the package — fixes the Qwen-collapse correctness bug.
- [x] **Compact NF4/asym-NF4 storage** (cat #4): scalars (`amax`, +`mean`) not expanded level table.
- [x] Make asym-NF4 the **recommended default**: `AutoConfig.build_cache(robust=True)` now wires
      `key_nf4_asym=True` + 2% outliers by default (was only reachable via `.robust()`). 82 tests green.

**Ground-truth audit (2026-06-29) — roadmap was stale; corrected:**
- cat #5 (speed): a real CUDA fused-decode kernel **already exists** (`kv_kernel.py` M1 block + M2
  warp, ~10x vs dequant, correctness-tested). The gap is *not* the kernel — it is a **latency +
  peak-memory benchmark vs fp16/vLLM** (none in-repo). NRP job, not new kernel work.
- cat #7 (integration): `cache_adapter.py`/`vllm_plugin.py` are **embedding/standalone** caches,
  **not** a `transformers.Cache` subclass. True one-line HF drop-in (`TurboQuantCache(Cache)`,
  transformers 5.11) is still missing — pure local code, CPU-verifiable.
- cat #6 (bit-depth): NF4/asym-NF4 are **4-bit only** (no NF3/8-bit); uniform is 2/3/4. Gap real.
- cat #9 (baselines): KVQuant/KIVI NOW in-repo; KVQuant runs end-to-end (validated on Atlas
  GV100, 2026-07-01) after fixing ~5 bugs (calibration wiring, post- & pre-RoPE circular deps,
  OOM via `autograd.grad`, the pre-RoPE layer-counter). BUT scores only **8.16** qasper on
  Llama-7B (vs published ~21) at 4-bit — infra done, *implementation not faithful*. Head-to-head
  still relies on external figures. See paper Limitations. Faithful calibration = future work.

**P1 — competitive proof (this week)**
- [~] **Same-harness KVQuant** implemented + runs end-to-end (8.16 qasper Llama-7B, NOT faithful
      vs ~21 → calibration fidelity is future work). KIVI code only. Head-to-head table DEFERRED
      until KVQuant reproduces published numbers; comparison uses external figures meanwhile.
- [ ] Finish the **expanded matrix** (running) + **WikiText ppl** (running) (cat #8).
- [ ] **GQA sweep** within the Qwen2.5 family (3B 8:1, 7B 7:1, 14B 5:1) to map the
      NF4-collapse cliff vs GQA ratio (cat #1) — turns the hypothesis into a curve.

**P2 — performance (next)**
- [ ] **Fused dequant kernel** (Triton) for NF4/asym-NF4 + dense-sparse scatter (cat #5).
- [ ] Decode-latency + peak-memory benchmarks vs fp16 and vLLM KV-quant, in-repo (cat #5/#8).

**P3 — reach (next)**
- [ ] **HF `Cache` subclass** for drop-in use; transformers ≥4.40 path (cat #7).
- [ ] **3-bit asym-NF** (NF3 grid) and **8-bit** path; optional 2-bit residual (cat #6).
- [ ] Reproducible **notebook** (#14): single-GPU subset + multi-GPU full; regenerates all tables.

**P4 — write-up**
- [ ] TMLR paper (#11) + practitioner guide (#13) finalized on the complete matrix.

## Definition of "success"
TurboQuant-Pro is the **default choice for calibration-free 4-bit KV** when you (a) don't
have/want calibration data and (b) need it to *not silently break* on your model — backed
by same-harness numbers vs KVQuant/KIVI, a real kernel + latency story, and one-line HF use.
