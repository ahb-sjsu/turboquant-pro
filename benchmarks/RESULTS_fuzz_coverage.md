# Randomized coverage sweep — running record

Harness: [`fuzz_plugins.py`](fuzz_plugins.py) (time-budgeted, seed-reproducible;
every failure line is a standalone repro via `NB_SEED`). Invariants: plugin
conformance at random shapes (GQA, S=1, odd D, 100× scale extremes), torch_xp
reference paths vs NumPy on CPU/CUDA, `TurboQuantKVCache.fused_decode`
exactness vs decompress-then-attend at random configs (CPU dispatch and,
where CuPy exists, the M2/M4 CUDA kernels), and invalid-shape-must-raise.

| host | arch | budget | cases | failures |
|---|---|---|---:|---:|
| lightning H100 80GB | Hopper | 30 min | **71,060** | **0** |
| lightning T4 16GB | Turing | soak | **216,237** | **0** |

(Early in the same session, 10/5,725 flags traced to the fuzzer's own
absolute tolerance at the 100× input-scale extreme — actual disagreement
5e-7..2e-5 *relative*, fp32 reassociation; the invariant is scale-aware
since. Formats covered: per_channel, polar, bnb_nf4, bnb_llm_int8, fp8_kv,
nvfp4_kv, gptq, awq.)

The Turing soak (216,237 cases, 0 failures) is the largest single-arch
sweep to date — same eight formats, same scale-aware invariants, no CUDA
kernel leg (T4 has no CuPy install here, so the CPU dispatch path only).

Pending rows: L40S + CuPy kernel-path leg (`tqp-cupy-fuzz`, running), H200
long-context. Not reachable and stated plainly: MPS, ROCm, Blackwell-native.
