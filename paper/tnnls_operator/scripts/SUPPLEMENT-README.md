# Supplement

Supplementary material, code and run records for the submission "Equal Weight Error, Unequal
Effect: Quantization Sensitivity at the Consuming Operator". Names, paths and the name of the
library the code was taken from are removed for double-anonymous review.

`supplementary_material.pdf` is the supplementary material the manuscript refers to as Sections
S-I to S-VI. Everything else in this archive is the code and the records.

## Check the paper's numbers (no GPU, a few minutes)

```
pip install numpy nbclient ipykernel
cd paper/tnnls_operator
jupyter execute reproduce.ipynb
```

`paper/tnnls_operator/reproduce.ipynb` recomputes every number the paper prints from the run
records in this archive and compares it with the printed value. Its last cell must report that
all comparisons pass. Run it from inside this directory tree: it finds the records by walking up
to the folder that contains `experiments/results_matched_bit`.

## Layout

| Path | Content |
|---|---|
| `supplementary_material.pdf` | the supplementary material, Sections S-I to S-VI |
| `paper/tnnls_operator/reproduce.ipynb` | the reproduction notebook; `scripts/build_notebook.py` generates it |
| `paper/tnnls_operator/scripts/` | scripts that derive the tables from the records, with their saved output, and the script for Figure 1 |
| `paper/tnnls_operator/lean/` | the Lean 4 file of formal checks and the record of its build |
| `experiments/*.py` | experiment code for attention projections, agreement, the tracer and routing |
| `experiments/results_matched_bit/` | run records for those experiments; `per_sequence/` holds the reruns with one value per sequence |
| `benchmarks/*.py` | experiment code for expert routing and state decay |
| `benchmarks/*.json`, `*.npz` | run records for routing and state decay |
| `opquant/` | the four library modules the experiment scripts import; `operator_trace.py` is the tracer of Section 6 |

## Which script wrote which record

| Record | Script |
|---|---|
| `behavioral_*.json` | `experiments/behavioral_metric_demo.py` |
| `agreement_repeats_*.json` | `experiments/agreement_reference_repeats.py` |
| `matched_bit_*.json` | `experiments/matched_bit_projection_sensitivity.py` |
| `llamacpp_factorial_*.json` | `experiments/llamacpp_factorial.py` |
| `score_error_*.json` | `experiments/score_error_decomposition.py` |
| `k_wavelength_*.json` | `experiments/k_wavelength_probe.py` |
| `k_row_protection_*.json` | `experiments/k_row_protection_controls.py` |
| `quantizer_control_*.json`, `k_row_protection_mse_clip_*.json` | `experiments/k_quantizer_control.py` |
| `gptq_control_projections_*.json` | `experiments/k_gptq_control.py` |
| `validate_rope_aware_k_*.json` | `experiments/validate_rope_aware_k.py` |
| `tracer_accuracy.json` | `experiments/tracer_accuracy.py` |
| `router_error_margin_*.json` | `experiments/router_error_margin.py` |
| `routing_margin_logits_*.json`, `routing_margins_*.npz` | `experiments/routing_margin_logits.py` |
| `results_mixtral_routing.json`, `results_olmoe_routing.json` | `benchmarks/validate_mixtral_routing.py`, `benchmarks/validate_olmoe_routing.py` |
| `results_mamba_decay.json` | `benchmarks/validate_mamba_decay.py` |
| `results_mamba_decay_census*.json` | `benchmarks/mamba_decay_grid_census.py` |
| `results_mamba_decay_tau_grid_*.json` | `benchmarks/mamba_decay_tau_grid.py` |
| `results_mamba_decay_intervals_*.json` | `benchmarks/mamba_decay_intervals.py` |
| `results_mamba_decay_rate_safe_*.json` | `benchmarks/mamba_decay_rate_safe.py` |
| `results_rwkv_decay_grid_*.json` | `benchmarks/rwkv_decay_grid.py` |

## Rerunning an experiment

The experiments need a GPU with 24 GB or more (the state-decay scripts run on a CPU), PyTorch,
`transformers`, `datasets`, `safetensors` and, for the factorial, `gguf` and a build of
llama.cpp. Each script's docstring gives its command and its environment variables. Run the
scripts from this directory so that `opquant` is importable, for example:

```
PYTHONPATH=. python experiments/matched_bit_projection_sensitivity.py
```

Two things differ from the code as it was run. First, `experiments/exp_io.py` has an optional
path, switched on by `NB_STRIPES`, `NB_MODEL_DIR` and `NB_TOKENS`, that reads weights and tokens
staged in advance on a cluster pod. The helpers behind it (`staged/`) are not in this archive.
The default path, which loads models and text from the hub, is complete. Second, one record,
`validate_rope_aware_k_Llama-3.2-3B.json`, named the cluster in its `gpu` field. That field is
rewritten here, so this record's hash in `scripts/numbers_derived.txt` is the hash of the
rewritten file. Every other record is a byte-for-byte copy.

Model weights come from the Hugging Face hub under each model's own licence. The quantizers are
deterministic, so a rerun on the same sequences reproduces a record up to floating-point
differences between hardware.

## Lean file

`paper/tnnls_operator/lean/OperatorChecks.lean` was compiled with Lean 4.32.2 and Mathlib
v4.32.2. `BUILD-RECORD.txt` gives the command, the file's hash and the list of theorems.
