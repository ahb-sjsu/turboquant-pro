# Model weight compression

Apply PCA-Matryoshka to model parameters for weight- or activation-space compression.

Part of [TurboQuant Pro](../../README.md).

### Model weight compression

PCA-Matryoshka applied to model parameters (inspired by [MatFormer](https://arxiv.org/abs/2310.07707) and [FLAT-LLM](https://arxiv.org/abs/2505.23966)). **Weight-space SVD** (fast, no data) or **activation-space PCA** (accurate, needs calibration — compress the directions that matter least for inference), with per-head granularity (some heads compress, others don't).

> **Caveat:** eigenspectrum analysis is *diagnostic*, not a performance guarantee. Keeping 95% of SVD variance does **not** mean keeping 95% of downstream accuracy. Always validate with `sweep()` + `eval_fn`.

```python
from turboquant_pro.model_compress import ModelCompressor

compressor = ModelCompressor(model)
report = compressor.analyze()                  # weight-space (fast)
report = compressor.analyze_activations(       # activation-space (accurate)
    calibration_data=texts, tokenizer=tokenizer, n_samples=64)
for head in report.heads:
    if head.compressible:
        print(f"{head.layer_name} head {head.head_idx}: "
              f"rank {head.effective_rank}/{head.head_dim} — COMPRESS")
compressed = compressor.compress_activations(target_ratio=0.5)

# ALWAYS validate on downstream tasks
results = compressor.sweep(ratios=[0.3, 0.5, 0.7],
    eval_fn=lambda m: evaluate_perplexity(m, test_set), mode="activation")
```
```bash
turboquant-pro model --model "meta-llama/Llama-3.2-1B"                       # weight-space
turboquant-pro model --model "meta-llama/Llama-3.2-1B" \
  --mode activation --calibration cal_data.txt --n-samples 64               # activation-space
```
