# [D] Training a 1D codebook on your actual embedding data reduces quantization error 22% vs the Gaussian assumption — quick experiment

Most scalar quantization for embeddings (including TurboQuant/PolarQuant) assumes that after random orthogonal rotation, coordinates are i.i.d. Gaussian. You use precomputed Lloyd-Max centroids for N(0, 1/sqrt(d)) and call it done.

I was curious how much that assumption actually costs you. So I ran a simple experiment:

1. Take a corpus of embeddings (1024-dim, BGE-M3-like variance structure)
2. Rotate them with the standard QR random rotation
3. Flatten all the rotated coordinates into one big 1D array
4. Run Lloyd's algorithm (iterative k-means, but 1D) on the actual distribution
5. Compare the trained codebook's centroids to the theoretical Lloyd-Max centroids

The trained centroids are noticeably different from the theoretical ones:

```
Theoretical (Gaussian):  [-0.154, -0.093, -0.044, -0.006, 0.006, 0.044, 0.093, 0.154]
Trained (actual data):   [-0.188, -0.118, -0.066, -0.021, 0.021, 0.066, 0.117, 0.187]
```

The tails are wider — real embeddings have heavier tails than a Gaussian after rotation. The trained codebook spreads its centroids further apart to better cover the actual distribution.

**The result:** 22% reduction in reconstruction error at the same 3-bit width. Cosine similarity goes from 0.978 to 0.983. That's free — same storage, same compression ratio, just better centroids.

I also tested this on KV cache tensors (head_dim=128) and got a similar improvement (0.978 -> 0.983).

**What's interesting theoretically:**

The PolarQuant paper (Zandieh et al. ICLR 2026) proves that random rotation makes coordinates *approximately* Gaussian, but the approximation quality depends on the effective dimensionality of the data. Real embedding models have spectral structure that survives rotation — the coordinates aren't perfectly Gaussian, and the deviation is systematic enough that a data-trained codebook exploits it.

This connects to older work on optimal quantizer design (Max 1960, Lloyd 1982) — the Lloyd-Max quantizer is only optimal for the distribution it was designed for. If your actual distribution deviates from the assumed one, re-running Lloyd on real data is always at least as good.

**Caveats:**

- The 22% number is on synthetic data with BGE-M3-like variance structure, not actual BGE-M3 embeddings. Real-world improvement may differ.
- You need ~1000 embeddings to train a stable codebook. Not a problem for most corpora but won't work for very small datasets.
- The trained codebook is specific to your embedding model. If you switch models, you need to retrain.

**What I'm still figuring out:**

- Does this compose with PCA dimension reduction? If you do PCA-384 first, then train the codebook on the reduced space, do you get a double win?
- Is there a closed-form correction you could apply to the Gaussian codebook without running full Lloyd iterations? Something like a tail-heaviness scaling factor?
- Has anyone measured how much the rotated-coordinate distribution varies across embedding models (BGE-M3 vs CLIP vs Whisper)?

If anyone has a corpus of real embeddings they'd be willing to share the rotated-coordinate histogram for, I'd love to compare.

Code for the experiment: https://github.com/ahb-sjsu/turboquant-pro (the `fit_codebook()` function in `learned_codebook.py`)
