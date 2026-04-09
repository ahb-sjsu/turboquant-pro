"""
Model weight compression via PCA-Matryoshka.

Two modes:
  1. Weight-space SVD (v0.6): SVD on weight matrices directly. Fast,
     no calibration data needed, but variance ≠ downstream performance.

  2. Activation-space PCA (v0.7, FLAT-LLM inspired): Run calibration
     data through the model, collect activations per layer/head, PCA
     those activations, truncate in the directions that matter least
     for actual inference. More accurate but requires calibration data.

Head-wise granularity: For attention layers, PCA each head separately.
Different heads may have different effective ranks — a head doing
positional encoding needs all its dimensions, while a head doing
broad semantic matching may be highly compressible.

References:
  - MatFormer (Devvrit et al., 2023): Nested FFN for elastic inference
  - FLAT-LLM (2025): Fine-grained low-rank activation space transform
  - PCA-Matryoshka (Bond, 2026): Training-free PCA for embeddings

Usage:
    from turboquant_pro.model_compress import ModelCompressor

    # Weight-space (fast, no calibration data)
    compressor = ModelCompressor(model)
    report = compressor.analyze()

    # Activation-space (accurate, needs calibration data)
    report = compressor.analyze_activations(calibration_data)
    compressed = compressor.compress_activations(target_ratio=0.5)

CLI:
    turboquant-pro model --model <name> --calibration <dataset>
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


# ─── Data structures ──────────────────────────────────────────


@dataclass
class LayerAnalysis:
    """PCA analysis of one layer's weight or activation matrix."""

    layer_name: str
    shape: tuple
    eigenvalues: list[float]
    variance_explained_50: float
    variance_explained_75: float
    variance_explained_90: float
    effective_rank: int  # dims needed for 95% variance
    condition_number: float
    mode: str = "weight"  # "weight" or "activation"


@dataclass
class HeadAnalysis:
    """PCA analysis of one attention head's activations."""

    layer_name: str
    head_idx: int
    head_dim: int
    effective_rank: int
    variance_explained_90: float
    compressible: bool  # effective_rank < 0.5 * head_dim


@dataclass
class CompressionReport:
    """Report from analyzing a model."""

    n_layers: int
    total_params: int
    ffn_params: int
    layers: list[LayerAnalysis] = field(default_factory=list)
    heads: list[HeadAnalysis] = field(default_factory=list)
    avg_effective_rank_ratio: float = 1.0
    recommended_ratio: float = 0.7
    estimated_speedup: float = 1.0
    mode: str = "weight"
    n_compressible_heads: int = 0
    n_total_heads: int = 0


# ─── SVD helpers ──────────────────────────────────────────────


def _analyze_matrix(
    name: str, matrix: np.ndarray, mode: str = "weight"
) -> LayerAnalysis:
    """Analyze one matrix via SVD."""
    try:
        s = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.ones(min(matrix.shape))

    total_var = np.sum(s**2)
    if total_var < 1e-30:
        total_var = 1.0
    cumvar = np.cumsum(s**2) / total_var

    effective_rank = int(np.searchsorted(cumvar, 0.95)) + 1

    n = len(s)
    var_50 = float(cumvar[max(0, n // 2 - 1)]) if n > 1 else 1.0
    var_75 = float(cumvar[max(0, 3 * n // 4 - 1)]) if n > 3 else 1.0
    var_90 = float(cumvar[max(0, 9 * n // 10 - 1)]) if n > 9 else 1.0

    cond = float(s[0] / max(s[-1], 1e-30))

    return LayerAnalysis(
        layer_name=name,
        shape=matrix.shape,
        eigenvalues=[float(v) for v in s[:20]],
        variance_explained_50=round(var_50, 4),
        variance_explained_75=round(var_75, 4),
        variance_explained_90=round(var_90, 4),
        effective_rank=effective_rank,
        condition_number=round(cond, 1),
        mode=mode,
    )


# ─── Main compressor ─────────────────────────────────────────


class ModelCompressor:
    """Compress model weights via PCA rotation.

    Supports two modes:
      - Weight-space SVD (fast, no data needed)
      - Activation-space PCA (FLAT-LLM style, needs calibration data)

    Args:
        model: A PyTorch model (HuggingFace or raw nn.Module).
    """

    def __init__(self, model=None) -> None:
        if torch is None:
            raise ImportError("PyTorch required for model compression")
        self._model = model
        self._ffn_layers: list[tuple[str, nn.Module]] = []
        self._attn_layers: list[tuple[str, nn.Module]] = []
        self._activation_bases: dict[str, np.ndarray] = {}
        if model is not None:
            self._find_layers()

    def _find_layers(self) -> None:
        """Find all FFN and attention weight matrices."""
        ffn_targets = ("up_proj", "down_proj", "gate_proj", "fc1", "fc2")
        attn_targets = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj")
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name for t in ffn_targets):
                    self._ffn_layers.append((name, module))
                elif any(t in name for t in attn_targets):
                    self._attn_layers.append((name, module))
        logger.info(
            "Found %d FFN layers, %d attention layers",
            len(self._ffn_layers),
            len(self._attn_layers),
        )

    # ── Weight-space analysis (v0.6) ──────────────────────────

    def analyze(self, sample_layers: int = 0) -> CompressionReport:
        """Analyze eigenspectrum of FFN weight matrices (weight-space SVD)."""
        layers_to_analyze = self._ffn_layers
        if 0 < sample_layers < len(self._ffn_layers):
            step = max(1, len(self._ffn_layers) // sample_layers)
            layers_to_analyze = self._ffn_layers[::step][:sample_layers]

        analyses = []
        total_params = sum(p.numel() for p in self._model.parameters())
        ffn_params = sum(m.weight.numel() for _, m in self._ffn_layers)

        for name, module in layers_to_analyze:
            w = module.weight.detach().float().cpu().numpy()
            analyses.append(_analyze_matrix(name, w, "weight"))

        avg_rank_ratio = (
            np.mean([a.effective_rank / max(a.shape) for a in analyses])
            if analyses
            else 1.0
        )
        recommended = min(0.9, max(0.3, avg_rank_ratio + 0.1))

        return CompressionReport(
            n_layers=len(self._ffn_layers),
            total_params=total_params,
            ffn_params=ffn_params,
            layers=analyses,
            avg_effective_rank_ratio=round(avg_rank_ratio, 3),
            recommended_ratio=round(recommended, 2),
            estimated_speedup=round(1.0 / recommended, 2),
            mode="weight",
        )

    # ── Activation-space analysis (v0.7, FLAT-LLM inspired) ──

    def analyze_activations(
        self,
        calibration_data: list[str] | None = None,
        tokenizer=None,
        n_samples: int = 128,
        max_length: int = 512,
        sample_layers: int = 0,
    ) -> CompressionReport:
        """Analyze using activations from calibration data.

        Runs calibration data through the model, collects activations
        at each FFN and attention layer, then PCA-analyzes those
        activations. This captures which dimensions actually matter
        for inference, not just weight structure.

        Args:
            calibration_data: List of text strings for calibration.
                If None, uses random noise (less accurate).
            tokenizer: HuggingFace tokenizer (required if calibration_data provided).
            n_samples: Number of calibration samples to use.
            max_length: Max token length per sample.
            sample_layers: If > 0, only analyze this many layers.

        Returns:
            CompressionReport with activation-based analysis.
        """
        self._model.eval()
        total_params = sum(p.numel() for p in self._model.parameters())
        ffn_params = sum(m.weight.numel() for _, m in self._ffn_layers)

        # Collect activations via hooks
        activations: dict[str, list] = {}
        hooks = []

        layers_to_hook = self._ffn_layers + self._attn_layers
        if 0 < sample_layers < len(layers_to_hook):
            step = max(1, len(layers_to_hook) // sample_layers)
            layers_to_hook = layers_to_hook[::step][:sample_layers]

        for name, module in layers_to_hook:
            activations[name] = []

            def make_hook(layer_name):
                def hook_fn(_module, _input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    # Flatten batch + seq dims, keep hidden dim
                    act = output.detach().float().cpu()
                    act = act.reshape(-1, act.shape[-1])
                    # Subsample to avoid memory explosion
                    if act.shape[0] > 1024:
                        idx = torch.randperm(act.shape[0])[:1024]
                        act = act[idx]
                    activations[layer_name].append(act.numpy())

                return hook_fn

            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        # Run calibration data
        device = next(self._model.parameters()).device
        n_processed = 0

        if calibration_data and tokenizer:
            for text in calibration_data[:n_samples]:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                with torch.no_grad():
                    self._model(**inputs)
                n_processed += 1
                if n_processed % 20 == 0:
                    logger.info("Calibration: %d/%d samples", n_processed, n_samples)
        else:
            # No calibration data — use random inputs
            logger.warning("No calibration data — using random noise (less accurate)")
            vocab_size = getattr(self._model.config, "vocab_size", 32000)
            for _ in range(min(n_samples, 32)):
                input_ids = torch.randint(0, vocab_size, (1, max_length), device=device)
                with torch.no_grad():
                    self._model(input_ids=input_ids)
                n_processed += 1

        # Remove hooks
        for h in hooks:
            h.remove()

        # Analyze collected activations
        analyses = []
        head_analyses = []

        for name, act_list in activations.items():
            if not act_list:
                continue
            # Concatenate all activations for this layer
            all_acts = np.concatenate(act_list, axis=0)
            # Subsample if too large
            if all_acts.shape[0] > 10000:
                idx = np.random.choice(all_acts.shape[0], 10000, replace=False)
                all_acts = all_acts[idx]

            analysis = _analyze_matrix(name, all_acts, "activation")
            analyses.append(analysis)

            # Head-wise analysis for attention layers
            is_attn = any(t in name for t in ("q_proj", "k_proj", "v_proj", "o_proj"))
            if is_attn:
                n_heads = getattr(self._model.config, "num_attention_heads", 0)
                head_dim = all_acts.shape[1] // max(n_heads, 1)
                if n_heads > 0 and head_dim > 0:
                    for h_idx in range(n_heads):
                        start = h_idx * head_dim
                        end = start + head_dim
                        head_acts = all_acts[:, start:end]
                        ha = _analyze_matrix(
                            f"{name}.head{h_idx}", head_acts, "activation"
                        )
                        compressible = ha.effective_rank < head_dim * 0.5
                        head_analyses.append(
                            HeadAnalysis(
                                layer_name=name,
                                head_idx=h_idx,
                                head_dim=head_dim,
                                effective_rank=ha.effective_rank,
                                variance_explained_90=ha.variance_explained_90,
                                compressible=compressible,
                            )
                        )

            # Store PCA basis for compression
            try:
                _, _, Vh = np.linalg.svd(all_acts, full_matrices=False)
                self._activation_bases[name] = Vh
            except np.linalg.LinAlgError:
                pass

        avg_rank_ratio = (
            np.mean([a.effective_rank / max(a.shape) for a in analyses])
            if analyses
            else 1.0
        )
        recommended = min(0.9, max(0.3, avg_rank_ratio + 0.1))
        n_compressible = sum(1 for ha in head_analyses if ha.compressible)

        logger.info(
            "Activation analysis: %d layers, %d heads (%d compressible), "
            "%d calibration samples",
            len(analyses),
            len(head_analyses),
            n_compressible,
            n_processed,
        )

        return CompressionReport(
            n_layers=len(self._ffn_layers),
            total_params=total_params,
            ffn_params=ffn_params,
            layers=analyses,
            heads=head_analyses,
            avg_effective_rank_ratio=round(avg_rank_ratio, 3),
            recommended_ratio=round(recommended, 2),
            estimated_speedup=round(1.0 / recommended, 2),
            mode="activation",
            n_compressible_heads=n_compressible,
            n_total_heads=len(head_analyses),
        )

    # ── Activation-space compression ──────────────────────────

    def compress_activations(
        self,
        target_ratio: float = 0.5,
        inplace: bool = False,
    ):
        """Compress using activation-space PCA bases.

        Must call analyze_activations() first to compute the bases.

        For each layer with a stored PCA basis V:
          1. Project weights into activation PCA space: W' = W @ V^T
          2. Truncate to top k dimensions
          3. Project back: W_compressed = W'[:,:k] @ V[:k,:]

        This is more accurate than weight-space SVD because the PCA
        basis reflects which directions matter during actual inference.

        Args:
            target_ratio: Fraction of dimensions to keep (0-1).
            inplace: If True, modify model in place.

        Returns:
            Compressed model.
        """
        if not self._activation_bases:
            raise RuntimeError(
                "Must call analyze_activations() first to compute PCA bases"
            )

        if not inplace:
            import copy

            model = copy.deepcopy(self._model)
            compressor = ModelCompressor(model)
            compressor._activation_bases = self._activation_bases
            return compressor.compress_activations(
                target_ratio=target_ratio, inplace=True
            )

        compressed_count = 0

        for name, module in self._ffn_layers + self._attn_layers:
            if name not in self._activation_bases:
                continue

            Vh = self._activation_bases[name]
            w = module.weight.detach().float().cpu()
            device = module.weight.device
            dtype = module.weight.dtype

            # Project into PCA space
            V = torch.from_numpy(Vh).float()
            k = max(1, int(V.shape[0] * target_ratio))

            # W' = W @ V^T (project columns into PCA space)
            # Then truncate and project back
            w_pca = w @ V.T  # [out, pca_dims]
            w_compressed = w_pca[:, :k] @ V[:k, :]  # [out, in]

            module.weight.data = w_compressed.to(device=device, dtype=dtype)
            compressed_count += 1

        logger.info(
            "Compressed %d layers via activation-space PCA (ratio=%.2f)",
            compressed_count,
            target_ratio,
        )
        return self._model

    # ── Weight-space compression (v0.6) ───────────────────────

    def compress(
        self,
        target_ratio: float = 0.5,
        inplace: bool = False,
    ):
        """Compress FFN layers via weight-space SVD truncation."""
        if not inplace:
            import copy

            model = copy.deepcopy(self._model)
            compressor = ModelCompressor(model)
            return compressor.compress(target_ratio=target_ratio, inplace=True)

        compressed_count = 0
        total_removed = 0

        for _name, module in self._ffn_layers:
            w = module.weight.detach().float()
            device = module.weight.device
            dtype = module.weight.dtype

            U, S, Vh = torch.linalg.svd(w.cpu(), full_matrices=False)
            k = max(1, int(len(S) * target_ratio))

            w_compressed = (U[:, :k] * S[:k]) @ Vh[:k, :]
            module.weight.data = w_compressed.to(device=device, dtype=dtype)

            total_removed += len(S) - k
            compressed_count += 1

        logger.info(
            "Compressed %d layers via weight SVD (ratio=%.2f, removed %d sv)",
            compressed_count,
            target_ratio,
            total_removed,
        )
        return self._model

    # ── Sweep ─────────────────────────────────────────────────

    def sweep(
        self,
        ratios: list[float] | None = None,
        eval_fn=None,
        mode: str = "weight",
    ) -> list[dict]:
        """Sweep compression ratios and evaluate quality.

        Args:
            ratios: List of target ratios. Default: [0.3..0.9]
            eval_fn: Function(model) -> float (higher = better).
            mode: "weight" or "activation".

        Returns:
            List of dicts with ratio, quality, speedup.
        """
        if ratios is None:
            ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        compress_fn = (
            self.compress_activations if mode == "activation" else self.compress
        )

        results = []
        for ratio in ratios:
            t0 = time.time()
            compressed = compress_fn(target_ratio=ratio, inplace=False)
            compress_time = time.time() - t0

            quality = eval_fn(compressed) if eval_fn else None
            n_params = sum(p.numel() for p in compressed.parameters())

            results.append(
                {
                    "ratio": ratio,
                    "quality": quality,
                    "params": n_params,
                    "speedup": round(1.0 / ratio, 2),
                    "compress_time_s": round(compress_time, 2),
                    "mode": mode,
                }
            )

            logger.info(
                "ratio=%.1f quality=%s speedup=%.1fx mode=%s",
                ratio,
                f"{quality:.4f}" if quality is not None else "N/A",
                1.0 / ratio,
                mode,
            )

        return results
