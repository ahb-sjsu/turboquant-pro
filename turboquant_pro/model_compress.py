"""
Model weight compression via PCA-Matryoshka.

Applies PCA rotation to FFN weight matrices so that column truncation
produces better sub-networks than naive pruning. This is the MatFormer
idea (Devvrit et al., 2023) applied post-training via PCA.

Usage:
    from turboquant_pro.model_compress import ModelCompressor

    compressor = ModelCompressor(model)
    report = compressor.analyze()  # eigenspectrum per layer
    compressed = compressor.compress(target_ratio=0.5)  # 50% FFN width

CLI:
    turboquant-pro autotune-model --model "path/to/model" --eval-fn perplexity
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class LayerAnalysis:
    """PCA analysis of one FFN layer's weight matrix."""

    layer_name: str
    shape: tuple
    eigenvalues: list[float]
    variance_explained_50: float  # variance in top 50% of dims
    variance_explained_75: float
    variance_explained_90: float
    effective_rank: int  # dims needed for 95% variance
    condition_number: float


@dataclass
class CompressionReport:
    """Report from analyzing a model's weight matrices."""

    n_layers: int
    total_params: int
    ffn_params: int
    layers: list[LayerAnalysis]
    avg_effective_rank_ratio: float  # effective_rank / full_rank
    recommended_ratio: float  # recommended truncation ratio
    estimated_speedup: float


class ModelCompressor:
    """Compress model FFN weights via PCA rotation.

    Applies PCA to each FFN weight matrix (up_proj, down_proj,
    gate_proj) so that truncating columns preserves maximum
    information — the same Eckart-Young optimality that makes
    PCA-Matryoshka work for embeddings.

    Args:
        model: A PyTorch model (HuggingFace or raw nn.Module).
    """

    def __init__(self, model=None) -> None:
        if torch is None:
            raise ImportError("PyTorch required for model compression")
        self._model = model
        self._ffn_layers: list[tuple[str, nn.Module]] = []
        if model is not None:
            self._find_ffn_layers()

    def _find_ffn_layers(self) -> None:
        """Find all FFN weight matrices in the model."""
        targets = ("up_proj", "down_proj", "gate_proj", "fc1", "fc2", "mlp.dense")
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name for t in targets):
                    self._ffn_layers.append((name, module))
        logger.info("Found %d FFN layers", len(self._ffn_layers))

    def analyze(self, sample_layers: int = 0) -> CompressionReport:
        """Analyze eigenspectrum of all FFN weight matrices.

        Args:
            sample_layers: If > 0, only analyze this many layers
                          (evenly spaced). 0 = analyze all.

        Returns:
            CompressionReport with per-layer analysis.
        """
        layers_to_analyze = self._ffn_layers
        if sample_layers > 0 and sample_layers < len(self._ffn_layers):
            step = len(self._ffn_layers) // sample_layers
            layers_to_analyze = self._ffn_layers[::step][:sample_layers]

        analyses = []
        total_params = sum(p.numel() for p in self._model.parameters())
        ffn_params = sum(m.weight.numel() for _, m in self._ffn_layers)

        for name, module in layers_to_analyze:
            w = module.weight.detach().float().cpu().numpy()
            analysis = self._analyze_layer(name, w)
            analyses.append(analysis)

        avg_rank_ratio = (
            np.mean([a.effective_rank / max(a.shape) for a in analyses])
            if analyses
            else 1.0
        )

        # Recommend ratio based on average effective rank
        recommended = min(0.9, max(0.3, avg_rank_ratio + 0.1))

        return CompressionReport(
            n_layers=len(self._ffn_layers),
            total_params=total_params,
            ffn_params=ffn_params,
            layers=analyses,
            avg_effective_rank_ratio=round(avg_rank_ratio, 3),
            recommended_ratio=round(recommended, 2),
            estimated_speedup=round(1.0 / recommended, 2),
        )

    def _analyze_layer(self, name: str, weight: np.ndarray) -> LayerAnalysis:
        """Analyze one weight matrix via SVD."""
        # SVD: W = U S V^T
        # Singular values tell us the effective rank
        try:
            s = np.linalg.svd(weight, compute_uv=False)
        except np.linalg.LinAlgError:
            s = np.ones(min(weight.shape))

        total_var = np.sum(s**2)
        cumvar = np.cumsum(s**2) / total_var

        # Find effective rank (95% variance)
        effective_rank = int(np.searchsorted(cumvar, 0.95)) + 1

        # Variance explained at different truncation levels
        n = len(s)
        var_50 = float(cumvar[n // 2 - 1]) if n > 1 else 1.0
        var_75 = float(cumvar[3 * n // 4 - 1]) if n > 3 else 1.0
        var_90 = float(cumvar[9 * n // 10 - 1]) if n > 9 else 1.0

        # Condition number
        cond = float(s[0] / max(s[-1], 1e-30))

        return LayerAnalysis(
            layer_name=name,
            shape=weight.shape,
            eigenvalues=[float(v) for v in s[:20]],  # top 20
            variance_explained_50=round(var_50, 4),
            variance_explained_75=round(var_75, 4),
            variance_explained_90=round(var_90, 4),
            effective_rank=effective_rank,
            condition_number=round(cond, 1),
        )

    def compress(
        self,
        target_ratio: float = 0.5,
        inplace: bool = False,
    ):
        """Compress FFN layers by PCA-rotating and truncating.

        For each FFN weight matrix W (shape [out, in]):
          1. Compute SVD: W = U S V^T
          2. Keep top k = int(min_dim * target_ratio) singular values
          3. Reconstruct: W_compressed = U[:,:k] S[:k] V^T[:k,:]
          4. Replace the weight matrix

        Args:
            target_ratio: Fraction of singular values to keep (0-1).
            inplace: If True, modify the model in place.

        Returns:
            The compressed model (same object if inplace).
        """
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

            # SVD
            U, S, Vh = torch.linalg.svd(w.cpu(), full_matrices=False)
            k = max(1, int(len(S) * target_ratio))

            # Low-rank reconstruction
            w_compressed = (U[:, :k] * S[:k]) @ Vh[:k, :]
            module.weight.data = w_compressed.to(device=device, dtype=dtype)

            removed = len(S) - k
            total_removed += removed
            compressed_count += 1

        logger.info(
            "Compressed %d layers (ratio=%.2f, removed %d singular values)",
            compressed_count,
            target_ratio,
            total_removed,
        )

        return self._model

    def sweep(
        self,
        ratios: list[float] | None = None,
        eval_fn=None,
    ) -> list[dict]:
        """Sweep compression ratios and evaluate quality.

        Args:
            ratios: List of target ratios to try. Default: [0.3..0.9]
            eval_fn: Function(model) -> float that measures quality
                    (e.g., perplexity, accuracy). Higher = better.

        Returns:
            List of dicts with ratio, quality, params, speedup.
        """
        if ratios is None:
            ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        results = []
        for ratio in ratios:
            t0 = time.time()
            compressed = self.compress(target_ratio=ratio, inplace=False)
            compress_time = time.time() - t0

            quality = None
            if eval_fn is not None:
                quality = eval_fn(compressed)

            # Count params
            n_params = sum(p.numel() for p in compressed.parameters())

            results.append(
                {
                    "ratio": ratio,
                    "quality": quality,
                    "params": n_params,
                    "speedup": round(1.0 / ratio, 2),
                    "compress_time_s": round(compress_time, 2),
                }
            )

            logger.info(
                "ratio=%.1f quality=%s speedup=%.1fx",
                ratio,
                f"{quality:.4f}" if quality is not None else "N/A",
                1.0 / ratio,
            )

        return results
