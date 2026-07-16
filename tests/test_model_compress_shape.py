"""Shape-soundness tests for activation-space weight compression.

These lock the fix for the non-square crash: the activation basis spans a
layer's *output* space, so compression must project output rows (``Vk^T Vk W``),
never ``W @ V^T`` — which requires ``in == out`` and crashes on real FFN
matrices (gate/up/down projections are all non-square).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int = 8, inter: int = 16) -> None:
        super().__init__()
        # Deliberately non-square: up_proj [inter, hidden], down_proj [hidden, inter].
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


def _orthonormal(dim: int, seed: int = 0) -> np.ndarray:
    """A [dim, dim] orthonormal basis (rows are principal directions)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q.astype(np.float32)


def _rank(mat: np.ndarray, tol: float = 1e-4) -> int:
    s = np.linalg.svd(mat, compute_uv=False)
    return int((s > tol * s[0]).sum()) if s[0] > 0 else 0


def test_compress_activations_handles_nonsquare():
    from turboquant_pro.model_compress import ModelCompressor

    model = _TinyMLP(hidden=8, inter=16)
    comp = ModelCompressor(model)
    # Bases span each layer's OUTPUT dim: up_proj out=16, down_proj out=8.
    comp._activation_bases = {
        "up_proj": _orthonormal(16, seed=1),
        "down_proj": _orthonormal(8, seed=2),
    }

    # Must not raise (the old ``w @ V^T`` crashed here for in != out).
    out = comp.compress_activations(target_ratio=0.5, inplace=True)
    assert out is model

    # Shapes are preserved (dense writeback), and rank is actually reduced.
    up = model.up_proj.weight.detach().cpu().numpy()
    down = model.down_proj.weight.detach().cpu().numpy()
    assert up.shape == (16, 8)
    assert down.shape == (8, 16)
    # target_ratio 0.5 keeps k = out//2 activation directions -> rank <= k.
    assert _rank(up) <= 8
    assert _rank(down) <= 4


def test_compress_activations_skips_mismatched_basis():
    from turboquant_pro.model_compress import ModelCompressor

    model = _TinyMLP(hidden=8, inter=16)
    comp = ModelCompressor(model)
    before = model.up_proj.weight.detach().cpu().numpy().copy()
    # Wrong-sized basis (spans 5 dims, weight output is 16): skip, don't corrupt.
    comp._activation_bases = {"up_proj": _orthonormal(5, seed=3)}
    comp.compress_activations(target_ratio=0.5, inplace=True)
    after = model.up_proj.weight.detach().cpu().numpy()
    np.testing.assert_array_equal(before, after)
