"""Ingestion: zero-copy paths and modality-blindness of _channels.

The consumer contract is a callable on channel vectors; everything in
front of the channels axis is operating points. That makes the bridge
multimodal by construction — ViT patch grids and audio frame stacks
flatten exactly like token sequences — and these tests pin that down so
a core rewrite is never needed for a new modality.
"""

import numpy as np
import pytest

from tqp_readscope.plugin import ReadscopeJacobianOperator, _channels

def test_vit_patch_grid_flattens_like_tokens():
    # (batch, H_patches, W_patches, d) — a ViT layout, not a text one
    a = np.random.default_rng(0).standard_normal((2, 7, 7, 32))
    pts, d = _channels(a)
    assert d == 32 and pts.shape == (2 * 7 * 7, 32)


def test_audio_frames_flatten_identically():
    a = np.random.default_rng(1).standard_normal((3, 400, 16))  # frames
    pts, d = _channels(a)
    assert pts.shape == (1200, 16)


def test_cpu_torch_tensor_is_zero_copy():
    torch = pytest.importorskip("torch", reason="DLPack test needs torch")
    t = torch.randn(5, 8, dtype=torch.float64)
    pts, d = _channels(t)
    assert d == 8
    # zero-copy: writing through numpy must be visible in torch
    pts[0, 0] = 1234.5
    assert float(t[0, 0]) == 1234.5


def test_cpu_torch_float32_dtype_preserved():
    torch = pytest.importorskip("torch", reason="DLPack test needs torch")
    t = torch.randn(4, 6, dtype=torch.float32)
    pts, _ = _channels(t)
    assert pts.dtype == np.float32   # no silent upcast-copy on the
    #                                  zero-copy path


def test_vit_consumer_end_to_end_blind_recovery():
    """A vision-flavored vector consumer: attention of 4 learned queries
    over a probed patch embedding — same math as the published text-head
    measurements, no core change required."""
    rng = np.random.default_rng(7)
    d, m = 24, 4
    Q = rng.standard_normal((m, d))

    def consumer(x):
        s = Q @ np.asarray(x, dtype=float)
        e = np.exp(s - s.max())
        return e / e.sum()

    patches = rng.standard_normal((1, 3, 3, d))  # one image, 3x3 grid
    op = ReadscopeJacobianOperator(consumer=consumer, seed=0,
                                   max_points=4)
    S = op.operator(patches)
    assert S.shape == (d, d)
    # read subspace of a softmax-over-queries consumer lies in span(Q)
    proj = Q.T @ np.linalg.pinv(Q.T)
    leak = np.linalg.norm(S - proj @ S @ proj) / np.linalg.norm(S)
    # tolerance sits above the O(eps^2) finite-difference floor
    # (~1e-6 at eps=1e-3); a wrong subspace would leak O(1)
    assert leak < 1e-4
