"""
Tests for RoPE-aware weight quantization in ModelCompressor (the section-2.3
mechanism of docs/notes/projection_sensitivity_deconfounded.md: protect the
long-wavelength (DC) rows of W^K instead of giving K more bits everywhere).

Usage:
    pytest tests/test_rope_aware_k.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from turboquant_pro.model_compress import (  # noqa: E402
    ModelCompressor,
    quantize_weight_rows,
    rope_protected_rows,
)

HIDDEN, N_HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 8
HALF = HEAD_DIM // 2
THETA = 1e4


class _Cfg:
    hidden_size = HIDDEN
    num_attention_heads = N_HEADS
    rope_theta = THETA


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inv = THETA ** (-torch.arange(HALF, dtype=torch.float32) / HALF)
        self.register_buffer("inv_freq", inv)


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)  # GQA
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.up_proj = nn.Linear(HIDDEN, 4 * HIDDEN, bias=False)
        self.down_proj = nn.Linear(4 * HIDDEN, HIDDEN, bias=False)


class _Model(nn.Module):
    config = _Cfg()

    def __init__(self, with_rotary: bool = True) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList([_Block(), _Block()])
        if with_rotary:
            self.rotary = _Rotary()


def _inv_freq() -> np.ndarray:
    return THETA ** (-np.arange(HALF, dtype=np.float64) / HALF)


class TestProtectedRowMask:
    def test_fraction_and_longest_wavelength(self) -> None:
        """protect_frac of frequency indices, the longest-wavelength ones."""
        mask = rope_protected_rows(
            KV_HEADS * HEAD_DIM, HEAD_DIM, _inv_freq(), protect_frac=0.25
        )
        # 25% of HALF=4 freqs -> 1 protected frequency; it pairs channels
        # f and f+HALF in every head -> 2 rows per head x KV_HEADS heads
        assert mask.sum() == 2 * KV_HEADS
        # longest wavelength = highest frequency index (smallest inv_freq)
        j = np.arange(KV_HEADS * HEAD_DIM)
        f = (j % HEAD_DIM) % HALF
        assert set(f[mask]) == {HALF - 1}

    def test_rows_pair_across_rotate_half(self) -> None:
        """Channels c and c+half share a frequency -> both protected."""
        mask = rope_protected_rows(HEAD_DIM, HEAD_DIM, _inv_freq(), 0.25)
        assert bool(mask[HALF - 1]) and bool(mask[HEAD_DIM - 1])

    def test_bad_inv_freq_length_raises(self) -> None:
        with pytest.raises(ValueError, match="head_dim/2"):
            rope_protected_rows(HEAD_DIM, HEAD_DIM, np.ones(3), 0.25)


class TestQuantizeWeights:
    def test_protected_rows_kept_exact(self) -> None:
        model = _Model()
        orig = {
            n: m.weight.detach().clone()
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear)
        }
        q = ModelCompressor(model).quantize_weights(bits=3, k_protect_frac=0.25)
        mask = rope_protected_rows(KV_HEADS * HEAD_DIM, HEAD_DIM, _inv_freq(), 0.25)
        for name, mod in q.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            W0, W1 = orig[name], mod.weight.detach()
            if "k_proj" in name:
                assert torch.equal(W1[mask], W0[mask]), "protected rows changed"
                assert not torch.allclose(
                    W1[~mask], W0[~mask]
                ), "unprotected K rows were not quantized"
            else:
                assert not torch.allclose(W1, W0), f"{name} not quantized"
                expect = quantize_weight_rows(W0.float(), 3).to(W0.dtype)
                assert torch.allclose(W1, expect)

    def test_protect_bits_mixed_precision(self) -> None:
        model = _Model()
        k0 = model.layers[0].self_attn.k_proj.weight.detach().clone()
        q = ModelCompressor(model).quantize_weights(
            bits=3, k_protect_frac=0.25, k_protect_bits=8
        )
        mask = torch.from_numpy(
            rope_protected_rows(KV_HEADS * HEAD_DIM, HEAD_DIM, _inv_freq(), 0.25)
        )
        k1 = q.layers[0].self_attn.k_proj.weight.detach()
        expect8 = quantize_weight_rows(k0.float(), 8).to(k0.dtype)
        expect3 = quantize_weight_rows(k0.float(), 3).to(k0.dtype)
        assert torch.allclose(k1[mask], expect8[mask])
        assert torch.allclose(k1[~mask], expect3[~mask])

    def test_disabled_quantizes_all_k_rows(self) -> None:
        model = _Model()
        k0 = model.layers[0].self_attn.k_proj.weight.detach().clone()
        q = ModelCompressor(model).quantize_weights(bits=3, rope_aware_k=False)
        k1 = q.layers[0].self_attn.k_proj.weight.detach()
        assert torch.allclose(k1, quantize_weight_rows(k0.float(), 3).to(k0.dtype))

    def test_inv_freq_buffer_preferred_over_config(self) -> None:
        """A (scaled) inv_freq buffer must win over config.rope_theta."""
        model = _Model()
        # scramble the buffer so the longest wavelength sits at index 0
        model.rotary.inv_freq = torch.flip(model.rotary.inv_freq, [0])
        q = ModelCompressor(model).quantize_weights(bits=3, k_protect_frac=0.25)
        k0 = _Model().layers[0].self_attn.k_proj.weight  # same seed
        k1 = q.layers[0].self_attn.k_proj.weight
        j = np.arange(KV_HEADS * HEAD_DIM)
        f = (j % HEAD_DIM) % HALF
        assert torch.equal(k1[f == 0], k0[f == 0])  # index 0 now protected

    def test_config_theta_fallback(self) -> None:
        model = _Model(with_rotary=False)  # no inv_freq buffer anywhere
        q = ModelCompressor(model).quantize_weights(bits=3, k_protect_frac=0.25)
        mask = rope_protected_rows(KV_HEADS * HEAD_DIM, HEAD_DIM, _inv_freq(), 0.25)
        k0 = _Model(with_rotary=False).layers[0].self_attn.k_proj.weight
        k1 = q.layers[0].self_attn.k_proj.weight
        assert torch.equal(k1[mask], k0[mask])

    def test_no_rope_geometry_warns_and_degrades(self, caplog) -> None:
        model = _Model(with_rotary=False)
        model.config = type(
            "C", (), {"hidden_size": HIDDEN, "num_attention_heads": N_HEADS}
        )()
        with caplog.at_level("WARNING"):
            q = ModelCompressor(model).quantize_weights(bits=3)
        assert "no rotary geometry" in caplog.text
        k0 = _Model(with_rotary=False).layers[0].self_attn.k_proj.weight
        k1 = q.layers[0].self_attn.k_proj.weight
        assert torch.allclose(k1, quantize_weight_rows(k0.float(), 3).to(k0.dtype))

    def test_not_inplace_by_default(self) -> None:
        model = _Model()
        k0 = model.layers[0].self_attn.k_proj.weight.detach().clone()
        ModelCompressor(model).quantize_weights(bits=3)
        assert torch.equal(model.layers[0].self_attn.k_proj.weight, k0)
