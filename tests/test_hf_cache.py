# TurboQuant Pro: tests for the HuggingFace drop-in KV cache.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""CPU, tiny-model, deterministic tests for ``turboquant_pro.hf_cache``.

No downloads: a tiny LlamaForCausalLM is built from a config. All runs are greedy
and seeded so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from turboquant_pro.hf_cache import (  # noqa: E402
    TurboQuantCache,
    enable_turboquant_cache,
)

VOCAB = 256
PROMPT_LEN = 12
NEW_TOKENS = 20


def _build_model(seed: int = 0):
    """Tiny GQA Llama (n_kv_heads < n_heads) on CPU, fixed weights, eval mode."""
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 2 kv heads shared across 4 query heads
        intermediate_size=128,
        vocab_size=VOCAB,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model, cfg


def _prompt(seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (1, PROMPT_LEN), generator=g)
    return ids


@torch.no_grad()
def _greedy(model, input_ids, cache):
    return model.generate(
        input_ids=input_ids,
        past_key_values=cache,
        max_new_tokens=NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )


def test_hot_only_exact_match():
    """hot_window large enough nothing is compressed -> bit-exact vs DynamicCache."""
    model, cfg = _build_model()
    input_ids = _prompt()

    ref = _greedy(model, input_ids, DynamicCache(config=cfg))
    # hot_window exceeds prompt + new tokens, so the cold path never triggers.
    got = _greedy(model, input_ids, TurboQuantCache(hot_window=4096))

    assert ref.shape == got.shape
    assert torch.equal(ref, got), (
        "Hot-only TurboQuantCache must reproduce DynamicCache token ids exactly "
        f"(diff at {torch.nonzero(ref != got).tolist()})"
    )


def test_compression_path_close():
    """Small hot_window forces cold compression; predictions stay close to fp32.

    The cold store quantizes keys to asym-NF4 (+2% fp16 outliers) and values to
    4-bit PolarQuant, so it is *not* bit-exact. We measure closeness on a single
    teacher-forced forward pass (per-position next-token argmax agreement), NOT on
    a free-running greedy trajectory: this tiny model is randomly initialized, so
    its per-step logits are near-ties and the *autoregressive* argmax path diverges
    chaotically under any perturbation (a property of the untrained model, not a
    cache bug). The single-pass agreement is the stable, defensible signal -- it is
    ~0.94 here; we require >=0.80, which fails loudly if the plumbing is broken
    while tolerating genuine quantization drift. (See also the bounded-MSE test.)
    """
    model, cfg = _build_model()
    # A longer sequence guarantees a substantial cold (compressed) region.
    g = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, 32), generator=g)

    # spill_block=1 compresses every token past the window, to exercise the codec.
    cache = TurboQuantCache(
        hot_window=4, key_bits=4, value_bits=4, outlier_frac=0.02, spill_block=1
    )
    with torch.no_grad():
        ref = model(input_ids, past_key_values=DynamicCache(config=cfg), use_cache=True)
        got = model(input_ids, past_key_values=cache, use_cache=True)

    assert ref.logits.shape == got.logits.shape
    agree = (ref.logits.argmax(-1) == got.logits.argmax(-1)).float().mean().item()
    assert agree >= 0.80, f"per-position argmax agreement {agree:.2f} too low"

    cos = torch.nn.functional.cosine_similarity(
        ref.logits.flatten(), got.logits.flatten(), dim=0
    ).item()
    assert cos >= 0.99, f"logit cosine similarity {cos:.4f} too low"

    # Cold compression actually happened on every layer (28 of 32 tokens spilled).
    assert len(cache.layers) == cfg.num_hidden_layers
    for layer in cache.layers:
        assert len(layer._cold_keys) > 0, "expected cold (compressed) chunks"
        assert layer.get_seq_length() == 32


def test_compression_logits_bounded_mse():
    """Single forward pass: compressed full K/V gives logits close to fp32."""
    model, cfg = _build_model()
    input_ids = _prompt()

    with torch.no_grad():
        ref_out = model(
            input_ids, past_key_values=DynamicCache(config=cfg), use_cache=True
        )
        cmp_out = model(
            input_ids,
            past_key_values=TurboQuantCache(hot_window=4, spill_block=1),
            use_cache=True,
        )
    # Prefill (seq=12) with hot_window=4 => 8 tokens spilled to cold store per layer.
    mse = torch.mean((ref_out.logits - cmp_out.logits) ** 2).item()
    var = torch.var(ref_out.logits).item()
    # Drift must be small relative to the natural logit variance.
    assert mse < 0.05 * var, f"logit MSE {mse:.4g} too high vs variance {var:.4g}"


def test_get_seq_length_tracks_steps():
    """get_seq_length advances by one per decode step and matches the model output."""
    model, cfg = _build_model()
    input_ids = _prompt()
    cache = TurboQuantCache(hot_window=8)

    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
        assert cache.get_seq_length() == PROMPT_LEN
        assert cache.get_seq_length(0) == PROMPT_LEN

        # Feed three more single tokens, length should advance one at a time.
        next_id = input_ids[:, -1:]
        for step in range(3):
            model(next_id, past_key_values=cache, use_cache=True)
            assert cache.get_seq_length() == PROMPT_LEN + step + 1


def test_reset_and_crop():
    """reset() empties the cache; crop() trims to a given length (cold+hot aware)."""
    model, cfg = _build_model()
    input_ids = _prompt()

    # Use a small hot window so both cold and hot tiers are populated.
    cache = TurboQuantCache(hot_window=4)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == PROMPT_LEN

    # crop to 6 tokens (spans cold + hot); length must update on every layer.
    cache.crop(6)
    assert cache.get_seq_length() == 6
    for layer in cache.layers:
        assert layer.get_seq_length() == 6
        k, v = layer._full()
        assert k.shape[-2] == 6 and v.shape[-2] == 6

    # negative crop drops tokens from the end.
    cache.crop(-2)
    assert cache.get_seq_length() == 4

    # reset clears everything but keeps the layer objects/quantizers.
    cache.reset()
    assert cache.get_seq_length() == 0
    for layer in cache.layers:
        assert layer.get_seq_length() == 0
        assert layer._kq is not None and layer._vq is not None


def test_crop_preserves_kept_tokens_exactly():
    """Cropping to within the hot window keeps the surviving K/V tensors lossless."""
    model, cfg = _build_model()
    input_ids = _prompt()

    cache = TurboQuantCache(hot_window=64)  # everything stays hot (lossless)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)

    before = [(*map(torch.clone, layer._full()),) for layer in cache.layers]
    cache.crop(5)
    for (k0, v0), layer in zip(before, cache.layers):
        k1, v1 = layer._full()
        assert torch.equal(k0[:, :, :5, :], k1)
        assert torch.equal(v0[:, :, :5, :], v1)


def test_enable_turboquant_cache_helper():
    """enable_turboquant_cache patches generate to inject a TurboQuantCache."""
    model, cfg = _build_model()
    input_ids = _prompt()

    ref = _greedy(model, input_ids, DynamicCache(config=cfg))

    enable_turboquant_cache(model, hot_window=4096)
    with torch.no_grad():
        got = model.generate(
            input_ids=input_ids,
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    # hot_window large -> lossless -> identical to the default dynamic cache.
    assert torch.equal(ref, got)


def test_gqa_shapes_roundtrip():
    """Compressed cold chunks round-trip the GQA [B, n_kv_heads, S, D] shape."""
    model, cfg = _build_model()
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads

    g = torch.Generator().manual_seed(7)
    key = torch.randn(1, n_kv, 10, head_dim, generator=g)
    val = torch.randn(1, n_kv, 10, head_dim, generator=g)

    cache = TurboQuantCache(hot_window=4, spill_block=1)
    k_full, v_full = cache.update(key, val, layer_idx=0)
    assert k_full.shape == (1, n_kv, 10, head_dim)
    assert v_full.shape == (1, n_kv, 10, head_dim)

    # The most-recent hot_window tokens are returned losslessly.
    assert torch.equal(k_full[:, :, -4:, :], key[:, :, -4:, :])
    assert torch.equal(v_full[:, :, -4:, :], val[:, :, -4:, :])

    # Cold (compressed) tokens are approximate but bounded.
    cold_err = (k_full[:, :, :6, :] - key[:, :, :6, :]).abs().mean().item()
    assert np.isfinite(cold_err) and cold_err < 0.5


def test_decode_spills_in_blocks_that_are_smaller_than_fp16():
    """Regression: a token-at-a-time spill stored every key element as an fp16
    outlier (about 7x the fp16 bytes). With the default block, 200 decode steps
    after a full window spill whole blocks, each smaller than the same tokens in
    fp16, keys and values both."""
    g = torch.Generator().manual_seed(3)
    b, h, d, window = 1, 4, 64, 16
    layer_cache = TurboQuantCache(hot_window=window)
    for _ in range(window + 200):
        k = torch.randn(b, h, 1, d, generator=g)
        v = torch.randn(b, h, 1, d, generator=g)
        layer_cache.update(k, v, layer_idx=0)
    layer = layer_cache.layers[0]
    assert layer.spill_block == 64
    assert layer._cold_lengths and all(n == 64 for n in layer._cold_lengths)
    assert window <= layer._hot_keys.shape[-2] < window + layer.spill_block
    assert layer.get_seq_length() == window + 200
    fp16 = b * h * d * 2  # bytes per token for one of K or V
    for n, ck, cv in zip(layer._cold_lengths, layer._cold_keys, layer._cold_values):
        assert ck.nbytes() < n * fp16, (n, ck.nbytes())
        assert cv.nbytes() < n * fp16, (n, cv.nbytes())


def test_a_one_token_spill_is_what_the_block_prevents():
    """The measured reason for blocks: the key codec on a 1-token block is larger
    than fp16, on a 64-token block far smaller."""
    from turboquant_pro.per_channel_kv import PerChannelKV

    q = PerChannelKV(head_dim=64, n_heads=4, bits=4, nf4_asym=True, outlier_frac=0.02)
    x = np.random.default_rng(0).standard_normal((1, 4, 64, 64)).astype(np.float32)
    fp16 = 4 * 64 * 2
    one = q.compress(x[:, :, :1], packed=True).nbytes()
    block = q.compress(x, packed=True).nbytes() / 64
    assert one > fp16 > 2 * block
