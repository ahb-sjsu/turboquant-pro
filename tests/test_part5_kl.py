"""Part V's reported KL to fp16 and fitting time (wikitext_ppl.py, KL_REF=1).

On a tiny random Qwen2 (CPU, prompts past the hot window, so keys are coded):

  kl_sum equals torch's KL in float64 and is zero for identical logits;
  the reference forward is the fp16 arm's forward (quantization off), leaves H.NOQUANT
      as it found it, and does not move the arm's own logits by one bit;
  a staged arm records the time it spent fitting its coding.

Usage:
    python -m pytest tests/test_part5_kl.py -q
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("datasets")  # wikitext_ppl imports it

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kvquant_matrix"))
sys.path.insert(0, str(ROOT / "tests"))

from test_key_coding import SHIPPED, _harness  # noqa: E402


def _unpatched_update():
    """DynamicCache.update as transformers defines it: earlier harness imports in this
    session wrapped it, and a wrapped update would quantize under the old module."""
    from transformers.cache_utils import DynamicCache

    f = DynamicCache.update
    while f.__name__ == "_patched_update":
        f = f.__globals__["_orig_update"]
    DynamicCache.update = f


def _setup(**env):
    _unpatched_update()
    h = _harness(**{**SHIPPED, "CODEBOOK": "uniform", "KEY_BITS": 3, **env})
    sys.modules["tq_paper_lb_shard"] = h
    import wikitext_ppl

    wp = importlib.reload(wikitext_ppl)
    cfg = transformers.Qwen2Config(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    model.config._attn_implementation = "sdpa"
    h.ensure_basis_calibration(model, None)  # registers W_O for the head weight
    chunk = torch.randint(0, 320, (1, 200), generator=torch.Generator().manual_seed(1))
    return h, wp, model, chunk


def test_kl_sum_matches_torch():
    _, wp, _, _ = _setup()
    g = torch.Generator().manual_seed(2)
    a = torch.randn(1, 300, 50, generator=g)
    b = a + 0.3 * torch.randn(1, 300, 50, generator=g)
    want = torch.nn.functional.kl_div(
        torch.log_softmax(b[0, :299].double(), -1),
        torch.log_softmax(a[0, :299].double(), -1),
        log_target=True,
        reduction="sum",
    )
    assert wp.kl_sum(a, b, 299) == pytest.approx(float(want), rel=1e-5)
    assert wp.kl_sum(a, a, 299) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("env", [{}, {"KEY_ALLOC": "head"}])
def test_reference_is_fp16_and_leaves_the_arm_unchanged(env):
    h, wp, model, chunk = _setup(**env)
    with torch.no_grad():
        arm_alone = model(chunk, use_cache=True).logits
    ref = wp.reference_logits(model, chunk)
    assert h.NOQUANT == 0
    with torch.no_grad():
        arm_after = model(chunk, use_cache=True).logits
    torch.testing.assert_close(arm_after, arm_alone, rtol=0, atol=0)
    h.NOQUANT = 1
    try:
        with torch.no_grad():
            fp16 = model(chunk, use_cache=True).logits
    finally:
        h.NOQUANT = 0
    torch.testing.assert_close(ref, fp16, rtol=0, atol=0)
    assert not torch.equal(ref, arm_alone)  # the arm's keys and values were coded
    assert wp.kl_sum(ref, arm_alone, chunk.shape[1] - 1) > 0


def test_staged_arm_records_its_fitting_time():
    h, _, model, chunk = _setup(KEY_ALLOC="head")
    h.KC.FIT_S = 0.0
    with torch.no_grad():
        model(chunk, use_cache=True)
    assert h.KC.FIT_S > 0
