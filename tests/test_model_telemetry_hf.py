"""model_telemetry on live Hugging Face models: tiny random-weight Llama (grouped-query
attention), Qwen2 and Gemma 3 (QK-norm, query_pre_attn_scalar, sliding-window layers),
built from configs so nothing is downloaded."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AttentionInterface  # noqa: E402
from transformers.integrations.sdpa_attention import (  # noqa: E402
    sdpa_attention_forward,
)

from turboquant_pro.hf_cache import TurboQuantCache, TurboQuantLayer  # noqa: E402
from turboquant_pro.model_telemetry import ModelMonitor, ReadPlan  # noqa: E402

V = 97


def _llama(attn):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )  # fmt: skip
    return LlamaForCausalLM(cfg).eval(), attn


def _qwen2(attn):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=1, max_position_embeddings=128,
    )  # fmt: skip
    return Qwen2ForCausalLM(cfg).eval(), attn


def _gemma3(attn):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        query_pre_attn_scalar=24, sliding_window=4, max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )  # fmt: skip
    return Gemma3ForCausalLM(cfg).eval(), attn


ARCHS = {"llama": _llama, "qwen2": _qwen2, "gemma3": _gemma3}


def _build(arch, attn="sdpa"):
    torch.manual_seed(0)
    model, attn = ARCHS[arch](attn)
    model.set_attn_implementation(attn)
    return model


def _ids(n=12, batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, V, (batch, n), generator=g)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_attaching_changes_no_output(arch):
    model = _build(arch)
    ids = _ids()
    with torch.no_grad():
        ref = model(ids).logits
        with ModelMonitor(model, window_tokens=10_000) as mon:
            got = model(ids).logits
    assert torch.equal(ref, got)
    assert not mon.errors
    assert model.config._attn_implementation == "sdpa"
    assert not [k for k in AttentionInterface._global_mapping if k.startswith("tqp_")]


def _capture(model):
    """Record the tensors every attention call is given (a test-only hook)."""
    seen = []

    def fn(module, q, k, v, mask, **kw):
        seen.append((module.layer_idx, q.detach().double(), k.detach().double(),
                     float(kw.get("scaling") or module.head_dim**-0.5)))  # fmt: skip
        return sdpa_attention_forward(module, q, k, v, mask, **kw)

    AttentionInterface.register("tqp_test_capture", fn)
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionMaskInterface.register("tqp_test_capture", sdpa_mask)
    model.set_attn_implementation("tqp_test_capture")
    return seen


def _reference(arch, ids, attention_mask=None):
    """The model's own attention weights (eager, output_attentions) and its captured
    queries and keys, from the same weights."""
    eager = _build(arch, "eager")
    with torch.no_grad():
        att = eager(
            ids, attention_mask=attention_mask, output_attentions=True
        ).attentions
    cap_model = _build(arch)
    seen = _capture(cap_model)
    with torch.no_grad():
        cap_model(ids, attention_mask=attention_mask)
    AttentionInterface._global_mapping.pop("tqp_test_capture", None)
    return [a.double() for a in att], seen


def _expected(att, seen, live_rows):
    """Per layer: the softmax operator, the logit operator and the entropy, per key
    head, computed by brute force from the model's own attention weights."""
    out = {}
    for li, q, k, s in seen:
        p = att[li]  # (B, Hq, S, S)
        B, Hq, S, D = q.shape
        Hkv = k.shape[1]
        G = Hq // Hkv
        Ps, Pl = np.zeros((Hkv, D, D)), np.zeros((Hkv, D, D))
        ent, n_rows = np.zeros(Hkv), np.zeros(Hkv)
        for b in range(B):
            for h in range(Hq):
                kv = h // G
                for i in range(S):
                    if not live_rows[b, i]:
                        continue
                    row = p[b, h, i].numpy()
                    vis = row > 0
                    a = 0.0
                    for t in np.flatnonzero(vis):
                        e = np.zeros(S)
                        e[t] = 1.0
                        a += float(((row * (e - row[t])) ** 2).sum())
                    qi = q[b, h, i].numpy()
                    Ps[kv] += a * s * s * np.outer(qi, qi)
                    Pl[kv] += vis.sum() * s * s * np.outer(qi, qi)
                    ent[kv] -= float((row[vis] * np.log2(row[vis])).sum())
                    n_rows[kv] += 1
        rows = n_rows / G
        out[li] = (Ps / rows[:, None, None], Pl / rows[:, None, None], ent / n_rows)
    return out


@pytest.mark.parametrize("arch", list(ARCHS))
@pytest.mark.parametrize("padded", [False, True])
def test_operators_match_brute_force_from_the_models_own_attention(arch, padded):
    ids = _ids()
    mask = None
    live = np.ones(ids.shape, dtype=bool)
    if padded:
        mask = torch.ones_like(ids)
        mask[0, :3] = 0  # left padding on the first sequence
        live = mask.numpy().astype(bool)
    att, seen = _reference(arch, ids, mask)
    # the captured queries and keys reproduce the model's attention exactly
    for li, q, k, s in seen:
        G = q.shape[1] // k.shape[1]
        z = torch.einsum("bhid,bhjd->bhij", q, k.repeat_interleave(G, 1)) * s
        ref = att[li]
        vis = ref > 0
        z = z.masked_fill(~vis, float("-inf"))
        p = torch.nan_to_num(torch.softmax(z, -1))
        rows = torch.from_numpy(live)[:, None, :, None].expand_as(p)
        assert torch.allclose(p[rows], ref[rows], atol=1e-5)
    expected = _expected(att, seen, live)

    model = _build(arch)
    with torch.no_grad(), ModelMonitor(model, window_tokens=10_000, max_rows=64) as mon:
        model(ids, attention_mask=mask)
    assert not mon.errors
    w = mon.windows[-1]
    for li, (Ps, Pl, ent) in expected.items():
        np.testing.assert_allclose(w.operator(li, "softmax"), Ps, rtol=1e-4, atol=1e-7)
        np.testing.assert_allclose(w.operator(li, "logit"), Pl, rtol=1e-4, atol=1e-7)
        L = w.layers[li]
        np.testing.assert_allclose(L["ent"] / L["ent_n"], ent, rtol=1e-4)


def test_windows_close_on_tokens_and_halves_alternate():
    model = _build("llama")
    ids = _ids(n=6, batch=1)
    with torch.no_grad(), ModelMonitor(model, window_tokens=4) as mon:
        model.generate(ids, max_new_tokens=9, do_sample=False, pad_token_id=0)
    toks = [w.tokens for w in mon.windows]
    assert sum(toks) == 6 + 8  # the prompt, then one token per decode call
    assert toks[0] == 6 and all(t == 4 for t in toks[1:-1])
    w = mon.windows[1]
    assert (w.layers[0]["rows"] > 0).all()  # both halves saw rows


def test_spill_hooks_see_each_first_coding_once():
    layer = TurboQuantLayer(hot_window=4, spill_block=4)
    calls = []
    layer.spill_hooks.append(lambda lay, k, kh: calls.append((k.clone(), kh.clone())))
    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, 2, 10, 8, generator=g)
    layer.update(k, k.clone())
    assert len(calls) == 1 and calls[0][0].shape[-2] == 6
    np.testing.assert_allclose(
        calls[0][1].numpy(), layer._kq.decompress(layer._cold_keys[0]), rtol=1e-6
    )
    layer.crop(8)  # a reload re-codes reconstructed keys: not observed
    assert len(calls) == 1


def test_key_distortion_is_measured_on_a_watched_cache():
    model = _build("llama")
    ids = _ids(n=10, batch=1)
    cache = TurboQuantCache(hot_window=4, spill_block=4)
    with torch.no_grad(), ModelMonitor(model, window_tokens=10_000) as mon:
        mon.watch_cache(cache)
        model.generate(ids, past_key_values=cache, max_new_tokens=6, do_sample=False,
                       pad_token_id=0)  # fmt: skip
    assert not mon.errors and cache.spill_hooks == []
    w = mon.windows[-1]
    for li in w.layers:
        assert w.layers[li]["err_n"] > 0
    rd = [r for r in mon.readings(w) if r["name"] == "model.kv.key_distortion"]
    assert rd and all(r["value"] > 0 for r in rd)


def test_readings_follow_the_schema_and_a_plan_scores_its_own_traffic(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    import turboquant_pro

    schema_path = os.path.join(
        os.path.dirname(turboquant_pro.__file__),
        "schemas",
        "metric_reading.schema.json",
    )
    schema = json.load(open(schema_path))
    model = _build("gemma3")
    with torch.no_grad(), ModelMonitor(model, window_tokens=10_000) as mon:
        model(_ids(n=24, batch=2))
    plan = mon.plan(bits_per_channel=4)
    plan.save(str(tmp_path / "p.npz"))
    plan = ReadPlan.load(str(tmp_path / "p.npz"))
    rs = mon.readings(mon.windows[-1], plan)
    names = {r["name"] for r in rs}
    assert {"model.read.trace", "model.plan.overlap", "model.plan.regret"} <= names
    for r in rs:
        jsonschema.validate(r, schema)
    for r in rs:
        if r["name"] == "model.plan.overlap" and r["value"] is not None:
            assert r["value"] == pytest.approx(1.0)
        if r["name"] == "model.plan.regret" and r["value"] is not None:
            assert r["value"] == pytest.approx(1.0, rel=1e-6)
