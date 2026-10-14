"""Expected Attention (Devoto, Jeblick and Jégou) through NVIDIA's kvpress, for Part V's
``ea`` and ``ea_u3`` baselines (docs/PREREG_full_consumer_kv.md, section 3).

Inactive unless ``EA`` is set; importing this module loads nothing from kvpress.

    EA=ea      evict keys and values at prefill, kept entries in fp16 (u3's coding off)
    EA=ea_u3   the same eviction, composed with u3's key and value coding
    EA_REF     u3's cell on the same model: each prompt is evicted to u3's stored bytes

The scoring and selection are kvpress 0.5.5's own code (``third_party/kvpress``, four
files byte-identical to the wheel, loaded under stub packages because the package needs
``transformers<5.3`` and imports presses this environment cannot load). Only the
attention hook is ours: transformers 5.5 no longer passes ``cache_position`` to the
attention layer, which kvpress's ``BasePress.forward_hook`` reads. It takes the same
steps: at prefill only, the layer's cached keys and values are scored by
``ExpectedAttentionPress.score``, the top ``compute_n_kept`` per head are kept (the steps
of ``ScorerPress.compress``), and the cache is overwritten with them.

Matched bytes, per prompt. u3 stores the settled tokens ``[0, T-HOT)`` coded (its keys at
the row's measured ``key_bits.total`` per element, its values at ``VAL_BITS`` plus a
16-bit minimum and a 16-bit scale per token and group) and the last ``HOT`` tokens in
fp16. Expected Attention stores its kept entries in fp16 (16 bits per key or value
element) and nothing else: kvpress compacts the cache and RoPE positions are already in
the keys. The kept fraction is u3's bits per element over 16, so the two store the same
bytes on every prompt.

Perplexity (owner, 2026-10-14). kvpress evicts after the prefill's attention, so the
harness's one-forward perplexity would equal fp16's. Every coded arm's perplexity reads
the cache as stored, by every query; so does this one. Pass 1 runs the chunk and records
which entries Expected Attention keeps in each layer and head, as it would at the end of
prefill. Pass 2 runs the chunk again with each layer's attention restricted, causally, to
those entries, and its loss is the perplexity.
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import sys
import types

import torch

MODE = os.environ.get("EA", "")
REF = os.environ.get("EA_REF", "")
HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "third_party", "kvpress")
# kvpress's defaults (ExpectedAttentionPress, 0.5.5), unchanged
N_FUTURE, N_SINK = 512, 4

_HOOKS: list = []
_S = types.SimpleNamespace(ratio=None, mode="evict", keep={}, n_kept={}, vb=None, hot=None,
                           u3=None, T=None, press=None, utils=None, enabled=None)


def active() -> bool:
    return MODE in ("ea", "ea_u3")


def codes_cache() -> bool:
    """Whether u3's coding runs under the eviction (ea_u3) or is off (ea)."""
    return MODE == "ea_u3"


def validate():
    if MODE not in ("", "ea", "ea_u3"):
        raise SystemExit(f"EA={MODE!r}: one of ea, ea_u3")
    if active() and not REF:
        raise SystemExit("EA needs EA_REF, u3's cell on the same model (matched bytes)")


def config() -> dict:
    return {"mode": MODE, "ref": REF, "n_future_positions": N_FUTURE, "n_sink": N_SINK,
            "kvpress": "0.5.5", "perplexity": "pass-1 selection, pass-2 masked"}


def vendored_files() -> list:
    return [os.path.join(VENDOR, "kvpress", p) for p in
            ("utils.py", "presses/base_press.py", "presses/scorer_press.py",
             "presses/expected_attention_press.py")]


def load_kvpress():
    """(ExpectedAttentionPress, kvpress.utils) from the vendored files. Stub packages stand
    in for ``kvpress`` and ``kvpress.presses`` so that their ``__init__`` (every press)
    never runs; each module's own imports resolve to the vendored files."""
    root = os.path.join(VENDOR, "kvpress")
    for name, path in (("kvpress", root), ("kvpress.presses", os.path.join(root, "presses"))):
        mod = sys.modules.get(name)
        if mod is None or getattr(mod, "__path__", None) != [path]:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod
    ea = importlib.import_module("kvpress.presses.expected_attention_press")
    return ea.ExpectedAttentionPress, importlib.import_module("kvpress.utils")


# ---------------------------------------------------------------- matched bytes
def value_bits(vb: int, d: int, group: int = 0) -> float:
    """u3's stored bits per value element: VAL_BITS plus a 16-bit minimum and a 16-bit
    scale per token and group (the harness's ``_qdq_val``; a group is the head unless
    VAL_GROUP is set)."""
    return vb + 32 / (group or d)


def u3_bits(T: int, kb: float | None, hot: int, vbe: float) -> float:
    """u3's stored bits per cached key or value element on a T-token prompt."""
    if T <= hot:
        return 16.0
    if kb is None:
        raise ValueError("u3 coded this prompt but recorded no key bits")
    n = T - hot
    return (n * (kb + vbe) + hot * 32) / (2 * T)


def ratio_for(u3b: float) -> float:
    """kvpress's compression ratio at which fp16 entries take u3's bytes."""
    return max(0.0, 1.0 - u3b / 16.0)


_REF_ROWS: dict = {}


def ref_key_bits(kind: str, key) -> float | None:
    """u3's recorded stored key bits per settled element: (task, document index) or
    ("ppl", chunk start). A prompt u3 never ran is an error, never a guess."""
    if kind not in _REF_ROWS:
        rows = {}
        if kind == "ppl":
            for line in open(os.path.join(REF, "ppl_chunks.jsonl"), encoding="utf-8"):
                o = json.loads(line)
                rows[o["start"]] = o
        else:
            for f in sorted(glob.glob(os.path.join(REF, f"{kind}.*.jsonl"))):
                for line in open(f, encoding="utf-8"):
                    o = json.loads(line)
                    rows[o["idx"]] = o
        _REF_ROWS[kind] = rows
    rows = _REF_ROWS[kind]
    if key not in rows:
        raise SystemExit(f"EA: u3 ({REF}) has no {kind} row {key}: run u3 first")
    kb = rows[key].get("key_bits")
    return kb["total"] if kb else None


def set_prompt(T: int, kb_ref: float | None, hot: int, vbe: float) -> float:
    """Fix the next prompt's eviction ratio from u3's row; returns the ratio."""
    _S.T, _S.hot, _S.vb = T, hot, vbe
    _S.u3 = u3_bits(T, kb_ref, hot, vbe)
    _S.ratio = ratio_for(_S.u3)
    _S.keep, _S.n_kept = {}, {}
    return _S.ratio


def kv_bits(kb_own: float | None = None) -> dict:
    """Stored bits per cached key or value element of the last prompt, as kept, against
    u3's. ``ea`` keeps fp16 entries; ``ea_u3`` keeps u3-coded settled entries (at this
    prompt's own measured key bits ``kb_own``) and fp16 hot ones."""
    T, hot = _S.T, _S.hot
    per = []
    for li, keep in _S.keep.items():  # (B, Hkv, T) bool
        k = keep.float().sum(-1)  # kept per head
        if MODE == "ea_u3" and T > hot:
            s = keep[..., : T - hot].float().sum(-1)
            bits = (s * (kb_own + _S.vb) + (k - s) * 32) / (2 * T)
        else:
            bits = 16 * k / T
        per.append(float(bits.mean()))
    arm = sum(per) / len(per) if per else (_S.u3 if MODE == "ea_u3" else 16.0)
    return {"arm": round(arm, 6), "u3": round(_S.u3, 6), "tokens": T,
            "ratio": round(_S.ratio, 6)}


# ---------------------------------------------------------------- hooks
def _select(mod, h, keys, values, kwargs):
    """kvpress's ScorerPress.compress steps, returning the kept indices (B, Hkv, n)."""
    scores = _S.press.score(mod, h, keys, values, None, kwargs)
    n = _S.utils.compute_n_kept(keys.shape[2], _S.press.compression_ratio)
    return scores.topk(n, dim=-1).indices


def _post(mod, args, kwargs, out):
    if not _S.enabled() or not _S.ratio or _S.mode == "mask":
        return out
    h = kwargs["hidden_states"]
    lay = kwargs["past_key_values"].layers[mod.layer_idx]
    if lay.keys.shape[2] != h.shape[1]:  # decoding: kvpress compresses at prefill only
        return out
    _S.press.compression_ratio = _S.ratio
    idx = _select(mod, h, lay.keys, lay.values, kwargs)
    keep = torch.zeros(lay.keys.shape[:3], dtype=torch.bool, device=idx.device)
    keep.scatter_(2, idx, True)
    _S.keep[mod.layer_idx] = keep
    if _S.mode == "evict":
        gi = idx.unsqueeze(-1).expand(-1, -1, -1, mod.head_dim)
        lay.keys = lay.keys.gather(2, gi).contiguous()
        lay.values = lay.values.gather(2, gi).contiguous()
    return out


def _pre(mod, args, kwargs):
    if not _S.enabled() or _S.mode != "mask" or mod.layer_idx not in _S.keep:
        return None
    keep = _S.keep[mod.layer_idx]  # (B, Hkv, T)
    T = keep.shape[-1]
    g = mod.config.num_attention_heads // keep.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=keep.device).tril()
    kwargs["attention_mask"] = causal[None, None] & keep.repeat_interleave(g, 1)[:, :, None, :]
    return args, kwargs


def install(model, enabled=lambda: True):
    """Hooks on every attention layer: evict at prefill (generation), or select and then
    mask (perplexity, ``select``). ``enabled`` is off for the KL reference pass."""
    remove()
    press_cls, utils = load_kvpress()
    _S.press = press_cls(compression_ratio=0.0, n_future_positions=N_FUTURE, n_sink=N_SINK)
    _S.utils, _S.enabled, _S.mode = utils, enabled, "evict"
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = model.model.rotary_emb  # as BasePress.__call__ does
        _HOOKS.append(layer.self_attn.register_forward_hook(_post, with_kwargs=True))
        _HOOKS.append(layer.self_attn.register_forward_pre_hook(_pre, with_kwargs=True))


def remove():
    while _HOOKS:
        _HOOKS.pop().remove()


def select(model, chunk):
    """Perplexity pass 1: record what Expected Attention keeps in every layer and head;
    the next forward (pass 2) is masked to it. The caller resets with ``done``."""
    _S.mode = "select"
    with torch.no_grad():
        model(chunk, use_cache=True)
    _S.mode = "mask"


def done():
    _S.mode = "evict"
