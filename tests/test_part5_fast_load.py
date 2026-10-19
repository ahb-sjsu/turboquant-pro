"""FAST_LOAD (fast_reads.reads): from_pretrained with whole-tensor preads, same model.

Part V's pods load each checkpoint through ``fast_reads.reads()``, which changes only
how the file's bytes are read. The checks here run on CPU: a model loaded with it must
equal the shipped load in every parameter and buffer (persistent or not), its config
and its generation config, and give the same logits; the reads must actually have gone
through pread; and without FAST_LOAD=1 nothing is patched. Tiny random models cover an
untied fp16 checkpoint and a tied bf16 one cast to fp16, each saved in several files;
the smoke's real checkpoint is compared too when it is cached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix")
)
import fast_reads as FR  # noqa: E402

CACHED = "Qwen/Qwen2.5-1.5B-Instruct"


def _tiny(tmp_path, kind):
    """A random model saved in several safetensors files with a generation config."""
    torch.manual_seed(0)
    if kind == "llama":
        cfg = transformers.LlamaConfig(
            vocab_size=96,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
        cls, dtype = transformers.LlamaForCausalLM, torch.float16
    else:
        cfg = transformers.Qwen2Config(
            vocab_size=96,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        cls, dtype = transformers.Qwen2ForCausalLM, torch.bfloat16
    model = cls(cfg).to(dtype)
    model.generation_config = transformers.GenerationConfig(
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=7,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
    )
    out = tmp_path / kind
    model.save_pretrained(out, max_shard_size="40KB")
    assert len(list(out.glob("*.safetensors"))) > 1
    return str(out)


def _load(path, fast, monkeypatch, **kw):
    monkeypatch.setenv("FAST_LOAD", "1" if fast else "0")
    with FR.reads():
        return transformers.AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **kw,
        ).eval()


def _same(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for n, t in pa.items():
        assert t.dtype == pb[n].dtype and t.shape == pb[n].shape, n
        assert torch.equal(t, pb[n]), n
    ba, bb = dict(a.named_buffers()), dict(b.named_buffers())
    assert ba.keys() == bb.keys()
    for n, t in ba.items():
        assert t.dtype == bb[n].dtype and torch.equal(t, bb[n]), n
    assert a.config.to_dict() == b.config.to_dict()
    assert a.generation_config.to_dict() == b.generation_config.to_dict()
    assert (a.lm_head.weight is a.model.embed_tokens.weight) == (
        b.lm_head.weight is b.model.embed_tokens.weight
    )
    ids = torch.arange(1, 25).unsqueeze(0) % a.config.vocab_size
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_fast_load_gives_the_shipped_model(tmp_path, monkeypatch, kind):
    path = _tiny(tmp_path, kind)
    shipped = _load(path, False, monkeypatch)
    before = dict(FR.stats)
    fast = _load(path, True, monkeypatch)
    n = sum(len(FR.header(str(f))[0]) for f in Path(path).glob("*.safetensors"))
    assert FR.stats["tensors"] - before["tensors"] == n
    assert FR.stats["bytes"] > before["bytes"]
    _same(shipped, fast)


def test_without_fast_load_nothing_is_patched(monkeypatch):
    from transformers import modeling_utils as mu

    real = mu.safe_open
    monkeypatch.delenv("FAST_LOAD", raising=False)
    with FR.reads():
        assert mu.safe_open is real
    monkeypatch.setenv("FAST_LOAD", "1")
    with FR.reads():
        assert mu.safe_open is not real
    assert mu.safe_open is real


def test_a_partial_slice_reads_the_real_file(tmp_path, monkeypatch):
    from safetensors import safe_open

    path = _tiny(tmp_path, "llama")
    f = next(Path(path).glob("*.safetensors"))
    name = next(iter(FR.header(str(f))[0]))
    monkeypatch.setenv("FAST_LOAD", "1")
    from transformers import modeling_utils as mu

    with FR.reads(), safe_open(str(f), framework="pt", device="cpu") as real:
        h = mu.safe_open(str(f), framework="pt", device="cpu")
        s = h.get_slice(name)
        assert torch.equal(s[...], real.get_tensor(name))
        assert torch.equal(s[0:1], real.get_slice(name)[0:1])
        assert s.get_shape() == real.get_slice(name).get_shape()
        h.__exit__(None, None, None)


def test_fast_load_of_the_cached_smoke_model(monkeypatch):
    try:
        shipped = _load(CACHED, False, monkeypatch, local_files_only=True)
    except OSError as e:
        pytest.skip(f"{CACHED} not cached: {e!r}")
    _same(shipped, _load(CACHED, True, monkeypatch, local_files_only=True))
