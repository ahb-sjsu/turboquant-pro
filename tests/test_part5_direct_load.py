"""DIRECT_LOAD (tq_paper_lb_shard.load_device): device_map loading, same weights.

Part V's 7B pods load straight to the GPU so the host never holds a full copy. The check
here runs on CPU, where the device-map dispatch path (accelerate) and the shipped path
(load, then move) can be compared without a GPU: every tensor must be bit-identical. It
needs a cached checkpoint and skips without one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _load(**kw):
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            **kw,
        )
    except OSError as e:
        pytest.skip(f"{MODEL} not cached: {e!r}")


def test_device_map_loading_gives_identical_weights():
    shipped = _load().state_dict()
    direct = _load(device_map={"": "cpu"}).state_dict()
    assert shipped.keys() == direct.keys()
    for name, t in shipped.items():
        d = direct[name]
        assert t.dtype == d.dtype and t.shape == d.shape, name
        assert torch.equal(t, d), name
