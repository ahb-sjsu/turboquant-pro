"""part5_cache_fp16: one harness-style load, saved as fp16, verified bit-identical.

Runs the script end to end on CPU with the smoke's model (Qwen2.5-1.5B-Instruct, bf16
on disk, so the fp16 cast is exercised): the saved copy must reload to the same
tensors, and the tokenizer files must travel with it. It skips without the checkpoint
in the local cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix")
)
import part5_cache_fp16  # noqa: E402

TINY = "Qwen/Qwen2.5-1.5B-Instruct"


def test_round_trip_is_verified_and_keeps_tokenizer_files(tmp_path):
    try:
        from huggingface_hub import snapshot_download

        src = snapshot_download(TINY, local_files_only=True)
    except Exception as e:  # not cached here
        pytest.skip(f"{TINY} not cached: {e!r}")
    dest = tmp_path / "fp16"
    rc = part5_cache_fp16.main(["--src", src, "--dest", str(dest), "--device", "cpu"])
    assert rc == 0
    files = set(os.listdir(dest))
    assert any(f.endswith(".safetensors") for f in files)
    assert "config.json" in files
    tok_files = {f for f in os.listdir(src) if f.startswith(("tokenizer", "vocab"))}
    assert tok_files <= files
