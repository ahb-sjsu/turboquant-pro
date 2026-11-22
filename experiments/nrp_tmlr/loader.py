# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Build a causal LM with no memory and read its weight stripes into the device.

This is nrp-kit's load_model (same author) with two changes. nrp-kit rebuilds the
modules whose buffers are computed at construction (rotary frequencies) from the
config, on the device; here they are rebuilt on the CPU and moved, as from_pretrained
builds them, because a GPU computes some frequencies one bit differently. Gemma 3's
scaled word embedding cannot be rebuilt from the config (its constructor takes no
config), but it keeps its scale as a plain float, so its buffer is refilled from that
float. The reader is nrp-kit's: one thread per stripe file, 64 MB pieces
into a pinned buffer, no memory-mapping, so a 7B model loads in a pod of 1 CPU and
2 GiB.

    model = load("/w0/mistral/model", sorted(glob.glob("/w?/mistral/s/*.safetensors")),
                 "cuda", torch.float16)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:  # in a pod, nrp-kit's module is copied beside this one
    from load_model import _computed, _read_file, _rebuild
except ImportError:  # on Atlas, the installed nrp-kit
    from nrp_kit.load_model import _computed, _read_file, _rebuild


def load(model_dir: str, shards: list[str], device: str, dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    scaled = [m for m in model.modules() if hasattr(m, "scalar_embed_scale")]
    computed = [
        n
        for n in _computed(model)
        if not hasattr(model.get_submodule(n), "scalar_embed_scale")
    ]
    model.to_empty(device=device)
    model.tie_weights()
    targets = model.state_dict(keep_vars=True)
    pinned = torch.device(device).type == "cuda"
    with ThreadPoolExecutor(max(1, len(shards))) as ex:
        per_file = list(ex.map(lambda p: _read_file(p, targets, pinned), shards))
    loaded = [k for ks in per_file for k in ks]
    if len(loaded) != len(set(loaded)):
        raise ValueError("a weight appears in more than one file")
    have = {id(targets[k]) for k in loaded}
    missing = [k for k, t in targets.items() if id(t) not in have]
    if missing:
        raise ValueError(f"weights missing from the files: {sorted(missing)[:5]}")
    # Computed on the CPU and then moved, as from_pretrained does: rotary frequencies
    # computed on a GPU differ in the last bit (2 of Qwen2.5's 64, 2026-11-22).
    _rebuild(model, computed, cfg, "cpu")
    for n in computed:
        model.get_submodule(n).to(device)
    for m in scaled:  # Gemma 3: the buffer from_pretrained fills from this float
        m.embed_scale.fill_(m.scalar_embed_scale)
    return model.eval()
