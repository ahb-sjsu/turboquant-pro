# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Engine-native KV connectors (2.0 Pillar 1 — ``docs/ROADMAP_2.0.md``).

``vllm_v1`` implements vLLM's first-class V1 connector interface so a
production cluster enables TurboQuant KV compression with configuration, not a
fork. The block store is engine-agnostic by design; the SGLang adapter mounts
the same store behind its hierarchical-cache offload hooks (roadmap M2).
"""

from .identity import IncompatibleProfile, KVIdentityProfile, prefix_block_hashes
from .vllm_v1 import TurboQuantBlockStore, TurboQuantKVConnector, register

__all__ = [
    "TurboQuantKVConnector",
    "TurboQuantBlockStore",
    "register",
    "KVIdentityProfile",
    "IncompatibleProfile",
    "prefix_block_hashes",
]
