# TurboQuant-KV: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
TurboQuant-KV: First open-source implementation of TurboQuant
(Zandieh et al., ICLR 2026) for LLM KV cache compression.

5x memory reduction with 0.978 cosine similarity.
"""

from .core import CompressedKV, TurboQuantKV, TurboQuantKVCache

__all__ = ["CompressedKV", "TurboQuantKV", "TurboQuantKVCache"]
__version__ = "0.1.0"
