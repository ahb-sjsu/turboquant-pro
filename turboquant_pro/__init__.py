# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
TurboQuant Pro: First open-source implementation of TurboQuant
(Zandieh et al., ICLR 2026) for LLM KV cache compression.

5x memory reduction with 0.978 cosine similarity.

Also provides TurboQuant compression for:
- pgvector embeddings (TurboQuantPGVector)
- NATS message bus transport (TurboQuantNATSCodec)
"""

from .core import CompressedKV, TurboQuantKV, TurboQuantKVCache
from .nats_codec import TurboQuantNATSCodec
from .pgvector import CompressedEmbedding, TurboQuantPGVector

__all__ = [
    "CompressedKV",
    "CompressedEmbedding",
    "TurboQuantKV",
    "TurboQuantKVCache",
    "TurboQuantNATSCodec",
    "TurboQuantPGVector",
]
__version__ = "0.2.0"
