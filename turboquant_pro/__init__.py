# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
TurboQuant Pro: Embedding compression for LLM inference and vector databases.

Features:
- PCA-Matryoshka dimension reduction (PCAMatryoshka, PCAMatryoshkaPipeline)
- TurboQuant scalar quantization (2/3/4-bit) with bit-packing
- KV cache compression for LLM inference (TurboQuantKVCache)
- pgvector embedding compression (TurboQuantPGVector)
- NATS message bus transport compression (TurboQuantNATSCodec)

Typical usage::

    from turboquant_pro import PCAMatryoshka

    pca = PCAMatryoshka(input_dim=1024, output_dim=384)
    pca.fit(sample_embeddings)
    pipeline = pca.with_quantizer(bits=3)  # ~27x compression
    compressed = pipeline.compress(embedding)
"""

from .autotune import run_autotune
from .cache_adapter import CompressedEmbeddingCache, InMemoryCacheBackend
from .core import CompressedKV, TurboQuantKV, TurboQuantKVCache
from .hnsw import CompressedHNSW
from .nats_codec import TurboQuantNATSCodec
from .pca import (
    PCACompressedEmbedding,
    PCAFitResult,
    PCAMatryoshka,
    PCAMatryoshkaPipeline,
)
from .pgvector import CompressedEmbedding, TurboQuantPGVector
from .vllm_plugin import TurboQuantKVManager

__all__ = [
    "CompressedKV",
    "CompressedEmbedding",
    "CompressedEmbeddingCache",
    "CompressedHNSW",
    "InMemoryCacheBackend",
    "PCACompressedEmbedding",
    "PCAFitResult",
    "PCAMatryoshka",
    "PCAMatryoshkaPipeline",
    "TurboQuantKV",
    "TurboQuantKVCache",
    "TurboQuantNATSCodec",
    "TurboQuantPGVector",
    "run_autotune",
    "TurboQuantKVManager",
]

# FAISS integration (lazy import — only available with faiss)
try:
    from .faiss_index import TurboQuantFAISS  # noqa: F401

    __all__.append("TurboQuantFAISS")
except Exception:
    pass

# GPU search functions (lazy import — only available with CuPy)
try:
    from .cuda_search import (
        gpu_adc_search,  # noqa: F401
        gpu_hamming_search,  # noqa: F401
        pack_binary,  # noqa: F401
    )

    __all__.extend(["gpu_adc_search", "gpu_hamming_search", "pack_binary"])
except Exception:
    pass
__version__ = "0.8.0"
