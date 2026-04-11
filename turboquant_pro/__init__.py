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

from .ans_codec import ANSCodec
from .auto_compress import AutoCompressResult, auto_compress
from .autoconfig import AutoConfig
from .autotune import run_autotune
from .cache_adapter import CompressedEmbeddingCache, InMemoryCacheBackend
from .core import CompressedKV, TurboQuantKV, TurboQuantKVCache
from .export import (
    GenericExporter,
    MilvusExporter,
    PineconeExporter,
    QdrantExporter,
    VectorDBExporter,
    WeaviateExporter,
    export_compressed,
    supported_formats,
)
from .hardware import HardwareInfo, HardwareProfile, detect_gpu, get_hardware_profile
from .hnsw import CompressedHNSW
from .modality import (
    ModalityPreset,
    get_modality_preset,
    get_presets_by_modality,
    list_modality_presets,
)
from .monitor import QualityMonitor
from .nats_codec import TurboQuantNATSCodec
from .pca import (
    EigenweightedPipeline,
    PCACompressedEmbedding,
    PCAFitResult,
    PCAMatryoshka,
    PCAMatryoshkaPipeline,
)
from .pgvector import CompressedEmbedding, TurboQuantPGVector
from .vllm_plugin import TurboQuantKVManager

__all__ = [
    "ANSCodec",
    "AutoCompressResult",
    "AutoConfig",
    "auto_compress",
    "CompressedKV",
    "CompressedEmbedding",
    "CompressedEmbeddingCache",
    "CompressedHNSW",
    "GenericExporter",
    "detect_gpu",
    "EigenweightedPipeline",
    "export_compressed",
    "get_hardware_profile",
    "HardwareInfo",
    "HardwareProfile",
    "InMemoryCacheBackend",
    "MilvusExporter",
    "PCACompressedEmbedding",
    "PineconeExporter",
    "PCAFitResult",
    "PCAMatryoshka",
    "PCAMatryoshkaPipeline",
    "QdrantExporter",
    "QualityMonitor",
    "TurboQuantKV",
    "TurboQuantKVCache",
    "TurboQuantNATSCodec",
    "supported_formats",
    "TurboQuantPGVector",
    "VectorDBExporter",
    "run_autotune",
    "TurboQuantKVManager",
    "WeaviateExporter",
    "ModalityPreset",
    "get_modality_preset",
    "get_presets_by_modality",
    "list_modality_presets",
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
__version__ = "1.0.0"
