# TurboQuant Pro: Multi-modal embedding compression presets
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Per-modality compression presets for multi-modal embeddings.

Provides recommended PCA dimension and bit-width settings for common
embedding models across text, vision, audio, and code modalities.
Each preset captures the optimal TurboQuant configuration discovered
through systematic benchmarking on representative corpora.

Typical usage::

    from turboquant_pro.modality import (
        get_modality_preset,
        get_presets_by_modality,
        list_modality_presets,
    )

    # Get a single preset by model name
    preset = get_modality_preset("clip-vit-l-14")
    print(preset.dim, preset.recommended_pca_dim, preset.recommended_bits)

    # List all available presets
    names = list_modality_presets()

    # Filter by modality category
    vision_presets = get_presets_by_modality("vision")
"""

from __future__ import annotations

from dataclasses import dataclass

# ------------------------------------------------------------------ #
# Preset dataclass                                                     #
# ------------------------------------------------------------------ #


@dataclass
class ModalityPreset:
    """Compression preset for a specific embedding modality.

    Attributes:
        name: Modality name (e.g., "clip-vit-l-14").
        modality: Category ("text", "vision", "audio", "code").
        dim: Embedding dimension.
        recommended_pca_dim: Optimal PCA output dimension (None = no PCA).
        recommended_bits: Optimal uniform bit-width.
        recommended_avg_bits: Optimal eigenweighted avg bits.
        expected_cosine: Expected cosine similarity with recommended config.
        notes: Human-readable notes.
    """

    name: str
    modality: str
    dim: int
    recommended_pca_dim: int | None
    recommended_bits: int
    recommended_avg_bits: float
    expected_cosine: float
    notes: str


# ------------------------------------------------------------------ #
# Built-in presets                                                     #
# ------------------------------------------------------------------ #

_MODALITY_PRESETS: dict[str, ModalityPreset] = {
    "bge-m3": ModalityPreset(
        name="bge-m3",
        modality="text",
        dim=1024,
        recommended_pca_dim=384,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.979,
        notes="BAAI BGE-M3 multi-lingual; log-decay eigenspectrum.",
    ),
    "e5-large-v2": ModalityPreset(
        name="e5-large-v2",
        modality="text",
        dim=1024,
        recommended_pca_dim=384,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.978,
        notes="Microsoft E5-large-v2; similar spectral profile to BGE-M3.",
    ),
    "ada-002": ModalityPreset(
        name="ada-002",
        modality="text",
        dim=1536,
        recommended_pca_dim=512,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.975,
        notes="OpenAI text-embedding-ada-002; high dim benefits from PCA.",
    ),
    "clip-vit-b-32": ModalityPreset(
        name="clip-vit-b-32",
        modality="vision",
        dim=512,
        recommended_pca_dim=None,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.980,
        notes="CLIP ViT-B/32; compact dim, uniform variance — skip PCA.",
    ),
    "clip-vit-l-14": ModalityPreset(
        name="clip-vit-l-14",
        modality="vision",
        dim=768,
        recommended_pca_dim=384,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.978,
        notes="CLIP ViT-L/14; moderate PCA reduction still beneficial.",
    ),
    "siglip-so400m": ModalityPreset(
        name="siglip-so400m",
        modality="vision",
        dim=1152,
        recommended_pca_dim=512,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.976,
        notes="Google SigLIP SO400M; high dim with moderate spectral decay.",
    ),
    "whisper-base": ModalityPreset(
        name="whisper-base",
        modality="audio",
        dim=512,
        recommended_pca_dim=None,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.980,
        notes="OpenAI Whisper base encoder; compact dim — skip PCA.",
    ),
    "whisper-large-v3": ModalityPreset(
        name="whisper-large-v3",
        modality="audio",
        dim=1280,
        recommended_pca_dim=512,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.977,
        notes="OpenAI Whisper large-v3 encoder; moderate eigenvalue decay.",
    ),
    "codebert-base": ModalityPreset(
        name="codebert-base",
        modality="code",
        dim=768,
        recommended_pca_dim=384,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.978,
        notes="Microsoft CodeBERT; text-like spectral profile.",
    ),
    "codellama-emb": ModalityPreset(
        name="codellama-emb",
        modality="code",
        dim=4096,
        recommended_pca_dim=1024,
        recommended_bits=3,
        recommended_avg_bits=3.0,
        expected_cosine=0.974,
        notes="Code Llama embedding; very high dim, large PCA gain.",
    ),
}

# Known modality categories for validation
_KNOWN_MODALITIES = frozenset({"text", "vision", "audio", "code"})


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #


def get_modality_preset(name: str) -> ModalityPreset:
    """Look up a modality preset by model name.

    Args:
        name: Model name (e.g., ``"bge-m3"``, ``"clip-vit-l-14"``).

    Returns:
        The corresponding :class:`ModalityPreset`.

    Raises:
        ValueError: If *name* is not a known preset.
    """
    try:
        return _MODALITY_PRESETS[name]
    except KeyError:
        available = ", ".join(sorted(_MODALITY_PRESETS))
        raise ValueError(
            f"Unknown modality preset {name!r}. " f"Available presets: {available}"
        ) from None


def list_modality_presets() -> list[str]:
    """Return a sorted list of all available preset names.

    Returns:
        List of preset name strings.
    """
    return sorted(_MODALITY_PRESETS)


def get_presets_by_modality(modality: str) -> list[ModalityPreset]:
    """Return all presets for a given modality category.

    Args:
        modality: One of ``"text"``, ``"vision"``, ``"audio"``, ``"code"``.

    Returns:
        List of matching :class:`ModalityPreset` objects, sorted by name.

    Raises:
        ValueError: If *modality* is not a known category.
    """
    if modality not in _KNOWN_MODALITIES:
        raise ValueError(
            f"Unknown modality {modality!r}. "
            f"Known modalities: {', '.join(sorted(_KNOWN_MODALITIES))}"
        )
    return sorted(
        (p for p in _MODALITY_PRESETS.values() if p.modality == modality),
        key=lambda p: p.name,
    )
