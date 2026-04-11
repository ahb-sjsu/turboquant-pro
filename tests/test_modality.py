"""
Unit tests for multi-modal compression presets.

Tests the modality preset registry, lookup functions, and validation
rules for the per-modality TurboQuant configuration presets.

Usage:
    pytest tests/test_modality.py -v
"""

from __future__ import annotations

import pytest

from turboquant_pro.modality import (
    _KNOWN_MODALITIES,
    ModalityPreset,
    get_modality_preset,
    get_presets_by_modality,
    list_modality_presets,
)

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_REQUIRED_FIELDS = {
    "name",
    "modality",
    "dim",
    "recommended_pca_dim",
    "recommended_bits",
    "recommended_avg_bits",
    "expected_cosine",
    "notes",
}


# ------------------------------------------------------------------ #
# TestModalityPresets                                                  #
# ------------------------------------------------------------------ #


class TestModalityPresets:
    """Tests for modality preset registry and lookup functions."""

    def test_list_presets(self) -> None:
        """list_modality_presets returns at least 10 presets."""
        names = list_modality_presets()
        assert len(names) >= 10

    def test_get_preset(self) -> None:
        """get_modality_preset('bge-m3') returns correct dim=1024."""
        preset = get_modality_preset("bge-m3")
        assert isinstance(preset, ModalityPreset)
        assert preset.dim == 1024
        assert preset.modality == "text"
        assert preset.name == "bge-m3"

    def test_unknown_preset_raises(self) -> None:
        """ValueError for unknown preset name."""
        with pytest.raises(ValueError, match="Unknown modality preset"):
            get_modality_preset("nonexistent-model-xyz")

    def test_get_by_modality(self) -> None:
        """get_presets_by_modality('vision') returns CLIP and SigLIP presets."""
        vision = get_presets_by_modality("vision")
        assert len(vision) >= 2
        names = {p.name for p in vision}
        assert "clip-vit-b-32" in names or "clip-vit-l-14" in names
        assert "siglip-so400m" in names
        # All returned presets should be vision modality
        for preset in vision:
            assert preset.modality == "vision"

    def test_all_presets_valid(self) -> None:
        """All presets have dim > 0, bits in (2,3,4), modality in known set."""
        for name in list_modality_presets():
            preset = get_modality_preset(name)
            assert preset.dim > 0, f"{name}: dim must be > 0"
            assert preset.recommended_bits in (
                2,
                3,
                4,
            ), f"{name}: bits must be 2, 3, or 4"
            assert (
                preset.modality in _KNOWN_MODALITIES
            ), f"{name}: unknown modality {preset.modality!r}"

    def test_text_presets_have_pca(self) -> None:
        """Text presets with dim >= 1024 have pca_dim set."""
        text_presets = get_presets_by_modality("text")
        for preset in text_presets:
            if preset.dim >= 1024:
                assert preset.recommended_pca_dim is not None, (
                    f"{preset.name}: text preset with dim={preset.dim} "
                    f"should have recommended_pca_dim set"
                )

    def test_preset_fields(self) -> None:
        """All required fields present in each preset."""
        for name in list_modality_presets():
            preset = get_modality_preset(name)
            actual_fields = set(vars(preset).keys())
            missing = _REQUIRED_FIELDS - actual_fields
            assert not missing, f"{name}: missing fields {missing}"
