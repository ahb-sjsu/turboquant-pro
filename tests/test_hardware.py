"""
Tests for hardware-aware compression profiles (Issue #11).

Usage:
    pytest tests/test_hardware.py -v
"""

from __future__ import annotations

from turboquant_pro.hardware import (
    HardwareInfo,
    HardwareProfile,
    detect_gpu,
    get_hardware_profile,
    profile_for_arch,
)


class TestProfileForArch:
    """Test hardware profile recommendations by architecture."""

    def test_volta(self) -> None:
        p = profile_for_arch("volta")
        assert p.recommended_bits == 3
        assert p.recommended_key_bits == 4
        assert p.recommended_value_bits == 3
        assert p.supports_fp8 is False
        assert p.supports_fp4 is False
        assert p.use_fused_kernel is True

    def test_ampere(self) -> None:
        p = profile_for_arch("ampere")
        assert p.recommended_key_bits == 4
        assert p.recommended_value_bits == 3
        assert p.supports_fp8 is False
        assert p.use_fused_kernel is True

    def test_hopper(self) -> None:
        p = profile_for_arch("hopper")
        assert p.recommended_key_bits == 4
        assert p.recommended_value_bits == 3
        assert p.supports_fp8 is True
        assert p.supports_fp4 is False

    def test_blackwell(self) -> None:
        p = profile_for_arch("blackwell")
        assert p.recommended_bits == 4
        assert p.recommended_key_bits == 4
        assert p.recommended_value_bits == 4
        assert p.supports_fp8 is True
        assert p.supports_fp4 is True

    def test_cpu(self) -> None:
        p = profile_for_arch("cpu")
        assert p.hardware.available is False
        assert p.use_fused_kernel is False
        assert p.recommended_key_bits == 4

    def test_all_have_notes(self) -> None:
        for arch in ["volta", "ampere", "hopper", "blackwell", "cpu"]:
            p = profile_for_arch(arch)
            assert len(p.notes) > 0


class TestDetectGPU:
    """Test GPU detection (works in CI without GPU)."""

    def test_detect_returns_hardware_info(self) -> None:
        hw = detect_gpu()
        assert isinstance(hw, HardwareInfo)

    def test_cpu_fallback(self) -> None:
        # Even without GPU, should return a valid HardwareInfo
        hw = detect_gpu()
        assert hw.arch in (
            "cpu",
            "volta",
            "ampere",
            "hopper",
            "blackwell",
            "unknown_7",
            "unknown_8",
            "unknown_9",
            "unknown_10",
        )


class TestGetHardwareProfile:
    """Test get_hardware_profile."""

    def test_returns_profile(self) -> None:
        p = get_hardware_profile()
        assert isinstance(p, HardwareProfile)
        assert p.recommended_bits in (2, 3, 4)

    def test_with_simulated_hardware(self) -> None:
        hw = HardwareInfo(
            available=True,
            name="Test Hopper",
            arch="hopper",
            compute_capability=(9, 0),
            memory_gb=80.0,
        )
        p = get_hardware_profile(hardware=hw)
        assert p.supports_fp8 is True
        assert p.hardware.name == "Test Hopper"


class TestAutoConfigIntegration:
    """Test hardware_profile integration with AutoConfig."""

    def test_with_hardware_tuning(self) -> None:
        from turboquant_pro.autoconfig import AutoConfig

        cfg = AutoConfig.from_pretrained("llama-3-8b")
        tuned = cfg.with_hardware_tuning()
        assert isinstance(tuned, AutoConfig)
        assert tuned.head_dim == cfg.head_dim
        assert tuned.n_layers == cfg.n_layers

    def test_hardware_profile_method(self) -> None:
        from turboquant_pro.autoconfig import AutoConfig

        cfg = AutoConfig.from_pretrained("llama-3-8b")
        p = cfg.hardware_profile()
        assert isinstance(p, HardwareProfile)
