# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Hardware-aware compression profiles.

Auto-detects GPU architecture and recommends optimal TurboQuant
compression parameters.  Different GPU generations have different
native low-precision capabilities:

- **Volta** (compute 7.x): INT8 tensor cores.  Best with 3-bit
  TurboQuant + custom CUDA kernels.
- **Ampere** (compute 8.x): INT8/INT4 tensor cores, TF32.  Same
  as Volta for TurboQuant but benefits from faster memory bandwidth.
- **Hopper** (compute 9.x): FP8 (E4M3/E5M2) tensor cores.  Can
  use native FP8 for intermediate values in the compression pipeline.
- **Blackwell** (compute 10.x): NVFP4 tensor cores.  Native 4-bit
  support makes 4-bit quantization nearly free.

Usage::

    from turboquant_pro.hardware import detect_gpu, get_hardware_profile

    gpu = detect_gpu()
    print(gpu)  # HardwareInfo(name='NVIDIA RTX 4090', arch='ada', ...)

    profile = get_hardware_profile()
    print(profile.recommended_bits)  # 3 for Volta, 4 for Blackwell
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


# ------------------------------------------------------------------ #
# GPU architecture mapping                                             #
# ------------------------------------------------------------------ #

_COMPUTE_TO_ARCH: dict[int, str] = {
    7: "volta",
    8: "ampere",
    9: "hopper",
    10: "blackwell",
}

_ARCH_DISPLAY: dict[str, str] = {
    "volta": "Volta",
    "ampere": "Ampere",
    "hopper": "Hopper",
    "blackwell": "Blackwell",
    "cpu": "CPU",
}


# ------------------------------------------------------------------ #
# Hardware info                                                        #
# ------------------------------------------------------------------ #


@dataclass
class HardwareInfo:
    """Detected GPU hardware information.

    Attributes:
        available: Whether a CUDA GPU was detected.
        name: GPU device name (e.g., ``"NVIDIA A100"``).
        arch: Architecture generation (``"volta"``, ``"ampere"``,
            ``"hopper"``, ``"blackwell"``, or ``"cpu"``).
        compute_capability: Tuple of (major, minor) compute capability.
        memory_gb: Total GPU memory in GB.
        device_id: CUDA device ordinal.
    """

    available: bool = False
    name: str = "CPU"
    arch: str = "cpu"
    compute_capability: tuple[int, int] = (0, 0)
    memory_gb: float = 0.0
    device_id: int = 0


@dataclass
class HardwareProfile:
    """Compression recommendations for detected hardware.

    Attributes:
        hardware: Detected GPU info.
        recommended_bits: Optimal default bit-width for this GPU.
        recommended_key_bits: Optimal key bit-width.
        recommended_value_bits: Optimal value bit-width.
        supports_fp8: Whether FP8 tensor cores are available.
        supports_fp4: Whether FP4/NVFP4 tensor cores are available.
        use_fused_kernel: Whether to use fused rotate+quantize kernels.
        notes: Human-readable notes about the recommendation.
    """

    hardware: HardwareInfo
    recommended_bits: int = 3
    recommended_key_bits: int = 4
    recommended_value_bits: int = 3
    supports_fp8: bool = False
    supports_fp4: bool = False
    use_fused_kernel: bool = True
    notes: str = ""


# ------------------------------------------------------------------ #
# Detection                                                            #
# ------------------------------------------------------------------ #


def detect_gpu(device_id: int = 0) -> HardwareInfo:
    """Detect the GPU at the given device ordinal.

    Returns :class:`HardwareInfo` with ``available=False`` if no GPU
    is found or CuPy is not installed.
    """
    if not _HAS_CUPY:
        return HardwareInfo()

    try:
        dev = cp.cuda.Device(device_id)
        cc = dev.compute_capability
        major = int(str(cc)[0]) if len(str(cc)) >= 2 else int(cc)
        minor = int(str(cc)[1:]) if len(str(cc)) >= 2 else 0
        mem_info = dev.mem_info
        total_gb = mem_info[1] / (1024**3)
        name = cp.cuda.runtime.getDeviceProperties(device_id)["name"].decode()

        arch = _COMPUTE_TO_ARCH.get(major, f"unknown_{major}")

        return HardwareInfo(
            available=True,
            name=name,
            arch=arch,
            compute_capability=(major, minor),
            memory_gb=round(total_gb, 1),
            device_id=device_id,
        )
    except Exception as e:
        logger.warning("GPU detection failed: %s", e)
        return HardwareInfo()


def get_hardware_profile(
    device_id: int = 0,
    hardware: HardwareInfo | None = None,
) -> HardwareProfile:
    """Get compression recommendations for the detected (or given) hardware.

    Args:
        device_id: CUDA device to detect (ignored if *hardware* given).
        hardware: Pre-detected hardware info.  If None, auto-detects.

    Returns:
        HardwareProfile with optimal compression parameters.
    """
    if hardware is None:
        hardware = detect_gpu(device_id)

    arch = hardware.arch

    if arch == "blackwell":
        return HardwareProfile(
            hardware=hardware,
            recommended_bits=4,
            recommended_key_bits=4,
            recommended_value_bits=4,
            supports_fp8=True,
            supports_fp4=True,
            use_fused_kernel=True,
            notes=(
                "Blackwell (compute 10.x): Native NVFP4 makes 4-bit "
                "nearly free. Use 4-bit for both K and V for best quality. "
                "FP8 intermediates available for pipeline acceleration."
            ),
        )

    if arch == "hopper":
        return HardwareProfile(
            hardware=hardware,
            recommended_bits=3,
            recommended_key_bits=4,
            recommended_value_bits=3,
            supports_fp8=True,
            supports_fp4=False,
            use_fused_kernel=True,
            notes=(
                "Hopper (compute 9.x): FP8 tensor cores available. "
                "K4/V3 asymmetric recommended — FP8 intermediates can "
                "accelerate the rotation step. Fused kernels effective."
            ),
        )

    if arch == "ampere":
        return HardwareProfile(
            hardware=hardware,
            recommended_bits=3,
            recommended_key_bits=4,
            recommended_value_bits=3,
            supports_fp8=False,
            supports_fp4=False,
            use_fused_kernel=True,
            notes=(
                "Ampere (compute 8.x): INT8/INT4 tensor cores, higher "
                "memory bandwidth than Volta. K4/V3 asymmetric recommended. "
                "Fused CUDA kernels provide best throughput."
            ),
        )

    if arch == "volta":
        return HardwareProfile(
            hardware=hardware,
            recommended_bits=3,
            recommended_key_bits=4,
            recommended_value_bits=3,
            supports_fp8=False,
            supports_fp4=False,
            use_fused_kernel=True,
            notes=(
                "Volta (compute 7.x): INT8 tensor cores. K4/V3 asymmetric "
                "recommended. Fused rotate+quantize CUDA kernels designed "
                "for this architecture."
            ),
        )

    # CPU or unknown
    return HardwareProfile(
        hardware=hardware,
        recommended_bits=3,
        recommended_key_bits=4,
        recommended_value_bits=3,
        supports_fp8=False,
        supports_fp4=False,
        use_fused_kernel=False,
        notes=(
            "CPU mode: No GPU detected. Using NumPy backend. "
            "K4/V3 asymmetric still recommended for quality. "
            "Consider installing CuPy for GPU acceleration."
        ),
    )


def profile_for_arch(arch: str) -> HardwareProfile:
    """Get a hardware profile for a named architecture (for testing).

    Args:
        arch: One of ``"volta"``, ``"ampere"``, ``"hopper"``,
            ``"blackwell"``, ``"cpu"``.
    """
    hw = HardwareInfo(
        available=arch != "cpu",
        name=f"Simulated {_ARCH_DISPLAY.get(arch, arch)}",
        arch=arch,
    )
    return get_hardware_profile(hardware=hw)
