"""Optional AVX2 ADC fast-scan kernel (C++/pybind11).

The kernel is an optional accelerator: if it is not compiled, :class:`ADCIndex`
falls back to a correct (slower) numpy implementation. Build it with::

    python -m turboquant_pro._adc        # or: pip install turboquant-pro[fast] && that

which compiles ``adc_scan.cpp`` (shipped beside this module) into a
``adc_scan`` extension next to it. Requires a C++17 compiler and ``pybind11``.
``-march=native`` enables the AVX2 ``pshufb`` path; otherwise the scalar path is
used automatically.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig

_HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    """Return the compiled kernel module, or ``None`` if unavailable."""
    try:
        from . import adc_scan  # type: ignore

        return adc_scan
    except Exception:
        return None


def is_available() -> bool:
    return load() is not None


def build(verbose: bool = True) -> str:
    """Compile the kernel into this package directory. Returns the output path."""
    src = os.path.join(_HERE, "adc_scan.cpp")
    if not os.path.exists(src):
        raise FileNotFoundError(f"kernel source not found: {src}")
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = os.path.join(_HERE, f"adc_scan{ext}")
    try:
        import pybind11

        inc = [pybind11.get_include()]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pybind11 is required to build the kernel: pip install turboquant-pro[fast]"
        ) from e
    py_inc = sysconfig.get_path("include")
    cmd = [
        os.environ.get("CXX", "g++"),
        "-O3",
        "-march=native",
        "-fopenmp",
        "-shared",
        "-fPIC",
        "-std=c++17",
        f"-I{py_inc}",
    ]
    cmd += [f"-I{i}" for i in inc]
    cmd += [src, "-o", out]
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    path = build()
    print(f"built {path}")
    print(
        "available" if is_available() else "build succeeded but import failed",
        file=sys.stderr,
    )
