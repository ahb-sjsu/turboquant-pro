"""Wheel build hook: bundle the console's terminal client (go/tqp-console).

An ordinary build is unchanged: a pure-Python ``py3-none-any`` wheel, no Go
needed. With ``TQP_WHEEL_GOARCH`` set to ``amd64`` or ``arm64`` the hook
cross-compiles the client for Linux on that architecture (static, no cgo),
places it at ``turboquant_pro/console/bin/tqp-console`` (where ``tqp console``
looks for it), and tags the wheel for that platform, so pip on Linux picks it
over the pure wheel:

    TQP_WHEEL_GOARCH=amd64 python -m build --wheel

The client is Linux-only (it uses Linux terminal ioctls), so there are no
macOS or Windows client wheels; there the pure wheel's ``tqp console --web``
still works.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Go's architecture name -> the wheel platform tag. A static Go binary needs no
# libc, so manylinux2014 (glibc 2.17) claims nothing it does not have.
PLATFORMS = {
    "amd64": "manylinux2014_x86_64",
    "arm64": "manylinux2014_aarch64",
}


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        goarch = os.environ.get("TQP_WHEEL_GOARCH", "")
        if self.target_name != "wheel" or version == "editable" or not goarch:
            return
        if goarch not in PLATFORMS:
            raise ValueError(
                f"TQP_WHEEL_GOARCH={goarch!r}: expected one of {sorted(PLATFORMS)}"
            )
        src = os.path.join(self.root, "go", "tqp-console")
        self._tmp = tempfile.mkdtemp(prefix="tqp-console-wheel-")
        out = os.path.join(self._tmp, "tqp-console")
        env = dict(os.environ, GOOS="linux", GOARCH=goarch, CGO_ENABLED="0")
        subprocess.run(
            ["go", "build", "-trimpath", "-ldflags=-s -w", "-o", out, "."],
            cwd=src,
            env=env,
            check=True,
        )
        os.chmod(out, 0o755)
        build_data["force_include"][out] = "turboquant_pro/console/bin/tqp-console"
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{PLATFORMS[goarch]}"

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        tmp = getattr(self, "_tmp", None)
        if tmp:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
