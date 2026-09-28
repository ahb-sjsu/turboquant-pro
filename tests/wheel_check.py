"""Check built wheels for the console's terminal client.

    python tests/wheel_check.py dist/*.whl

A pure wheel (``py3-none-any``) must not carry the client. A Linux wheel
(``py3-none-manylinux2014_<arch>``) must carry it at
``turboquant_pro/console/bin/tqp-console``, marked executable, built for the
wheel's own architecture, and declare itself not pure Python. Exits non-zero on
any failure.
"""

from __future__ import annotations

import re
import sys
import zipfile

CLIENT = "turboquant_pro/console/bin/tqp-console"
ELF_MACHINE = {"x86_64": 62, "aarch64": 183}  # e_machine in the ELF header


def check(path: str) -> list[str]:
    bad = []
    tag = re.search(r"-(py3-none-[A-Za-z0-9_]+)\.whl$", path)
    if not tag:
        return [f"{path}: not a py3-none wheel"]
    tag = tag.group(1)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        wheel = next(n for n in names if n.endswith(".dist-info/WHEEL"))
        meta = z.read(wheel).decode()
        if tag == "py3-none-any":
            if CLIENT in names:
                bad.append("the pure wheel carries the client")
            if "Root-Is-Purelib: true" not in meta:
                bad.append("the pure wheel is not marked purelib")
            return bad
        arch = "x86_64" if tag.endswith("x86_64") else tag.rsplit("_", 1)[-1]
        if arch not in ELF_MACHINE:
            return [f"unexpected platform tag {tag}"]
        if CLIENT not in names:
            return [f"{tag}: no {CLIENT}"]
        info = z.getinfo(CLIENT)
        mode = info.external_attr >> 16
        if not mode & 0o111:
            bad.append(f"{tag}: the client is not executable (mode {oct(mode)})")
        head = z.read(CLIENT)[:20]
        if head[:4] != b"\x7fELF":
            bad.append(f"{tag}: the client is not an ELF binary")
        elif int.from_bytes(head[18:20], "little") != ELF_MACHINE[arch]:
            bad.append(f"{tag}: the client is built for another architecture")
        if f"Tag: {tag}" not in meta:
            bad.append(f"{tag}: WHEEL does not declare Tag: {tag}")
        if "Root-Is-Purelib: false" not in meta:
            bad.append(f"{tag}: a wheel with a binary is marked purelib")
    return bad


def main(paths: list[str]) -> int:
    fails = 0
    for p in paths:
        bad = check(p)
        fails += bool(bad)
        print(("FAIL " if bad else "ok   ") + p + "".join("\n  " + b for b in bad))
    return 1 if fails or not paths else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
