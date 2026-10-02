# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Generate the TQPW golden corpus (run from the repo root).

One small ``golden.tqpw`` holding a matrix at each of several widths (including the
byte-straddling 3, 5 and 6 bits), the float32 tensors it decodes to
(``expected.npz``), and its sha256 in ``manifest.json``. A reader that reproduces
``expected.npz`` from the bytes alone conforms (``contrib/tqpw_reader.py`` is the
dependency-free one); the in-tree writer must keep producing the same bytes.

Deterministic: fixed seed, the grid drawn in float16, sorted-key JSON. If the file
already exists the script re-encodes it and fails unless the bytes are identical;
it never rewrites a pinned file.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

from turboquant_pro import packed_weights as PW

HERE = pathlib.Path(__file__).parent
SEED = 20261002
CASES = [  # (name, (out, in), bits)
    ("layers.0.self_attn.q_proj", (4, 256), 3),
    ("layers.0.self_attn.o_proj", (2, 128), 5),
    ("layers.0.mlp.down_proj", (3, 384), 8),
    ("layers.1.mlp.up_proj", (2, 128), 2),
    ("layers.1.mlp.gate_proj", (3, 256), 6),
    ("layers.1.self_attn.v_proj", (1, 128), 4),
]
META = {"codec": "gptq", "model": "golden"}


def build() -> list:
    rng = np.random.default_rng(SEED)
    out = []
    for name, (rows, cols), bits in CASES:
        codes = rng.integers(0, 2**bits, size=(rows, cols))
        lo = rng.normal(0, 0.05, size=(rows, cols // 128)).astype(np.float16)
        step = rng.uniform(1e-3, 2e-2, size=(rows, cols // 128)).astype(np.float16)
        out.append(PW.pack_matrix(name, codes, lo, step, bits))
    return out


def main() -> int:
    matrices = build()
    target = HERE / "golden.tqpw"
    if target.exists():
        tmp = HERE / "check.tqpw"
        PW.write(str(tmp), matrices, META)
        same = tmp.read_bytes() == target.read_bytes()
        tmp.unlink()
        if not same:
            print("golden.tqpw would change: a format break", file=sys.stderr)
            return 1
        print("golden.tqpw reproduced byte for byte")
        return 0
    PW.write(str(target), matrices, META)
    np.savez(HERE / "expected.npz", **{m.name: PW.decode(m) for m in matrices})
    blob = target.read_bytes()
    manifest = {
        "format": PW.FORMAT,
        "files": {
            "golden": {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote golden.tqpw ({len(blob)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
