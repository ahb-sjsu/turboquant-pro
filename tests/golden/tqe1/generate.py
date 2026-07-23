# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Regenerate the TQE1 golden corpus (run from the repo root).

The corpus is the cross-implementation conformance target for the TQE1 record
format (ROADMAP_2.0 Pillar 2): small committed ``.tqe`` files plus the exact
decoded tensors the in-tree implementation produces, with sha256 hashes pinned
in ``manifest.json``. Any third-party reader that reproduces ``expected.npz``
from the ``.tqe`` bytes alone conforms; the in-tree writer must keep producing
byte-identical files (format stability is a release gate, not a hope).

Deterministic by construction: fixed vector seed, fixed quantizer seeds.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from turboquant_pro.format import pack_batch
from turboquant_pro.pgvector import TurboQuantPGVector

HERE = pathlib.Path(__file__).parent
N, DIM = 8, 32
VEC_SEED = 20260723

CASES = [
    # (name, bits, rotation, quantizer seed) — v1 records for qr, v2 for hadamard
    ("qr_b2", 2, "qr", 7),
    ("qr_b3", 3, "qr", 11),
    ("qr_b4", 4, "qr", 13),
    ("hadamard_b3", 3, "hadamard", 17),
]


def main() -> None:
    rng = np.random.default_rng(VEC_SEED)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    manifest: dict = {
        "n": N,
        "dim": DIM,
        "vector_seed": VEC_SEED,
        "files": {},
    }
    expected: dict[str, np.ndarray] = {}
    for name, bits, rotation, qseed in CASES:
        q = TurboQuantPGVector(dim=DIM, bits=bits, seed=qseed, rotation=rotation)
        ces = [q.compress_embedding(v) for v in vectors]
        blob = pack_batch(ces)
        (HERE / f"{name}.tqe").write_bytes(blob)
        expected[name] = np.stack([q.decompress_embedding(ce) for ce in ces])
        manifest["files"][name] = {
            "bits": bits,
            "rotation": rotation,
            "seed": qseed,
            "version": 1 if rotation == "qr" else 2,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
        }
        print(f"{name}: {len(blob)} bytes, sha256 {manifest['files'][name]['sha256']}")
    np.savez(HERE / "expected.npz", **expected)
    manifest["expected_sha256"] = hashlib.sha256(
        (HERE / "expected.npz").read_bytes()
    ).hexdigest()
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(CASES)} golden files + expected.npz + manifest.json")


if __name__ == "__main__":
    main()
