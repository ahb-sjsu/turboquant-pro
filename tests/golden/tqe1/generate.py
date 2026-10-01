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

Extending, not rewriting: a case already in ``manifest.json`` is re-encoded and
must reproduce its pinned bytes exactly (the script fails otherwise), and its
committed expected tensor is kept as is, so adding a case never re-derives an
old expectation under a different BLAS. Only new cases are written.
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
    # (name, bits, rotation, quantizer seed, codebook). v1 records for qr with the
    # legacy codebook, v2 for another rotation, v3 for a non-legacy codebook.
    ("qr_b2", 2, "qr", 7, "legacy"),
    ("qr_b3", 3, "qr", 11, "legacy"),
    ("qr_b4", 4, "qr", 13, "legacy"),
    ("hadamard_b3", 3, "hadamard", 17, "legacy"),
    ("lloydmax_qr_b3", 3, "qr", 19, "lloyd-max"),
    ("lloydmax_qr_b4", 4, "qr", 23, "lloyd-max"),
    ("lloydmax_hadamard_b2", 2, "hadamard", 29, "lloyd-max"),
]


def _version(rotation: str, codebook: str) -> int:
    if codebook != "legacy":
        return 3
    return 1 if rotation == "qr" else 2


def main() -> None:
    rng = np.random.default_rng(VEC_SEED)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    mpath = HERE / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {"files": {}}
    old_expected = (
        dict(np.load(HERE / "expected.npz")) if (HERE / "expected.npz").exists() else {}
    )
    manifest: dict = {
        "n": N,
        "dim": DIM,
        "vector_seed": VEC_SEED,
        "files": {},
    }
    expected: dict[str, np.ndarray] = {}
    for name, bits, rotation, qseed, codebook in CASES:
        q = TurboQuantPGVector(
            dim=DIM, bits=bits, seed=qseed, rotation=rotation, codebook=codebook
        )
        ces = [q.compress_embedding(v) for v in vectors]
        blob = pack_batch(ces)
        sha = hashlib.sha256(blob).hexdigest()
        entry = {
            "bits": bits,
            "rotation": rotation,
            "seed": qseed,
            "version": _version(rotation, codebook),
            "sha256": sha,
            "bytes": len(blob),
        }
        if codebook != "legacy":
            entry["codebook"] = codebook
        if name in old["files"]:
            if old["files"][name]["sha256"] != sha or name not in old_expected:
                raise SystemExit(
                    f"{name}: the writer no longer reproduces the pinned bytes; "
                    "that is a format break, not a regeneration"
                )
            expected[name] = old_expected[name]
            status = "verified"
        else:
            (HERE / f"{name}.tqe").write_bytes(blob)
            expected[name] = np.stack([q.decompress_embedding(ce) for ce in ces])
            status = "new"
        manifest["files"][name] = entry
        print(f"{name}: {len(blob)} bytes, sha256 {sha} ({status})")
    np.savez(HERE / "expected.npz", **expected)
    manifest["expected_sha256"] = hashlib.sha256(
        (HERE / "expected.npz").read_bytes()
    ).hexdigest()
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(CASES)} golden files + expected.npz + manifest.json")


if __name__ == "__main__":
    main()
