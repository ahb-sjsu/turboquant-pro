"""Write the cross-language conformance corpus for the Rust reader.

    python rust/tqformat/conformance/make_corpus.py OUT_DIR

Python writes, Rust reads (rust/tqformat/tests/conformance.rs): TQE1 batch files with
the in-tree decoder's output as expected values, and persisted indexes with the
expected bytes of every section. The corpus covers what the committed golden corpus
(tests/golden/tqe1/) does not: dimensions other than 32 (including a 3-bit tail that
is not a whole group), the structured `qr` rotation above 4096 dimensions (one with
an odd dimension, so the 32-bit draw buffer carries from the sign draws into the
permutation), large Hadamard rotations, seeds 0 and 2**32 - 1, a zero vector, and
index format versions 2, 3 and 4 with and without deletions.

It is generated afresh in CI rather than committed, so it also checks the reader
against the NumPy and BLAS that CI has: the `qr` expectations are only reproducible
to a tolerance across BLAS builds (see FINDINGS.md).
"""

from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np

import turboquant_pro
from turboquant_pro.format import pack_batch
from turboquant_pro.index import TQEIndex
from turboquant_pro.packed_codes import unpack_rows
from turboquant_pro.pgvector import TurboQuantPGVector

# (name, dim, bits, rotation, codebook, seed)
RECORD_CASES = [
    ("qr_d100_b3", 100, 3, "qr", "legacy", 5),
    ("qr_d256_b2", 256, 2, "qr", "legacy", 0),
    ("qr_d256_lm_b3", 256, 3, "qr", "lloyd-max", 4294967295),
    ("qr_d1024_b4", 1024, 4, "qr", "legacy", 42),
    (
        "qr_d4096_b3",
        4096,
        3,
        "qr",
        "legacy",
        3,
    ),  # the largest dense Q (about 80 s to regenerate in Rust)
    (
        "structured_d4097_b2",
        4097,
        2,
        "qr",
        "legacy",
        11,
    ),  # the smallest structured case, odd
    ("structured_d4100_b3", 4100, 3, "qr", "legacy", 7),
    ("structured_d5003_b4", 5003, 4, "qr", "legacy", 123456),
    ("hadamard_d64_b2", 64, 2, "hadamard", "legacy", 1),
    ("hadamard_d1024_b3", 1024, 3, "hadamard", "legacy", 99),
    ("hadamard_d8192_b4", 8192, 4, "hadamard", "legacy", 2026),
    ("hadamard_d512_lm_b4", 512, 4, "hadamard", "lloyd-max", 31337),
]
N_VECTORS = 4


def crc(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF


def write_records(out: str) -> dict:
    cases = {}
    rng = np.random.default_rng(20261126)
    for name, dim, bits, rotation, codebook, seed in RECORD_CASES:
        q = TurboQuantPGVector(
            dim=dim, bits=bits, seed=seed, rotation=rotation, codebook=codebook
        )
        vecs = rng.standard_normal((N_VECTORS, dim)).astype(np.float32) * np.float32(
            3.0
        )
        if name == "qr_d256_b2":
            vecs[1] = 0.0  # a zero vector: norm 0, decodes to zeros
        ces = [q.compress_embedding(v) for v in vecs]
        blob = pack_batch(ces)
        with open(os.path.join(out, f"{name}.tqe"), "wb") as f:
            f.write(blob)
        decoded = np.stack([q.decompress_embedding(ce) for ce in ces]).astype("<f4")
        with open(os.path.join(out, f"{name}.f32"), "wb") as f:
            f.write(decoded.tobytes())
        cases[name] = {
            "dim": dim,
            "bits": bits,
            "rotation": rotation,
            "codebook": codebook,
            "seed": seed,
            "n": N_VECTORS,
            "bytes": len(blob),
            "codelen": len(ces[0].packed_bytes),
        }
        print(f"{name}: {len(blob)} bytes")
    return cases


def describe_index(path: str) -> dict:
    """What a conforming reader must find in the file, from Python's own reader."""
    from turboquant_pro.index_file import read_container

    version, sections = read_container(path)
    meta = json.loads(sections["meta"].decode("utf-8"))
    arrays = {
        name: {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "crc32": crc(sections[name]),
        }
        for name, spec in meta["arrays"].items()
    }
    idx = TQEIndex.open(path)
    codes = np.ascontiguousarray(idx._adc._codes, dtype=np.uint8)
    cp = meta.get("codes_packed")
    if cp is not None:  # check the unpacking independently of open()
        packed = np.frombuffer(sections["codes"], dtype=np.uint8).reshape(
            meta["arrays"]["codes"]["shape"]
        )
        assert np.array_equal(unpack_rows(packed, cp["dim"], cp["slot_bits"]), codes)
    return {
        "version": version,
        "sections": [[name, len(blob), crc(blob)] for name, blob in sections.items()],
        "meta_crc32": crc(sections["meta"]),
        "arrays": arrays,
        "codes": {
            "rows": int(codes.shape[0]),
            "dim": int(codes.shape[1]),
            "crc32": crc(codes.tobytes()),
        },
        "ids_crc32": crc(np.ascontiguousarray(idx._ids, dtype="<i8").tobytes()),
        "tombstones_crc32": crc(
            np.ascontiguousarray(idx._tomb, dtype=np.uint8).tobytes()
        ),
        "n_ids": int(len(idx._ids)),
    }


def write_indexes(out: str) -> dict:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((400, 48)).astype(np.float32)
    built = {}

    a = TQEIndex.create(
        x, output_dim=32, bits=3, seed=5, keep_originals=False
    )  # v3, implicit ids
    a.save(os.path.join(out, "index_v3_implicit.tqix"))
    built["index_v3_implicit"] = None

    b = TQEIndex.create(x, output_dim=32, bits=4, seed=6, keep_originals=True)
    b.delete(np.array([3, 17, 250]))  # tombstones: ids are written explicitly
    b.save(os.path.join(out, "index_v3_deleted.tqix"))
    built["index_v3_deleted"] = None

    c = TQEIndex.create(
        x, output_dim=32, bits=2, seed=8, codebook="lloyd-max", keep_originals=False
    )  # v4
    c.save(os.path.join(out, "index_v4_lloydmax.tqix"))
    built["index_v4_lloydmax"] = None

    d = TQEIndex.create(x, output_dim=16, bits=3, seed=9, keep_originals=False)
    # Pin the older layout for old readers ("a writer can pin format_version=2",
    # docs/FORMATS.md section 2). TQEIndex.migrate cannot downgrade, so this sets the
    # version the constructor would have been given.
    d._format_version = 2
    d.save(os.path.join(out, "index_v2.tqix"))
    built["index_v2"] = None

    return {name: describe_index(os.path.join(out, f"{name}.tqix")) for name in built}


def main() -> None:
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    manifest = {
        "numpy": np.__version__,
        "turboquant_pro": turboquant_pro.__version__,
        "records": write_records(out),
        "indexes": write_indexes(out),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
