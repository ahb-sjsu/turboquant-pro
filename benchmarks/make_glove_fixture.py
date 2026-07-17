# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Regenerate the hermetic tiny GloVe fixture from the real dataset.

Deterministically subsamples a small **real** glove-100-angular corpus + query
set into ``benchmarks/fixtures/glove_tiny.npz`` — the CI-safe stand-in used by
``benchmarks/canonical_glove.py --small``. Kept a genuine subset (same
distribution, same 100-d geometry) rather than synthetic so the hermetic claim
is a faithful, if small, proxy of the full one.

Run on a machine that has the dataset (e.g. Atlas), not in CI:

    TQP_GLOVE_HDF5=/archive/cache/glove-100-angular.hdf5 \\
        python benchmarks/make_glove_fixture.py --n 3000 --queries 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "fixtures", "glove_tiny.npz")
GLOVE_URL = "http://ann-benchmarks.com/glove-100-angular.hdf5"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the tiny GloVe CI fixture.")
    ap.add_argument("--n", type=int, default=2000, help="corpus vectors to keep")
    ap.add_argument("--queries", type=int, default=100, help="query vectors to keep")
    ap.add_argument("--seed", type=int, default=0, help="subsample seed")
    args = ap.parse_args(argv)

    import h5py

    path = os.environ.get("TQP_GLOVE_HDF5") or "glove-100-angular.hdf5"
    if not os.path.exists(path):
        import urllib.request

        print(f"downloading {GLOVE_URL} -> {path} ...", flush=True)
        urllib.request.urlretrieve(GLOVE_URL, path)
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    ti = np.sort(rng.choice(len(train), size=min(args.n, len(train)), replace=False))
    qi = np.sort(
        rng.choice(len(test), size=min(args.queries, len(test)), replace=False)
    )
    # float16 keeps the bundled fixture small; the tiny recall gate is a smoke
    # check, and half-precision GloVe is a negligible perturbation for it.
    sub_train = train[ti].astype(np.float16)
    sub_test = test[qi].astype(np.float16)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, train=sub_train, test=sub_test)
    size_kb = os.path.getsize(OUT) / 1024
    print(
        f"wrote {OUT}: train={sub_train.shape} test={sub_test.shape} "
        f"({size_kb:.1f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
