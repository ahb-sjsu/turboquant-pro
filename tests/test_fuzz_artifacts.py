"""Replay artifacts are deterministic and reject untrusted evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro.fuzz.artifacts import (
    ReplayBundleError,
    load_replay_bundle,
    write_replay_bundle,
)


def _write(tmp_path, name="case"):
    return write_replay_bundle(
        tmp_path / name,
        case={"seed": 42, "codec": {"bits": 3}, "tolerances": {"atol": 0.0}},
        geometry={
            "schema": "turboquant-pro/fuzz-geometry-profile",
            "schema_version": 1,
        },
        arrays={"queries": np.array([[1.0, 2.0]], dtype=np.float32)},
        documents={"expected_exact.json": {"top_k": [[0, 1]]}},
    )


def test_replay_bundle_is_byte_deterministic_and_round_trips(tmp_path):
    first = _write(tmp_path, "first")
    second = _write(tmp_path, "second")

    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    loaded = load_replay_bundle(first)
    assert loaded["documents"]["case.json"]["seed"] == 42
    np.testing.assert_array_equal(loaded["arrays"]["queries"], [[1.0, 2.0]])


@pytest.mark.parametrize("mutation", ["corrupt", "incomplete", "incompatible"])
def test_replay_bundle_fails_closed_for_corrupt_incomplete_and_incompatible_inputs(
    tmp_path, mutation
):
    bundle = _write(tmp_path)
    if mutation == "corrupt":
        (bundle / "queries.npz").write_bytes(b"not an npz")
    elif mutation == "incomplete":
        (bundle / "geometry.json").unlink()
    else:
        manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
        manifest["schema_version"] = 999
        (bundle / "bundle.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    with pytest.raises(ReplayBundleError):
        load_replay_bundle(bundle)
