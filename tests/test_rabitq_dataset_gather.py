"""Dataset._gather_pool's sequential sweep returns exactly what scattered reads do."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from rabitq_public import datasets as ds_mod  # noqa: E402
from rabitq_public.datasets import Dataset  # noqa: E402


@pytest.fixture
def root(tmp_path):
    """A two-part npy corpus matching the smoke spec."""
    rng = np.random.default_rng(0)
    d = tmp_path / "smoke"
    d.mkdir()
    for i, n in enumerate((700, 500)):
        np.save(d / f"part_{i:03d}.npy", rng.standard_normal((n, 32), dtype=np.float32))
    return str(tmp_path)


@pytest.mark.parametrize("count", [1, 7, 120, 900])
def test_sweep_and_scatter_agree(root, monkeypatch, count):
    ds = Dataset("smoke-npy", root)
    rows = np.sort(
        np.random.default_rng(count).choice(int(ds._offsets[-1]), count, replace=False)
    )
    monkeypatch.setattr(ds_mod, "SWEEP_RATIO", 0)  # never sweep
    scattered = ds._gather_pool(rows)
    monkeypatch.setattr(ds_mod, "SWEEP_RATIO", 10**9)  # always sweep
    swept = ds._gather_pool(rows)
    np.testing.assert_array_equal(swept, scattered)


def test_sweep_keeps_the_caller_s_row_order(root, monkeypatch):
    """take() passes unsorted positions; each output row must match its request."""
    ds = Dataset("smoke-npy", root)
    rows = np.array([900, 3, 1100, 42, 699, 700], dtype=np.int64)
    monkeypatch.setattr(ds_mod, "SWEEP_RATIO", 10**9)
    swept = ds._gather_pool(rows)
    for i, r in enumerate(rows):
        part = 0 if r < 700 else 1
        local = r if part == 0 else r - 700
        np.testing.assert_array_equal(swept[i], ds._parts[part][local])


def test_sweep_block_boundaries(root, monkeypatch):
    """Rows landing exactly on and around a sweep block edge are all copied once."""
    monkeypatch.setattr(ds_mod, "SWEEP_BLOCK", 64)
    monkeypatch.setattr(ds_mod, "SWEEP_RATIO", 10**9)
    ds = Dataset("smoke-npy", root)
    rows = np.array([0, 63, 64, 65, 127, 128, 699, 700, 701, 1199], dtype=np.int64)
    swept = ds._gather_pool(rows)
    monkeypatch.setattr(ds_mod, "SWEEP_RATIO", 0)
    np.testing.assert_array_equal(swept, ds._gather_pool(rows))


def test_a_contiguous_range_sweeps_only_itself(root, monkeypatch):
    """blocks() asks for contiguous ranges; sweeping the whole part would reread it."""
    ds = Dataset("smoke-npy", root)
    swept = []
    original = Dataset._sweep

    def spy(self, src, want, out, dest):
        swept.append(int(want[-1]) - int(want[0]) + 1)
        return original(self, src, want, out, dest)

    monkeypatch.setattr(Dataset, "_sweep", spy)
    ds._gather_pool(np.arange(250, 500))
    assert swept == [250], swept  # the range itself, not the 700-row part


def test_a_sparse_request_keeps_the_scattered_read(root, monkeypatch):
    ds = Dataset("smoke-npy", root)
    swept = []
    monkeypatch.setattr(
        Dataset, "_sweep", lambda self, *a: swept.append(1) or Dataset._sweep(self, *a)
    )
    ds._gather_pool(np.array([3, 200, 640], dtype=np.int64))
    assert swept == []
