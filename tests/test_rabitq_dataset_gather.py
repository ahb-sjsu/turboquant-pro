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
    for i, n in enumerate((5000, 3000)):
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
    rows = np.array([5200, 3, 7100, 42, 4999, 5000], dtype=np.int64)
    monkeypatch.setattr(ds_mod, "SWEEP_RATIO", 10**9)
    swept = ds._gather_pool(rows)
    for i, r in enumerate(rows):
        part = 0 if r < 5000 else 1
        local = r if part == 0 else r - 5000
        np.testing.assert_array_equal(swept[i], ds._parts[part][local])


def test_sweep_block_boundaries(root, monkeypatch):
    """Rows landing exactly on and around a sweep block edge are all copied once."""
    monkeypatch.setattr(ds_mod, "SWEEP_BLOCK", 64)
    monkeypatch.setattr(ds_mod, "SWEEP_RATIO", 10**9)
    ds = Dataset("smoke-npy", root)
    rows = np.array([0, 63, 64, 65, 127, 128, 4999, 5000, 5001, 7999], dtype=np.int64)
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
    assert swept == [250], swept  # the range itself, not the 5000-row part


def test_a_sparse_request_keeps_the_scattered_read(root, monkeypatch):
    """Three rows spread over thousands: their span is far more than they are."""
    ds = Dataset("smoke-npy", root)
    swept = []
    original = Dataset._sweep

    def spy(self, src, want, out, dest):
        swept.append(1)
        return original(self, src, want, out, dest)

    monkeypatch.setattr(Dataset, "_sweep", spy)
    ds._gather_pool(np.array([3, 2000, 4800], dtype=np.int64))
    assert swept == []


def test_prefetched_blocks_match_a_plain_pass(root):
    """The read-ahead thread must not change what blocks() yields, or its order."""
    ds = Dataset("smoke-npy", root)
    got = [(s, np.array(b, copy=True)) for s, b in ds.blocks(block=333)]
    assert [s for s, _ in got] == list(range(0, ds.n, 333))
    for s, b in got:
        np.testing.assert_array_equal(b, ds.take(np.arange(s, min(ds.n, s + 333))))
    assert sum(len(b) for _, b in got) == ds.n
