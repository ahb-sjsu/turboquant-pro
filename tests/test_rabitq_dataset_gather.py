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
    monkeypatch.setattr(ds_mod, "_sweep_is_cheaper", lambda *a: False)  # never sweep
    scattered = ds._gather_pool(rows)
    monkeypatch.setattr(ds_mod, "_sweep_is_cheaper", lambda *a: True)  # always sweep
    swept = ds._gather_pool(rows)
    np.testing.assert_array_equal(swept, scattered)


def test_sweep_keeps_the_caller_s_row_order(root, monkeypatch):
    """take() passes unsorted positions; each output row must match its request."""
    ds = Dataset("smoke-npy", root)
    rows = np.array([5200, 3, 7100, 42, 4999, 5000], dtype=np.int64)
    monkeypatch.setattr(ds_mod, "_sweep_is_cheaper", lambda *a: True)
    swept = ds._gather_pool(rows)
    for i, r in enumerate(rows):
        part = 0 if r < 5000 else 1
        local = r if part == 0 else r - 5000
        np.testing.assert_array_equal(swept[i], ds._parts[part][local])


def test_sweep_block_boundaries(root, monkeypatch):
    """Rows landing exactly on and around a sweep block edge are all copied once."""
    monkeypatch.setattr(ds_mod, "SWEEP_BLOCK", 64)
    monkeypatch.setattr(ds_mod, "_sweep_is_cheaper", lambda *a: True)
    ds = Dataset("smoke-npy", root)
    rows = np.array([0, 63, 64, 65, 127, 128, 4999, 5000, 5001, 7999], dtype=np.int64)
    swept = ds._gather_pool(rows)
    monkeypatch.setattr(ds_mod, "_sweep_is_cheaper", lambda *a: False)
    np.testing.assert_array_equal(swept, ds._gather_pool(rows))


def test_a_contiguous_range_sweeps_only_itself(root, monkeypatch):
    """blocks() asks for contiguous ranges; sweeping the whole part would reread it."""
    ds = Dataset("smoke-npy", root)
    swept = []
    original = Dataset._sweep

    def spy(self, src, want, out, dest, layout=None):
        swept.append(int(want[-1]) - int(want[0]) + 1)
        return original(self, src, want, out, dest, layout)

    monkeypatch.setattr(Dataset, "_sweep", spy)
    ds._gather_pool(np.arange(250, 500))
    assert swept == [250], swept  # the range itself, not the 5000-row part


def test_the_cost_model_picks_the_cheaper_read():
    """At the measured rates: stream a dense span, fetch sparse rows one by one.

    Counting bytes gets this backwards. 200k training rows out of a million are
    800 MiB against a 4 GiB sweep, but at one ~76 ms round trip each they cost far
    more time than the stream does, and that is what stalled the wiki cells.
    """
    rng = np.random.default_rng(0)

    def decide(rows, dim=1024):
        rows = np.sort(np.asarray(rows))
        runs = int(np.count_nonzero(np.diff(rows) != 1)) + 1
        return ds_mod._sweep_is_cheaper(len(rows), runs, rows, dim)

    assert decide(np.arange(250_000))  # a contiguous block streams
    assert decide(
        rng.choice(1_000_000, 200_000, replace=False)
    )  # a training sample streams
    assert not decide(rng.choice(1_000_000, 100, replace=False))  # 100 rows are fetched
    assert not decide([5, 999_000])  # two rows a million apart are fetched


@pytest.mark.parametrize("count", [1, 2, 37, 400])
def test_the_concurrent_reader_matches_the_memory_map(root, monkeypatch, count):
    """Rows read with file handles must equal rows read through the map."""
    ds = Dataset("smoke-npy", root)
    pool = int(ds._offsets[-1])
    rows = np.random.default_rng(count).choice(pool, count, replace=False)
    monkeypatch.setattr(
        ds_mod, "_sweep_is_cheaper", lambda *a: False
    )  # force the scattered path
    got = ds._gather_pool(np.sort(rows))
    for i, r in enumerate(np.sort(rows)):
        part = 0 if r < 5000 else 1
        local = r if part == 0 else r - 5000
        np.testing.assert_array_equal(got[i], ds._parts[part][local])


def test_the_concurrent_reader_is_skipped_for_an_unreadable_layout(root, monkeypatch):
    """A part whose layout will not parse falls back to the map, not an error."""
    ds = Dataset("smoke-npy", root)
    monkeypatch.setattr(ds_mod, "_sweep_is_cheaper", lambda *a: False)
    ds._layout = [None] * len(ds._layout)
    rows = np.array([3, 2000, 6000], dtype=np.int64)
    got = ds._gather_pool(rows)
    assert got.shape == (3, ds.dim)


def test_the_training_sample_cache_returns_identical_rows(root):
    """The cached rows must equal the gathered ones, and prefixes must stay nested."""
    ds = Dataset("smoke-npy", root)
    first = ds.train_sample(seed=1, size=120)
    cache = os.path.join(root, "trainsample", "smoke-npy-s1.npy")
    assert os.path.exists(cache), "the first call should leave the sample behind"
    second = ds.train_sample(seed=1, size=120)  # now served from the cache
    np.testing.assert_array_equal(second, first)
    prefix = ds.train_sample(seed=1, size=40)
    np.testing.assert_array_equal(prefix, first[:40])
    other = ds.train_sample(seed=2, size=40)
    assert not np.array_equal(other, prefix)  # a different seed is a different draw


def test_an_unreadable_cache_is_rebuilt_not_fatal(root):
    ds = Dataset("smoke-npy", root)
    want = ds.train_sample(seed=3, size=50)
    cache = os.path.join(root, "trainsample", "smoke-npy-s3.npy")
    with open(cache, "wb") as fh:
        fh.write(b"not an npy file")
    np.testing.assert_array_equal(ds.train_sample(seed=3, size=50), want)


def test_the_cache_is_skipped_when_the_volume_is_nearly_full(root, monkeypatch):
    """A full volume costs more than the cache saves; gathering still works."""
    import collections

    ds = Dataset("smoke-npy", root)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(ds_mod.shutil, "disk_usage", lambda p: usage(0, 0, 1024))
    rows = ds.train_sample(seed=7, size=60)
    assert rows.shape == (60, ds.dim)
    assert not os.path.exists(os.path.join(root, "trainsample", "smoke-npy-s7.npy"))


def test_a_bigger_request_than_the_cache_rebuilds_it(root):
    """The cache holds the largest size asked for so far, and never more."""
    ds = Dataset("smoke-npy", root)
    small = ds.train_sample(seed=5, size=30)
    cache = os.path.join(root, "trainsample", "smoke-npy-s5.npy")
    assert len(np.load(cache, mmap_mode="r")) == 30
    big = ds.train_sample(seed=5, size=90)
    np.testing.assert_array_equal(
        big[:30], small
    )  # nested prefixes survive the rebuild
    assert len(np.load(cache, mmap_mode="r")) == 90
