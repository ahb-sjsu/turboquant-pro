# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Unit tests for the ANS codec and HNSW save/load.

Tests delta+varint integer compression, neighbor list encoding, and
full HNSW index serialization/deserialization round-trips.

Usage:
    pytest tests/test_ans_codec.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from turboquant_pro.ans_codec import ANSCodec
from turboquant_pro.hnsw import CompressedHNSW
from turboquant_pro.pgvector import TurboQuantPGVector

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture()
def codec() -> ANSCodec:
    return ANSCodec()


@pytest.fixture()
def tq() -> TurboQuantPGVector:
    return TurboQuantPGVector(dim=64, bits=3, seed=0)


# ------------------------------------------------------------------ #
# Varint / integer encoding tests                                      #
# ------------------------------------------------------------------ #


class TestVarIntEncoding:
    """Delta + varint encoding of integer lists."""

    def test_small_integers(self, codec: ANSCodec) -> None:
        """Encode/decode [0, 1, 2, 3] round-trips exactly."""
        ids = [0, 1, 2, 3]
        data = codec.encode_integers(ids)
        recovered = codec.decode_integers(data, n=len(ids))
        assert recovered == sorted(ids)

    def test_large_integers(self, codec: ANSCodec) -> None:
        """Encode/decode [100000, 200000, 300000] round-trips exactly."""
        ids = [100000, 200000, 300000]
        data = codec.encode_integers(ids)
        recovered = codec.decode_integers(data, n=len(ids))
        assert recovered == sorted(ids)

    def test_empty_list(self, codec: ANSCodec) -> None:
        """Encode/decode empty list returns empty."""
        data = codec.encode_integers([])
        recovered = codec.decode_integers(data, n=0)
        assert recovered == []

    def test_single_element(self, codec: ANSCodec) -> None:
        """Encode/decode [42] round-trips exactly."""
        ids = [42]
        data = codec.encode_integers(ids)
        recovered = codec.decode_integers(data, n=1)
        assert recovered == [42]

    def test_roundtrip_random(self, codec: ANSCodec) -> None:
        """1000 random non-negative ints round-trip correctly."""
        rng = np.random.default_rng(42)
        ids = rng.integers(0, 1_000_000, size=1000).tolist()
        data = codec.encode_integers(ids)
        recovered = codec.decode_integers(data, n=len(ids))
        assert recovered == sorted(ids)

    def test_compression_ratio(self, codec: ANSCodec) -> None:
        """Sorted IDs with locality compress to < 4 bytes per ID."""
        # Simulate HNSW neighbor IDs: clustered around a base value.
        rng = np.random.default_rng(99)
        base = 50000
        ids = sorted([base + x for x in rng.integers(0, 500, size=100).tolist()])
        data = codec.encode_integers(ids)
        bytes_per_id = len(data) / len(ids)
        assert bytes_per_id < 4.0, f"Expected < 4 bytes/ID, got {bytes_per_id:.2f}"

    def test_duplicate_ids(self, codec: ANSCodec) -> None:
        """Duplicate IDs are preserved after sort (deltas of 0)."""
        ids = [5, 5, 10, 10, 10]
        data = codec.encode_integers(ids)
        recovered = codec.decode_integers(data, n=len(ids))
        assert recovered == sorted(ids)

    def test_unsorted_input(self, codec: ANSCodec) -> None:
        """Unsorted input is sorted before encoding and recovered sorted."""
        ids = [99, 3, 42, 7, 1000]
        data = codec.encode_integers(ids)
        recovered = codec.decode_integers(data, n=len(ids))
        assert recovered == sorted(ids)

    def test_zero_only(self, codec: ANSCodec) -> None:
        """[0] encodes and decodes correctly."""
        data = codec.encode_integers([0])
        recovered = codec.decode_integers(data, n=1)
        assert recovered == [0]

    def test_large_single_value(self, codec: ANSCodec) -> None:
        """Large single value encodes correctly."""
        ids = [2**28]
        data = codec.encode_integers(ids)
        recovered = codec.decode_integers(data, n=1)
        assert recovered == ids


# ------------------------------------------------------------------ #
# Neighbor list encoding tests                                         #
# ------------------------------------------------------------------ #


class TestNeighborListEncoding:
    """Neighbor list (id, distance) pair encoding."""

    def test_roundtrip(self, codec: ANSCodec) -> None:
        """Encode/decode a neighbor list preserves (id, distance) pairs."""
        neighbors = [(10, 0.1), (42, 0.3), (100, 0.5), (5, 0.8)]
        packed = codec.encode_neighbor_list(neighbors)
        restored = codec.decode_neighbor_list(packed)

        # IDs are sorted during encoding.  The distances are reordered
        # to follow the sorted ID order, so build a lookup from the
        # original list and verify each restored pair.
        original_by_id = {nid: dist for nid, dist in neighbors}
        assert len(restored) == len(neighbors)
        restored_ids = [r[0] for r in restored]
        assert restored_ids == sorted(original_by_id.keys())
        for rid, rdist in restored:
            assert rid in original_by_id
            assert abs(rdist - original_by_id[rid]) < 1e-6

    def test_empty_neighbors(self, codec: ANSCodec) -> None:
        """Empty neighbor list round-trips correctly."""
        packed = codec.encode_neighbor_list([])
        restored = codec.decode_neighbor_list(packed)
        assert restored == []

    def test_distances_preserved(self, codec: ANSCodec) -> None:
        """Float32 distances match exactly after round-trip."""
        import struct

        # Use specific float32 values that are exactly representable.
        neighbors = [(1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)]
        packed = codec.encode_neighbor_list(neighbors)
        restored = codec.decode_neighbor_list(packed)

        restored_sorted = sorted(restored, key=lambda x: x[0])
        original_sorted = sorted(neighbors, key=lambda x: x[0])
        for (_, rdist), (_, odist) in zip(restored_sorted, original_sorted):
            # Pack/unpack through float32 to get exact representation.
            expected = struct.unpack("<f", struct.pack("<f", odist))[0]
            actual = struct.unpack("<f", struct.pack("<f", rdist))[0]
            assert actual == expected

    def test_single_neighbor(self, codec: ANSCodec) -> None:
        """Single-entry neighbor list round-trips."""
        neighbors = [(77, 0.42)]
        packed = codec.encode_neighbor_list(neighbors)
        restored = codec.decode_neighbor_list(packed)
        assert len(restored) == 1
        assert restored[0][0] == 77
        assert abs(restored[0][1] - 0.42) < 1e-6

    def test_compressed_smaller(self, codec: ANSCodec) -> None:
        """Compressed neighbor list is smaller than uncompressed."""
        neighbors = [(i * 10, float(i) * 0.1) for i in range(32)]
        compressed_sz = codec.compressed_size(neighbors)
        uncompressed_sz = codec.uncompressed_size(neighbors)
        assert compressed_sz < uncompressed_sz


# ------------------------------------------------------------------ #
# HNSW save/load tests                                                 #
# ------------------------------------------------------------------ #


class TestHNSWSaveLoad:
    """HNSW index serialization and deserialization."""

    def test_save_load_roundtrip(self, tq: TurboQuantPGVector) -> None:
        """Build index with 100 vectors, save, load, search produces
        same top-5 results."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((100, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        query = rng.standard_normal(64).astype(np.float32)
        results_before = index.search(query, k=5, rerank=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_index.tqhnsw")
            index.save(path)

            loaded = CompressedHNSW.load(path, tq)
            results_after = loaded.search(query, k=5, rerank=False)

        # Same IDs in same order.
        ids_before = [r[0] for r in results_before]
        ids_after = [r[0] for r in results_after]
        assert ids_before == ids_after

        # Scores should be identical.
        for (_, s1), (_, s2) in zip(results_before, results_after):
            assert abs(s1 - s2) < 1e-6

    def test_save_load_preserves_structure(self, tq: TurboQuantPGVector) -> None:
        """n_vectors, top_layer, entry_point match after load."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((100, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_index.tqhnsw")
            index.save(path)
            loaded = CompressedHNSW.load(path, tq)

        assert loaded.n_vectors == index.n_vectors
        assert loaded._top_layer == index._top_layer
        assert loaded._entry_point == index._entry_point

    def test_save_load_empty_index(self, tq: TurboQuantPGVector) -> None:
        """Empty index can be saved and loaded."""
        index = CompressedHNSW(tq, M=8, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "empty.tqhnsw")
            index.save(path)
            loaded = CompressedHNSW.load(path, tq)

        assert loaded.n_vectors == 0
        assert loaded._entry_point is None
        results = loaded.search(np.zeros(64, dtype=np.float32), k=5)
        assert results == []

    def test_save_load_rerank_consistency(self, tq: TurboQuantPGVector) -> None:
        """Reranked search on loaded index matches original."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((100, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        query = rng.standard_normal(64).astype(np.float32)
        results_before = index.search(query, k=5, rerank=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_rerank.tqhnsw")
            index.save(path)
            loaded = CompressedHNSW.load(path, tq)
            results_after = loaded.search(query, k=5, rerank=True)

        ids_before = [r[0] for r in results_before]
        ids_after = [r[0] for r in results_after]
        assert ids_before == ids_after

    def test_load_wrong_magic_raises(self, tq: TurboQuantPGVector) -> None:
        """Loading a file with wrong magic raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bad_magic.tqhnsw")
            with open(path, "wb") as f:
                f.write(b"BADMAGIC" + b"\x00" * 100)

            with pytest.raises(ValueError, match="Invalid magic"):
                CompressedHNSW.load(path, tq)

    def test_load_wrong_tq_params_raises(self, tq: TurboQuantPGVector) -> None:
        """Loading with mismatched TQ parameters raises ValueError."""
        index = CompressedHNSW(tq, M=8, seed=42)
        rng = np.random.default_rng(42)
        index.insert(0, rng.standard_normal(64).astype(np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mismatch.tqhnsw")
            index.save(path)

            # Create a TQ with different seed.
            tq_other = TurboQuantPGVector(dim=64, bits=3, seed=999)
            with pytest.raises(ValueError, match="TQ parameter mismatch"):
                CompressedHNSW.load(path, tq_other)

    def test_file_size_smaller_than_naive(self, tq: TurboQuantPGVector) -> None:
        """Saved file is smaller than naive storage (float32 embeddings
        + 12 bytes per neighbor edge)."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((200, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "size_test.tqhnsw")
            index.save(path)
            file_size = Path(path).stat().st_size

        # Naive: 200 vectors * 64 dims * 4 bytes = 51200 bytes just for
        # embeddings.  The file uses compressed embeddings + compressed
        # neighbor lists, so it should be significantly smaller.
        naive_embedding_bytes = 200 * 64 * 4
        assert file_size < naive_embedding_bytes


# ------------------------------------------------------------------ #
# Memory usage reporting                                               #
# ------------------------------------------------------------------ #


class TestMemoryUsage:
    """Updated memory_usage_bytes reports both compressed and
    uncompressed graph sizes."""

    def test_memory_usage_returns_dict(self, tq: TurboQuantPGVector) -> None:
        """memory_usage_bytes returns a dict with expected keys."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        for i in range(50):
            index.insert(i, rng.standard_normal(64).astype(np.float32))

        mem = index.memory_usage_bytes()
        assert isinstance(mem, dict)
        expected_keys = {
            "embeddings",
            "indices",
            "graph_uncompressed",
            "graph_compressed",
            "total",
        }
        assert set(mem.keys()) == expected_keys

    def test_compressed_smaller_than_uncompressed(self, tq: TurboQuantPGVector) -> None:
        """Compressed graph size is smaller than uncompressed."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        for i in range(200):
            index.insert(i, rng.standard_normal(64).astype(np.float32))

        mem = index.memory_usage_bytes()
        assert mem["graph_compressed"] < mem["graph_uncompressed"]

    def test_stats_includes_memory_detail(self, tq: TurboQuantPGVector) -> None:
        """stats() includes both memory_bytes and memory_detail."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        for i in range(50):
            index.insert(i, rng.standard_normal(64).astype(np.float32))

        stats = index.stats()
        assert "memory_bytes" in stats
        assert "memory_detail" in stats
        assert stats["memory_bytes"] == stats["memory_detail"]["total"]
