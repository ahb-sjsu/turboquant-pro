# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
ANS-inspired lossless integer codec for HNSW graph compression.

Implements delta + varint encoding for compressing sorted integer ID lists
(HNSW neighbor IDs).  This achieves good compression on neighbor lists
where IDs exhibit locality -- sorted delta values are typically small and
encode efficiently with variable-length integers.

A January 2025 Meta paper demonstrated that lossless compression of vector
IDs using ANS coding gives ~7x compression on IDs and ~30% overall index
reduction.  This module provides a simpler delta+varint scheme that
captures most of the benefit with straightforward implementation.

Wire format for neighbor lists::

    [2 bytes] n_neighbors  (uint16 LE)
    [2 bytes] id_bytes_len (uint16 LE)
    [N bytes] varint-encoded delta IDs
    [M bytes] float32 distances (4 * n_neighbors bytes)

Usage::

    from turboquant_pro.ans_codec import ANSCodec

    codec = ANSCodec()

    # Compress a list of integer IDs
    data = codec.encode_integers([10, 42, 43, 100, 105])
    ids = codec.decode_integers(data, n=5)

    # Compress a full neighbor list
    neighbors = [(10, 0.1), (42, 0.3), (100, 0.5)]
    packed = codec.encode_neighbor_list(neighbors)
    restored = codec.decode_neighbor_list(packed)
"""

from __future__ import annotations

import logging
import struct

logger = logging.getLogger(__name__)


class ANSCodec:
    """Asymmetric Numeral Systems codec for lossless integer compression.

    Encodes sequences of non-negative integers using a delta + varint
    scheme.  Designed for compressing HNSW neighbor ID lists where values
    are sorted integer IDs with exploitable locality.

    The encoding pipeline:
    1. Sort the IDs (neighbor lists are typically unsorted).
    2. Delta-encode: ``[ids[0], ids[1]-ids[0], ids[2]-ids[1], ...]``
    3. Varint-encode each delta (7 bits per byte, high bit = continuation).

    This is lossless and achieves good compression when IDs are clustered,
    which is common in HNSW graphs where nearby nodes share neighbors.
    """

    # ------------------------------------------------------------------ #
    # Varint primitives                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode a single non-negative integer as a varint.

        Uses 7 bits per byte with the high bit as a continuation flag.
        """
        if value < 0:
            raise ValueError(f"Cannot varint-encode negative value: {value}")
        parts: list[int] = []
        while value > 0x7F:
            parts.append((value & 0x7F) | 0x80)
            value >>= 7
        parts.append(value & 0x7F)
        return bytes(parts)

    @staticmethod
    def _decode_varint(data: bytes | memoryview, offset: int) -> tuple[int, int]:
        """Decode a single varint starting at *offset*.

        Returns:
            Tuple of (decoded_value, new_offset).
        """
        result = 0
        shift = 0
        while True:
            if offset >= len(data):
                raise ValueError("Unexpected end of data while decoding varint")
            byte = data[offset]
            result |= (byte & 0x7F) << shift
            offset += 1
            if not (byte & 0x80):
                break
            shift += 7
        return result, offset

    # ------------------------------------------------------------------ #
    # Integer list encoding                                                #
    # ------------------------------------------------------------------ #

    def encode_integers(self, ids: list[int]) -> bytes:
        """Encode a list of non-negative integers using delta + varint.

        The IDs are sorted, then delta-encoded, then each delta is
        varint-encoded.

        Args:
            ids: List of non-negative integers to encode.

        Returns:
            Compact bytes representation.
        """
        if not ids:
            return b""

        sorted_ids = sorted(ids)
        parts: list[bytes] = []

        # First value is encoded directly.
        parts.append(self._encode_varint(sorted_ids[0]))

        # Subsequent values are delta-encoded.
        for i in range(1, len(sorted_ids)):
            delta = sorted_ids[i] - sorted_ids[i - 1]
            parts.append(self._encode_varint(delta))

        return b"".join(parts)

    def decode_integers(self, data: bytes, n: int) -> list[int]:
        """Decode *n* integers from varint-encoded delta bytes.

        Args:
            data: Bytes from :meth:`encode_integers`.
            n: Number of integers to decode.

        Returns:
            List of original integers (sorted).
        """
        if n == 0:
            return []

        offset = 0
        values: list[int] = []

        # First value is stored directly.
        value, offset = self._decode_varint(data, offset)
        values.append(value)

        # Subsequent values are cumulative sums of deltas.
        for _ in range(1, n):
            delta, offset = self._decode_varint(data, offset)
            value = values[-1] + delta
            values.append(value)

        return values

    # ------------------------------------------------------------------ #
    # Neighbor list encoding                                               #
    # ------------------------------------------------------------------ #

    def encode_neighbor_list(self, neighbors: list[tuple[int, float]]) -> bytes:
        """Encode a neighbor list of ``(node_id, distance)`` pairs.

        Format::

            [2 bytes] n_neighbors  (uint16 LE)
            [2 bytes] id_bytes_len (uint16 LE)
            [N bytes] delta+varint encoded IDs
            [M bytes] float32 distances (4 * n_neighbors bytes, LE)

        Args:
            neighbors: List of ``(node_id, distance)`` tuples.

        Returns:
            Compact bytes representation.
        """
        n = len(neighbors)
        if n == 0:
            return struct.pack("<HH", 0, 0)

        # Sort by ID so delta encoding produces small deltas.
        sorted_neighbors = sorted(neighbors, key=lambda x: x[0])
        ids = [nid for nid, _ in sorted_neighbors]
        distances = [dist for _, dist in sorted_neighbors]

        # Delta+varint encode the (already sorted) IDs.
        id_bytes = self.encode_integers(ids)
        id_bytes_len = len(id_bytes)

        # Pack distances as float32 LE.
        dist_bytes = struct.pack(f"<{n}f", *distances)

        # Header: n_neighbors (2 bytes) + id_bytes_len (2 bytes).
        header = struct.pack("<HH", n, id_bytes_len)

        return header + id_bytes + dist_bytes

    def decode_neighbor_list(self, data: bytes) -> list[tuple[int, float]]:
        """Decode a neighbor list from bytes.

        Args:
            data: Bytes from :meth:`encode_neighbor_list`.

        Returns:
            List of ``(node_id, distance)`` tuples.
        """
        if len(data) < 4:
            raise ValueError(f"Neighbor list data too short: {len(data)} < 4")

        n, id_bytes_len = struct.unpack("<HH", data[:4])

        if n == 0:
            return []

        # Decode IDs.
        id_data = data[4 : 4 + id_bytes_len]
        ids = self.decode_integers(id_data, n)

        # Decode distances.
        dist_start = 4 + id_bytes_len
        dist_end = dist_start + n * 4
        distances = list(struct.unpack(f"<{n}f", data[dist_start:dist_end]))

        # Reconstruct pairs.  The IDs were sorted during encoding, but the
        # distances correspond to the sorted order, so we return them as-is.
        return list(zip(ids, distances))

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def compressed_size(self, neighbors: list[tuple[int, float]]) -> int:
        """Return the compressed size in bytes for a neighbor list.

        Args:
            neighbors: List of ``(node_id, distance)`` tuples.

        Returns:
            Size in bytes after compression.
        """
        return len(self.encode_neighbor_list(neighbors))

    def uncompressed_size(self, neighbors: list[tuple[int, float]]) -> int:
        """Return the uncompressed (logical) size in bytes.

        Each entry is ``(int id, float distance)`` = 12 bytes.

        Args:
            neighbors: List of ``(node_id, distance)`` tuples.

        Returns:
            Size in bytes at 12 bytes per entry.
        """
        return len(neighbors) * 12
