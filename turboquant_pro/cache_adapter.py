# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Compressed embedding cache with pluggable backends.

Provides an L2 RAM cache for TurboQuant-compressed embeddings, eliminating
redundant compression/decompression for frequently accessed vectors. The
cache stores embeddings in their compressed wire format (pgbytea), so memory
usage is ~10x lower than caching raw float32 vectors.

Two backends are included:

  - **InMemoryCacheBackend**: Pure-Python LRU cache backed by
    ``collections.OrderedDict``. Suitable for single-process workloads
    with a bounded number of hot embeddings.

  - **RedisCacheBackend**: Delegates to a ``redis.Redis`` client for
    shared, multi-process caching with optional TTL expiry. Requires the
    ``redis`` package (``pip install redis``).

The high-level ``CompressedEmbeddingCache`` wraps a backend together with a
``TurboQuantPGVector`` instance, handling compression, serialization, and
hit/miss statistics transparently.

Usage::

    from turboquant_pro.pgvector import TurboQuantPGVector
    from turboquant_pro.cache_adapter import (
        CompressedEmbeddingCache,
        InMemoryCacheBackend,
    )

    tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
    backend = InMemoryCacheBackend(max_entries=10_000)
    cache = CompressedEmbeddingCache(tq, backend)

    cache.put("doc:42", embedding_float32)
    result = cache.get("doc:42")  # np.ndarray or None
    print(cache.stats())
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .pgvector import CompressedEmbedding, TurboQuantPGVector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Abstract backend                                                    #
# ------------------------------------------------------------------ #


class CacheBackend(ABC):
    """Abstract base class for cache storage backends."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Returns:
            The stored bytes, or ``None`` if the key is absent.
        """

    @abstractmethod
    def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """Store a value under *key*.

        Args:
            key: Cache key.
            value: Raw bytes to store.
            ttl: Time-to-live in seconds.  ``None`` means no expiry.
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key.

        Returns:
            ``True`` if the key existed and was removed.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether *key* is present in the cache."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of entries in the cache."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""

    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """Estimate the total memory consumed by stored values.

        Returns:
            Byte count, or ``-1`` if the metric is unavailable.
        """


# ------------------------------------------------------------------ #
# In-memory LRU backend                                               #
# ------------------------------------------------------------------ #


class InMemoryCacheBackend(CacheBackend):
    """LRU cache backed by :class:`collections.OrderedDict`.

    On :meth:`get`, the accessed entry is moved to the end (most-recently
    used).  On :meth:`set`, when the cache is at capacity the oldest
    entry is evicted from the front.

    Args:
        max_entries: Maximum number of entries.  ``None`` means unlimited.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        self._max_entries = max_entries
        self._store: OrderedDict[str, bytes] = OrderedDict()

    # -- CacheBackend interface ----------------------------------------

    def get(self, key: str) -> bytes | None:
        if key not in self._store:
            return None
        # LRU refresh: move to end
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        # TTL is accepted for interface compatibility but not enforced
        # in this simple in-memory backend.
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
        else:
            if (
                self._max_entries is not None
                and len(self._store) >= self._max_entries
            ):
                # Evict the least-recently used (front) entry
                self._store.popitem(last=False)
            self._store[key] = value

    def delete(self, key: str) -> bool:
        try:
            del self._store[key]
            return True
        except KeyError:
            return False

    def exists(self, key: str) -> bool:
        return key in self._store

    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def memory_usage_bytes(self) -> int:
        return sum(len(v) for v in self._store.values())


# ------------------------------------------------------------------ #
# Redis backend                                                       #
# ------------------------------------------------------------------ #


class RedisCacheBackend(CacheBackend):
    """Cache backend delegating to a :class:`redis.Redis` client.

    The ``redis`` package is imported lazily so that the rest of the
    library works without it.

    Args:
        redis_client: An already-connected ``redis.Redis`` instance.
        prefix: Key prefix applied to every cache key (default ``"tq:"``).
    """

    def __init__(self, redis_client: Any, prefix: str = "tq:") -> None:
        try:
            import redis  # noqa: F401  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'redis' package is required for RedisCacheBackend. "
                "Install it with: pip install redis"
            ) from exc

        self._client = redis_client
        self._prefix = prefix

    def _prefixed(self, key: str) -> str:
        return f"{self._prefix}{key}"

    # -- CacheBackend interface ----------------------------------------

    def get(self, key: str) -> bytes | None:
        value = self._client.get(self._prefixed(key))
        return value  # bytes or None

    def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        pkey = self._prefixed(key)
        if ttl is not None:
            self._client.setex(pkey, ttl, value)
        else:
            self._client.set(pkey, value)

    def delete(self, key: str) -> bool:
        return bool(self._client.delete(self._prefixed(key)))

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(self._prefixed(key)))

    def size(self) -> int:
        # Count keys matching the prefix pattern
        cursor, keys = self._client.scan(match=f"{self._prefix}*", count=10000)
        total = len(keys)
        while cursor != 0:
            cursor, keys = self._client.scan(
                cursor=cursor, match=f"{self._prefix}*", count=10000
            )
            total += len(keys)
        return total

    def clear(self) -> None:
        cursor, keys = self._client.scan(match=f"{self._prefix}*", count=10000)
        if keys:
            self._client.delete(*keys)
        while cursor != 0:
            cursor, keys = self._client.scan(
                cursor=cursor, match=f"{self._prefix}*", count=10000
            )
            if keys:
                self._client.delete(*keys)

    def memory_usage_bytes(self) -> int:
        # Server-side memory usage is not easily accessible per-prefix.
        return -1


# ------------------------------------------------------------------ #
# High-level compressed embedding cache                               #
# ------------------------------------------------------------------ #


class CompressedEmbeddingCache:
    """Compressed embedding cache wrapping a backend and a quantizer.

    Stores embeddings in their TurboQuant-compressed pgbytea wire format,
    achieving ~10x memory reduction compared to float32 caching while
    keeping decompression latency under a microsecond per vector.

    Args:
        tq: A :class:`TurboQuantPGVector` instance that defines the
            compression parameters (dim, bits, seed).
        backend: A :class:`CacheBackend` for actual storage.
        default_ttl: Default time-to-live in seconds for cached entries.
            ``None`` means no expiry.
    """

    def __init__(
        self,
        tq: TurboQuantPGVector,
        backend: CacheBackend,
        default_ttl: int | None = None,
    ) -> None:
        self._tq = tq
        self._backend = backend
        self._default_ttl = default_ttl
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # Single-entry operations                                             #
    # ------------------------------------------------------------------ #

    def put(
        self,
        key: str,
        embedding: np.ndarray,
        ttl: int | None = None,
    ) -> None:
        """Compress and cache a single embedding.

        Args:
            key: Cache key.
            embedding: Float32 array of shape ``(dim,)``.
            ttl: Per-entry TTL override.  Falls back to *default_ttl*.
        """
        compressed = self._tq.compress_embedding(embedding)
        data = compressed.to_pgbytea()
        self._backend.set(key, data, ttl if ttl is not None else self._default_ttl)

    def get(self, key: str) -> np.ndarray | None:
        """Retrieve and decompress a cached embedding.

        Args:
            key: Cache key.

        Returns:
            Approximate float32 embedding of shape ``(dim,)``, or ``None``
            on a cache miss.
        """
        data = self._backend.get(key)
        if data is None:
            self._miss_count += 1
            return None
        self._hit_count += 1
        compressed = CompressedEmbedding.from_pgbytea(
            data, self._tq.dim, self._tq.bits
        )
        return self._tq.decompress_embedding(compressed)

    # ------------------------------------------------------------------ #
    # Batch operations                                                    #
    # ------------------------------------------------------------------ #

    def put_batch(
        self,
        keys: Sequence[str],
        embeddings: np.ndarray,
    ) -> None:
        """Compress and cache a batch of embeddings.

        Args:
            keys: Sequence of cache keys (one per embedding).
            embeddings: Float32 array of shape ``(n, dim)``.
        """
        compressed_list = self._tq.compress_batch(np.asarray(embeddings, dtype=np.float32))
        ttl = self._default_ttl
        for key, compressed in zip(keys, compressed_list):
            self._backend.set(key, compressed.to_pgbytea(), ttl)

    def get_batch(self, keys: Sequence[str]) -> list[np.ndarray | None]:
        """Retrieve and decompress a batch of cached embeddings.

        Args:
            keys: Sequence of cache keys.

        Returns:
            List of float32 arrays (or ``None`` for cache misses),
            in the same order as *keys*.
        """
        results: list[np.ndarray | None] = []
        for key in keys:
            results.append(self.get(key))
        return results

    # ------------------------------------------------------------------ #
    # Statistics                                                          #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Return cache performance and memory statistics.

        Returns:
            Dict with keys:

            - ``n_entries``: Number of cached embeddings.
            - ``memory_bytes``: Approximate bytes used by the backend.
            - ``equivalent_float32_bytes``: How much memory the same
              number of embeddings would consume as raw float32.
            - ``effective_compression_ratio``: float32 / compressed.
            - ``hit_count``: Total cache hits.
            - ``miss_count``: Total cache misses.
            - ``hit_rate``: Hit count / (hit + miss), or 0.0 if no
              accesses yet.
        """
        n_entries = self._backend.size()
        memory_bytes = self._backend.memory_usage_bytes()
        equivalent_float32_bytes = n_entries * self._tq.dim * 4
        total_accesses = self._hit_count + self._miss_count

        if memory_bytes > 0:
            effective_compression_ratio = equivalent_float32_bytes / memory_bytes
        else:
            effective_compression_ratio = 0.0

        return {
            "n_entries": n_entries,
            "memory_bytes": memory_bytes,
            "equivalent_float32_bytes": equivalent_float32_bytes,
            "effective_compression_ratio": round(effective_compression_ratio, 2),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(
                self._hit_count / total_accesses, 4
            )
            if total_accesses > 0
            else 0.0,
        }
