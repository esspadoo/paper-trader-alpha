"""Thread-safe generic TTL cache implementations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class _TTLCacheEntry(Generic[V]):
    """Internal cache entry with value and expiration metadata."""

    value: V
    expires_at: float


class TTLCache(Generic[K, V]):
    """In-memory TTL cache with defensive copies for mutable values."""

    def __init__(self, *, default_ttl_seconds: float = 300.0) -> None:
        """Initialize the cache with a default entry TTL."""

        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be greater than 0")

        self._default_ttl_seconds = default_ttl_seconds
        self._entries: dict[K, _TTLCacheEntry[V]] = {}
        self._lock = RLock()

    def get(self, key: K) -> V | None:
        """Return a cached value copy when present and not expired."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= monotonic():
                self._entries.pop(key, None)
                return None
            return deepcopy(entry.value)

    def set(self, key: K, value: V, *, ttl_seconds: float | None = None) -> None:
        """Store a value copy under the provided key."""

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be greater than 0")

        with self._lock:
            self._entries[key] = _TTLCacheEntry(value=deepcopy(value), expires_at=monotonic() + ttl)

    def contains(self, key: K) -> bool:
        """Return whether a non-expired entry exists for the key."""

        return self.get(key) is not None

    def pop(self, key: K) -> V | None:
        """Remove and return a cached value copy when present."""

        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None or entry.expires_at <= monotonic():
                return None
            return deepcopy(entry.value)

    def clear(self) -> None:
        """Remove all entries from the cache."""

        with self._lock:
            self._entries.clear()
