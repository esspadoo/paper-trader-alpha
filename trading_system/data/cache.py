"""Thread-safe DataFrame cache implementations for market data requests."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(slots=True)
class _CacheEntry:
    """Internal cache record with expiry metadata."""

    frame: "pd.DataFrame"
    expires_at: float


class DataFrameCache(ABC):
    """Abstract cache for pandas DataFrames."""

    @abstractmethod
    def get(self, key: str) -> "pd.DataFrame | None":
        """Return a copy of the cached frame if the entry is still valid."""

    @abstractmethod
    def set(self, key: str, frame: "pd.DataFrame", *, ttl_seconds: float | None = None) -> None:
        """Store a DataFrame under the given key."""

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """Remove a cache entry if it exists."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all cache entries."""


class TTLDataFrameCache(DataFrameCache):
    """In-memory cache with per-entry TTL and defensive DataFrame copies."""

    def __init__(self, *, default_ttl_seconds: float = 300.0) -> None:
        """Initialize the cache with a default TTL."""

        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be greater than 0")

        self._default_ttl_seconds = default_ttl_seconds
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = RLock()

    def get(self, key: str) -> "pd.DataFrame | None":
        """Return a deep copy of the cached frame when present and unexpired."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            if entry.expires_at <= time.monotonic():
                self._entries.pop(key, None)
                return None

            return entry.frame.copy(deep=True)

    def set(self, key: str, frame: "pd.DataFrame", *, ttl_seconds: float | None = None) -> None:
        """Store a deep copy of the frame under the cache key."""

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be greater than 0")

        with self._lock:
            self._entries[key] = _CacheEntry(
                frame=frame.copy(deep=True),
                expires_at=time.monotonic() + ttl,
            )

    def invalidate(self, key: str) -> None:
        """Remove a cached frame by key."""

        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        """Clear the cache."""

        with self._lock:
            self._entries.clear()
