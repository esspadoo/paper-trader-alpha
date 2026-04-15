"""High-level market data access service with caching and async APIs."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

from trading_system.data.cache import DataFrameCache, TTLDataFrameCache
from trading_system.data.providers.base import MarketDataProvider
from trading_system.data.schemas import BarRequest

if TYPE_CHECKING:
    import pandas as pd


class MarketDataStream:
    """Fetch normalized market data frames through a provider-backed API."""

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        cache: DataFrameCache | None = None,
        intraday_cache_ttl_seconds: float = 60.0,
        daily_cache_ttl_seconds: float = 3600.0,
    ) -> None:
        """Initialize the stream with a provider and optional cache."""

        self._provider = provider
        self._cache = cache or TTLDataFrameCache(default_ttl_seconds=intraday_cache_ttl_seconds)
        self._intraday_cache_ttl_seconds = intraday_cache_ttl_seconds
        self._daily_cache_ttl_seconds = daily_cache_ttl_seconds
        self._connection_lock = asyncio.Lock()
        self._connected = False

    @property
    def provider_name(self) -> str:
        """Return the underlying provider name."""

        return self._provider.provider_name

    async def connect(self) -> None:
        """Connect the underlying provider once."""

        async with self._connection_lock:
            if self._connected:
                return

            await self._provider.connect()
            self._connected = True

    async def disconnect(self) -> None:
        """Disconnect the underlying provider once."""

        async with self._connection_lock:
            if not self._connected:
                return

            await self._provider.disconnect()
            self._connected = False

    async def fetch_intraday_ohlcv(
        self,
        tickers: Sequence[str],
        *,
        period: str = "5d",
        start: datetime | None = None,
        end: datetime | None = None,
        include_prepost: bool = False,
        force_refresh: bool = False,
    ) -> "pd.DataFrame":
        """Fetch cached or live 5-minute OHLCV bars for multiple tickers."""

        request = BarRequest(
            tickers=tuple(tickers),
            interval="5m",
            start=start,
            end=end,
            period=period,
            include_prepost=include_prepost,
        )
        return await self._fetch(request, ttl_seconds=self._intraday_cache_ttl_seconds, force_refresh=force_refresh)

    async def fetch_daily_ohlcv(
        self,
        tickers: Sequence[str],
        *,
        period: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        force_refresh: bool = False,
    ) -> "pd.DataFrame":
        """Fetch cached or live daily OHLCV bars for multiple tickers."""

        request = BarRequest(
            tickers=tuple(tickers),
            interval="1d",
            start=start,
            end=end,
            period=period,
        )
        return await self._fetch(request, ttl_seconds=self._daily_cache_ttl_seconds, force_refresh=force_refresh)

    async def __aenter__(self) -> "MarketDataStream":
        """Connect the stream when entering an async context."""

        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Disconnect the stream when leaving an async context."""

        await self.disconnect()

    async def _fetch(self, request: BarRequest, *, ttl_seconds: float, force_refresh: bool) -> "pd.DataFrame":
        """Fetch a normalized DataFrame with transparent cache lookups."""

        await self.connect()

        if not force_refresh:
            cached = self._cache.get(request.cache_key)
            if cached is not None:
                return cached

        frame = await self._provider.fetch_bars(request)
        self._cache.set(request.cache_key, frame, ttl_seconds=ttl_seconds)
        return frame.copy(deep=True)
