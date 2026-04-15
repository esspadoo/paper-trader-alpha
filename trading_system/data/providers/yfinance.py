"""Development market data provider backed by yfinance."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from trading_system.data.dependencies import require_pandas, require_yfinance
from trading_system.data.exceptions import DataValidationError, ProviderConfigurationError
from trading_system.data.providers.base import MarketDataProvider
from trading_system.data.schemas import BarRequest

if TYPE_CHECKING:
    import pandas as pd


class YFinanceMarketDataProvider(MarketDataProvider):
    """Async-compatible market data provider backed by yfinance downloads."""

    def __init__(
        self,
        *,
        batch_size: int = 25,
        enable_threads: bool = True,
        request_timeout_seconds: float = 20.0,
    ) -> None:
        """Initialize the provider with safe batching defaults."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than 0")

        self._batch_size = batch_size
        self._enable_threads = enable_threads
        self._request_timeout_seconds = request_timeout_seconds
        self._connected = False

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""

        return "yfinance"

    async def connect(self) -> None:
        """Validate that yfinance is available for downloads."""

        require_yfinance()
        self._connected = True

    async def disconnect(self) -> None:
        """Release provider state."""

        self._connected = False

    async def fetch_bars(self, request: BarRequest) -> "pd.DataFrame":
        """Fetch normalized OHLCV bars for the request."""

        self._ensure_connected()

        batches = [request.tickers[index : index + self._batch_size] for index in range(0, len(request.tickers), self._batch_size)]
        frames = await asyncio.gather(
            *[
                asyncio.to_thread(
                    self._download_batch,
                    batch,
                    request,
                )
                for batch in batches
            ]
        )

        pd = require_pandas()
        if not frames:
            return self._empty_frame(pd)

        non_empty_frames = [frame for frame in frames if not frame.empty]
        if not non_empty_frames:
            return self._empty_frame(pd)

        return pd.concat(non_empty_frames, axis=0).sort_index()

    def _download_batch(self, tickers: Sequence[str], request: BarRequest) -> "pd.DataFrame":
        """Download and normalize a single yfinance batch."""

        pd = require_pandas()
        yf = require_yfinance()

        download_kwargs = {
            "tickers": " ".join(tickers),
            "interval": request.interval,
            "start": request.start,
            "end": request.end,
            "group_by": "ticker",
            "auto_adjust": request.auto_adjust,
            "actions": False,
            "progress": False,
            "threads": self._enable_threads,
            "prepost": request.include_prepost,
            "timeout": self._request_timeout_seconds,
        }
        if request.start is None and request.end is None and request.period is not None:
            download_kwargs["period"] = request.period

        try:
            raw = yf.download(**download_kwargs)
        except Exception as exc:
            raise DataValidationError(
                f"provider {self.provider_name} failed to download bars for {','.join(tickers)}"
            ) from exc

        if raw is None:
            raise DataValidationError(
                f"provider {self.provider_name} returned no data object for {','.join(tickers)}"
            )
        if raw.empty:
            return self._empty_frame(pd)

        if isinstance(raw.columns, pd.MultiIndex):
            frames = [self._normalize_symbol_frame(raw[symbol].copy(), symbol) for symbol in tickers if symbol in raw.columns.get_level_values(0)]
        else:
            frames = [self._normalize_symbol_frame(raw.copy(), tickers[0])]

        non_empty_frames = [frame for frame in frames if not frame.empty]
        if not non_empty_frames:
            return self._empty_frame(pd)

        return pd.concat(non_empty_frames, axis=0).sort_index()

    def _normalize_symbol_frame(self, frame: "pd.DataFrame", symbol: str) -> "pd.DataFrame":
        """Normalize a single-symbol yfinance frame to the shared schema."""

        pd = require_pandas()
        if frame.empty:
            return self._empty_frame(pd)

        column_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        normalized = frame.rename(columns=column_map)
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [column for column in required_columns if column not in normalized.columns]
        if missing_columns:
            raise DataValidationError(
                f"provider {self.provider_name} returned incomplete OHLCV columns for {symbol}: {missing_columns}"
            )

        normalized = normalized[required_columns].copy()
        normalized.index = pd.to_datetime(normalized.index, utc=True)
        normalized["symbol"] = symbol
        normalized = normalized.reset_index(names="timestamp")
        normalized = normalized.set_index(["symbol", "timestamp"])
        normalized.index.names = ["symbol", "timestamp"]

        return normalized.sort_index()

    def _empty_frame(self, pd: object) -> "pd.DataFrame":
        """Return an empty normalized OHLCV frame."""

        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
        return empty

    def _ensure_connected(self) -> None:
        """Ensure the provider has been connected before use."""

        if not self._connected:
            raise ProviderConfigurationError("provider must be connected before fetching bars")
