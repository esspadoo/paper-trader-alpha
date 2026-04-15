"""Daily universe selection over normalized market data frames."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Sequence

from trading_system.data.cache import DataFrameCache, TTLDataFrameCache
from trading_system.data.dependencies import require_pandas
from trading_system.data.market_data import MarketDataStream
from trading_system.data.schemas import UniverseSelectionCriteria

if TYPE_CHECKING:
    import pandas as pd


class UniverseSelector:
    """Select a daily trading universe using liquidity and volatility filters."""

    def __init__(
        self,
        market_data_stream: MarketDataStream,
        *,
        default_criteria: UniverseSelectionCriteria | None = None,
        cache: DataFrameCache | None = None,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        """Initialize the selector with a market data source and cache."""

        self._market_data_stream = market_data_stream
        self._default_criteria = default_criteria or UniverseSelectionCriteria()
        self._cache = cache or TTLDataFrameCache(default_ttl_seconds=cache_ttl_seconds)
        self._cache_ttl_seconds = cache_ttl_seconds

    async def select(
        self,
        candidate_tickers: Sequence[str],
        *,
        as_of: datetime | None = None,
        criteria: UniverseSelectionCriteria | None = None,
        force_refresh: bool = False,
    ) -> "pd.DataFrame":
        """Return the top liquid and volatile stocks for the session date."""

        pd = require_pandas()
        active_criteria = criteria or self._default_criteria
        as_of_utc = self._normalize_as_of(as_of)
        cache_key = self._build_cache_key(candidate_tickers, as_of_utc, active_criteria)

        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        start = as_of_utc - timedelta(days=max(active_criteria.lookback_days * 3, 60))
        end = as_of_utc + timedelta(days=1)
        bars = await self._market_data_stream.fetch_daily_ohlcv(
            candidate_tickers,
            start=start,
            end=end,
            force_refresh=force_refresh,
        )
        selection = self._build_selection_frame(bars, active_criteria, as_of_utc, pd)
        self._cache.set(cache_key, selection, ttl_seconds=self._cache_ttl_seconds)
        return selection.copy(deep=True)

    def _build_selection_frame(
        self,
        bars: "pd.DataFrame",
        criteria: UniverseSelectionCriteria,
        as_of: datetime,
        pd: object,
    ) -> "pd.DataFrame":
        """Build the ranked universe DataFrame from daily bars."""

        columns = [
            "as_of",
            "symbol",
            "close",
            "average_volume",
            "average_dollar_volume",
            "realized_volatility",
            "rank",
        ]
        if bars.empty:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, object]] = []
        grouped = bars.sort_index().groupby(level="symbol", sort=False)
        for symbol, symbol_frame in grouped:
            frame = symbol_frame.droplevel("symbol").dropna(subset=["close", "volume"])
            if len(frame) < criteria.lookback_days:
                continue

            window = frame.tail(criteria.lookback_days)
            returns = window["close"].pct_change().dropna()
            if returns.empty:
                continue

            close = float(window["close"].iloc[-1])
            average_volume = float(window["volume"].mean())
            realized_volatility = float(returns.std(ddof=0))
            average_dollar_volume = close * average_volume

            if close <= criteria.min_price:
                continue
            if average_volume <= criteria.min_average_volume:
                continue
            if realized_volatility <= criteria.min_daily_volatility:
                continue

            rows.append(
                {
                    "as_of": as_of.date().isoformat(),
                    "symbol": symbol,
                    "close": close,
                    "average_volume": average_volume,
                    "average_dollar_volume": average_dollar_volume,
                    "realized_volatility": realized_volatility,
                }
            )

        if not rows:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame(rows)
        frame = frame.sort_values(
            by=["average_dollar_volume", "realized_volatility", "symbol"],
            ascending=[False, False, True],
        ).head(criteria.selection_size)
        frame.insert(len(frame.columns), "rank", range(1, len(frame) + 1))
        return frame.reset_index(drop=True)

    def _build_cache_key(
        self,
        candidate_tickers: Sequence[str],
        as_of: datetime,
        criteria: UniverseSelectionCriteria,
    ) -> str:
        """Return a deterministic cache key for a universe request."""

        tickers = ",".join(sorted({ticker.strip().upper() for ticker in candidate_tickers if ticker.strip()}))
        return f"as_of={as_of.date().isoformat()}|tickers={tickers}|{criteria.cache_key}"

    def _normalize_as_of(self, value: datetime | None) -> datetime:
        """Normalize selection timestamps to UTC."""

        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
