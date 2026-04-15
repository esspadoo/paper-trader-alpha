"""Request and selection schemas for the market data layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence


def _normalize_timestamp(value: datetime | None) -> datetime | None:
    """Normalize timestamps to UTC for provider interoperability."""

    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    """Normalize ticker symbols while preserving caller order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)

    if not normalized:
        raise ValueError("at least one ticker symbol is required")

    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class BarRequest:
    """Normalized request for OHLCV bars across one or more symbols."""

    tickers: tuple[str, ...]
    interval: str
    start: datetime | None = None
    end: datetime | None = None
    period: str | None = None
    include_prepost: bool = False
    auto_adjust: bool = False

    def __post_init__(self) -> None:
        """Normalize timestamps and enforce a valid request shape."""

        start = _normalize_timestamp(self.start)
        end = _normalize_timestamp(self.end)
        period = self.period

        object.__setattr__(self, "tickers", _normalize_tickers(self.tickers))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if start is not None or end is not None:
            period = None
        object.__setattr__(self, "period", period)

        if not self.interval:
            raise ValueError("interval must be provided")
        if start is not None and end is not None and start >= end:
            raise ValueError("start must be earlier than end")
        if start is None and end is None and period is None:
            raise ValueError("period must be provided when start and end are omitted")

    @property
    def cache_key(self) -> str:
        """Return a deterministic cache key for the request."""

        return (
            f"tickers={','.join(self.tickers)}|"
            f"interval={self.interval}|"
            f"start={self.start.isoformat() if self.start else ''}|"
            f"end={self.end.isoformat() if self.end else ''}|"
            f"period={self.period or ''}|"
            f"prepost={self.include_prepost}|"
            f"auto_adjust={self.auto_adjust}"
        )


@dataclass(frozen=True, slots=True)
class UniverseSelectionCriteria:
    """Filtering and ranking rules for the daily stock universe."""

    selection_size: int = 100
    min_price: float = 10.0
    min_average_volume: float = 1_000_000.0
    min_daily_volatility: float = 0.02
    lookback_days: int = 20

    def __post_init__(self) -> None:
        """Validate the universe selection parameters."""

        if self.selection_size <= 0:
            raise ValueError("selection_size must be greater than 0")
        if self.min_price < 0:
            raise ValueError("min_price cannot be negative")
        if self.min_average_volume < 0:
            raise ValueError("min_average_volume cannot be negative")
        if self.min_daily_volatility < 0:
            raise ValueError("min_daily_volatility cannot be negative")
        if self.lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")

    @property
    def cache_key(self) -> str:
        """Return a deterministic cache key for the criteria."""

        return (
            f"selection_size={self.selection_size}|"
            f"min_price={self.min_price}|"
            f"min_average_volume={self.min_average_volume}|"
            f"min_daily_volatility={self.min_daily_volatility}|"
            f"lookback_days={self.lookback_days}"
        )
