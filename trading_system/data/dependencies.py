"""Optional dependency helpers for the market data layer."""

from __future__ import annotations

from typing import Any

from trading_system.data.exceptions import DependencyNotAvailableError


def require_pandas() -> Any:
    """Return the pandas module or raise a dependency error."""

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise DependencyNotAvailableError(
            "pandas is required for the market data layer. Install the 'market-data' extra."
        ) from exc

    return pd


def require_yfinance() -> Any:
    """Return the yfinance module or raise a dependency error."""

    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise DependencyNotAvailableError(
            "yfinance is required for the development market data provider. Install the 'market-data' extra."
        ) from exc

    return yf
