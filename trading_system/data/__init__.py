"""Async-compatible market data APIs, providers, and selection utilities."""

from trading_system.data.base import BaseDataStream
from trading_system.data.cache import DataFrameCache, TTLDataFrameCache
from trading_system.data.exceptions import (
    DataLayerError,
    DataValidationError,
    DependencyNotAvailableError,
    NewsParsingError,
    NewsSourceError,
    NewsTimeoutError,
    ProviderConfigurationError,
)
from trading_system.data.market_data import MarketDataStream
from trading_system.data.news import BaseNewsSource, JSONNewsSource, NewsArticle, NewsDeduplicator, RSSNewsSource
from trading_system.data.providers import (
    IBKRMarketDataProvider,
    MarketDataProvider,
    PolygonMarketDataProvider,
    YFinanceMarketDataProvider,
)
from trading_system.data.schemas import BarRequest, UniverseSelectionCriteria
from trading_system.data.universe import UniverseSelector

__all__ = [
    "BaseDataStream",
    "DataFrameCache",
    "TTLDataFrameCache",
    "DataLayerError",
    "DependencyNotAvailableError",
    "ProviderConfigurationError",
    "DataValidationError",
    "NewsSourceError",
    "NewsParsingError",
    "NewsTimeoutError",
    "BarRequest",
    "UniverseSelectionCriteria",
    "NewsArticle",
    "BaseNewsSource",
    "RSSNewsSource",
    "JSONNewsSource",
    "NewsDeduplicator",
    "MarketDataProvider",
    "YFinanceMarketDataProvider",
    "PolygonMarketDataProvider",
    "IBKRMarketDataProvider",
    "MarketDataStream",
    "UniverseSelector",
]
