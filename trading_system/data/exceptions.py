"""Exceptions raised by the market data layer."""

from __future__ import annotations


class DataLayerError(Exception):
    """Base exception for market data failures."""


class DependencyNotAvailableError(DataLayerError):
    """Raised when an optional dependency required by a provider is unavailable."""


class ProviderConfigurationError(DataLayerError):
    """Raised when a provider is used before being configured or connected."""


class DataValidationError(DataLayerError):
    """Raised when upstream market data cannot be normalized safely."""


class NewsSourceError(DataLayerError):
    """Raised when a news source cannot be read or configured correctly."""


class NewsParsingError(DataLayerError):
    """Raised when RSS or JSON news payloads cannot be normalized safely."""


class NewsTimeoutError(DataLayerError):
    """Raised when a local news source exceeds its timeout budget."""
