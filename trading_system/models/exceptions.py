"""Exceptions raised by the model layer."""

from __future__ import annotations


class ModelLayerError(Exception):
    """Base exception for model-layer failures."""


class ModelDependencyError(ModelLayerError):
    """Raised when an optional ML dependency is unavailable."""


class ModelNotFittedError(ModelLayerError):
    """Raised when inference is attempted before fitting the model."""


class FeatureEngineeringError(ModelLayerError):
    """Raised when an OHLCV frame cannot be normalized or transformed safely."""


class TrainingDataError(ModelLayerError):
    """Raised when the training set is empty, invalid, or too small."""


class LLMBackendError(ModelLayerError):
    """Raised when the local LLM backend cannot complete a request."""


class LLMTimeoutError(LLMBackendError):
    """Raised when a local LLM request exceeds its timeout budget."""


class StrictJSONParseError(ModelLayerError):
    """Raised when the LLM response is not strict, valid JSON."""
