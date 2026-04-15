"""Execution-layer exceptions for brokers and order management."""

from __future__ import annotations


class ExecutionError(Exception):
    """Base exception for execution-layer failures."""


class ExecutionDependencyError(ExecutionError):
    """Raised when an optional execution dependency is unavailable."""


class BrokerConnectionError(ExecutionError):
    """Raised when the broker session cannot be opened or maintained."""


class OrderSubmissionError(ExecutionError):
    """Raised when an order cannot be submitted safely."""


class OrderCancellationError(ExecutionError):
    """Raised when an order cancellation fails."""


class OrderNotFoundError(ExecutionError):
    """Raised when a requested order id cannot be found."""
