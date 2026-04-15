"""Abstract model interfaces for inference and state updates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from trading_system.core.events import BaseEvent


class BaseModel(ABC):
    """Abstract interface for predictive or decision-support models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the stable name of the model implementation."""

    @abstractmethod
    async def warmup(self) -> None:
        """Prepare model resources before live or simulated trading."""

    @abstractmethod
    def predict(self, features: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a prediction or decision artifact for the provided features."""

    @abstractmethod
    def update(self, event: BaseEvent) -> None:
        """Update model state from a normalized event."""

    @abstractmethod
    def reset(self) -> None:
        """Reset transient state between sessions or backtest runs."""
