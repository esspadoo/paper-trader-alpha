"""Abstract data stream interfaces for upstream market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from trading_system.core.events import BaseEvent


class BaseDataStream(ABC):
    """Abstract interface for a normalized market data stream."""

    @property
    @abstractmethod
    def stream_name(self) -> str:
        """Return the unique name of the upstream data stream."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the upstream connection used to ingest data."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the upstream connection and release related resources."""

    @abstractmethod
    async def subscribe(self, symbols: Sequence[str]) -> None:
        """Subscribe the stream to the provided instrument symbols."""

    @abstractmethod
    async def unsubscribe(self, symbols: Sequence[str]) -> None:
        """Remove subscriptions for the provided instrument symbols."""

    @abstractmethod
    def events(self) -> AsyncIterator[BaseEvent]:
        """Yield normalized events produced by the upstream source."""
