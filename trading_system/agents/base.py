"""Abstract agent interfaces for strategy orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from trading_system.core.events import BaseEvent


class BaseAgent(ABC):
    """Abstract interface for an event-driven trading agent."""

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Return the stable identifier for the agent instance."""

    @property
    @abstractmethod
    def subscribed_topics(self) -> Sequence[str]:
        """Return the event topics the agent expects to consume."""

    @abstractmethod
    async def start(self) -> None:
        """Prepare the agent for intraday processing."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources owned by the agent."""

    @abstractmethod
    async def on_event(self, event: BaseEvent) -> None:
        """Consume a normalized event emitted by the system."""

    @abstractmethod
    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of the agent state."""
