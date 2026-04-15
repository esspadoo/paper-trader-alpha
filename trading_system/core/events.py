"""Immutable event models for the trading system core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4


def _freeze_payload(payload: Mapping[str, Any] | None) -> MappingProxyType[str, Any]:
    """Return an immutable view of the event payload."""

    return MappingProxyType(dict(payload or {}))


@dataclass(frozen=True, slots=True)
class BaseEvent(ABC):
    """Base immutable event exchanged by the event-driven runtime."""

    source: str
    payload: Mapping[str, Any] | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: UUID = field(default_factory=uuid4)
    correlation_id: UUID | None = None

    def __post_init__(self) -> None:
        """Normalize event metadata before the event enters the runtime."""

        if not self.source:
            raise ValueError("source must be a non-empty string")

        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")

        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(timezone.utc))
        object.__setattr__(self, "payload", _freeze_payload(self.payload))

    @property
    @abstractmethod
    def topic(self) -> str:
        """Return the logical topic used for routing and observability."""

    @property
    def event_type(self) -> str:
        """Return the normalized event type name."""

        return type(self).__name__


@dataclass(frozen=True, slots=True)
class MarketEvent(BaseEvent):
    """Market data update normalized by the ingress layer."""

    symbol: str = ""
    venue: str = ""
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    volume: float | None = None

    @property
    def topic(self) -> str:
        """Return the market event routing topic."""

        return "market"


@dataclass(frozen=True, slots=True)
class NewsEvent(BaseEvent):
    """News or alternative-data update affecting instruments or sectors."""

    headline: str = ""
    symbols: tuple[str, ...] = ()
    body: str | None = None
    urgency: str = "normal"

    def __post_init__(self) -> None:
        """Normalize symbol lists while preserving the base event contract."""

        BaseEvent.__post_init__(self)
        object.__setattr__(self, "symbols", tuple(self.symbols))

    @property
    def topic(self) -> str:
        """Return the news event routing topic."""

        return "news"


@dataclass(frozen=True, slots=True)
class SignalEvent(BaseEvent):
    """Trading signal emitted by a model or agent."""

    symbol: str = ""
    strategy_id: str = ""
    signal_type: str = ""
    confidence: float = 0.0
    target_price: float | None = None

    @property
    def topic(self) -> str:
        """Return the signal event routing topic."""

        return "signal"


@dataclass(frozen=True, slots=True)
class OrderEvent(BaseEvent):
    """Order lifecycle event emitted by strategy or execution workflows."""

    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    status: str = ""
    limit_price: float | None = None

    @property
    def topic(self) -> str:
        """Return the order event routing topic."""

        return "order"
