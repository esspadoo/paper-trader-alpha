"""Abstract broker interfaces for order routing and account state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BrokerInterface(ABC):
    """Abstract execution gateway for brokers and exchanges."""

    @property
    @abstractmethod
    def venue(self) -> str:
        """Return the broker or execution venue identifier."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the broker session required for order management."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the broker session and release related resources."""

    @abstractmethod
    async def submit_order(self, order: Mapping[str, Any]) -> str:
        """Submit a normalized order request and return the broker order id."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        """Cancel a live order by its broker-assigned identifier."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Mapping[str, Any] | None:
        """Return the latest normalized state for a specific order when available."""

    @abstractmethod
    async def get_open_orders(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the currently active orders keyed by broker order id."""

    @abstractmethod
    async def get_positions(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the current positions keyed by instrument symbol."""

    @abstractmethod
    async def get_account_snapshot(self) -> Mapping[str, Any]:
        """Return a normalized snapshot of account-level state."""
