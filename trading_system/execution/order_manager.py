"""Order lifecycle management for limit orders and stale-order controls."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter_ns
from typing import Any

from trading_system.execution.broker import BrokerInterface
from trading_system.execution.journal import OrderJournal
from trading_system.infra import LatencyRecorder


class OrderManager:
    """Submit, monitor, and cancel live broker orders."""

    def __init__(
        self,
        broker: BrokerInterface,
        *,
        stale_order_seconds: float = 60.0,
        journal: OrderJournal | None = None,
        logger: logging.Logger | None = None,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        """Initialize the order manager and its stale-order policy."""

        if stale_order_seconds <= 0.0:
            raise ValueError("stale_order_seconds must be greater than 0")

        self._broker = broker
        self._stale_order_seconds = stale_order_seconds
        self._journal = journal
        self._logger = logger or logging.getLogger(__name__)
        self._latency_recorder = latency_recorder
        self._last_seen_orders: dict[str, dict[str, Any]] = {}

    async def submit_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        tif: str = "DAY",
        exchange: str = "SMART",
        currency: str = "USD",
        smart_routing: bool = True,
        outside_rth: bool = False,
        order_ref: str | None = None,
        account: str | None = None,
        client_order_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Submit a limit order and return the initial normalized order snapshot."""

        started_at_ns = perf_counter_ns()
        normalized_client_order_key = self._normalize_client_order_key(client_order_key)
        if normalized_client_order_key is not None:
            existing = await self._existing_submission(normalized_client_order_key)
            if existing is not None:
                self._logger.warning(
                    "suppressing duplicate order submission client_order_key=%s order_id=%s status=%s",
                    normalized_client_order_key,
                    existing.get("order_id", ""),
                    existing.get("status", "UNKNOWN"),
                )
                if self._latency_recorder is not None:
                    self._latency_recorder.record_ns("order_manager_dispatch", perf_counter_ns() - started_at_ns)
                return existing

        order_request = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "limit_price": limit_price,
            "tif": tif,
            "exchange": exchange,
            "currency": currency,
            "smart_routing": smart_routing,
            "outside_rth": outside_rth,
            "order_ref": order_ref,
            "account": account,
        }
        order_id = await self._broker.submit_order(order_request)
        snapshot = await self.get_order_status(order_id)
        if snapshot is None:
            raise RuntimeError(f"broker accepted order {order_id} but did not expose an order snapshot")
        if normalized_client_order_key is not None and self._journal is not None:
            await self._journal.upsert(
                client_order_key=normalized_client_order_key,
                broker_order_id=str(order_id),
                snapshot=dict(snapshot),
                order_ref=order_ref,
            )
        if self._latency_recorder is not None:
            self._latency_recorder.record_ns("order_manager_dispatch", perf_counter_ns() - started_at_ns)
        return snapshot

    async def get_order_status(self, order_id: str) -> Mapping[str, Any] | None:
        """Return the latest snapshot for an order and log fill transitions."""

        snapshot = await self._broker.get_order_status(order_id)
        if snapshot is None:
            return None
        normalized = dict(snapshot)
        self._record_order_transition(normalized)
        await self._sync_journal_from_snapshot(normalized)
        return normalized

    async def sync_open_orders(self) -> Mapping[str, Mapping[str, Any]]:
        """Refresh and return all active open orders."""

        open_orders = await self._broker.get_open_orders()
        normalized: dict[str, Mapping[str, Any]] = {}
        for order_id, snapshot in open_orders.items():
            state = dict(snapshot)
            self._record_order_transition(state)
            await self._sync_journal_from_snapshot(state)
            normalized[str(order_id)] = state
        return normalized

    async def reconcile(self) -> Mapping[str, Mapping[str, Any]]:
        """Refresh open broker orders and align persisted journal state."""

        open_orders = await self.sync_open_orders()
        if self._journal is None:
            return open_orders

        journal_records = await self._journal.list_records()
        for record in journal_records:
            if record.broker_order_id is None:
                continue
            if record.broker_order_id in open_orders:
                continue
            snapshot = await self._broker.get_order_status(record.broker_order_id)
            if snapshot is None:
                continue
            normalized = dict(snapshot)
            self._record_order_transition(normalized)
            await self._sync_journal_from_snapshot(normalized, order_ref=record.order_ref)
        return open_orders

    async def cancel_stale_orders(self) -> list[str]:
        """Cancel open orders that have exceeded the stale-order threshold."""

        open_orders = await self.sync_open_orders()
        cancelled: list[str] = []
        for order_id, snapshot in open_orders.items():
            age_seconds = snapshot.get("age_seconds")
            if not isinstance(age_seconds, (int, float)) or float(age_seconds) < self._stale_order_seconds:
                continue

            self._logger.warning(
                "cancelling stale order order_id=%s age_seconds=%.2f status=%s filled_quantity=%.4f remaining_quantity=%.4f",
                order_id,
                float(age_seconds),
                snapshot.get("status", "UNKNOWN"),
                float(snapshot.get("filled_quantity", 0.0)),
                float(snapshot.get("remaining_quantity", 0.0)),
            )
            await self._broker.cancel_order(order_id)
            cancelled.append(order_id)
            cached = self._last_seen_orders.get(order_id)
            if cached is not None:
                cached["status"] = "Cancelled"
                cached["fill_state"] = "partial" if float(cached.get("filled_quantity", 0.0)) > 0.0 else "unfilled"
                cached["is_active"] = False
                await self._sync_journal_from_snapshot(cached)
        return cancelled

    def _record_order_transition(self, snapshot: dict[str, Any]) -> None:
        """Track order-state changes and emit lifecycle logs."""

        order_id = str(snapshot.get("order_id", "")).strip()
        if not order_id:
            raise ValueError("order snapshots must include a non-empty order_id")

        previous = self._last_seen_orders.get(order_id)
        self._last_seen_orders[order_id] = snapshot

        if previous is None:
            self._logger.info(
                "tracking order order_id=%s symbol=%s side=%s quantity=%.4f limit_price=%.4f",
                order_id,
                snapshot.get("symbol", ""),
                snapshot.get("side", ""),
                float(snapshot.get("quantity", 0.0)),
                float(snapshot.get("limit_price", 0.0)),
            )
            return

        previous_filled = float(previous.get("filled_quantity", 0.0))
        current_filled = float(snapshot.get("filled_quantity", 0.0))
        previous_status = str(previous.get("status", ""))
        current_status = str(snapshot.get("status", ""))
        previous_fill_state = str(previous.get("fill_state", ""))
        current_fill_state = str(snapshot.get("fill_state", ""))

        if current_filled > previous_filled:
            self._logger.info(
                "fill update order_id=%s filled_quantity=%.4f remaining_quantity=%.4f fill_state=%s",
                order_id,
                current_filled,
                float(snapshot.get("remaining_quantity", 0.0)),
                current_fill_state,
            )

        if previous_fill_state != current_fill_state and current_fill_state == "partial":
            self._logger.warning(
                "partial fill detected order_id=%s filled_quantity=%.4f remaining_quantity=%.4f",
                order_id,
                current_filled,
                float(snapshot.get("remaining_quantity", 0.0)),
            )

        if previous_status != current_status:
            self._logger.info("order status changed order_id=%s status=%s", order_id, current_status)

    async def _existing_submission(self, client_order_key: str) -> Mapping[str, Any] | None:
        """Return an existing broker or journal snapshot for a client order key."""

        if self._journal is None:
            return None

        record = await self._journal.get(client_order_key)
        if record is None:
            return None

        if record.broker_order_id is not None:
            broker_snapshot = await self._broker.get_order_status(record.broker_order_id)
            if broker_snapshot is not None:
                normalized = dict(broker_snapshot)
                self._record_order_transition(normalized)
                await self._sync_journal_from_snapshot(normalized, order_ref=record.order_ref)
                return normalized

        return dict(record.snapshot)

    async def _sync_journal_from_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        order_ref: str | None = None,
    ) -> None:
        """Persist a broker snapshot when it carries a client order key."""

        if self._journal is None:
            return
        resolved_order_ref = str(order_ref or snapshot.get("order_ref", "")).strip()
        client_order_key = self._client_order_key_from_order_ref(resolved_order_ref)
        if client_order_key is None:
            return
        await self._journal.upsert(
            client_order_key=client_order_key,
            broker_order_id=str(snapshot.get("order_id", "")).strip() or None,
            snapshot=dict(snapshot),
            order_ref=resolved_order_ref,
        )

    def _normalize_client_order_key(self, client_order_key: str | None) -> str | None:
        """Return a normalized client order key when provided."""

        if client_order_key is None:
            return None
        normalized = str(client_order_key).strip()
        if not normalized:
            raise ValueError("client_order_key must be a non-empty string when provided")
        return normalized

    def _client_order_key_from_order_ref(self, order_ref: str | None) -> str | None:
        """Extract the client order key from a normalized order reference."""

        if order_ref is None:
            return None
        normalized = str(order_ref).strip()
        if not normalized.startswith("client-order:"):
            return None
        client_order_key = normalized.split(":", 1)[1].strip()
        return client_order_key or None
