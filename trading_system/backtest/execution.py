"""Event-driven execution simulation for intraday backtests."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from trading_system.backtest.portfolio import PortfolioLedger
from trading_system.backtest.types import BacktestConfig, FillRecord, SimulatedOrder
from trading_system.core.engine import Engine
from trading_system.core.events import BaseEvent, MarketEvent, OrderEvent


class BacktestExecutionSimulator:
    """Simulate realistic order execution with latency, slippage, and commissions."""

    def __init__(
        self,
        *,
        config: BacktestConfig,
        portfolio: PortfolioLedger,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the execution simulator and pending-order book."""

        self._config = config
        self._portfolio = portfolio
        self._logger = logger or logging.getLogger(__name__)
        self._engine: Engine | None = None
        self._pending_orders: dict[str, SimulatedOrder] = {}
        self._next_order_id = 1

    def bind_engine(self, engine: Engine) -> None:
        """Bind the simulator to an engine for fill event publication."""

        self._engine = engine

    @property
    def portfolio(self) -> PortfolioLedger:
        """Return the underlying portfolio ledger."""

        return self._portfolio

    def has_active_order(self, symbol: str) -> bool:
        """Return whether a symbol currently has a pending order."""

        symbol_key = symbol.strip().upper()
        return any(order.symbol == symbol_key for order in self._pending_orders.values())

    async def on_event(self, event: BaseEvent) -> None:
        """Consume market and order events emitted by the backtest runtime."""

        if isinstance(event, OrderEvent):
            await self.on_order(event)
        elif isinstance(event, MarketEvent):
            await self.on_market(event)

    async def on_order(self, event: OrderEvent) -> None:
        """Track a newly submitted order for future market-event fills."""

        if event.status.strip().upper() != "NEW":
            return

        if event.quantity <= 0.0:
            return

        order_id = event.order_id.strip() or f"BT-{self._next_order_id:06d}"
        if order_id in self._pending_orders:
            raise ValueError(f"duplicate pending order id {order_id}")
        self._next_order_id += 1

        limit_price = event.limit_price if event.limit_price is not None else self._payload_float(event.payload, "limit_price")
        if limit_price is None or limit_price <= 0.0:
            raise ValueError("OrderEvent requires a positive limit price for backtest execution")

        side = event.side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("OrderEvent side must be BUY or SELL for backtest execution")

        eligible_after = event.occurred_at + timedelta(milliseconds=self._config.latency_ms)
        payload = dict(event.payload or {})
        stop_loss = self._payload_float(payload, "stop_loss")

        pending = SimulatedOrder(
            order_id=order_id,
            symbol=event.symbol.strip().upper(),
            side=side,
            quantity=float(event.quantity),
            remaining_quantity=float(event.quantity),
            limit_price=float(limit_price),
            submitted_at=event.occurred_at,
            eligible_after=eligible_after,
            stop_loss=stop_loss,
            strategy_id=str(payload.get("strategy_id", "")).strip(),
            reason=str(payload.get("reason", payload.get("reasoning", ""))).strip(),
            source=event.source,
            correlation_id=event.correlation_id,
        )
        self._pending_orders[order_id] = pending
        self._logger.info(
            "accepted simulated order order_id=%s symbol=%s side=%s quantity=%.4f limit_price=%.4f eligible_after=%s",
            pending.order_id,
            pending.symbol,
            pending.side,
            pending.quantity,
            pending.limit_price,
            pending.eligible_after.isoformat(),
        )

    async def on_market(self, event: MarketEvent) -> None:
        """Process stop exits, pending order fills, and mark-to-market equity."""

        symbol = event.symbol.strip().upper()
        bar = self._normalize_bar(event.payload)
        await self._process_stop_loss(symbol=symbol, timestamp=event.occurred_at, bar=bar)
        await self._process_pending_orders(symbol=symbol, timestamp=event.occurred_at, bar=bar)
        self._portfolio.mark_to_market(event.occurred_at, {symbol: float(bar["close"])})

    async def flatten_positions(self, *, timestamp: datetime) -> None:
        """Force-close all open positions at the current marked prices."""

        for symbol, position in list(self._portfolio.positions().items()):
            close_price = self._portfolio.current_price(symbol)
            side = "SELL" if position.quantity > 0.0 else "BUY"
            quantity = abs(position.quantity)
            commission = self._commission(quantity)
            fill = FillRecord(
                timestamp=timestamp,
                symbol=symbol,
                order_id=f"FORCE-{symbol}-{timestamp.strftime('%Y%m%d%H%M%S')}",
                side=side,
                quantity=quantity,
                price=close_price,
                commission=commission,
                reason="forced_end_of_backtest_exit",
            )
            self._portfolio.record_fill(fill)
            await self._publish_fill_event(
                fill=fill,
                status="FILLED",
                remaining_quantity=0.0,
                stop_loss=None,
            )

    async def _process_stop_loss(self, *, symbol: str, timestamp: datetime, bar: Mapping[str, float]) -> None:
        """Exit positions when the ATR stop is breached."""

        position = self._portfolio.position_for(symbol)
        if position is None or position.stop_loss is None:
            return

        slip = self._config.slippage_bps / 10_000.0
        stop_loss = float(position.stop_loss)
        if position.quantity > 0.0:
            triggered = float(bar["low"]) <= stop_loss
            if not triggered:
                return
            candidate = min(stop_loss, float(bar["open"]) * (1.0 - slip))
            fill_price = max(float(bar["low"]), candidate)
            side = "SELL"
        else:
            triggered = float(bar["high"]) >= stop_loss
            if not triggered:
                return
            candidate = max(stop_loss, float(bar["open"]) * (1.0 + slip))
            fill_price = min(float(bar["high"]), candidate)
            side = "BUY"

        fill = FillRecord(
            timestamp=timestamp,
            symbol=symbol,
            order_id=f"STOP-{symbol}-{timestamp.strftime('%Y%m%d%H%M%S')}",
            side=side,
            quantity=abs(position.quantity),
            price=fill_price,
            commission=self._commission(abs(position.quantity)),
            reason="stop_loss_triggered",
        )
        self._portfolio.record_fill(fill)
        await self._publish_fill_event(fill=fill, status="STOP_FILLED", remaining_quantity=0.0, stop_loss=stop_loss)
        self._cancel_pending_orders_for_symbol(symbol)

    async def _process_pending_orders(self, *, symbol: str, timestamp: datetime, bar: Mapping[str, float]) -> None:
        """Fill eligible limit orders against the current bar."""

        volume_capacity = max(float(bar["volume"]) * self._config.max_participation_rate, 0.0)
        used_volume = 0.0
        for order in list(self._pending_orders.values()):
            if order.symbol != symbol or order.eligible_after > timestamp:
                continue

            if not self._is_limit_reached(order=order, bar=bar):
                continue

            remaining_volume = max(volume_capacity - used_volume, 0.0)
            fill_quantity = min(order.remaining_quantity, remaining_volume)
            if fill_quantity <= 0.0:
                continue

            fill_price = self._limit_fill_price(order=order, bar=bar)
            commission = self._commission(fill_quantity)
            fill = FillRecord(
                timestamp=timestamp,
                symbol=symbol,
                order_id=order.order_id,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                commission=commission,
                reason="limit_fill",
                correlation_id=order.correlation_id,
            )
            self._portfolio.record_fill(fill, stop_loss=order.stop_loss)
            order.remaining_quantity -= fill_quantity
            used_volume += fill_quantity

            status = "FILLED" if order.remaining_quantity <= 1e-12 else "PARTIALLY_FILLED"
            await self._publish_fill_event(
                fill=fill,
                status=status,
                remaining_quantity=max(order.remaining_quantity, 0.0),
                stop_loss=order.stop_loss,
            )
            if order.remaining_quantity <= 1e-12:
                self._pending_orders.pop(order.order_id, None)

    def _normalize_bar(self, payload: Mapping[str, Any] | None) -> dict[str, float]:
        """Return a normalized OHLCV bar mapping from the event payload."""

        bar = dict((payload or {}).get("bar") or {})
        required = ("open", "high", "low", "close", "volume")
        missing = [field for field in required if field not in bar]
        if missing:
            raise ValueError(f"market event payload is missing required bar fields: {missing}")

        normalized: dict[str, float] = {}
        for field in required:
            numeric = float(bar[field])
            if not math.isfinite(numeric):
                raise ValueError(f"bar field {field} must be finite")
            if field != "volume" and numeric <= 0.0:
                raise ValueError(f"bar field {field} must be greater than 0")
            if field == "volume" and numeric < 0.0:
                raise ValueError("bar field volume must be greater than or equal to 0")
            normalized[field] = numeric

        if normalized["low"] > normalized["high"]:
            raise ValueError("bar low cannot exceed bar high")
        return normalized

    def _is_limit_reached(self, *, order: SimulatedOrder, bar: Mapping[str, float]) -> bool:
        """Return whether the current bar is deep enough to fill the limit order."""

        slip = self._config.slippage_bps / 10_000.0
        if order.side == "BUY":
            return float(bar["low"]) <= order.limit_price * (1.0 - slip)
        return float(bar["high"]) >= order.limit_price * (1.0 + slip)

    def _limit_fill_price(self, *, order: SimulatedOrder, bar: Mapping[str, float]) -> float:
        """Return a slippage-aware fill price that still respects the limit."""

        slip = self._config.slippage_bps / 10_000.0
        if order.side == "BUY":
            candidate = float(bar["open"]) * (1.0 + slip)
            return min(order.limit_price, max(float(bar["low"]), candidate))
        candidate = float(bar["open"]) * (1.0 - slip)
        return max(order.limit_price, min(float(bar["high"]), candidate))

    def _commission(self, quantity: float) -> float:
        """Return the commission for a single fill."""

        per_share = quantity * self._config.commission_per_share
        return float(max(per_share, self._config.minimum_commission))

    async def _publish_fill_event(
        self,
        *,
        fill: FillRecord,
        status: str,
        remaining_quantity: float,
        stop_loss: float | None,
    ) -> None:
        """Publish an execution update through the bound engine when available."""

        if self._engine is None:
            return

        payload: dict[str, Any] = {
            "fill_price": fill.price,
            "commission": fill.commission,
            "remaining_quantity": remaining_quantity,
            "reason": fill.reason,
        }
        if stop_loss is not None:
            payload["stop_loss"] = stop_loss

        await self._engine.publish(
            OrderEvent(
                source="backtest-execution",
                order_id=fill.order_id,
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                status=status,
                limit_price=fill.price,
                payload=payload,
                occurred_at=fill.timestamp,
                correlation_id=fill.correlation_id,
            )
        )

    def _cancel_pending_orders_for_symbol(self, symbol: str) -> None:
        """Remove pending orders for a symbol after a stop exit."""

        symbol_key = symbol.strip().upper()
        stale_ids = [order_id for order_id, order in self._pending_orders.items() if order.symbol == symbol_key]
        for order_id in stale_ids:
            self._pending_orders.pop(order_id, None)

    def _payload_float(self, payload: Mapping[str, Any], key: str) -> float | None:
        """Return a float from an event payload key when present."""

        if key not in payload:
            return None
        if payload[key] is None:
            return None
        value = float(payload[key])
        if not math.isfinite(value):
            raise ValueError(f"payload field {key} must be finite")
        return value
