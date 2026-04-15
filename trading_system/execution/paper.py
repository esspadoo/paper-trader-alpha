"""In-memory paper broker for end-to-end system integration and demos."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic
from typing import Any

from trading_system.execution.broker import BrokerInterface
from trading_system.execution.exceptions import BrokerConnectionError, OrderNotFoundError, OrderSubmissionError


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """Configuration for the in-memory paper broker."""

    initial_capital: float = 100_000.0
    venue: str = "paper"
    account: str = "DU-PAPER"
    fill_latency_seconds: float = 0.0
    default_currency: str = "USD"

    def __post_init__(self) -> None:
        """Validate paper-broker configuration."""

        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be greater than 0")
        if not self.venue.strip():
            raise ValueError("venue must be a non-empty string")
        if not self.account.strip():
            raise ValueError("account must be a non-empty string")
        if self.fill_latency_seconds < 0.0:
            raise ValueError("fill_latency_seconds must be greater than or equal to 0")
        if not self.default_currency.strip():
            raise ValueError("default_currency must be a non-empty string")


class PaperBroker(BrokerInterface):
    """Simple asynchronous paper broker with immediate limit-order fills."""

    def __init__(
        self,
        *,
        config: PaperBrokerConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the paper broker state."""

        self._config = config or PaperBrokerConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._connected = False
        self._cash = float(self._config.initial_capital)
        self._positions: dict[str, dict[str, float | str]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._marks: dict[str, float] = {}
        self._submitted_at: dict[str, float] = {}
        self._next_order_id = 1

    @property
    def venue(self) -> str:
        """Return the paper venue identifier."""

        return self._config.venue

    @property
    def paper_trading(self) -> bool:
        """Return whether the broker is in paper mode."""

        return True

    async def connect(self) -> None:
        """Activate the paper broker session."""

        async with self._lock:
            self._connected = True
            self._logger.info(
                "connected to paper broker account=%s initial_capital=%.2f",
                self._config.account,
                self._config.initial_capital,
            )

    async def disconnect(self) -> None:
        """Deactivate the paper broker session."""

        async with self._lock:
            self._connected = False
            self._logger.info("disconnected from paper broker")

    async def submit_order(self, order: dict[str, Any] | Any) -> str:
        """Accept and immediately fill a normalized limit order."""

        request = self._normalize_order_request(order)
        async with self._lock:
            self._ensure_connected()
            order_id = f"PAPER-{self._next_order_id:06d}"
            self._next_order_id += 1

            if self._config.fill_latency_seconds > 0.0:
                await asyncio.sleep(self._config.fill_latency_seconds)

            fill_price = request["limit_price"]
            self._apply_fill(
                symbol=request["symbol"],
                side=request["side"],
                quantity=request["quantity"],
                price=fill_price,
            )
            submitted_at = monotonic()
            self._submitted_at[order_id] = submitted_at
            snapshot = {
                "order_id": order_id,
                "symbol": request["symbol"],
                "side": request["side"],
                "quantity": request["quantity"],
                "limit_price": fill_price,
                "status": "Filled",
                "filled_quantity": request["quantity"],
                "remaining_quantity": 0.0,
                "average_fill_price": fill_price,
                "exchange": request["exchange"],
                "currency": request["currency"],
                "fill_state": "filled",
                "is_active": False,
                "age_seconds": 0.0,
                "order_ref": request["order_ref"],
                "outside_rth": request["outside_rth"],
                "paper_trading": True,
            }
            self._orders[order_id] = snapshot
            self._logger.info(
                "paper fill order_id=%s symbol=%s side=%s quantity=%.4f limit_price=%.4f",
                order_id,
                request["symbol"],
                request["side"],
                request["quantity"],
                fill_price,
            )
            return order_id

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an order when still active."""

        async with self._lock:
            self._ensure_connected()
            snapshot = self._orders.get(order_id)
            if snapshot is None:
                raise OrderNotFoundError(f"paper order {order_id} was not found")
            if not bool(snapshot.get("is_active", False)):
                self._logger.info("skipping cancel for inactive paper order_id=%s", order_id)
                return
            snapshot["status"] = "Cancelled"
            snapshot["fill_state"] = "unfilled"
            snapshot["is_active"] = False

    async def get_order_status(self, order_id: str) -> dict[str, Any] | None:
        """Return the latest normalized order snapshot."""

        async with self._lock:
            self._ensure_connected()
            return self._snapshot_for_order_locked(order_id)

    async def get_open_orders(self) -> dict[str, dict[str, Any]]:
        """Return the currently active paper orders."""

        async with self._lock:
            self._ensure_connected()
            return {
                order_id: snapshot
                for order_id in self._orders
                if (snapshot := self._snapshot_for_order_locked(order_id)) is not None
                and bool(snapshot.get("is_active", False))
            }

    def _snapshot_for_order_locked(self, order_id: str) -> dict[str, Any] | None:
        """Return a normalized snapshot for an order while the lock is held."""

        snapshot = self._orders.get(order_id)
        if snapshot is None:
            return None
        normalized = dict(snapshot)
        normalized["age_seconds"] = max(monotonic() - self._submitted_at.get(order_id, monotonic()), 0.0)
        return normalized

    async def get_positions(self) -> dict[str, dict[str, Any]]:
        """Return current normalized paper positions."""

        async with self._lock:
            self._ensure_connected()
            return {
                symbol: dict(position)
                for symbol, position in self._positions.items()
                if float(position["quantity"]) != 0.0
            }

    async def get_account_snapshot(self) -> dict[str, Any]:
        """Return the current paper-account snapshot."""

        async with self._lock:
            self._ensure_connected()
            gross_exposure = 0.0
            market_value = 0.0
            for position in self._positions.values():
                quantity = float(position["quantity"])
                price = float(position["market_price"])
                gross_exposure += abs(quantity * price)
                market_value += quantity * price

            capital = self._cash + market_value
            return {
                "account": self._config.account,
                "venue": self.venue,
                "paper_trading": True,
                "capital": float(capital),
                "cash": float(self._cash),
                "gross_exposure": float(gross_exposure),
                "currency": self._config.default_currency,
            }

    async def update_market_price(self, symbol: str, price: float) -> None:
        """Update the marked market price for a symbol."""

        async with self._lock:
            self._ensure_connected()
            symbol_key = str(symbol).strip().upper()
            price_value = self._positive_float(price, field_name="price")
            self._marks[symbol_key] = price_value
            position = self._positions.get(symbol_key)
            if position is None:
                return

            average_cost = float(position["average_cost"])
            quantity = float(position["quantity"])
            position["market_price"] = price_value
            position["market_value"] = quantity * price_value
            position["unrealized_pnl"] = (price_value - average_cost) * quantity

    def _ensure_connected(self) -> None:
        """Ensure the broker is connected before servicing requests."""

        if not self._connected:
            raise BrokerConnectionError("paper broker is not connected")

    def _normalize_order_request(self, order: dict[str, Any] | Any) -> dict[str, Any]:
        """Validate and normalize a paper-order request."""

        if not isinstance(order, dict):
            raise OrderSubmissionError("order must be a mapping")

        symbol = str(order.get("symbol", "")).strip().upper()
        side = str(order.get("side") or order.get("action") or "").strip().upper()
        exchange = str(order.get("exchange", "SMART")).strip().upper()
        currency = str(order.get("currency", self._config.default_currency)).strip().upper()
        tif = str(order.get("tif", "DAY")).strip().upper()
        order_ref = str(order.get("order_ref", "")).strip() or None
        outside_rth = bool(order.get("outside_rth", False))
        if not symbol:
            raise OrderSubmissionError("symbol must be a non-empty string")
        if side not in {"BUY", "SELL"}:
            raise OrderSubmissionError("side must be BUY or SELL")
        if not exchange:
            raise OrderSubmissionError("exchange must be a non-empty string")
        if not currency:
            raise OrderSubmissionError("currency must be a non-empty string")
        if not tif:
            raise OrderSubmissionError("tif must be a non-empty string")

        quantity = self._positive_float(order.get("quantity"), field_name="quantity")
        limit_price = self._positive_float(order.get("limit_price"), field_name="limit_price")
        return {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "limit_price": limit_price,
            "exchange": exchange,
            "currency": currency,
            "tif": tif,
            "order_ref": order_ref,
            "outside_rth": outside_rth,
        }

    def _apply_fill(self, *, symbol: str, side: str, quantity: float, price: float) -> None:
        """Apply an immediate paper fill to cash and positions."""

        signed_quantity = quantity if side == "BUY" else -quantity
        notional = quantity * price
        if side == "BUY":
            self._cash -= notional
        else:
            self._cash += notional

        current = self._positions.get(symbol)
        if current is None:
            self._positions[symbol] = {
                "symbol": symbol,
                "quantity": signed_quantity,
                "average_cost": price,
                "market_price": self._marks.get(symbol, price),
                "market_value": signed_quantity * self._marks.get(symbol, price),
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "exchange": "SMART",
                "currency": self._config.default_currency,
            }
            return

        current_quantity = float(current["quantity"])
        current_average_cost = float(current["average_cost"])
        updated_quantity = current_quantity + signed_quantity
        if current_quantity == 0.0 or (current_quantity > 0.0) == (signed_quantity > 0.0):
            total_quantity = abs(current_quantity) + quantity
            weighted_average = (
                (abs(current_quantity) * current_average_cost) + (quantity * price)
            ) / total_quantity
            current["quantity"] = updated_quantity
            current["average_cost"] = weighted_average
        else:
            closed_quantity = min(abs(current_quantity), quantity)
            realized = (price - current_average_cost) * closed_quantity * (1.0 if current_quantity > 0.0 else -1.0)
            current["realized_pnl"] = float(current.get("realized_pnl", 0.0)) + realized
            if updated_quantity == 0.0:
                current["quantity"] = 0.0
            elif (current_quantity > 0.0 and updated_quantity > 0.0) or (current_quantity < 0.0 and updated_quantity < 0.0):
                current["quantity"] = updated_quantity
            else:
                current["quantity"] = updated_quantity
                current["average_cost"] = price

        market_price = self._marks.get(symbol, price)
        quantity_now = float(current["quantity"])
        current["market_price"] = market_price
        current["market_value"] = quantity_now * market_price
        current["unrealized_pnl"] = (market_price - float(current["average_cost"])) * quantity_now
        if quantity_now == 0.0:
            self._positions.pop(symbol, None)

    def _positive_float(self, value: Any, *, field_name: str) -> float:
        """Return a validated positive float."""

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise OrderSubmissionError(f"{field_name} must be numeric") from exc
        if numeric <= 0.0:
            raise OrderSubmissionError(f"{field_name} must be greater than 0")
        return numeric
