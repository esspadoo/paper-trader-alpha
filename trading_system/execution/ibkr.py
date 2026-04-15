"""Interactive Brokers execution adapter backed by `ib_insync`."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any, TypeVar

from trading_system.execution.broker import BrokerInterface
from trading_system.execution.dependencies import require_ib_insync
from trading_system.execution.exceptions import (
    BrokerConnectionError,
    OrderCancellationError,
    OrderNotFoundError,
    OrderSubmissionError,
)

T = TypeVar("T")

RetryableOperation = Callable[[], T | Awaitable[T]]
IBClientFactory = Callable[[], Any]
ContractFactory = Callable[..., Any]
LimitOrderFactory = Callable[..., Any]
SleepFunction = Callable[[float], Awaitable[None]]

TERMINAL_ORDER_STATUSES = {"APICANCELLED", "API_CANCELLED", "CANCELLED", "FILLED", "INACTIVE"}


@dataclass(frozen=True, slots=True)
class IBKRBrokerConfig:
    """Connection and retry configuration for the IBKR broker adapter."""

    host: str = "127.0.0.1"
    port: int | None = None
    client_id: int = 1
    account: str | None = None
    paper_trading: bool = True
    read_only: bool = False
    connection_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    default_exchange: str = "SMART"
    default_currency: str = "USD"

    def __post_init__(self) -> None:
        """Validate broker configuration values."""

        if not self.host.strip():
            raise ValueError("host must be a non-empty string")
        if self.port is not None and self.port <= 0:
            raise ValueError("port must be greater than 0 when provided")
        if self.client_id < 0:
            raise ValueError("client_id must be greater than or equal to 0")
        if self.connection_timeout_seconds <= 0.0:
            raise ValueError("connection_timeout_seconds must be greater than 0")
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be greater than 0")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds must be greater than or equal to 0")
        if not self.default_exchange.strip():
            raise ValueError("default_exchange must be a non-empty string")
        if not self.default_currency.strip():
            raise ValueError("default_currency must be a non-empty string")

    @property
    def resolved_port(self) -> int:
        """Return the explicit or paper/live default TWS port."""

        if self.port is not None:
            return self.port
        return 7497 if self.paper_trading else 7496


class IBKRBroker(BrokerInterface):
    """Production-oriented Interactive Brokers broker implementation."""

    def __init__(
        self,
        *,
        config: IBKRBrokerConfig | None = None,
        logger: logging.Logger | None = None,
        ib_client_factory: IBClientFactory | None = None,
        contract_factory: ContractFactory | None = None,
        limit_order_factory: LimitOrderFactory | None = None,
        sleep: SleepFunction | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the broker adapter with injected runtime factories when needed."""

        self._config = config or IBKRBrokerConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._ib_client_factory = ib_client_factory
        self._contract_factory = contract_factory
        self._limit_order_factory = limit_order_factory
        self._sleep = sleep or asyncio.sleep
        self._monotonic = monotonic_fn or monotonic
        self._lock = asyncio.Lock()
        self._client: Any | None = None
        self._tracked_trades: dict[str, Any] = {}
        self._submitted_at: dict[str, float] = {}

    @property
    def venue(self) -> str:
        """Return the broker venue identifier."""

        return "ibkr"

    @property
    def paper_trading(self) -> bool:
        """Return whether the broker is configured for paper trading."""

        return self._config.paper_trading

    @property
    def is_connected(self) -> bool:
        """Return whether the underlying IB client is connected."""

        return bool(self._client is not None and self._client.isConnected())

    async def connect(self) -> None:
        """Connect to TWS or IB Gateway with retry and logging."""

        async with self._lock:
            client = self._ensure_client()
            if client.isConnected():
                return

            try:
                await self._run_with_retries(
                    "connect to IBKR",
                    lambda: client.connectAsync(
                        self._config.host,
                        self._config.resolved_port,
                        clientId=self._config.client_id,
                        timeout=self._config.connection_timeout_seconds,
                        readonly=self._config.read_only,
                        account=self._config.account or "",
                    ),
                    timeout_seconds=self._config.connection_timeout_seconds,
                )
            except Exception as exc:
                raise BrokerConnectionError(
                    f"unable to connect to IBKR at {self._config.host}:{self._config.resolved_port}"
                ) from exc

            if not client.isConnected():
                raise BrokerConnectionError("IBKR connection attempt completed without an active session")

            self._logger.info(
                "connected to IBKR host=%s port=%s client_id=%s paper_trading=%s",
                self._config.host,
                self._config.resolved_port,
                self._config.client_id,
                self._config.paper_trading,
            )

    async def disconnect(self) -> None:
        """Disconnect from the broker session."""

        async with self._lock:
            if self._client is None or not self._client.isConnected():
                return

            self._client.disconnect()
            self._logger.info("disconnected from IBKR")

    async def submit_order(self, order: Mapping[str, Any]) -> str:
        """Submit a normalized limit order through Interactive Brokers."""

        request = self._normalize_order_request(order)

        async with self._lock:
            client = self._require_connected_client()
            contract = self._build_contract(request)
            limit_order = self._build_limit_order(request)

            try:
                trade = await self._run_with_retries(
                    f"submit order for {request['symbol']}",
                    lambda: client.placeOrder(contract, limit_order),
                )
            except Exception as exc:
                raise OrderSubmissionError(
                    f"unable to submit {request['side']} limit order for {request['symbol']}"
                ) from exc

            order_id = self._extract_order_id(trade)
            self._tracked_trades[order_id] = trade
            self._submitted_at[order_id] = self._monotonic()
            self._logger.info(
                "submitted limit order order_id=%s symbol=%s side=%s quantity=%.4f limit_price=%.4f route=%s paper_trading=%s",
                order_id,
                request["symbol"],
                request["side"],
                request["quantity"],
                request["limit_price"],
                request["exchange"],
                self._config.paper_trading,
            )
            return order_id

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by broker order id."""

        async with self._lock:
            client = self._require_connected_client()
            trade = self._find_trade_locked(order_id)
            if trade is None:
                raise OrderNotFoundError(f"order {order_id} was not found in the broker trade cache")

            snapshot = self._normalize_trade(trade)
            if not snapshot["is_active"]:
                self._logger.info("skipping cancel for inactive order_id=%s status=%s", order_id, snapshot["status"])
                return

            try:
                await self._run_with_retries(
                    f"cancel order {order_id}",
                    lambda: client.cancelOrder(trade.order),
                )
            except Exception as exc:
                raise OrderCancellationError(f"unable to cancel order {order_id}") from exc

            self._logger.info("cancelled order order_id=%s", order_id)

    async def get_order_status(self, order_id: str) -> Mapping[str, Any] | None:
        """Return the latest normalized order snapshot when available."""

        async with self._lock:
            self._require_connected_client()
            trade = self._find_trade_locked(order_id)
            if trade is None:
                return None
            return self._normalize_trade(trade)

    async def get_open_orders(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the active orders keyed by broker order id."""

        async with self._lock:
            self._require_connected_client()
            self._refresh_trade_cache_locked()
            open_orders: dict[str, Mapping[str, Any]] = {}
            for order_id, trade in self._tracked_trades.items():
                snapshot = self._normalize_trade(trade)
                if snapshot["is_active"]:
                    open_orders[order_id] = snapshot
            return open_orders

    async def get_positions(self) -> Mapping[str, Mapping[str, Any]]:
        """Return normalized current positions."""

        async with self._lock:
            client = self._require_connected_client()
            positions = await self._run_with_retries("request positions", client.positions)

            normalized: dict[str, Mapping[str, Any]] = {}
            for position in positions:
                contract = getattr(position, "contract", None)
                symbol = str(getattr(contract, "symbol", "")).strip().upper()
                if not symbol:
                    continue

                normalized[symbol] = {
                    "symbol": symbol,
                    "exchange": str(getattr(contract, "exchange", "")).strip().upper(),
                    "currency": str(getattr(contract, "currency", "")).strip().upper(),
                    "quantity": self._safe_float(getattr(position, "position", 0.0)),
                    "average_cost": self._safe_float(getattr(position, "avgCost", 0.0)),
                    "market_price": self._safe_float(getattr(position, "marketPrice", 0.0)),
                    "market_value": self._safe_float(getattr(position, "marketValue", 0.0)),
                    "unrealized_pnl": self._safe_float(getattr(position, "unrealizedPNL", 0.0)),
                    "realized_pnl": self._safe_float(getattr(position, "realizedPNL", 0.0)),
                }
            return normalized

    async def get_account_snapshot(self) -> Mapping[str, Any]:
        """Return a normalized account summary snapshot."""

        async with self._lock:
            client = self._require_connected_client()
            summary_items = await self._run_with_retries(
                "request account summary",
                lambda: client.accountSummaryAsync(self._config.account or ""),
                timeout_seconds=self._config.request_timeout_seconds,
            )

            snapshot: dict[str, Any] = {
                "account": self._config.account or "",
                "venue": self.venue,
                "paper_trading": self._config.paper_trading,
            }
            for item in summary_items:
                tag = str(getattr(item, "tag", "")).strip()
                if not tag:
                    continue

                raw_value = getattr(item, "value", "")
                normalized_value = self._maybe_float(raw_value)
                snapshot[tag] = raw_value if normalized_value is None else normalized_value

                account = str(getattr(item, "account", "")).strip()
                if account and not snapshot["account"]:
                    snapshot["account"] = account

            if "NetLiquidation" in snapshot:
                snapshot["capital"] = self._safe_float(snapshot["NetLiquidation"])
            if "GrossPositionValue" in snapshot:
                snapshot["gross_exposure"] = abs(self._safe_float(snapshot["GrossPositionValue"]))
            if "AvailableFunds" in snapshot:
                snapshot["available_funds"] = self._safe_float(snapshot["AvailableFunds"])
            if "BuyingPower" in snapshot:
                snapshot["buying_power"] = self._safe_float(snapshot["BuyingPower"])

            return snapshot

    def _ensure_client(self) -> Any:
        """Return an instantiated IB client, creating one when necessary."""

        if self._client is not None:
            return self._client

        ib_client_factory = self._ib_client_factory
        contract_factory = self._contract_factory
        limit_order_factory = self._limit_order_factory
        if ib_client_factory is None or contract_factory is None or limit_order_factory is None:
            ib_insync = require_ib_insync()
            ib_client_factory = ib_client_factory or ib_insync.IB
            contract_factory = contract_factory or ib_insync.Stock
            limit_order_factory = limit_order_factory or ib_insync.LimitOrder
            self._ib_client_factory = ib_client_factory
            self._contract_factory = contract_factory
            self._limit_order_factory = limit_order_factory

        self._client = ib_client_factory()
        return self._client

    def _require_connected_client(self) -> Any:
        """Return the connected IB client or raise a connection error."""

        client = self._ensure_client()
        if not client.isConnected():
            raise BrokerConnectionError("IBKR client is not connected")
        return client

    async def _run_with_retries(
        self,
        label: str,
        operation: RetryableOperation[T],
        *,
        timeout_seconds: float | None = None,
    ) -> T:
        """Run an operation with bounded retry logic and logging."""

        last_error: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                result = operation()
                if inspect.isawaitable(result):
                    if timeout_seconds is not None:
                        return await asyncio.wait_for(result, timeout=timeout_seconds)
                    return await result
                return result
            except (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt >= self._config.max_retries:
                    break

                delay = self._config.retry_backoff_seconds * attempt
                self._logger.warning(
                    "%s failed on attempt %s/%s: %s; retrying in %.2fs",
                    label,
                    attempt,
                    self._config.max_retries,
                    exc,
                    delay,
                )
                if delay > 0.0:
                    await self._sleep(delay)

        assert last_error is not None
        raise last_error

    def _normalize_order_request(self, order: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize a limit order request."""

        symbol = str(order.get("symbol", "")).strip().upper()
        side = str(order.get("side") or order.get("action") or "").strip().upper()
        if not symbol:
            raise OrderSubmissionError("symbol must be a non-empty string")
        if side not in {"BUY", "SELL"}:
            raise OrderSubmissionError("side must be BUY or SELL")

        quantity = self._positive_float(order.get("quantity"), field_name="quantity")
        limit_price = self._positive_float(
            order.get("limit_price", order.get("lmtPrice")),
            field_name="limit_price",
        )
        order_type = str(order.get("order_type", "LIMIT")).strip().upper()
        if order_type not in {"LIMIT", "LMT"}:
            raise OrderSubmissionError("IBKRBroker only supports limit orders")

        smart_routing = bool(order.get("smart_routing", True))
        exchange = "SMART" if smart_routing else str(
            order.get("exchange", self._config.default_exchange)
        ).strip().upper()
        if not exchange:
            raise OrderSubmissionError("exchange must be a non-empty string")

        tif = str(order.get("tif", "DAY")).strip().upper()
        if not tif:
            raise OrderSubmissionError("tif must be a non-empty string")

        currency = str(order.get("currency", self._config.default_currency)).strip().upper()
        if not currency:
            raise OrderSubmissionError("currency must be a non-empty string")

        account = str(order.get("account", self._config.account or "")).strip() or None
        order_ref = str(order.get("order_ref", "")).strip() or None
        outside_rth = bool(order.get("outside_rth", False))

        return {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "limit_price": limit_price,
            "exchange": exchange,
            "currency": currency,
            "tif": tif,
            "account": account,
            "order_ref": order_ref,
            "outside_rth": outside_rth,
            "smart_routing": smart_routing,
        }

    def _build_contract(self, request: Mapping[str, Any]) -> Any:
        """Build an IB contract for a stock routed through SMART or a direct venue."""

        assert self._contract_factory is not None
        return self._contract_factory(request["symbol"], request["exchange"], request["currency"])

    def _build_limit_order(self, request: Mapping[str, Any]) -> Any:
        """Build an IB limit order from a normalized request."""

        assert self._limit_order_factory is not None
        kwargs: dict[str, Any] = {
            "tif": request["tif"],
            "transmit": True,
            "outsideRth": request["outside_rth"],
        }
        if request["account"] is not None:
            kwargs["account"] = request["account"]
        if request["order_ref"] is not None:
            kwargs["orderRef"] = request["order_ref"]
        return self._limit_order_factory(request["side"], request["quantity"], request["limit_price"], **kwargs)

    def _find_trade_locked(self, order_id: str) -> Any | None:
        """Return the tracked trade for the requested order id when available."""

        normalized_id = str(order_id).strip()
        if not normalized_id:
            raise OrderNotFoundError("order_id must be a non-empty string")

        trade = self._tracked_trades.get(normalized_id)
        if trade is not None:
            return trade

        self._refresh_trade_cache_locked()
        return self._tracked_trades.get(normalized_id)

    def _refresh_trade_cache_locked(self) -> None:
        """Refresh the tracked trade cache from the client when available."""

        client = self._require_connected_client()
        trade_sources: list[Any] = []
        if hasattr(client, "openTrades"):
            trade_sources.extend(client.openTrades())
        if hasattr(client, "trades"):
            for trade in client.trades():
                if trade not in trade_sources:
                    trade_sources.append(trade)

        for trade in trade_sources:
            try:
                order_id = self._extract_order_id(trade)
            except OrderNotFoundError:
                continue
            self._tracked_trades[order_id] = trade

    def _extract_order_id(self, trade: Any) -> str:
        """Return the broker order id from a trade object."""

        order = getattr(trade, "order", None)
        order_id = getattr(order, "orderId", None)
        if order_id is None:
            raise OrderNotFoundError("trade did not expose an orderId")
        return str(order_id)

    def _normalize_trade(self, trade: Any) -> dict[str, Any]:
        """Return a normalized order snapshot from an IB trade object."""

        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        order_status = getattr(trade, "orderStatus", None)
        order_id = self._extract_order_id(trade)

        quantity = self._safe_float(getattr(order, "totalQuantity", 0.0))
        filled_quantity = self._safe_float(getattr(order_status, "filled", 0.0))
        raw_remaining = getattr(order_status, "remaining", None)
        if raw_remaining is None:
            remaining_quantity = max(quantity - filled_quantity, 0.0)
        else:
            remaining_quantity = max(self._safe_float(raw_remaining), 0.0)

        status = str(getattr(order_status, "status", "UNKNOWN")).strip() or "UNKNOWN"
        normalized_status = status.replace(" ", "").upper()
        if filled_quantity > 0.0 and remaining_quantity > 0.0:
            fill_state = "partial"
        elif filled_quantity > 0.0 and remaining_quantity <= 0.0:
            fill_state = "filled"
        else:
            fill_state = "unfilled"

        is_active = normalized_status not in TERMINAL_ORDER_STATUSES and remaining_quantity > 0.0
        submitted_at = self._submitted_at.get(order_id)
        age_seconds = None if submitted_at is None else max(self._monotonic() - submitted_at, 0.0)

        return {
            "order_id": order_id,
            "symbol": str(getattr(contract, "symbol", "")).strip().upper(),
            "side": str(getattr(order, "action", "")).strip().upper(),
            "quantity": quantity,
            "filled_quantity": filled_quantity,
            "remaining_quantity": remaining_quantity,
            "average_fill_price": self._safe_float(getattr(order_status, "avgFillPrice", 0.0)),
            "limit_price": self._safe_float(getattr(order, "lmtPrice", 0.0)),
            "status": status,
            "fill_state": fill_state,
            "is_active": is_active,
            "exchange": str(getattr(contract, "exchange", "")).strip().upper(),
            "currency": str(getattr(contract, "currency", "")).strip().upper(),
            "paper_trading": self._config.paper_trading,
            "age_seconds": age_seconds,
            "order_ref": str(getattr(order, "orderRef", "")).strip(),
        }

    def _positive_float(self, value: Any, *, field_name: str) -> float:
        """Return a validated positive float."""

        numeric = self._safe_float(value)
        if numeric <= 0.0:
            raise OrderSubmissionError(f"{field_name} must be greater than 0")
        return numeric

    def _safe_float(self, value: Any) -> float:
        """Coerce a value to float, defaulting invalid values to 0.0."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _maybe_float(self, value: Any) -> float | None:
        """Return a float for numeric values or `None` otherwise."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
