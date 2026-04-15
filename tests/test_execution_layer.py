"""Offline tests for the IBKR execution layer and order manager."""

from __future__ import annotations

import unittest

from trading_system.execution import IBKRBroker, IBKRBrokerConfig, OrderManager


class FakeClock:
    """Mutable monotonic clock used by stale-order tests."""

    def __init__(self, start: float = 0.0) -> None:
        """Initialize the clock at a deterministic point."""

        self._value = float(start)

    def __call__(self) -> float:
        """Return the current monotonic time."""

        return self._value

    def advance(self, seconds: float) -> None:
        """Advance the clock by the requested duration."""

        self._value += float(seconds)


async def no_sleep(seconds: float) -> None:
    """Zero-cost sleep stub used by retry tests."""


class FakeContract:
    """Minimal stock contract stub compatible with the broker adapter."""

    def __init__(self, symbol: str, exchange: str, currency: str) -> None:
        """Store the contract routing parameters."""

        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency


class FakeLimitOrder:
    """Minimal IB limit order stub."""

    def __init__(
        self,
        action: str,
        totalQuantity: float,
        lmtPrice: float,
        *,
        tif: str = "DAY",
        transmit: bool = True,
        outsideRth: bool = False,
        account: str | None = None,
        orderRef: str | None = None,
    ) -> None:
        """Store the limit-order parameters expected by the broker."""

        self.action = action
        self.totalQuantity = float(totalQuantity)
        self.lmtPrice = float(lmtPrice)
        self.tif = tif
        self.transmit = transmit
        self.outsideRth = outsideRth
        self.account = account
        self.orderRef = orderRef
        self.orderId: int | None = None


class FakeOrderStatus:
    """Minimal IB order-status stub."""

    def __init__(
        self,
        *,
        status: str = "Submitted",
        filled: float = 0.0,
        remaining: float = 0.0,
        avgFillPrice: float = 0.0,
    ) -> None:
        """Store the order lifecycle attributes."""

        self.status = status
        self.filled = float(filled)
        self.remaining = float(remaining)
        self.avgFillPrice = float(avgFillPrice)


class FakeTrade:
    """Minimal IB trade stub composed of contract, order, and status."""

    def __init__(self, contract: FakeContract, order: FakeLimitOrder) -> None:
        """Initialize the trade in Submitted state."""

        self.contract = contract
        self.order = order
        self.orderStatus = FakeOrderStatus(remaining=order.totalQuantity)


class FakeAccountValue:
    """Minimal IB account summary record."""

    def __init__(self, tag: str, value: str, account: str = "DU123456") -> None:
        """Store the account summary item."""

        self.tag = tag
        self.value = value
        self.account = account


class FakePosition:
    """Minimal IB position record."""

    def __init__(
        self,
        *,
        symbol: str,
        exchange: str,
        currency: str,
        position: float,
        avg_cost: float,
    ) -> None:
        """Store the position data."""

        self.contract = FakeContract(symbol, exchange, currency)
        self.position = float(position)
        self.avgCost = float(avg_cost)
        self.marketPrice = float(avg_cost) + 1.5
        self.marketValue = self.position * self.marketPrice
        self.unrealizedPNL = self.position * 1.5
        self.realizedPNL = 0.0


class FakeIB:
    """In-memory IB client stub with deterministic retry/fill behavior."""

    def __init__(self, *, connect_failures: int = 0, place_failures: int = 0) -> None:
        """Initialize the fake client and failure counters."""

        self._connected = False
        self.connect_failures = connect_failures
        self.place_failures = place_failures
        self.connect_calls: list[dict[str, object]] = []
        self.place_calls: list[tuple[FakeContract, FakeLimitOrder]] = []
        self.cancel_calls: list[int] = []
        self._next_order_id = 1
        self._trades: list[FakeTrade] = []
        self._positions = [
            FakePosition(symbol="AAPL", exchange="SMART", currency="USD", position=25.0, avg_cost=100.0)
        ]
        self._account_summary = [
            FakeAccountValue("NetLiquidation", "100000"),
            FakeAccountValue("GrossPositionValue", "5000"),
            FakeAccountValue("AvailableFunds", "95000"),
            FakeAccountValue("BuyingPower", "400000"),
        ]

    def isConnected(self) -> bool:
        """Return the fake connection state."""

        return self._connected

    async def connectAsync(
        self,
        host: str,
        port: int,
        *,
        clientId: int,
        timeout: float,
        readonly: bool,
        account: str,
    ) -> "FakeIB":
        """Connect with optional transient failures."""

        self.connect_calls.append(
            {
                "host": host,
                "port": port,
                "client_id": clientId,
                "timeout": timeout,
                "readonly": readonly,
                "account": account,
            }
        )
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise OSError("temporary connect failure")

        self._connected = True
        return self

    def disconnect(self) -> None:
        """Disconnect the fake session."""

        self._connected = False

    def placeOrder(self, contract: FakeContract, order: FakeLimitOrder) -> FakeTrade:
        """Place an order with optional transient failures."""

        self.place_calls.append((contract, order))
        if self.place_failures > 0:
            self.place_failures -= 1
            raise OSError("temporary submission failure")

        if order.orderId is None:
            order.orderId = self._next_order_id
            self._next_order_id += 1

        trade = FakeTrade(contract, order)
        self._trades.append(trade)
        return trade

    def cancelOrder(self, order: FakeLimitOrder) -> FakeTrade:
        """Cancel an order and mark the trade as inactive."""

        if order.orderId is None:
            raise KeyError("orderId is required")

        self.cancel_calls.append(order.orderId)
        for trade in self._trades:
            if trade.order.orderId == order.orderId:
                trade.orderStatus.status = "Cancelled"
                trade.orderStatus.remaining = max(trade.order.totalQuantity - trade.orderStatus.filled, 0.0)
                return trade
        raise KeyError(order.orderId)

    def openTrades(self) -> list[FakeTrade]:
        """Return currently active trades."""

        return [
            trade
            for trade in self._trades
            if trade.orderStatus.status not in {"Cancelled", "Filled", "Inactive"}
        ]

    def trades(self) -> list[FakeTrade]:
        """Return all tracked trades."""

        return list(self._trades)

    def positions(self) -> list[FakePosition]:
        """Return the fake positions."""

        return list(self._positions)

    async def accountSummaryAsync(self, account: str) -> list[FakeAccountValue]:
        """Return the fake account summary."""

        return list(self._account_summary)


class ExecutionLayerTests(unittest.IsolatedAsyncioTestCase):
    """Verify IBKR broker and order-manager behavior without a live broker."""

    async def test_ibkr_broker_uses_paper_port_and_smart_routing(self) -> None:
        """Paper-trading connections should default to port 7497 and SMART routing."""

        fake_ib = FakeIB()
        broker = IBKRBroker(
            config=IBKRBrokerConfig(paper_trading=True, max_retries=2, retry_backoff_seconds=0.0),
            ib_client_factory=lambda: fake_ib,
            contract_factory=FakeContract,
            limit_order_factory=FakeLimitOrder,
            sleep=no_sleep,
        )

        await broker.connect()
        order_id = await broker.submit_order(
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "limit_price": 101.25,
                "smart_routing": True,
            }
        )
        status = await broker.get_order_status(order_id)
        positions = await broker.get_positions()
        account = await broker.get_account_snapshot()
        await broker.disconnect()

        self.assertEqual(fake_ib.connect_calls[0]["port"], 7497)
        self.assertEqual(fake_ib.place_calls[0][0].exchange, "SMART")
        self.assertEqual(status["status"], "Submitted")
        self.assertTrue(status["paper_trading"])
        self.assertEqual(positions["AAPL"]["quantity"], 25.0)
        self.assertEqual(account["capital"], 100000.0)
        self.assertEqual(account["gross_exposure"], 5000.0)

    async def test_ibkr_broker_retries_transient_connect_and_submit_failures(self) -> None:
        """Transient broker failures should be retried before surfacing an error."""

        fake_ib = FakeIB(connect_failures=1, place_failures=1)
        broker = IBKRBroker(
            config=IBKRBrokerConfig(max_retries=2, retry_backoff_seconds=0.0),
            ib_client_factory=lambda: fake_ib,
            contract_factory=FakeContract,
            limit_order_factory=FakeLimitOrder,
            sleep=no_sleep,
        )

        await broker.connect()
        order_id = await broker.submit_order(
            {
                "symbol": "MSFT",
                "side": "SELL",
                "quantity": 5,
                "limit_price": 250.0,
            }
        )

        self.assertEqual(len(fake_ib.connect_calls), 2)
        self.assertEqual(len(fake_ib.place_calls), 2)
        self.assertEqual(order_id, "1")

    async def test_order_manager_tracks_partial_fills_and_cancels_stale_orders(self) -> None:
        """The order manager should expose partial fills and cancel stale orders."""

        clock = FakeClock()
        fake_ib = FakeIB()
        broker = IBKRBroker(
            config=IBKRBrokerConfig(max_retries=2, retry_backoff_seconds=0.0),
            ib_client_factory=lambda: fake_ib,
            contract_factory=FakeContract,
            limit_order_factory=FakeLimitOrder,
            sleep=no_sleep,
            monotonic_fn=clock,
        )
        manager = OrderManager(broker, stale_order_seconds=30.0)

        await broker.connect()
        initial = await manager.submit_limit_order(symbol="AAPL", side="BUY", quantity=10, limit_price=100.0)
        trade = fake_ib.trades()[0]
        trade.orderStatus.filled = 4.0
        trade.orderStatus.remaining = 6.0
        trade.orderStatus.avgFillPrice = 99.75
        trade.orderStatus.status = "Submitted"

        open_orders = await manager.sync_open_orders()
        clock.advance(31.0)
        cancelled = await manager.cancel_stale_orders()
        final_status = await broker.get_order_status(initial["order_id"])
        await broker.disconnect()

        self.assertEqual(open_orders[initial["order_id"]]["fill_state"], "partial")
        self.assertEqual(open_orders[initial["order_id"]]["filled_quantity"], 4.0)
        self.assertEqual(cancelled, [initial["order_id"]])
        self.assertEqual(final_status["status"], "Cancelled")
        self.assertFalse(final_status["is_active"])
