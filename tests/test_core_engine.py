"""Async tests for the event-driven core runtime."""

from __future__ import annotations

import asyncio
import threading
import unittest

from trading_system.core import BaseEvent, Engine, MarketEvent, NewsEvent, OrderEvent, SignalEvent
from trading_system.core.example import run_example_flow


class CoreEngineTests(unittest.IsolatedAsyncioTestCase):
    """Verify the queue-based event runtime behaves correctly."""

    async def test_example_flow_produces_expected_event_chain(self) -> None:
        """Market events should fan out into signals and orders."""

        events = await run_example_flow()

        self.assertEqual([event.event_type for event in events], ["MarketEvent", "SignalEvent", "OrderEvent"])
        self.assertIsInstance(events[0], MarketEvent)
        self.assertIsInstance(events[1], SignalEvent)
        self.assertIsInstance(events[2], OrderEvent)

    async def test_base_event_subscription_receives_all_subclasses(self) -> None:
        """A BaseEvent subscription should observe every concrete event type."""

        engine = Engine()
        observed: list[str] = []
        completed = asyncio.Event()

        async def audit(event: BaseEvent) -> None:
            observed.append(event.event_type)
            if len(observed) == 2:
                completed.set()

        async with engine:
            await engine.subscribe(BaseEvent, audit)
            await engine.publish(MarketEvent(source="feed", symbol="AAPL", venue="NASDAQ"))
            await engine.publish(NewsEvent(source="news", headline="Upgrade", symbols=("AAPL",)))
            await asyncio.wait_for(completed.wait(), timeout=1.0)
            await engine.wait_until_idle()

        self.assertEqual(observed, ["MarketEvent", "NewsEvent"])

    async def test_threadsafe_publish_accepts_foreign_thread_events(self) -> None:
        """Thread-safe publishing should bridge producer threads into the engine loop."""

        engine = Engine()
        handled = asyncio.Event()
        observed_symbols: list[str] = []

        async def on_market(event: MarketEvent) -> None:
            observed_symbols.append(event.symbol)
            handled.set()

        async with engine:
            await engine.subscribe(MarketEvent, on_market)

            def producer() -> None:
                future = engine.publish_threadsafe(
                    MarketEvent(source="threaded-feed", symbol="MSFT", venue="NASDAQ", last_price=410.5)
                )
                future.result(timeout=3.0)

            thread = threading.Thread(target=producer, name="market-producer", daemon=True)
            thread.start()
            await asyncio.wait_for(handled.wait(), timeout=3.0)
            thread.join(timeout=3.0)
            await engine.wait_until_idle()

        self.assertEqual(observed_symbols, ["MSFT"])
