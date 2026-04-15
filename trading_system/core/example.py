"""Runnable example showing an end-to-end event flow through the engine."""

from __future__ import annotations

import asyncio

from trading_system.core.engine import Engine
from trading_system.core.events import BaseEvent, MarketEvent, OrderEvent, SignalEvent


async def run_example_flow() -> list[BaseEvent]:
    """Run a simple market-to-signal-to-order flow and return the observed events."""

    engine = Engine()
    observed_events: list[BaseEvent] = []
    completed = asyncio.Event()

    async def audit(event: BaseEvent) -> None:
        observed_events.append(event)
        if isinstance(event, OrderEvent):
            completed.set()

    async def on_market(event: MarketEvent) -> None:
        if event.last_price is None or event.last_price <= 100.0:
            return

        await engine.publish(
            SignalEvent(
                source="alpha-agent",
                symbol=event.symbol,
                strategy_id="mean-reversion",
                signal_type="BUY",
                confidence=0.91,
                payload={"trigger": "price-breakout", "market_event_id": str(event.event_id)},
                correlation_id=event.event_id,
            )
        )

    async def on_signal(event: SignalEvent) -> None:
        await engine.publish(
            OrderEvent(
                source="execution-agent",
                order_id="ORD-0001",
                symbol=event.symbol,
                side=event.signal_type,
                quantity=100.0,
                status="NEW",
                payload={"strategy_id": event.strategy_id, "confidence": event.confidence},
                correlation_id=event.correlation_id or event.event_id,
            )
        )

    async with engine:
        await engine.subscribe(BaseEvent, audit)
        await engine.subscribe(MarketEvent, on_market)
        await engine.subscribe(SignalEvent, on_signal)

        await engine.publish(
            MarketEvent(
                source="market-data-feed",
                symbol="AAPL",
                venue="NASDAQ",
                last_price=101.25,
                bid=101.20,
                ask=101.30,
                volume=15_000,
                payload={"session": "regular"},
            )
        )

        await completed.wait()
        await engine.wait_until_idle()

    return observed_events


async def main() -> None:
    """Run the example flow and print the event sequence."""

    for event in await run_example_flow():
        print(f"{event.event_type}: topic={event.topic} source={event.source} correlation={event.correlation_id}")


if __name__ == "__main__":
    asyncio.run(main())
