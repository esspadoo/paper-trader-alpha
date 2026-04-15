"""Lifecycle manager for the event-driven trading runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import Future

from trading_system.core.bus import BaseEventBus, EventBus, EventHandler, EventType
from trading_system.core.events import BaseEvent
from trading_system.core.exceptions import LifecycleStateError

EngineBootstrap = Callable[["Engine"], Awaitable[None]]


class Engine:
    """Own the event loop resources and runtime lifecycle for the trading system."""

    def __init__(
        self,
        *,
        event_bus: BaseEventBus | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the engine with an event bus implementation."""

        self._bus = event_bus or EventBus()
        self._logger = logger or logging.getLogger(__name__)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    @property
    def bus(self) -> BaseEventBus:
        """Return the engine-owned event bus."""

        return self._bus

    @property
    def is_running(self) -> bool:
        """Return whether the engine is active on an event loop."""

        return self._started

    async def start(self) -> None:
        """Start the engine and the underlying event bus."""

        if self._started:
            raise LifecycleStateError("engine is already running")

        self._loop = asyncio.get_running_loop()
        await self._bus.start()
        self._started = True
        self._logger.debug("engine started")

    async def stop(self, *, drain: bool = True) -> None:
        """Stop the engine and underlying event bus."""

        if not self._started:
            return

        await self._bus.stop(drain=drain)
        self._loop = None
        self._started = False
        self._logger.debug("engine stopped")

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event through the engine-owned bus."""

        self._ensure_running()
        await self._bus.publish(event)

    def publish_threadsafe(self, event: BaseEvent) -> Future[None]:
        """Publish an event from a foreign thread."""

        self._ensure_running()
        publisher = getattr(self._bus, "publish_threadsafe", None)
        if not callable(publisher):
            raise TypeError("thread-safe publishing requires an event bus with publish_threadsafe support")
        return publisher(event)

    async def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""

        await self._bus.subscribe(event_type, handler)

    async def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""

        await self._bus.unsubscribe(event_type, handler)

    async def wait_until_idle(self) -> None:
        """Wait until queued events have been processed."""

        self._ensure_running()
        await self._bus.join()

    async def run(
        self,
        *,
        bootstrap: EngineBootstrap | None = None,
        until: asyncio.Event | None = None,
    ) -> None:
        """Start the engine, optionally run bootstrap work, and wait for completion."""

        await self.start()
        try:
            if bootstrap is not None:
                await bootstrap(self)

            if until is None:
                await self.wait_until_idle()
            else:
                await until.wait()
                await self.wait_until_idle()
        finally:
            await self.stop(drain=True)

    async def __aenter__(self) -> "Engine":
        """Start the engine when entering an async context manager."""

        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Stop the engine when leaving an async context manager."""

        await self.stop(drain=True)

    def _ensure_running(self) -> None:
        """Ensure the engine has been started before use."""

        if not self._started or self._loop is None:
            raise LifecycleStateError("engine has not been started")
