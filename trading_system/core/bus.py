"""Queue-backed publish/subscribe runtime for trading system events."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from time import perf_counter_ns
from typing import TypeAlias

from trading_system.core.events import BaseEvent
from trading_system.core.exceptions import HandlerExecutionError, LifecycleStateError
from trading_system.infra import LatencyRecorder

HandlerResult: TypeAlias = Awaitable[None] | None
EventHandler: TypeAlias = Callable[[BaseEvent], HandlerResult]
EventType: TypeAlias = type[BaseEvent]


class BaseEventBus(ABC):
    """Abstract event bus interface for decoupled communication."""

    @abstractmethod
    async def start(self) -> None:
        """Start the event bus workers on the current asyncio loop."""

    @abstractmethod
    async def stop(self, *, drain: bool = True) -> None:
        """Stop the event bus and optionally drain queued events first."""

    @abstractmethod
    async def publish(self, event: BaseEvent) -> None:
        """Enqueue an event for asynchronous dispatch."""

    @abstractmethod
    async def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for an event type."""

    @abstractmethod
    async def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from an event type subscription."""

    @abstractmethod
    async def join(self) -> None:
        """Wait until all queued events have been processed."""


class EventBus(BaseEventBus):
    """Production-grade event bus with queue-based asynchronous dispatch."""

    def __init__(
        self,
        *,
        worker_count: int = 4,
        queue_maxsize: int = 10_000,
        handler_timeout: float | None = None,
        raise_on_handler_error: bool = False,
        logger: logging.Logger | None = None,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        """Initialize the event bus and its dispatch controls."""

        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        if queue_maxsize < 1:
            raise ValueError("queue_maxsize must be at least 1")
        if handler_timeout is not None and handler_timeout <= 0:
            raise ValueError("handler_timeout must be greater than 0 when provided")

        self._worker_count = worker_count
        self._handler_timeout = handler_timeout
        self._raise_on_handler_error = raise_on_handler_error
        self._logger = logger or logging.getLogger(__name__)
        self._latency_recorder = latency_recorder
        self._subscriptions: dict[EventType, list[EventHandler]] = {}
        self._subscription_lock = threading.RLock()
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._workers: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._stopping = False
        self._enqueued_at_ns: dict[object, int] = {}

    @property
    def is_running(self) -> bool:
        """Return whether the bus is actively dispatching events."""

        return self._started and not self._stopping

    @property
    def queue_size(self) -> int:
        """Return the current event backlog."""

        return self._queue.qsize()

    async def start(self) -> None:
        """Start worker tasks on the current running event loop."""

        if self._started:
            raise LifecycleStateError("event bus is already running")

        self._loop = asyncio.get_running_loop()
        self._stopping = False
        self._workers = [
            self._loop.create_task(self._worker(worker_id), name=f"event-bus-worker-{worker_id}")
            for worker_id in range(self._worker_count)
        ]
        self._started = True

    async def stop(self, *, drain: bool = True) -> None:
        """Stop dispatch workers and optionally wait for queued work to finish."""

        if not self._started:
            return

        self._stopping = True

        if drain:
            await self.join()

        for worker in self._workers:
            worker.cancel()

        results = await asyncio.gather(*self._workers, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                self._logger.error(
                    "event bus worker exited with an unexpected error",
                    exc_info=(type(result), result, result.__traceback__),
                )

        self._workers.clear()
        self._loop = None
        self._started = False
        self._stopping = False

    async def publish(self, event: BaseEvent) -> None:
        """Enqueue an event for non-blocking asynchronous processing."""

        self._ensure_running()
        if self._latency_recorder is not None:
            self._enqueued_at_ns[event.event_id] = perf_counter_ns()
        await self._queue.put(event)

    def publish_threadsafe(self, event: BaseEvent) -> Future[None]:
        """Publish an event from a non-event-loop thread."""

        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(self.publish(event), self._loop)

    async def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for the given event type."""

        self._validate_subscription(event_type, handler)

        with self._subscription_lock:
            handlers = self._subscriptions.setdefault(event_type, [])
            if handler not in handlers:
                handlers.append(handler)

    def subscribe_threadsafe(self, event_type: EventType, handler: EventHandler) -> Future[None]:
        """Register a handler from a non-event-loop thread."""

        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(self.subscribe(event_type, handler), self._loop)

    async def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from the given event type."""

        self._validate_subscription(event_type, handler)

        with self._subscription_lock:
            handlers = self._subscriptions.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)
                if not handlers:
                    self._subscriptions.pop(event_type, None)

    async def join(self) -> None:
        """Wait until the queue has been fully processed."""

        await self._queue.join()

    async def _worker(self, worker_id: int) -> None:
        """Continuously dispatch events from the internal queue."""

        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                if self._latency_recorder is not None:
                    published_at_ns = self._enqueued_at_ns.pop(event.event_id, None)
                    if published_at_ns is not None:
                        receive_latency_ns = max(perf_counter_ns() - published_at_ns, 0)
                        self._latency_recorder.record_ns("event_queue_latency", receive_latency_ns)
                        if event.event_type == "MarketEvent":
                            self._latency_recorder.record_ns("market_event_receive_latency", receive_latency_ns)
                await self._dispatch(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error(
                    "worker %s failed while processing %s",
                    worker_id,
                    event.event_type,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            finally:
                self._queue.task_done()

    async def _dispatch(self, event: BaseEvent) -> None:
        """Dispatch a single event to all matching handlers."""

        handlers = self._matching_handlers(type(event))
        if not handlers:
            self._logger.debug("dropping event %s with no subscribers", event.event_type)
            return

        tasks = [
            asyncio.create_task(
                self._invoke_handler(handler, event),
                name=f"dispatch-{event.event_type}-{index}",
            )
            for index, handler in enumerate(handlers)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            for failure in failures:
                self._logger.error(
                    "event handler failed for %s",
                    event.event_type,
                    exc_info=(type(failure), failure, failure.__traceback__),
                )
            if self._raise_on_handler_error:
                raise HandlerExecutionError(event=event, errors=tuple(failures))

    async def _invoke_handler(self, handler: EventHandler, event: BaseEvent) -> None:
        """Execute a handler without blocking the event loop."""

        if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, "__call__", None)):
            operation = handler(event)
        else:
            operation = asyncio.to_thread(handler, event)

        if self._handler_timeout is None:
            result = await operation
        else:
            result = await asyncio.wait_for(operation, timeout=self._handler_timeout)

        if inspect.isawaitable(result):
            await result

    def _matching_handlers(self, event_type: EventType) -> list[EventHandler]:
        """Return handlers for the event type and its subscribed base classes."""

        with self._subscription_lock:
            deduplicated: list[EventHandler] = []
            seen: set[int] = set()
            for parent in event_type.mro():
                if not issubclass(parent, BaseEvent):
                    continue

                for handler in self._subscriptions.get(parent, []):
                    identity = id(handler)
                    if identity not in seen:
                        deduplicated.append(handler)
                        seen.add(identity)

        return deduplicated

    def _validate_subscription(self, event_type: EventType, handler: EventHandler) -> None:
        """Validate a subscription request before it mutates the registry."""

        if not issubclass(event_type, BaseEvent):
            raise TypeError("event_type must inherit from BaseEvent")
        if not callable(handler):
            raise TypeError("handler must be callable")

    def _ensure_running(self) -> None:
        """Ensure the bus is in a state that can accept new events."""

        if not self._started:
            raise LifecycleStateError("event bus has not been started")
        if self._stopping:
            raise LifecycleStateError("event bus is stopping and cannot accept new events")

    def _ensure_loop(self) -> None:
        """Ensure the bus has an active loop for thread-safe operations."""

        if self._loop is None or not self._started:
            raise LifecycleStateError("event bus must be running before thread-safe operations")
