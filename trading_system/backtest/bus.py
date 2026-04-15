"""Deterministic event-bus variant for backtest simulation."""

from __future__ import annotations

import asyncio

from trading_system.core.bus import EventBus
from trading_system.core.events import BaseEvent
from trading_system.core.exceptions import HandlerExecutionError


class SequentialEventBus(EventBus):
    """Dispatch handlers sequentially to preserve deterministic simulation order."""

    async def _dispatch(self, event: BaseEvent) -> None:
        """Dispatch a single event in subscription order."""

        handlers = self._matching_handlers(type(event))
        if not handlers:
            self._logger.debug("dropping event %s with no subscribers", event.event_type)
            return

        failures: list[Exception] = []
        for handler in handlers:
            try:
                await self._invoke_handler(handler, event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures.append(exc)
                self._logger.error(
                    "event handler failed for %s",
                    event.event_type,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        if failures and self._raise_on_handler_error:
            raise HandlerExecutionError(event=event, errors=tuple(failures))
