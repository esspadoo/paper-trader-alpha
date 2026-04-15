"""Core exceptions raised by the event-driven runtime."""

from __future__ import annotations

from dataclasses import dataclass

from trading_system.core.events import BaseEvent


class CoreRuntimeError(Exception):
    """Base exception for event runtime failures."""


class LifecycleStateError(CoreRuntimeError):
    """Raised when lifecycle methods are called in an invalid state."""


@dataclass(slots=True)
class HandlerExecutionError(CoreRuntimeError):
    """Raised when one or more event handlers fail during dispatch."""

    event: BaseEvent
    errors: tuple[Exception, ...]

    def __str__(self) -> str:
        """Return a concise description of the handler failure."""

        return f"{len(self.errors)} handler(s) failed while processing {self.event.event_type}"
