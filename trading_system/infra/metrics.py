"""Lightweight runtime metrics registry for health and operations snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from trading_system.infra.latency import LatencyReport


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Serializable runtime metrics snapshot."""

    generated_at: str
    counters: dict[str, float]
    gauges: dict[str, float]
    latency: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics payload."""

        return dict(asdict(self))


class MetricsRegistry:
    """Thread-safe in-memory metrics registry."""

    def __init__(self) -> None:
        """Initialize the registry."""

        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._lock = RLock()

    def increment_counter(self, name: str, delta: float = 1.0) -> None:
        """Increase a named counter."""

        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("counter name must be a non-empty string")
        with self._lock:
            self._counters[normalized_name] = self._counters.get(normalized_name, 0.0) + float(delta)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a named gauge."""

        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("gauge name must be a non-empty string")
        with self._lock:
            self._gauges[normalized_name] = float(value)

    def snapshot(self, *, latency_report: LatencyReport | None = None) -> MetricsSnapshot:
        """Return a stable metrics snapshot."""

        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
        return MetricsSnapshot(
            generated_at=_utc_now_iso(),
            counters=counters,
            gauges=gauges,
            latency=None if latency_report is None else latency_report.to_dict(),
        )
