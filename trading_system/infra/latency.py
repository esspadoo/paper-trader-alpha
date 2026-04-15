"""Low-overhead latency instrumentation utilities for runtime hot paths."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import ceil
from statistics import mean
from threading import RLock
from time import perf_counter_ns


@dataclass(frozen=True, slots=True)
class LatencyMetricSummary:
    """Summary statistics for a single latency series."""

    name: str
    count: int
    mean_ms: float
    p95_ms: float
    p99_ms: float
    worst_ms: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable metric payload."""

        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class LatencyReport:
    """Grouped latency report with bottleneck ranking."""

    metrics: tuple[LatencyMetricSummary, ...]
    top_bottlenecks: tuple[LatencyMetricSummary, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report payload."""

        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "top_bottlenecks": [metric.to_dict() for metric in self.top_bottlenecks],
        }


class LatencyRecorder:
    """Thread-safe bounded recorder for latency series."""

    def __init__(self, *, sample_limit: int = 8_192) -> None:
        """Initialize the recorder with a bounded sample history."""

        if sample_limit < 128:
            raise ValueError("sample_limit must be at least 128")
        self._sample_limit = sample_limit
        self._samples: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=self._sample_limit))
        self._lock = RLock()

    @property
    def sample_limit(self) -> int:
        """Return the configured per-metric sample bound."""

        return self._sample_limit

    def record_ns(self, name: str, duration_ns: int) -> None:
        """Record a latency sample in nanoseconds."""

        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("latency metric name must be a non-empty string")
        if duration_ns < 0:
            raise ValueError("duration_ns must be greater than or equal to 0")
        with self._lock:
            self._samples[normalized_name].append(int(duration_ns))

    def record_ms(self, name: str, duration_ms: float) -> None:
        """Record a latency sample in milliseconds."""

        self.record_ns(name, int(duration_ms * 1_000_000.0))

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Record elapsed time for a code region."""

        started_at = perf_counter_ns()
        try:
            yield
        finally:
            self.record_ns(name, perf_counter_ns() - started_at)

    def snapshot(self) -> dict[str, tuple[int, ...]]:
        """Return an immutable copy of the currently recorded samples."""

        with self._lock:
            return {
                name: tuple(samples)
                for name, samples in self._samples.items()
            }

    def report(self, *, top_limit: int = 5) -> LatencyReport:
        """Return a summarized latency report."""

        if top_limit < 1:
            raise ValueError("top_limit must be at least 1")

        metrics = tuple(
            sorted(
                (self._summarize_metric(name, tuple(samples)) for name, samples in self.snapshot().items() if samples),
                key=lambda metric: metric.name,
            )
        )
        top_bottlenecks = tuple(
            sorted(metrics, key=lambda metric: (metric.p95_ms, metric.mean_ms, metric.worst_ms), reverse=True)[:top_limit]
        )
        return LatencyReport(metrics=metrics, top_bottlenecks=top_bottlenecks)

    def _summarize_metric(self, name: str, samples_ns: tuple[int, ...]) -> LatencyMetricSummary:
        """Summarize a single metric from bounded samples."""

        ordered = tuple(sorted(samples_ns))
        return LatencyMetricSummary(
            name=name,
            count=len(ordered),
            mean_ms=self._ns_to_ms(mean(ordered)),
            p95_ms=self._ns_to_ms(self._percentile(ordered, 0.95)),
            p99_ms=self._ns_to_ms(self._percentile(ordered, 0.99)),
            worst_ms=self._ns_to_ms(ordered[-1]),
        )

    def _percentile(self, ordered_samples: tuple[int, ...], percentile: float) -> int:
        """Return a nearest-rank percentile from ordered samples."""

        if not ordered_samples:
            return 0
        index = max(ceil(len(ordered_samples) * percentile) - 1, 0)
        return int(ordered_samples[index])

    def _ns_to_ms(self, value_ns: float | int) -> float:
        """Convert nanoseconds to milliseconds."""

        return float(value_ns) / 1_000_000.0
