"""Asynchronous JSONL journal helpers for non-blocking operational persistence."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

Serializer = Callable[[Any], Mapping[str, Any] | dict[str, Any]]


class AsyncJsonlWriter:
    """Queue-backed JSONL writer that keeps file I/O off the hot path."""

    def __init__(
        self,
        path: str | Path,
        *,
        serializer: Serializer | None = None,
        flush_interval_seconds: float = 0.25,
        batch_size: int = 64,
        queue_maxsize: int = 4_096,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the journal writer."""

        if flush_interval_seconds <= 0.0:
            raise ValueError("flush_interval_seconds must be greater than 0")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if queue_maxsize < 1:
            raise ValueError("queue_maxsize must be at least 1")

        self._path = Path(path)
        self._serializer = serializer or self._default_serializer
        self._flush_interval_seconds = float(flush_interval_seconds)
        self._batch_size = batch_size
        self._queue: asyncio.Queue[Any | None] = asyncio.Queue(maxsize=queue_maxsize)
        self._logger = logger or logging.getLogger(__name__)
        self._worker: asyncio.Task[None] | None = None
        self._started = False
        self._dropped_records = 0
        self._drop_lock = Lock()

    @property
    def path(self) -> Path:
        """Return the journal path."""

        return self._path

    @property
    def dropped_records(self) -> int:
        """Return the number of records dropped because the queue was full."""

        with self._drop_lock:
            return self._dropped_records

    async def start(self) -> None:
        """Start the background writer task."""

        if self._started:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._worker = asyncio.create_task(self._writer_loop(), name=f"jsonl-writer:{self._path.name}")
        self._started = True

    async def stop(self) -> None:
        """Drain the queue and stop the background writer."""

        if not self._started:
            return
        await self._queue.join()
        await self._queue.put(None)
        if self._worker is not None:
            await self._worker
        self._worker = None
        self._started = False

    async def append(self, record: Any) -> None:
        """Append a record, waiting for queue capacity when required."""

        if not self._started:
            raise RuntimeError("AsyncJsonlWriter.start must be called before append")
        await self._queue.put(record)

    def append_nowait(self, record: Any) -> bool:
        """Append a record without blocking, dropping when the queue is full."""

        if not self._started:
            raise RuntimeError("AsyncJsonlWriter.start must be called before append_nowait")
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            with self._drop_lock:
                self._dropped_records += 1
            self._logger.warning("dropping journal record because the queue is full path=%s", self._path)
            return False
        return True

    async def _writer_loop(self) -> None:
        """Continuously flush queued records to disk."""

        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            batch: list[Any] = [item]
            deadline = monotonic() + self._flush_interval_seconds
            while len(batch) < self._batch_size:
                timeout_seconds = deadline - monotonic()
                if timeout_seconds <= 0.0:
                    break
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
                except TimeoutError:
                    break
                if next_item is None:
                    self._queue.task_done()
                    await self._flush_batch(batch)
                    for _ in batch:
                        self._queue.task_done()
                    return
                batch.append(next_item)

            await self._flush_batch(batch)
            for _ in batch:
                self._queue.task_done()

    async def _flush_batch(self, batch: list[Any]) -> None:
        """Serialize a batch and append it to the journal."""

        lines = [self._serialize_line(record) for record in batch]
        await asyncio.to_thread(self._append_lines_sync, lines)

    def _serialize_line(self, record: Any) -> str:
        """Serialize a queued record to a compact JSON line."""

        payload = self._serializer(record)
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))

    def _append_lines_sync(self, lines: list[str]) -> None:
        """Synchronously append a batch of lines to the journal file."""

        with self._path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")

    def _default_serializer(self, record: Any) -> Mapping[str, Any] | dict[str, Any]:
        """Return a JSON-ready record when no serializer was provided."""

        if isinstance(record, Mapping):
            return dict(record)
        raise TypeError("journal records must be mappings when no serializer is configured")
