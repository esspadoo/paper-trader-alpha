"""Persistent operational state used for runtime recoverability and controls."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    """Persisted kill-switch state."""

    engaged: bool
    reason: str | None
    engaged_at: str | None
    source: str | None
    strategy_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return dict(asdict(self))

    @classmethod
    def disengaged(cls) -> "KillSwitchState":
        """Return the default non-engaged state."""

        return cls(engaged=False, reason=None, engaged_at=None, source=None, strategy_id=None)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "KillSwitchState":
        """Build a kill-switch state from a stored payload."""

        if not isinstance(payload, dict):
            return cls.disengaged()
        return cls(
            engaged=bool(payload.get("engaged", False)),
            reason=None if payload.get("reason") in {None, ""} else str(payload.get("reason")),
            engaged_at=None if payload.get("engaged_at") in {None, ""} else str(payload.get("engaged_at")),
            source=None if payload.get("source") in {None, ""} else str(payload.get("source")),
            strategy_id=None if payload.get("strategy_id") in {None, ""} else str(payload.get("strategy_id")),
        )


class RuntimeStateStore:
    """SQLite-backed state store for runtime health, metrics, and reconciliation state."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the state-store path and lock."""

        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._connection: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        """Return the state-store path."""

        return self._path

    async def write_record(self, key: str, payload: dict[str, Any]) -> None:
        """Persist a named JSON payload."""

        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("key must be a non-empty string")
        async with self._lock:
            connection = self._connection_locked()
            connection.execute(
                """
                INSERT INTO runtime_state (state_key, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    normalized_key,
                    _utc_now_iso(),
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
            connection.commit()

    async def read_record(self, key: str) -> dict[str, Any] | None:
        """Return a previously stored JSON payload when available."""

        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("key must be a non-empty string")
        async with self._lock:
            connection = self._connection_locked()
            row = connection.execute(
                """
                SELECT payload_json
                FROM runtime_state
                WHERE state_key = ?
                """,
                (normalized_key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["payload_json"]))

    async def list_records(self) -> dict[str, dict[str, Any]]:
        """Return all stored runtime-state records."""

        async with self._lock:
            connection = self._connection_locked()
            rows = connection.execute(
                """
                SELECT state_key, payload_json
                FROM runtime_state
                ORDER BY state_key ASC
                """
            ).fetchall()
        return {
            str(row["state_key"]): json.loads(str(row["payload_json"]))
            for row in rows
        }

    async def write_kill_switch(self, state: KillSwitchState) -> None:
        """Persist the current kill-switch state."""

        await self.write_record("kill_switch", state.to_dict())

    async def read_kill_switch(self) -> KillSwitchState:
        """Return the persisted kill-switch state."""

        return KillSwitchState.from_payload(await self.read_record("kill_switch"))

    async def write_health_snapshot(self, payload: dict[str, Any]) -> None:
        """Persist the latest health snapshot."""

        await self.write_record("health", payload)

    async def read_health_snapshot(self) -> dict[str, Any] | None:
        """Return the latest persisted health snapshot."""

        return await self.read_record("health")

    async def write_metrics_snapshot(self, payload: dict[str, Any]) -> None:
        """Persist the latest metrics snapshot."""

        await self.write_record("metrics", payload)

    async def read_metrics_snapshot(self) -> dict[str, Any] | None:
        """Return the latest persisted metrics snapshot."""

        return await self.read_record("metrics")

    async def write_runtime_snapshot(self, payload: dict[str, Any]) -> None:
        """Persist the latest full runtime snapshot."""

        await self.write_record("runtime_snapshot", payload)

    async def read_runtime_snapshot(self) -> dict[str, Any] | None:
        """Return the latest persisted full runtime snapshot."""

        return await self.read_record("runtime_snapshot")

    async def write_trading_state(self, payload: dict[str, Any]) -> None:
        """Persist the latest lightweight trading-state snapshot."""

        await self.write_record("trading_state", payload)

    async def read_trading_state(self) -> dict[str, Any] | None:
        """Return the latest persisted lightweight trading-state snapshot."""

        return await self.read_record("trading_state")

    async def write_reconciliation(self, payload: dict[str, Any]) -> None:
        """Persist the latest reconciliation status."""

        await self.write_record("reconciliation", payload)

    async def read_reconciliation(self) -> dict[str, Any] | None:
        """Return the latest reconciliation payload."""

        return await self.read_record("reconciliation")

    def _connection_locked(self) -> sqlite3.Connection:
        """Return the SQLite connection, creating schema when required."""

        if self._connection is not None:
            return self._connection

        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_state (
                state_key TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.commit()
        self._connection = connection
        return connection
