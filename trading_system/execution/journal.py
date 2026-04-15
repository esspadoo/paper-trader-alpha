"""Persistent order journal used for idempotency and restart recovery."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class OrderJournalRecord:
    """Persisted order state used for replay-safe submission."""

    client_order_key: str
    broker_order_id: str | None
    symbol: str
    side: str
    quantity: float
    limit_price: float
    status: str
    order_ref: str | None
    updated_at: str
    snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record payload."""

        return dict(asdict(self))

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "OrderJournalRecord":
        """Build a journal record from a SQLite row."""

        return cls(
            client_order_key=str(row["client_order_key"]),
            broker_order_id=None if row["broker_order_id"] in {None, ""} else str(row["broker_order_id"]),
            symbol=str(row["symbol"]).strip().upper(),
            side=str(row["side"]).strip().upper(),
            quantity=float(row["quantity"]),
            limit_price=float(row["limit_price"]),
            status=str(row["status"]).strip(),
            order_ref=None if row["order_ref"] in {None, ""} else str(row["order_ref"]).strip(),
            updated_at=str(row["updated_at"]),
            snapshot=json.loads(str(row["snapshot_json"])),
        )


class OrderJournal:
    """SQLite-backed journal for order idempotency and recovery."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the journal path and in-process lock."""

        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._connection: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        """Return the journal storage path."""

        return self._path

    async def get(self, client_order_key: str) -> OrderJournalRecord | None:
        """Return the stored record for a client order key when available."""

        normalized_key = str(client_order_key).strip()
        if not normalized_key:
            raise ValueError("client_order_key must be a non-empty string")

        async with self._lock:
            connection = self._connection_locked()
            row = connection.execute(
                """
                SELECT client_order_key, broker_order_id, symbol, side, quantity, limit_price, status, order_ref, updated_at, snapshot_json
                FROM order_journal
                WHERE client_order_key = ?
                """,
                (normalized_key,),
            ).fetchone()
        return None if row is None else OrderJournalRecord.from_row(row)

    async def upsert(
        self,
        *,
        client_order_key: str,
        broker_order_id: str | None,
        snapshot: dict[str, Any],
        order_ref: str | None = None,
    ) -> OrderJournalRecord:
        """Insert or replace a journal record for the provided client order key."""

        normalized_key = str(client_order_key).strip()
        if not normalized_key:
            raise ValueError("client_order_key must be a non-empty string")

        normalized_snapshot = dict(snapshot)
        symbol = str(normalized_snapshot.get("symbol", "")).strip().upper()
        side = str(normalized_snapshot.get("side", "")).strip().upper()
        if not symbol:
            raise ValueError("order journal snapshots must include a symbol")
        if side not in {"BUY", "SELL"}:
            raise ValueError("order journal snapshots must include a BUY or SELL side")

        record = OrderJournalRecord(
            client_order_key=normalized_key,
            broker_order_id=None if broker_order_id is None else str(broker_order_id),
            symbol=symbol,
            side=side,
            quantity=float(normalized_snapshot.get("quantity", 0.0)),
            limit_price=float(normalized_snapshot.get("limit_price", 0.0)),
            status=str(normalized_snapshot.get("status", "")).strip() or "UNKNOWN",
            order_ref=None if order_ref in {None, ""} else str(order_ref).strip(),
            updated_at=_utc_now().isoformat(),
            snapshot=normalized_snapshot,
        )

        async with self._lock:
            connection = self._connection_locked()
            connection.execute(
                """
                INSERT INTO order_journal (
                    client_order_key,
                    broker_order_id,
                    symbol,
                    side,
                    quantity,
                    limit_price,
                    status,
                    order_ref,
                    updated_at,
                    snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_key) DO UPDATE SET
                    broker_order_id = excluded.broker_order_id,
                    symbol = excluded.symbol,
                    side = excluded.side,
                    quantity = excluded.quantity,
                    limit_price = excluded.limit_price,
                    status = excluded.status,
                    order_ref = excluded.order_ref,
                    updated_at = excluded.updated_at,
                    snapshot_json = excluded.snapshot_json
                """,
                (
                    record.client_order_key,
                    record.broker_order_id,
                    record.symbol,
                    record.side,
                    record.quantity,
                    record.limit_price,
                    record.status,
                    record.order_ref,
                    record.updated_at,
                    json.dumps(record.snapshot, sort_keys=True),
                ),
            )
            connection.commit()
        return record

    async def list_records(self) -> tuple[OrderJournalRecord, ...]:
        """Return all persisted order records."""

        async with self._lock:
            connection = self._connection_locked()
            rows = connection.execute(
                """
                SELECT client_order_key, broker_order_id, symbol, side, quantity, limit_price, status, order_ref, updated_at, snapshot_json
                FROM order_journal
                ORDER BY updated_at ASC
                """
            ).fetchall()
        return tuple(OrderJournalRecord.from_row(row) for row in rows)

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
            CREATE TABLE IF NOT EXISTS order_journal (
                client_order_key TEXT PRIMARY KEY,
                broker_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                limit_price REAL NOT NULL,
                status TEXT NOT NULL,
                order_ref TEXT,
                updated_at TEXT NOT NULL,
                snapshot_json TEXT NOT NULL
            )
            """
        )
        connection.commit()
        self._connection = connection
        return connection
