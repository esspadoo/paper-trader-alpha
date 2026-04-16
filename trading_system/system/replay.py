"""Event serialization helpers for audit journaling and deterministic replay."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import UUID

from trading_system.core.events import BaseEvent, MarketEvent, NewsEvent, OrderEvent, SignalEvent


def serialize_event_record(event: BaseEvent, *, replay_history_bars: int = 64) -> dict[str, Any]:
    """Return a JSON-serializable audit record for an event."""

    payload = _serialize_value(_normalize_event_payload(event=event, replay_history_bars=replay_history_bars))
    record: dict[str, Any] = {
        "record_type": "event",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "event": {
            "event_type": event.event_type,
            "source": event.source,
            "payload": payload,
            "occurred_at": event.occurred_at.isoformat(),
            "event_id": str(event.event_id),
            "correlation_id": None if event.correlation_id is None else str(event.correlation_id),
        },
    }
    if isinstance(event, MarketEvent):
        record["event"] |= {
            "symbol": event.symbol,
            "venue": event.venue,
            "bid": event.bid,
            "ask": event.ask,
            "last_price": event.last_price,
            "volume": event.volume,
        }
    elif isinstance(event, NewsEvent):
        record["event"] |= {
            "headline": event.headline,
            "symbols": list(event.symbols),
            "body": event.body,
            "urgency": event.urgency,
        }
    elif isinstance(event, SignalEvent):
        record["event"] |= {
            "symbol": event.symbol,
            "strategy_id": event.strategy_id,
            "signal_type": event.signal_type,
            "confidence": event.confidence,
            "target_price": event.target_price,
        }
    elif isinstance(event, OrderEvent):
        record["event"] |= {
            "order_id": event.order_id,
            "symbol": event.symbol,
            "side": event.side,
            "quantity": event.quantity,
            "status": event.status,
            "limit_price": event.limit_price,
        }
    return record


def deserialize_event_record(record: Mapping[str, Any]) -> BaseEvent:
    """Deserialize an event audit record back into an event object."""

    payload = record.get("event", record)
    if not isinstance(payload, Mapping):
        raise ValueError("event record must contain an 'event' mapping")

    event_type = str(payload.get("event_type", "")).strip()
    common: dict[str, Any] = {
        "source": str(payload.get("source", "")).strip(),
        "payload": _deserialize_value(payload.get("payload")),
        "occurred_at": datetime.fromisoformat(str(payload.get("occurred_at"))),
        "event_id": UUID(str(payload.get("event_id"))),
        "correlation_id": None
        if payload.get("correlation_id") in {None, ""}
        else UUID(str(payload.get("correlation_id"))),
    }
    if event_type == "MarketEvent":
        return MarketEvent(
            symbol=str(payload.get("symbol", "")).strip().upper(),
            venue=str(payload.get("venue", "")).strip(),
            bid=_maybe_float(payload.get("bid")),
            ask=_maybe_float(payload.get("ask")),
            last_price=_maybe_float(payload.get("last_price")),
            volume=_maybe_float(payload.get("volume")),
            **common,
        )
    if event_type == "NewsEvent":
        return NewsEvent(
            headline=str(payload.get("headline", "")).strip(),
            symbols=tuple(str(symbol).strip().upper() for symbol in payload.get("symbols", [])),
            body=None if payload.get("body") in {None, ""} else str(payload.get("body")),
            urgency=str(payload.get("urgency", "normal")).strip(),
            **common,
        )
    if event_type == "SignalEvent":
        return SignalEvent(
            symbol=str(payload.get("symbol", "")).strip().upper(),
            strategy_id=str(payload.get("strategy_id", "")).strip(),
            signal_type=str(payload.get("signal_type", "")).strip(),
            confidence=float(payload.get("confidence", 0.0)),
            target_price=_maybe_float(payload.get("target_price")),
            **common,
        )
    if event_type == "OrderEvent":
        return OrderEvent(
            order_id=str(payload.get("order_id", "")).strip(),
            symbol=str(payload.get("symbol", "")).strip().upper(),
            side=str(payload.get("side", "")).strip().upper(),
            quantity=float(payload.get("quantity", 0.0)),
            status=str(payload.get("status", "")).strip(),
            limit_price=_maybe_float(payload.get("limit_price")),
            **common,
        )
    raise ValueError(f"unsupported event_type for replay: {event_type}")


def load_replay_market_events(path: str | Path) -> list[MarketEvent]:
    """Load replayable market events from an audit journal."""

    events: list[MarketEvent] = []
    audit_path = Path(path)
    if not audit_path.exists():
        raise FileNotFoundError(f"audit journal was not found: {audit_path}")
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            event = deserialize_event_record(record)
            if isinstance(event, MarketEvent):
                events.append(event)
    return events


def _normalize_event_payload(*, event: BaseEvent, replay_history_bars: int) -> dict[str, Any]:
    """Return a replay-safe payload for event auditing."""

    payload = dict(event.payload or {})
    if not isinstance(event, MarketEvent):
        return payload

    ohlcv = payload.get("ohlcv")
    if ohlcv is not None:
        try:
            import pandas as pd

            if isinstance(ohlcv, pd.DataFrame):
                payload["ohlcv"] = ohlcv.tail(replay_history_bars).copy(deep=True)
        except ImportError:
            payload.pop("ohlcv", None)
    return payload


def _serialize_value(value: Any) -> Any:
    """Recursively serialize an object into JSON-safe data."""

    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, UUID):
        return {"__type__": "uuid", "value": str(value)}
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}

    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return {
                "__type__": "dataframe",
                "columns": [_serialize_value(column) for column in value.columns.tolist()],
                "index": [_serialize_value(index) for index in value.index.tolist()],
                "data": [[_serialize_value(cell) for cell in row] for row in value.to_numpy().tolist()],
            }
    except ImportError:
        pass

    return value


def _deserialize_value(value: Any) -> Any:
    """Recursively restore JSON-safe data into runtime objects."""

    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    marker = value.get("__type__")
    if marker == "datetime":
        return datetime.fromisoformat(str(value["value"]))
    if marker == "uuid":
        return UUID(str(value["value"]))
    if marker == "path":
        return Path(str(value["value"]))
    if marker == "dataframe":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required to deserialize replay market events") from exc
        columns = [_deserialize_value(column) for column in value["columns"]]
        index = [_deserialize_value(index_value) for index_value in value["index"]]
        data = [[_deserialize_value(cell) for cell in row] for row in value["data"]]
        frame = pd.DataFrame(data=data, columns=columns)
        frame.index = index
        return frame
    return {str(key): _deserialize_value(inner) for key, inner in value.items()}


def _maybe_float(value: Any) -> float | None:
    """Return a nullable float from a serialized value."""

    if value is None:
        return None
    return float(value)
