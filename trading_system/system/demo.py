"""Deterministic local demo components for the integrated trading system."""

from __future__ import annotations

import json
import math
from datetime import datetime

from trading_system.core.events import MarketEvent
from trading_system.models import LocalLLMBackend
from trading_system.models.dependencies import require_pandas
from trading_system.system.config import MarketConfig


class DemoLocalLLMBackend(LocalLLMBackend):
    """Deterministic local backend for runnable offline demos and tests."""

    POSITIVE_KEYWORDS = ("beats", "upgrade", "growth", "strong", "partnership", "launch", "demand")
    NEGATIVE_KEYWORDS = ("miss", "downgrade", "weak", "lawsuit", "recall", "cut", "investigation")

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""

        return "demo"

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        timeout_seconds: float,
    ) -> str:
        """Return deterministic strict JSON derived from keyword heuristics."""

        user_message = next(
            (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
            prompt,
        )
        relevant_lines = []
        for line in user_message.splitlines():
            lowered = line.lower()
            if lowered.startswith("title:") or lowered.startswith("summary:") or lowered.startswith("content:"):
                relevant_lines.append(line.split(":", 1)[1].strip())
        content = " ".join(relevant_lines).lower()
        sentiment = 0.0
        impact = 0.25
        event_type = "other"
        summary = "Neutral market color."

        if any(keyword in content for keyword in self.POSITIVE_KEYWORDS):
            sentiment = 0.65
            impact = 0.70
            event_type = "product_launch" if "launch" in content else "partnership"
            summary = "Constructive catalyst with positive sentiment."
        elif any(keyword in content for keyword in self.NEGATIVE_KEYWORDS):
            sentiment = -0.70
            impact = 0.75
            event_type = "litigation" if "lawsuit" in content or "investigation" in content else "guidance"
            summary = "Negative catalyst with elevated downside risk."

        return json.dumps(
            {
                "sentiment": sentiment,
                "impact": impact,
                "event_type": event_type,
                "summary": summary,
            },
            separators=(",", ":"),
        )


def build_demo_market_frame(config: MarketConfig) -> "object":
    """Build a deterministic intraday OHLCV frame for training and streaming."""

    pd = require_pandas()

    total_bars = config.training_bars + config.stream_bars
    timestamps = pd.date_range(
        start=pd.Timestamp(config.start_timestamp),
        periods=total_bars,
        freq=f"{config.bar_interval_minutes}min",
        tz="UTC",
    )
    records: list[dict[str, float | str | datetime]] = []
    previous_close = config.base_price
    for index, timestamp in enumerate(timestamps):
        drift = config.trend_per_bar * index
        cycle = math.sin(index / 5.0) * config.amplitude
        impulse = 1.4 if config.training_bars <= index < config.training_bars + 8 else 0.0
        close = config.base_price + drift + cycle + impulse
        open_price = previous_close
        high = max(open_price, close) + 0.35 + (0.05 * math.cos(index / 3.0))
        low = min(open_price, close) - 0.35 - (0.05 * math.sin(index / 4.0))
        volume = config.base_volume + (120_000.0 * math.sin(index / 6.0)) + (40_000.0 * (index % 5))
        records.append(
            {
                "symbol": config.symbol,
                "timestamp": timestamp,
                "open": round(open_price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": round(max(volume, 250_000.0), 2),
            }
        )
        previous_close = close

    return pd.DataFrame.from_records(records)


def build_market_events(frame: "object", *, warmup_bars: int) -> list[MarketEvent]:
    """Convert a market frame into sequenced MarketEvent objects with rolling history."""

    pd = require_pandas()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    if warmup_bars < 1 or warmup_bars >= len(frame):
        raise ValueError("warmup_bars must be at least 1 and smaller than the frame length")

    working = frame.copy(deep=True)
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working = working.sort_values("timestamp").reset_index(drop=True)

    events: list[MarketEvent] = []
    for index in range(warmup_bars, len(working)):
        history = working.iloc[: index + 1].copy(deep=True)
        symbol = str(history["symbol"].iloc[-1]).strip().upper()
        history = history.set_index("timestamp").loc[:, ["open", "high", "low", "close", "volume"]]
        row = working.iloc[index]
        events.append(
            MarketEvent(
                source="demo-market-feed",
                symbol=symbol,
                venue="SIM",
                bid=float(row["close"]),
                ask=float(row["close"]),
                last_price=float(row["close"]),
                volume=float(row["volume"]),
                payload={
                    "bar": {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    },
                    "ohlcv": history,
                },
                occurred_at=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            )
        )
    return events
