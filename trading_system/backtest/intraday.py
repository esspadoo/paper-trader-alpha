"""Event-driven intraday backtester integrated with the agent pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from trading_system.agents import CriticAgent, DecisionAgent, RiskAgent
from trading_system.agents.base import BaseAgent
from trading_system.backtest.bus import SequentialEventBus
from trading_system.backtest.execution import BacktestExecutionSimulator
from trading_system.backtest.portfolio import PortfolioLedger
from trading_system.backtest.strategy import AgentBacktestCoordinator
from trading_system.backtest.types import BacktestConfig, BacktestResult
from trading_system.core.engine import Engine
from trading_system.core.events import BaseEvent, MarketEvent, NewsEvent, OrderEvent
from trading_system.models.dependencies import require_pandas

if TYPE_CHECKING:
    import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class IntradayBacktester:
    """Run a deterministic event-driven intraday simulation over historical bars."""

    def __init__(
        self,
        *,
        config: BacktestConfig | None = None,
        market_agent: BaseAgent | None = None,
        news_agent: BaseAgent | None = None,
        decision_agent: DecisionAgent | None = None,
        critic_agent: CriticAgent | None = None,
        risk_agent: RiskAgent | None = None,
        strategy_id: str = "intraday-backtest",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the backtester and its agent-driven execution pipeline."""

        self._config = config or BacktestConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._portfolio = PortfolioLedger(initial_capital=self._config.initial_capital)
        self._execution = BacktestExecutionSimulator(config=self._config, portfolio=self._portfolio, logger=self._logger)
        self._strategy = AgentBacktestCoordinator(
            execution=self._execution,
            market_agent=market_agent,
            news_agent=news_agent,
            decision_agent=decision_agent,
            critic_agent=critic_agent,
            risk_agent=risk_agent,
            strategy_id=strategy_id,
            logger=self._logger,
        )
        self._strategy_id = strategy_id

    @property
    def portfolio(self) -> PortfolioLedger:
        """Return the portfolio ledger used by the backtester."""

        return self._portfolio

    async def run_async(
        self,
        market_data: "pd.DataFrame",
        *,
        news_events: Sequence[NewsEvent | Mapping[str, Any]] | "pd.DataFrame" | None = None,
    ) -> BacktestResult:
        """Run the intraday backtest asynchronously and return the result."""

        pd = require_pandas()
        normalized_market = self._normalize_market_data(market_data, pd)
        normalized_news = self._normalize_news_events(news_events, pd)
        engine = Engine(
            event_bus=SequentialEventBus(
                worker_count=1,
                queue_maxsize=100_000,
                raise_on_handler_error=True,
                logger=self._logger,
            ),
            logger=self._logger,
        )
        self._execution.bind_engine(engine)
        self._strategy.bind_engine(engine)

        observed_events: list[str] = []

        async def audit(event: BaseEvent) -> None:
            observed_events.append(self._event_label(event))

        async with engine:
            await engine.subscribe(MarketEvent, self._execution.on_event)
            await engine.subscribe(OrderEvent, self._execution.on_event)
            await engine.subscribe(NewsEvent, self._strategy.on_event)
            await engine.subscribe(MarketEvent, self._strategy.on_event)
            await engine.subscribe(BaseEvent, audit)

            await self._strategy.start()
            try:
                for item in self._build_timeline(normalized_market, normalized_news, pd):
                    await engine.publish(item)
                    await engine.wait_until_idle()

                if not normalized_market.empty:
                    last_timestamp = normalized_market.index.get_level_values("timestamp")[-1]
                    await self._execution.flatten_positions(timestamp=last_timestamp)
                    current_marks = self._portfolio.current_marks()
                    if current_marks:
                        self._portfolio.mark_to_market(last_timestamp, current_marks)
                    await engine.wait_until_idle()
            finally:
                await self._strategy.stop()

        metrics = self._portfolio.compute_metrics(config=self._config)
        metadata = {
            "strategy_id": self._strategy_id,
            "initial_capital": self._config.initial_capital,
            "final_equity": self._portfolio.equity(prices=self._portfolio.current_marks()) if self._portfolio.current_marks() else self._config.initial_capital,
            "realized_pnl": self._portfolio.realized_pnl,
            "open_positions": {symbol: position.quantity for symbol, position in self._portfolio.positions().items()},
        }
        return BacktestResult(
            metrics=metrics,
            equity_curve=self._portfolio.equity_curve,
            fills=self._portfolio.fills,
            trades=self._portfolio.trades,
            observed_events=tuple(observed_events),
            metadata=metadata,
        )

    def run(
        self,
        market_data: "pd.DataFrame",
        *,
        news_events: Sequence[NewsEvent | Mapping[str, Any]] | "pd.DataFrame" | None = None,
    ) -> BacktestResult:
        """Run the intraday backtest synchronously and return the result."""

        return asyncio.run(self.run_async(market_data, news_events=news_events))

    def _normalize_market_data(self, market_data: "pd.DataFrame", pd: Any) -> "pd.DataFrame":
        """Normalize OHLCV market data to a sorted symbol-timestamp MultiIndex."""

        if not isinstance(market_data, pd.DataFrame):
            raise ValueError("market_data must be a pandas DataFrame")
        if market_data.empty:
            raise ValueError("market_data cannot be empty")

        frame = market_data.copy(deep=True)
        frame.columns = [str(column).strip().lower() for column in frame.columns]

        if isinstance(frame.index, pd.MultiIndex):
            if frame.index.nlevels != 2:
                raise ValueError("MultiIndex market_data must contain exactly symbol and timestamp levels")
            frame.index = frame.index.set_names(["symbol", "timestamp"])
        elif {"symbol", "timestamp"} <= set(frame.columns):
            frame["symbol"] = frame["symbol"].map(lambda value: str(value).strip().upper())
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index(["symbol", "timestamp"])
        else:
            timestamps = pd.to_datetime(frame.index, utc=True)
            symbol = frame["symbol"].map(lambda value: str(value).strip().upper()) if "symbol" in frame.columns else "__DEFAULT__"
            frame = frame.reset_index(drop=True)
            frame["symbol"] = symbol
            frame["timestamp"] = timestamps
            frame = frame.set_index(["symbol", "timestamp"])

        symbols = frame.index.get_level_values("symbol").map(lambda value: str(value).strip().upper())
        timestamps = pd.to_datetime(frame.index.get_level_values("timestamp"), utc=True)
        frame.index = pd.MultiIndex.from_arrays([symbols, timestamps], names=["symbol", "timestamp"])

        missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"market_data is missing required OHLCV columns: {missing}")

        return frame.sort_index()

    def _normalize_news_events(
        self,
        news_events: Sequence[NewsEvent | Mapping[str, Any]] | "pd.DataFrame" | None,
        pd: Any,
    ) -> list[NewsEvent]:
        """Return normalized news events sorted by occurrence time."""

        if news_events is None:
            return []

        if isinstance(news_events, pd.DataFrame):
            records: Iterable[Any] = news_events.to_dict(orient="records")
        else:
            records = news_events

        normalized: list[NewsEvent] = []
        for item in records:
            if isinstance(item, NewsEvent):
                normalized.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError("news_events must contain NewsEvent instances or mappings")

            occurred_at = item.get("occurred_at")
            if occurred_at is None:
                raise ValueError("news event mappings must include occurred_at")

            symbols = item.get("symbols", ())
            if isinstance(symbols, str):
                symbols = tuple(symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip())
            else:
                symbols = tuple(str(symbol).strip().upper() for symbol in symbols)

            normalized.append(
                NewsEvent(
                    source=str(item.get("source", "historical-news")).strip(),
                    headline=str(item.get("headline", "")).strip(),
                    symbols=symbols,
                    body=str(item.get("body", "") or ""),
                    urgency=str(item.get("urgency", "normal")).strip() or "normal",
                    payload=dict(item.get("payload") or {}),
                    occurred_at=self._to_utc_datetime(occurred_at, pd),
                )
            )

        return sorted(normalized, key=lambda event: event.occurred_at)

    def _build_timeline(
        self,
        market_data: "pd.DataFrame",
        news_events: Sequence[NewsEvent],
        pd: Any,
    ) -> list[BaseEvent]:
        """Return a merged market/news event timeline in timestamp order."""

        symbol_frames = {
            symbol: market_data.xs(symbol, level="symbol").sort_index()
            for symbol in market_data.index.get_level_values("symbol").unique()
        }
        symbol_offsets: dict[str, int] = {symbol: 0 for symbol in symbol_frames}

        market_events: list[BaseEvent] = []
        for symbol, timestamp in market_data.index:
            symbol_frame = symbol_frames[str(symbol)]
            offset = symbol_offsets[str(symbol)]
            history = symbol_frame.iloc[: offset + 1].loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy(deep=True)
            row = symbol_frame.iloc[offset]
            symbol_offsets[str(symbol)] += 1

            extras = {
                column: row[column]
                for column in symbol_frame.columns
                if column not in REQUIRED_OHLCV_COLUMNS
            }
            payload: dict[str, Any] = {
                "bar": {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                },
                "ohlcv": history,
            }
            if extras:
                payload["extras"] = dict(extras)

            market_signal = self._market_signal_from_extras(extras)
            if market_signal is not None:
                payload["market_signal"] = market_signal

            market_events.append(
                MarketEvent(
                    source="historical-market",
                    symbol=str(symbol),
                    venue="BACKTEST",
                    last_price=float(row["close"]),
                    bid=float(row["close"]),
                    ask=float(row["close"]),
                    volume=float(row["volume"]),
                    payload=payload,
                    occurred_at=self._to_utc_datetime(timestamp, pd),
                )
            )

        combined: list[tuple[datetime, int, BaseEvent]] = []
        combined.extend((event.occurred_at, 0, event) for event in news_events)
        combined.extend((event.occurred_at, 1, event) for event in market_events)
        combined.sort(key=lambda item: (item[0], item[1]))
        return [event for _, _, event in combined]

    def _market_signal_from_extras(self, extras: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return a normalized market-signal payload from extra row columns when present."""

        if "signal" not in extras:
            return None

        features = {
            key: float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else value
            for key, value in extras.items()
            if key not in {"signal", "confidence"}
        }
        return {
            "signal": float(extras["signal"]),
            "confidence": float(extras.get("confidence", 1.0)),
            "features": features,
        }

    def _event_label(self, event: BaseEvent) -> str:
        """Return a stable string label for an observed event."""

        if isinstance(event, OrderEvent):
            return f"{event.event_type}:{event.status}"
        return event.event_type

    def _to_utc_datetime(self, value: Any, pd: Any) -> datetime:
        """Return a timezone-aware UTC datetime."""

        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.to_pydatetime()
