"""Agent-coordinated trading logic for the intraday backtester."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from trading_system.agents import CriticAgent, DecisionAgent, RiskAgent
from trading_system.agents._validation import normalize_market_signal, normalize_news_analysis
from trading_system.agents.base import BaseAgent
from trading_system.backtest.execution import BacktestExecutionSimulator
from trading_system.core.engine import Engine
from trading_system.core.events import BaseEvent, MarketEvent, NewsEvent, OrderEvent, SignalEvent


class AgentBacktestCoordinator:
    """Coordinate market, news, decision, critic, and risk agents in a backtest."""

    def __init__(
        self,
        *,
        execution: BacktestExecutionSimulator,
        market_agent: BaseAgent | None = None,
        news_agent: BaseAgent | None = None,
        decision_agent: DecisionAgent | None = None,
        critic_agent: CriticAgent | None = None,
        risk_agent: RiskAgent | None = None,
        strategy_id: str = "intraday-backtest",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the coordinator with the agent pipeline."""

        self._execution = execution
        self._market_agent = market_agent
        self._news_agent = news_agent
        self._decision_agent = decision_agent
        self._critic_agent = critic_agent
        self._risk_agent = risk_agent
        self._strategy_id = strategy_id
        self._logger = logger or logging.getLogger(__name__)
        self._engine: Engine | None = None
        self._latest_news: dict[str, dict[str, float | str]] = {}
        self._us_eastern = ZoneInfo("America/New_York")

    def bind_engine(self, engine: Engine) -> None:
        """Bind the coordinator to an engine for event publication."""

        self._engine = engine

    async def start(self) -> None:
        """Start the configured agents."""

        for agent in self._agents():
            await agent.start()

    async def stop(self) -> None:
        """Stop the configured agents."""

        for agent in reversed(list(self._agents())):
            await agent.stop()

    async def on_event(self, event: BaseEvent) -> None:
        """Consume market and news events from the backtest engine."""

        if isinstance(event, NewsEvent):
            await self._handle_news(event)
        elif isinstance(event, MarketEvent):
            await self._handle_market(event)

    async def _handle_news(self, event: NewsEvent) -> None:
        """Update the latest normalized news state for the affected symbols."""

        analysis = None
        if self._news_agent is not None:
            await self._news_agent.on_event(event)
            snapshot = await self._news_agent.snapshot_state()
            candidate = snapshot.get("last_output")
            if isinstance(candidate, Mapping):
                analysis = normalize_news_analysis(candidate)

        if analysis is None and "sentiment" in dict(event.payload or {}):
            analysis = normalize_news_analysis(dict(event.payload or {}))

        if analysis is None:
            return

        targets = tuple(symbol.strip().upper() for symbol in event.symbols if symbol.strip()) or ("__GLOBAL__",)
        for symbol in targets:
            self._latest_news[symbol] = dict(analysis)

    async def _handle_market(self, event: MarketEvent) -> None:
        """Run the agent stack on a market event and emit orders when warranted."""

        if self._engine is None:
            raise RuntimeError("AgentBacktestCoordinator must be bound to an engine before use")

        symbol = event.symbol.strip().upper()
        payload = dict(event.payload or {})
        ohlcv = payload.get("ohlcv")
        if ohlcv is None:
            return

        market_signal = await self._resolve_market_signal(event)
        if market_signal is None:
            return
        market_signal = self._augment_market_signal(event=event, market_signal=market_signal)

        await self._engine.publish(
            SignalEvent(
                source=self._market_agent.agent_id if self._market_agent is not None else "market-signal",
                symbol=symbol,
                strategy_id=self._strategy_id,
                signal_type="MARKET_SIGNAL",
                confidence=float(market_signal["confidence"]),
                payload=dict(market_signal),
                occurred_at=event.occurred_at,
                correlation_id=event.event_id,
            )
        )

        news_analysis = self._latest_news.get(symbol) or self._latest_news.get("__GLOBAL__") or {
            "sentiment": 0.0,
            "impact": 0.0,
            "event_type": "none",
            "summary": "no recent news",
        }

        if self._decision_agent is None:
            raise ValueError("decision_agent is required for intraday backtesting")

        decision = await self._decision_agent.decide(
            market_signal=market_signal,
            news_analysis=news_analysis,
            symbol=symbol,
            occurred_at=event.occurred_at,
        )
        await self._engine.publish(
            SignalEvent(
                source=self._decision_agent.agent_id,
                symbol=symbol,
                strategy_id=self._strategy_id,
                signal_type=str(decision["action"]),
                confidence=float(market_signal["confidence"]),
                payload=dict(decision),
                occurred_at=event.occurred_at,
                correlation_id=event.event_id,
            )
        )

        if self._critic_agent is not None:
            review = await self._critic_agent.review(
                decision,
                market_signal=market_signal,
                news_analysis=news_analysis,
                symbol=symbol,
                occurred_at=event.occurred_at,
            )
            if not bool(review["approved"]):
                self._logger.info("critic rejected %s on %s: %s", symbol, event.occurred_at.isoformat(), review["reason"])
                return

        if self._risk_agent is None:
            raise ValueError("risk_agent is required for intraday backtesting")

        account_state = self._execution.portfolio.account_state(timestamp=event.occurred_at)
        entry_price = float(event.last_price if event.last_price is not None else payload["bar"]["close"])
        try:
            risk = await self._risk_agent.assess(
                decision,
                market_signal=market_signal,
                account_state=account_state,
                ohlcv=ohlcv,
                symbol=symbol,
                entry_price=entry_price,
            )
        except ValueError as exc:
            self._logger.info("risk assessment skipped for %s on %s: %s", symbol, event.occurred_at.isoformat(), exc)
            return

        final_action = str(risk["final_action"]).upper()
        if final_action == "HOLD":
            return

        existing_position = self._execution.portfolio.position_for(symbol)
        if self._execution.has_active_order(symbol):
            return

        order_quantity = float(risk["position_size"])
        order_side = final_action
        order_reason = str(decision["reasoning"])
        stop_loss = float(risk["stop_loss"])

        if existing_position is not None:
            current_side = "BUY" if existing_position.quantity > 0.0 else "SELL"
            if current_side == final_action:
                return
            order_quantity = abs(existing_position.quantity)
            order_side = "SELL" if existing_position.quantity > 0.0 else "BUY"
            stop_loss = 0.0
            order_reason = f"close existing {current_side} position before reversal"

        if order_quantity <= 0.0:
            return

        await self._engine.publish(
            OrderEvent(
                source=self._risk_agent.agent_id,
                order_id=f"{symbol}-{event.occurred_at.strftime('%Y%m%d%H%M%S%f')}",
                symbol=symbol,
                side=order_side,
                quantity=order_quantity,
                status="NEW",
                limit_price=entry_price,
                payload={
                    "strategy_id": self._strategy_id,
                    "reason": order_reason,
                    "stop_loss": stop_loss if stop_loss > 0.0 else None,
                },
                occurred_at=event.occurred_at,
                correlation_id=event.event_id,
            )
        )

    async def _resolve_market_signal(self, event: MarketEvent) -> dict[str, Any] | None:
        """Return a normalized market signal from the configured source."""

        if self._market_agent is not None:
            await self._market_agent.on_event(event)
            snapshot = await self._market_agent.snapshot_state()
            candidate = snapshot.get("last_output")
            if isinstance(candidate, Mapping):
                return normalize_market_signal(candidate)

        payload = dict(event.payload or {})
        if "market_signal" in payload and isinstance(payload["market_signal"], Mapping):
            return normalize_market_signal(payload["market_signal"])
        if "signal" in payload:
            return normalize_market_signal(payload)
        return None

    def _agents(self) -> tuple[BaseAgent, ...]:
        """Return the configured agents in pipeline order."""

        return tuple(agent for agent in (self._market_agent, self._news_agent, self._decision_agent, self._critic_agent, self._risk_agent) if agent is not None)

    def _augment_market_signal(self, *, event: MarketEvent, market_signal: Mapping[str, Any]) -> dict[str, Any]:
        """Attach lightweight session and spread context to the market signal."""

        enriched = dict(market_signal)
        features = dict(market_signal["features"])
        features.setdefault("session_phase", self._session_phase(event.occurred_at))
        if event.bid is not None and event.ask is not None:
            bid = float(event.bid)
            ask = float(event.ask)
            mid = (bid + ask) / 2.0
            if bid > 0.0 and ask > 0.0 and ask >= bid and mid > 0.0:
                features.setdefault("spread_bps", ((ask - bid) / mid) * 10_000.0)
        if "dollar_volume" not in features and event.last_price is not None and event.volume is not None:
            features["dollar_volume"] = float(event.last_price) * float(event.volume)
        enriched["features"] = features
        if event.symbol:
            enriched["symbol"] = event.symbol.strip().upper()
        return enriched

    def _session_phase(self, timestamp: datetime) -> str:
        """Return a coarse US cash-session phase."""

        eastern = timestamp.astimezone(self._us_eastern)
        clock = eastern.time()
        if clock < datetime(2000, 1, 1, 9, 30).time() or clock >= datetime(2000, 1, 1, 16, 0).time():
            return "outside_rth"
        if clock < datetime(2000, 1, 1, 10, 0).time():
            return "open"
        if clock < datetime(2000, 1, 1, 12, 0).time():
            return "morning"
        if clock < datetime(2000, 1, 1, 14, 0).time():
            return "lunch"
        if clock < datetime(2000, 1, 1, 15, 30).time():
            return "afternoon"
        return "close"
