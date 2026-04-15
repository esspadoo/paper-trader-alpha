"""Integrated async runtime for the multi-agent trading workflow."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from time import perf_counter_ns
from typing import Any

from trading_system.agents import BaseAgent, CriticAgent, DecisionAgent, NewsAgent, RiskAgent
from trading_system.agents._validation import normalize_market_signal, normalize_news_analysis
from trading_system.backtest.bus import SequentialEventBus
from trading_system.core.engine import Engine
from trading_system.core.events import BaseEvent, MarketEvent, NewsEvent, OrderEvent, SignalEvent
from trading_system.execution import BrokerInterface, OrderManager
from trading_system.infra import LatencyRecorder, LatencyReport
from trading_system.system.config import ExecutionConfig, RuntimeConfig

NEUTRAL_NEWS_ANALYSIS = {
    "sentiment": 0.0,
    "impact": 0.0,
    "event_type": "none",
    "summary": "No material news available.",
}


@dataclass(frozen=True, slots=True)
class _NewsContext:
    """Latest analyzed news payload with publication metadata."""

    analysis: dict[str, float | str]
    published_at: datetime


class ExecutionService:
    """Submit new orders to the broker and keep account marks updated."""

    def __init__(
        self,
        *,
        broker: BrokerInterface,
        order_manager: OrderManager,
        config: ExecutionConfig,
        logger: logging.Logger,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        """Initialize the execution service."""

        self._broker = broker
        self._order_manager = order_manager
        self._config = config
        self._logger = logger
        self._latency_recorder = latency_recorder
        self._engine: Engine | None = None
        self._executions: list[dict[str, Any]] = []

    @property
    def executions(self) -> tuple[dict[str, Any], ...]:
        """Return normalized execution snapshots."""

        return tuple(dict(item) for item in self._executions)

    def bind_engine(self, engine: Engine) -> None:
        """Bind the service to the system engine."""

        self._engine = engine

    async def on_market_event(self, event: MarketEvent) -> None:
        """Refresh account marks and stale orders on each market update."""

        update_price = getattr(self._broker, "update_market_price", None)
        if callable(update_price) and event.last_price is not None:
            await update_price(event.symbol, float(event.last_price))

        cancelled = await self._order_manager.cancel_stale_orders()
        if cancelled:
            self._logger.warning("cancelled stale orders: %s", ",".join(cancelled))

    async def on_order_event(self, event: OrderEvent) -> None:
        """Submit new strategy orders and publish execution updates."""

        if event.status.strip().upper() != "NEW":
            return
        if self._engine is None:
            raise RuntimeError("ExecutionService must be bound to an engine before use")

        self._logger.info(
            "execution step symbol=%s side=%s quantity=%.4f limit_price=%.4f",
            event.symbol,
            event.side,
            event.quantity,
            float(event.limit_price or 0.0),
        )
        started_at_ns = perf_counter_ns()
        snapshot = await self._order_manager.submit_limit_order(
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            limit_price=float(event.limit_price or 0.0),
            tif=self._config.tif,
            exchange=self._config.exchange,
            currency=self._config.currency,
            smart_routing=self._config.smart_routing,
            outside_rth=self._config.outside_rth,
            order_ref=str(event.payload.get("order_ref", "")) or None,
            account=self._config.account,
            client_order_key=str(event.payload.get("client_order_key", "")) or None,
        )
        if self._latency_recorder is not None:
            self._latency_recorder.record_ns("order_submission_path", perf_counter_ns() - started_at_ns)
            signal_started_ns = event.payload.get("signal_path_started_ns")
            if isinstance(signal_started_ns, int):
                self._latency_recorder.record_ns(
                    "signal_to_order_latency",
                    max(perf_counter_ns() - signal_started_ns, 0),
                )
        normalized = dict(snapshot)
        if event.payload.get("client_order_key") is not None:
            normalized["client_order_key"] = str(event.payload["client_order_key"])
        self._executions.append(normalized)
        await self._engine.publish(
            OrderEvent(
                source="execution-service",
                order_id=str(normalized["order_id"]),
                symbol=event.symbol,
                side=event.side,
                quantity=float(normalized.get("filled_quantity", event.quantity)),
                status=str(normalized.get("status", "Submitted")).upper(),
                limit_price=float(normalized.get("average_fill_price", event.limit_price or 0.0)),
                payload={
                    "broker_snapshot": normalized,
                    "reason": str(event.payload.get("reason", "")),
                    "client_order_key": str(event.payload.get("client_order_key", "")) or None,
                },
                occurred_at=event.occurred_at,
                correlation_id=event.correlation_id,
            )
        )
        self._logger.info(
            "order executed broker_order_id=%s status=%s filled_quantity=%.4f",
            normalized["order_id"],
            normalized.get("status", "UNKNOWN"),
            float(normalized.get("filled_quantity", 0.0)),
        )


class IntegratedTradingSystem:
    """Coordinate the async event-driven trading workflow end to end."""

    def __init__(
        self,
        *,
        market_agent: BaseAgent,
        news_agent: NewsAgent,
        decision_agent: DecisionAgent,
        critic_agent: CriticAgent,
        risk_agent: RiskAgent,
        broker: BrokerInterface,
        order_manager: OrderManager,
        execution_config: ExecutionConfig,
        runtime_config: RuntimeConfig | None = None,
        news_max_age_seconds: float = 14_400.0,
        strategy_id: str = "integrated-intraday",
        logger: logging.Logger | None = None,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        """Initialize the integrated runtime and engine."""

        self._logger = logger or logging.getLogger(__name__)
        self._market_agent = market_agent
        self._news_agent = news_agent
        self._decision_agent = decision_agent
        self._critic_agent = critic_agent
        self._risk_agent = risk_agent
        self._broker = broker
        self._order_manager = order_manager
        self._latency_recorder = latency_recorder
        self._execution_service = ExecutionService(
            broker=broker,
            order_manager=order_manager,
            config=execution_config,
            logger=self._logger,
            latency_recorder=self._latency_recorder,
        )
        self._strategy_id = strategy_id
        self._runtime_config = runtime_config or RuntimeConfig(strategy_id=strategy_id)
        self._news_max_age_seconds = float(news_max_age_seconds)
        if self._news_max_age_seconds <= 0.0:
            raise ValueError("news_max_age_seconds must be greater than 0")
        self._engine = Engine(
            event_bus=SequentialEventBus(
                worker_count=1,
                queue_maxsize=10_000,
                raise_on_handler_error=True,
                logger=self._logger,
                latency_recorder=self._latency_recorder,
            ),
            logger=self._logger,
        )
        self._execution_service.bind_engine(self._engine)
        self._latest_news: dict[str, _NewsContext] = {}
        self._deferred_articles: dict[str, Any] = {}
        self._suppressed_article_fingerprints: set[str] = set()
        self._pending_news_refresh_at: datetime | None = None
        self._pending_news_correlation_id: str | None = None
        self._news_refresh_task: asyncio.Task[None] | None = None
        self._daily_start_capital: dict[date, float] = {}
        self._observed_events: list[str] = []
        self._started = False

    @property
    def executions(self) -> tuple[dict[str, Any], ...]:
        """Return execution snapshots recorded by the runtime."""

        return self._execution_service.executions

    @property
    def latency_report(self) -> LatencyReport | None:
        """Return a summarized latency report when profiling is enabled."""

        if self._latency_recorder is None:
            return None
        return self._latency_recorder.report()

    async def start(self) -> None:
        """Start the engine, broker, and all agents."""

        if self._started:
            return
        await self._broker.connect()
        if self._runtime_config.reconcile_on_start:
            await self._order_manager.reconcile()
        for agent in self._agents():
            await agent.start()
        await self._engine.start()
        await self._engine.subscribe(MarketEvent, self._execution_service.on_market_event)
        await self._engine.subscribe(MarketEvent, self._handle_market_event)
        await self._engine.subscribe(OrderEvent, self._execution_service.on_order_event)
        await self._engine.subscribe(BaseEvent, self._audit_event)
        self._started = True
        self._logger.info("integrated trading system started strategy_id=%s", self._strategy_id)

    async def stop(self) -> None:
        """Stop the engine, agents, and broker."""

        if not self._started:
            return
        await self._drain_background_news()
        await self._engine.stop(drain=True)
        for agent in reversed(self._agents()):
            await agent.stop()
        await self._broker.disconnect()
        self._started = False
        self._logger.info("integrated trading system stopped strategy_id=%s", self._strategy_id)

    async def run(
        self,
        market_events: Sequence[MarketEvent],
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        """Publish a stream of market events and wait for the pipeline to settle."""

        if not self._started:
            raise RuntimeError("IntegratedTradingSystem.start must be called before run")

        for event in market_events:
            await self._engine.publish(event)
            await self._engine.wait_until_idle()
            if delay_seconds > 0.0:
                await asyncio.sleep(delay_seconds)

    async def snapshot_state(self) -> dict[str, Any]:
        """Return a serializable system snapshot for the end of a run."""

        if self._started:
            await self._drain_background_news()
            await self._engine.wait_until_idle()
        account_snapshot = dict(await self._broker.get_account_snapshot())
        positions = {
            symbol: dict(position)
            for symbol, position in (await self._broker.get_positions()).items()
        }
        return {
            "strategy_id": self._strategy_id,
            "account": account_snapshot,
            "positions": positions,
            "executions": list(self.executions),
            "observed_events": list(self._observed_events),
            "market_agent": dict(await self._market_agent.snapshot_state()),
            "news_agent": dict(await self._news_agent.snapshot_state()),
            "decision_agent": dict(await self._decision_agent.snapshot_state()),
            "critic_agent": dict(await self._critic_agent.snapshot_state()),
            "risk_agent": dict(await self._risk_agent.snapshot_state()),
        }

    async def _handle_market_event(self, event: MarketEvent) -> None:
        """Run the full decision pipeline for a single market event."""

        symbol = event.symbol.strip().upper()
        self._logger.info("market event arrived symbol=%s occurred_at=%s", symbol, event.occurred_at.isoformat())
        event_started_ns = perf_counter_ns()
        self._schedule_news_refresh(now=event.occurred_at, correlation_id=event.event_id)

        if self._latency_recorder is None:
            market_signal = await self._run_market_step(event)
        else:
            with self._latency_recorder.span("market_agent_total"):
                market_signal = await self._run_market_step(event)
        news_analysis = await self._run_news_step(event)

        decision_started_ns = perf_counter_ns()
        decision = await self._run_decision_step(symbol=symbol, event=event, market_signal=market_signal, news_analysis=news_analysis)
        review = await self._run_critic_step(symbol=symbol, market_signal=market_signal, news_analysis=news_analysis, decision=decision)
        if not bool(review["approved"]):
            if self._latency_recorder is not None:
                self._latency_recorder.record_ns("decision_critic_risk_path", perf_counter_ns() - decision_started_ns)
            self._logger.info("critic rejected symbol=%s reason=%s", symbol, self._truncate_for_log(str(review["reason"])))
            return

        risk = await self._run_risk_step(
            symbol=symbol,
            event=event,
            market_signal=market_signal,
            decision=decision,
        )
        if self._latency_recorder is not None:
            self._latency_recorder.record_ns("decision_critic_risk_path", perf_counter_ns() - decision_started_ns)
        final_action = str(risk["final_action"]).upper()
        self._logger.info(
            "risk step symbol=%s final_action=%s position_size=%.4f stop_loss=%.4f",
            symbol,
            final_action,
            float(risk["position_size"]),
            float(risk["stop_loss"]),
        )
        if final_action == "HOLD":
            return

        await self._publish_order(symbol=symbol, event=event, decision=decision, risk=risk, signal_path_started_ns=event_started_ns)

    async def _run_market_step(self, event: MarketEvent) -> dict[str, Any]:
        """Run the market agent for the current bar."""

        await self._market_agent.on_event(event)
        candidate = getattr(self._market_agent, "last_output", None)
        if not isinstance(candidate, Mapping):
            snapshot = await self._market_agent.snapshot_state()
            candidate = snapshot.get("last_output")
        if not isinstance(candidate, Mapping):
            raise ValueError("market agent did not emit a signal payload")
        market_signal = normalize_market_signal(candidate)
        self._logger.info(
            "market step symbol=%s signal=%.4f confidence=%.4f",
            event.symbol,
            float(market_signal["signal"]),
            float(market_signal["confidence"]),
        )
        await self._engine.publish(
            SignalEvent(
                source=self._market_agent.agent_id,
                symbol=event.symbol,
                strategy_id=self._strategy_id,
                signal_type="MARKET_SIGNAL",
                confidence=float(market_signal["confidence"]),
                payload=dict(market_signal),
                occurred_at=event.occurred_at,
                correlation_id=event.event_id,
            )
        )
        return market_signal

    async def _run_news_step(self, event: MarketEvent) -> dict[str, float | str]:
        """Return the current non-stale news context without blocking the market path."""

        return self._current_news_analysis(symbol=event.symbol.strip().upper(), now=event.occurred_at)

    async def _run_decision_step(
        self,
        *,
        symbol: str,
        event: MarketEvent,
        market_signal: Mapping[str, Any],
        news_analysis: Mapping[str, Any],
    ) -> dict[str, float | str]:
        """Run the deterministic decision agent."""

        if self._latency_recorder is None:
            decision = await self._decision_agent.decide(market_signal=market_signal, news_analysis=news_analysis)
        else:
            with self._latency_recorder.span("decision_step"):
                decision = await self._decision_agent.decide(market_signal=market_signal, news_analysis=news_analysis)
        self._logger.info(
            "decision step symbol=%s action=%s score=%.4f",
            symbol,
            decision["action"],
            float(decision["score"]),
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
        return decision

    async def _run_critic_step(
        self,
        *,
        symbol: str,
        market_signal: Mapping[str, Any],
        news_analysis: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> dict[str, bool | str]:
        """Run the critic agent on the proposed trade."""

        if self._latency_recorder is None:
            review = await self._critic_agent.review(
                decision,
                market_signal=market_signal,
                news_analysis=news_analysis,
            )
        else:
            with self._latency_recorder.span("critic_step"):
                review = await self._critic_agent.review(
                    decision,
                    market_signal=market_signal,
                    news_analysis=news_analysis,
                )
        self._logger.info(
            "critic step symbol=%s approved=%s reason=%s",
            symbol,
            review["approved"],
            self._truncate_for_log(str(review["reason"])),
        )
        return review

    async def _run_risk_step(
        self,
        *,
        symbol: str,
        event: MarketEvent,
        market_signal: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> dict[str, float | str]:
        """Run the risk agent using the latest account and OHLCV context."""

        payload = dict(event.payload or {})
        account_state = await self._account_state(now=event.occurred_at)
        if self._latency_recorder is None:
            return await self._risk_agent.assess(
                decision,
                market_signal=market_signal,
                account_state=account_state,
                ohlcv=payload["ohlcv"],
                symbol=symbol,
                entry_price=float(event.last_price or payload["bar"]["close"]),
                timestamp=event.occurred_at,
                bar=payload.get("bar"),
            )
        with self._latency_recorder.span("risk_step"):
            return await self._risk_agent.assess(
                decision,
                market_signal=market_signal,
                account_state=account_state,
                ohlcv=payload["ohlcv"],
                symbol=symbol,
                entry_price=float(event.last_price or payload["bar"]["close"]),
                timestamp=event.occurred_at,
                bar=payload.get("bar"),
            )

    async def _publish_order(
        self,
        *,
        symbol: str,
        event: MarketEvent,
        decision: Mapping[str, Any],
        risk: Mapping[str, Any],
        signal_path_started_ns: int,
    ) -> None:
        """Publish a normalized order event when the symbol is tradeable."""

        open_orders = await self._broker.get_open_orders()
        for snapshot in open_orders.values():
            if str(snapshot.get("symbol", "")).strip().upper() == symbol:
                self._logger.info("skipping symbol=%s due to active open order", symbol)
                return

        positions = await self._broker.get_positions()
        position = positions.get(symbol)
        final_action = str(risk["final_action"]).upper()
        quantity = float(risk["position_size"])
        stop_loss = float(risk["stop_loss"])
        order_side = final_action
        reason = str(decision["reasoning"])
        if position is not None:
            current_quantity = float(position.get("quantity", 0.0))
            if current_quantity > 0.0 and final_action == "BUY":
                self._logger.info("skipping symbol=%s because the account is already long", symbol)
                return
            if current_quantity < 0.0 and final_action == "SELL":
                self._logger.info("skipping symbol=%s because the account is already short", symbol)
                return
            if current_quantity != 0.0:
                quantity = abs(current_quantity)
                order_side = "SELL" if current_quantity > 0.0 else "BUY"
                stop_loss = 0.0
                reason = "close existing position before reversal"

        if quantity <= 0.0:
            self._logger.info("skipping symbol=%s because quantity resolved to zero", symbol)
            return

        limit_price = float(event.last_price or 0.0)
        client_order_key = self._build_client_order_key(
            symbol=symbol,
            occurred_at=event.occurred_at,
            side=order_side,
            limit_price=limit_price,
        )
        await self._engine.publish(
            OrderEvent(
                source=self._risk_agent.agent_id,
                order_id=client_order_key,
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                status="NEW",
                limit_price=limit_price,
                payload={
                    "strategy_id": self._strategy_id,
                    "reason": reason,
                    "client_order_key": client_order_key,
                    "order_ref": f"client-order:{client_order_key}",
                    "signal_path_started_ns": signal_path_started_ns,
                    "stop_loss": stop_loss if stop_loss > 0.0 else None,
                },
                occurred_at=event.occurred_at,
                correlation_id=event.event_id,
            )
        )
        self._logger.info(
            "order step symbol=%s side=%s quantity=%.4f stop_loss=%.4f client_order_key=%s reason=%s",
            symbol,
            order_side,
            quantity,
            stop_loss,
            client_order_key,
            self._truncate_for_log(reason),
        )

    async def _collect_due_articles(self, now: datetime) -> list[Any]:
        """Return newly due news articles up to the current market timestamp."""

        due: list[Any] = []
        carry_over: dict[str, Any] = {}
        batch_fingerprints: set[str] = set()
        for fingerprint, article in self._deferred_articles.items():
            if fingerprint in self._suppressed_article_fingerprints:
                continue
            published_at = getattr(article, "published_at", None)
            if published_at is not None and published_at > now:
                carry_over[fingerprint] = article
            else:
                due.append(article)
                batch_fingerprints.add(fingerprint)

        self._deferred_articles = carry_over
        for article in await self._news_agent.ingest():
            if article.fingerprint in self._suppressed_article_fingerprints:
                continue
            if article.fingerprint in batch_fingerprints:
                continue
            published_at = article.published_at
            if published_at is not None and published_at > now:
                self._deferred_articles.setdefault(article.fingerprint, article)
                continue
            due.append(article)
            batch_fingerprints.add(article.fingerprint)

        due.sort(key=lambda article: article.published_at or now)
        return due

    async def _drain_background_news(self) -> None:
        """Wait for any in-flight background news refresh to complete."""

        if self._news_refresh_task is None:
            return
        try:
            await self._news_refresh_task
        finally:
            self._news_refresh_task = None

    def _schedule_news_refresh(self, *, now: datetime, correlation_id: str | None) -> None:
        """Schedule a background refresh of due news articles."""

        if self._pending_news_refresh_at is None or now > self._pending_news_refresh_at:
            self._pending_news_refresh_at = now
            self._pending_news_correlation_id = correlation_id
        elif self._pending_news_correlation_id is None and correlation_id is not None:
            self._pending_news_correlation_id = correlation_id

        if self._news_refresh_task is None or self._news_refresh_task.done():
            self._news_refresh_task = asyncio.create_task(
                self._run_news_refresh_loop(),
                name=f"{self._strategy_id}-news-refresh",
            )

    async def _run_news_refresh_loop(self) -> None:
        """Continuously refresh due news until the queued target time is satisfied."""

        while True:
            refresh_at = self._pending_news_refresh_at
            correlation_id = self._pending_news_correlation_id
            self._pending_news_refresh_at = None
            self._pending_news_correlation_id = None
            if refresh_at is None:
                return

            started_at_ns = perf_counter_ns()
            try:
                await self._refresh_news_until(now=refresh_at, correlation_id=correlation_id)
            except Exception:
                self._logger.exception("background news refresh failed strategy_id=%s", self._strategy_id)
            finally:
                if self._latency_recorder is not None:
                    self._latency_recorder.record_ns("news_agent_processing", perf_counter_ns() - started_at_ns)

            if self._pending_news_refresh_at is None:
                return

    async def _refresh_news_until(self, *, now: datetime, correlation_id: str | None) -> None:
        """Analyze and publish all due articles up to the provided market timestamp."""

        due_articles = await self._collect_due_articles(now)
        for article in due_articles:
            published_at = article.published_at or now
            news_age_seconds = max((now - published_at).total_seconds(), 0.0)
            if news_age_seconds > self._news_max_age_seconds:
                self._suppressed_article_fingerprints.add(article.fingerprint)
                self._logger.info(
                    "dropping stale news article source=%s symbols=%s age_seconds=%.2f",
                    article.source,
                    ",".join(article.symbols or ("__GLOBAL__",)),
                    news_age_seconds,
                )
                continue

            analysis = normalize_news_analysis(await self._news_agent.analyze_article(article))
            targets = article.symbols or ("__GLOBAL__",)
            for symbol in targets:
                self._latest_news[symbol] = _NewsContext(analysis=dict(analysis), published_at=published_at)
            await self._engine.publish(
                NewsEvent(
                    source=article.source,
                    headline=article.title,
                    symbols=tuple(targets),
                    body=article.content or article.summary,
                    payload=dict(analysis) | {"url": article.url},
                    occurred_at=published_at,
                    correlation_id=correlation_id,
                )
            )
            self._logger.info(
                "news step source=%s symbols=%s sentiment=%.4f impact=%.4f",
                article.source,
                ",".join(targets),
                float(analysis["sentiment"]),
                float(analysis["impact"]),
            )

    def _current_news_analysis(self, *, symbol: str, now: datetime) -> dict[str, float | str]:
        """Return the freshest cached news context for a symbol or the global fallback."""

        selected = self._latest_news.get(symbol) or self._latest_news.get("__GLOBAL__")
        if selected is None or self._news_context_is_stale(selected, now=now):
            return dict(NEUTRAL_NEWS_ANALYSIS)
        return dict(selected.analysis)

    async def _account_state(self, *, now: datetime) -> dict[str, float]:
        """Return account state normalized for the risk layer."""

        snapshot = dict(await self._broker.get_account_snapshot())
        capital = float(snapshot.get("capital", snapshot.get("equity", 0.0)))
        if capital <= 0.0:
            raise ValueError("broker account snapshot must expose a positive capital value")
        self._daily_start_capital.setdefault(now.date(), capital)
        daily_loss = max(self._daily_start_capital[now.date()] - capital, 0.0)
        gross_exposure = float(snapshot.get("gross_exposure", snapshot.get("exposure", 0.0)))
        return {
            "capital": capital,
            "daily_loss": daily_loss,
            "gross_exposure": gross_exposure,
        }

    async def _audit_event(self, event: BaseEvent) -> None:
        """Record observed events for diagnostics."""

        label = event.event_type if not isinstance(event, OrderEvent) else f"{event.event_type}:{event.status}"
        self._observed_events.append(label)

    def _agents(self) -> tuple[BaseAgent, ...]:
        """Return all managed agents in startup order."""

        return (
            self._market_agent,
            self._news_agent,
            self._decision_agent,
            self._critic_agent,
            self._risk_agent,
        )

    def _news_context_is_stale(self, context: _NewsContext, *, now: datetime) -> bool:
        """Return whether analyzed news has exceeded the configured recency budget."""

        return max((now - context.published_at).total_seconds(), 0.0) > self._news_max_age_seconds

    def _build_client_order_key(
        self,
        *,
        symbol: str,
        occurred_at: datetime,
        side: str,
        limit_price: float,
    ) -> str:
        """Build a deterministic idempotency key for a strategy order intent."""

        payload = "|".join(
            [
                self._strategy_id,
                symbol,
                side,
                occurred_at.isoformat(),
                f"{limit_price:.6f}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _truncate_for_log(self, value: str) -> str:
        """Truncate potentially sensitive reasoning text before logging."""

        limit = self._runtime_config.max_log_reason_length
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."
