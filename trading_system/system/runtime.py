"""Integrated async runtime for the multi-agent trading workflow."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from time import monotonic, perf_counter_ns
from typing import Any
from zoneinfo import ZoneInfo

from trading_system.agents import BaseAgent, CriticAgent, DecisionAgent, NewsAgent, RiskAgent
from trading_system.agents._validation import normalize_market_signal, normalize_news_analysis
from trading_system.agents.news_scoring import apply_freshness_decay
from trading_system.agents.schemas import NewsAnalysisSchema
from trading_system.backtest.bus import SequentialEventBus
from trading_system.core.engine import Engine
from trading_system.core.events import BaseEvent, MarketEvent, NewsEvent, OrderEvent, SignalEvent
from trading_system.execution import BrokerInterface, OrderManager
from trading_system.infra import AsyncJsonlWriter, LatencyRecorder, LatencyReport, MetricsRegistry
from trading_system.system.config import (
    ExecutionConfig,
    ObservabilityConfig,
    PersistenceConfig,
    RuntimeConfig,
    SafetyConfig,
)
from trading_system.system.replay import serialize_event_record
from trading_system.system.state import KillSwitchState, RuntimeStateStore

_US_EASTERN = ZoneInfo("America/New_York")
NEUTRAL_NEWS_ANALYSIS = normalize_news_analysis(
    {
        "sentiment": 0.0,
        "impact": 0.0,
        "event_type": "none",
        "summary": "No material news available.",
    }
)


@dataclass(frozen=True, slots=True)
class _NewsContext:
    """Latest analyzed news payload with publication metadata."""

    analysis: dict[str, Any]
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

    async def cancel_all_open_orders(self) -> list[str]:
        """Cancel every currently open order exposed by the broker."""

        open_orders = await self._broker.get_open_orders()
        cancelled: list[str] = []
        for order_id in open_orders:
            await self._broker.cancel_order(order_id)
            cancelled.append(str(order_id))
        return cancelled


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
        safety_config: SafetyConfig | None = None,
        persistence_config: PersistenceConfig | None = None,
        observability_config: ObservabilityConfig | None = None,
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
        self._metrics = MetricsRegistry()
        self._execution_service = ExecutionService(
            broker=broker,
            order_manager=order_manager,
            config=execution_config,
            logger=self._logger,
            latency_recorder=self._latency_recorder,
        )
        self._strategy_id = strategy_id
        self._runtime_config = runtime_config or RuntimeConfig(strategy_id=strategy_id)
        self._safety_config = safety_config or SafetyConfig()
        self._persistence_config = persistence_config or PersistenceConfig()
        self._observability_config = observability_config or ObservabilityConfig()
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

        self._state_store = RuntimeStateStore(self._persistence_config.state_store_path)
        self._audit_journal = (
            AsyncJsonlWriter(
                self._persistence_config.audit_journal_path,
                serializer=lambda event: serialize_event_record(
                    event,
                    replay_history_bars=self._persistence_config.replay_history_bars,
                ),
                logger=self._logger,
            )
            if self._persistence_config.enable_event_audit
            else None
        )
        self._dead_letter_journal = (
            AsyncJsonlWriter(self._persistence_config.dead_letter_path, logger=self._logger)
            if self._persistence_config.enable_dead_letter_journal
            else None
        )
        self._alert_journal = (
            AsyncJsonlWriter(self._persistence_config.alert_journal_path, logger=self._logger)
            if self._persistence_config.enable_alert_journal
            else None
        )

        self._latest_news: dict[str, _NewsContext] = {}
        self._deferred_articles: dict[str, Any] = {}
        self._suppressed_article_fingerprints: set[str] = set()
        self._pending_news_refresh_at: datetime | None = None
        self._pending_news_correlation_id: Any | None = None
        self._news_refresh_task: asyncio.Task[None] | None = None
        self._ops_task: asyncio.Task[None] | None = None

        self._daily_start_capital: dict[date, float] = {}
        self._observed_events: list[str] = []
        self._kill_switch = KillSwitchState.disengaged()
        self._last_reconciliation: dict[str, Any] | None = None
        self._last_market_event_at: datetime | None = None
        self._last_order_event_at: datetime | None = None
        self._last_snapshot_at: datetime | None = None
        self._latest_account_snapshot: dict[str, Any] | None = None
        self._latest_position_snapshot: dict[str, dict[str, Any]] = {}
        self._last_latency_alert_signature: tuple[str, float] | None = None
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
        """Start the engine, broker, agents, and operational background tasks."""

        if self._started:
            return

        await self._start_writers()
        try:
            await self._broker.connect()
            await self._restore_persisted_state()
            if self._runtime_config.reconcile_on_start:
                await self._order_manager.reconcile()
            await self._reconcile_startup_state()
            for agent in self._agents():
                await agent.start()
            await self._engine.start()
            await self._engine.subscribe(MarketEvent, self._on_execution_market_event_safe)
            await self._engine.subscribe(MarketEvent, self._on_market_event_safe)
            await self._engine.subscribe(OrderEvent, self._on_order_event_safe)
            await self._engine.subscribe(BaseEvent, self._audit_event)
            self._started = True
            self._ops_task = asyncio.create_task(self._ops_loop(), name=f"{self._strategy_id}-ops")
            await self._persist_operational_state(reason="start", include_trading_state=True)
            self._logger.info("integrated trading system started strategy_id=%s", self._strategy_id)
        except Exception:
            try:
                if self._engine.is_running:
                    await self._engine.stop(drain=False)
                for agent in reversed(self._agents()):
                    try:
                        await agent.stop()
                    except Exception:
                        self._logger.exception("failed to stop agent during startup rollback agent_id=%s", agent.agent_id)
            finally:
                try:
                    await self._broker.disconnect()
                finally:
                    await self._stop_writers()
            raise

    async def stop(self) -> None:
        """Stop the runtime and persist final operational state."""

        if not self._started:
            return

        await self._cancel_ops_task()
        await self._drain_background_news()
        await self._engine.wait_until_idle()
        final_snapshot = await self._build_snapshot()
        await self._state_store.write_runtime_snapshot(final_snapshot)
        await self._persist_operational_state(reason="stop", include_trading_state=True)
        await self._engine.stop(drain=True)
        for agent in reversed(self._agents()):
            await agent.stop()
        await self._broker.disconnect()
        await self._stop_writers()
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
        """Return a serializable system snapshot."""

        if self._started:
            await self._drain_background_news()
            await self._engine.wait_until_idle()
        return await self._build_snapshot()

    async def engage_kill_switch(self, reason: str, *, source: str = "runtime", persist: bool = True) -> dict[str, Any]:
        """Engage the kill switch and cancel open orders when configured."""

        if not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self._kill_switch.engaged and self._kill_switch.reason == reason and self._kill_switch.source == source:
            return self._kill_switch.to_dict()

        self._kill_switch = KillSwitchState(
            engaged=True,
            reason=reason.strip(),
            engaged_at=datetime.now(timezone.utc).isoformat(),
            source=source.strip() or "runtime",
            strategy_id=self._strategy_id,
        )
        self._metrics.increment_counter("kill_switch_engaged_total")
        self._logger.critical("kill switch engaged source=%s reason=%s", self._kill_switch.source, reason)
        if persist:
            await self._state_store.write_kill_switch(self._kill_switch)
        if self._safety_config.cancel_open_orders_on_kill_switch:
            try:
                cancelled = await self._execution_service.cancel_all_open_orders()
                if cancelled:
                    self._logger.warning("kill switch cancelled open orders order_ids=%s", ",".join(cancelled))
            except Exception as exc:
                await self._emit_alert(
                    kind="kill_switch_cancel_failure",
                    severity="critical",
                    message=f"failed to cancel open orders after kill switch engagement: {exc}",
                    payload={"reason": reason, "source": source},
                )
        await self._emit_alert(
            kind="kill_switch_engaged",
            severity="critical",
            message=reason.strip(),
            payload={"source": source.strip() or "runtime"},
        )
        await self._persist_operational_state(reason="kill_switch", include_trading_state=True)
        return self._kill_switch.to_dict()

    async def release_kill_switch(self, *, reason: str, source: str = "operator", persist: bool = True) -> dict[str, Any]:
        """Release the kill switch."""

        if not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not self._kill_switch.engaged:
            if persist:
                await self._state_store.write_kill_switch(self._kill_switch)
            return self._kill_switch.to_dict()

        self._kill_switch = KillSwitchState.disengaged()
        self._metrics.increment_counter("kill_switch_released_total")
        self._logger.warning("kill switch released source=%s reason=%s", source, reason)
        if persist:
            await self._state_store.write_kill_switch(self._kill_switch)
        await self._emit_alert(
            kind="kill_switch_released",
            severity="warning",
            message=reason.strip(),
            payload={"source": source.strip() or "operator"},
        )
        await self._persist_operational_state(reason="kill_switch_release", include_trading_state=True)
        return self._kill_switch.to_dict()

    async def health_status(self) -> dict[str, Any]:
        """Return a health snapshot for the live runtime."""

        queue_size = getattr(self._engine.bus, "queue_size", 0) if self._started else 0
        self._metrics.set_gauge("event_queue_size", float(queue_size))
        self._metrics.set_gauge("kill_switch_engaged", 1.0 if self._kill_switch.engaged else 0.0)
        return {
            "strategy_id": self._strategy_id,
            "mode": str(getattr(self._runtime_config.mode, "value", self._runtime_config.mode)),
            "started": self._started,
            "kill_switch": self._kill_switch.to_dict(),
            "queue_size": int(queue_size),
            "news_refresh_in_flight": bool(self._news_refresh_task is not None and not self._news_refresh_task.done()),
            "last_market_event_at": None if self._last_market_event_at is None else self._last_market_event_at.isoformat(),
            "last_order_event_at": None if self._last_order_event_at is None else self._last_order_event_at.isoformat(),
            "last_snapshot_at": None if self._last_snapshot_at is None else self._last_snapshot_at.isoformat(),
            "last_reconciliation": self._last_reconciliation,
            "latest_account": self._latest_account_snapshot,
            "latest_positions": self._latest_position_snapshot,
            "degraded": bool(self._kill_switch.engaged),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def metrics_snapshot(self) -> dict[str, Any]:
        """Return a metrics snapshot for the live runtime."""

        return self._metrics.snapshot(latency_report=self.latency_report).to_dict()

    async def _on_market_event_safe(self, event: MarketEvent) -> None:
        """Run the market pipeline with dead-letter and kill-switch protection."""

        try:
            await self._handle_market_event(event)
        except Exception as exc:
            await self._handle_handler_failure(
                event=event,
                stage="market_pipeline",
                exc=exc,
                engage_kill_switch=True,
            )
            raise

    async def _on_execution_market_event_safe(self, event: MarketEvent) -> None:
        """Run execution-side market maintenance with failure handling."""

        try:
            await self._execution_service.on_market_event(event)
        except Exception as exc:
            await self._handle_handler_failure(
                event=event,
                stage="execution_market_maintenance",
                exc=exc,
                engage_kill_switch=True,
            )
            raise

    async def _on_order_event_safe(self, event: OrderEvent) -> None:
        """Submit orders with failure handling."""

        try:
            await self._execution_service.on_order_event(event)
        except Exception as exc:
            await self._handle_handler_failure(
                event=event,
                stage="execution_order_submission",
                exc=exc,
                engage_kill_switch=True,
            )
            raise

    async def _handle_market_event(self, event: MarketEvent) -> None:
        """Run the full decision pipeline for a single market event."""

        self._metrics.increment_counter("market_events_total")
        self._last_market_event_at = event.occurred_at
        symbol = event.symbol.strip().upper()
        self._logger.info("market event arrived symbol=%s occurred_at=%s", symbol, event.occurred_at.isoformat())

        if self._kill_switch.engaged:
            self._metrics.increment_counter("market_events_skipped_kill_switch_total")
            self._logger.warning("skipping symbol=%s because the kill switch is engaged", symbol)
            return

        event_started_ns = perf_counter_ns()
        self._schedule_news_refresh(now=event.occurred_at, correlation_id=event.event_id)

        if self._latency_recorder is None:
            market_signal = await self._run_market_step(event)
        else:
            with self._latency_recorder.span("market_agent_total"):
                market_signal = await self._run_market_step(event)
        news_analysis = await self._run_news_step(event)

        decision_started_ns = perf_counter_ns()
        decision = await self._run_decision_step(
            symbol=symbol,
            event=event,
            market_signal=market_signal,
            news_analysis=news_analysis,
        )
        review = await self._run_critic_step(
            symbol=symbol,
            market_signal=market_signal,
            news_analysis=news_analysis,
            decision=decision,
            occurred_at=event.occurred_at,
        )
        if not bool(review["approved"]):
            self._metrics.increment_counter("critic_rejections_total")
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
        if self._kill_switch.engaged:
            self._metrics.increment_counter("risk_holds_total")
            self._logger.warning("risk path exited because the kill switch is engaged symbol=%s", symbol)
            return
        if final_action == "HOLD":
            self._metrics.increment_counter("risk_holds_total")
            return

        await self._publish_order(
            symbol=symbol,
            event=event,
            decision=decision,
            risk=risk,
            signal_path_started_ns=event_started_ns,
        )

    async def _run_market_step(self, event: MarketEvent) -> dict[str, Any]:
        """Run the market agent for the current bar."""

        await self._market_agent.on_event(event)
        candidate = getattr(self._market_agent, "last_output", None)
        if not isinstance(candidate, Mapping):
            snapshot = await self._market_agent.snapshot_state()
            candidate = snapshot.get("last_output")
        if not isinstance(candidate, Mapping):
            raise ValueError("market agent did not emit a signal payload")
        market_signal = self._augment_market_signal(event=event, market_signal=normalize_market_signal(candidate))
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

    async def _run_news_step(self, event: MarketEvent) -> dict[str, Any]:
        """Return the current non-stale news context without blocking the market path."""

        return self._current_news_analysis(symbol=event.symbol.strip().upper(), now=event.occurred_at)

    async def _run_decision_step(
        self,
        *,
        symbol: str,
        event: MarketEvent,
        market_signal: Mapping[str, Any],
        news_analysis: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run the deterministic decision agent."""

        if self._latency_recorder is None:
            decision = await self._decision_agent.decide(
                market_signal=market_signal,
                news_analysis=news_analysis,
                symbol=symbol,
                occurred_at=event.occurred_at,
            )
        else:
            with self._latency_recorder.span("decision_step"):
                decision = await self._decision_agent.decide(
                    market_signal=market_signal,
                    news_analysis=news_analysis,
                    symbol=symbol,
                    occurred_at=event.occurred_at,
                )
        self._logger.info(
            "decision step symbol=%s action=%s score=%.4f confidence=%.4f expected_edge_bps=%.2f",
            symbol,
            decision["action"],
            float(decision["score"]),
            float(decision["confidence"]),
            float(decision["expected_edge"]) * 10_000.0,
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
        occurred_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Run the critic agent on the proposed trade."""

        if self._latency_recorder is None:
            review = await self._critic_agent.review(
                decision,
                market_signal=market_signal,
                news_analysis=news_analysis,
                symbol=symbol,
                occurred_at=occurred_at,
            )
        else:
            with self._latency_recorder.span("critic_step"):
                review = await self._critic_agent.review(
                    decision,
                    market_signal=market_signal,
                    news_analysis=news_analysis,
                    symbol=symbol,
                    occurred_at=occurred_at,
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

        if self._kill_switch.engaged:
            self._metrics.increment_counter("orders_blocked_kill_switch_total")
            return

        open_orders = await self._broker.get_open_orders()
        for snapshot in open_orders.values():
            if str(snapshot.get("symbol", "")).strip().upper() == symbol:
                self._metrics.increment_counter("orders_suppressed_open_order_total")
                self._logger.info("skipping symbol=%s due to active open order", symbol)
                return

        positions = {
            position_symbol: dict(position)
            for position_symbol, position in (await self._broker.get_positions()).items()
        }
        self._latest_position_snapshot = positions
        position = positions.get(symbol)
        final_action = str(risk["final_action"]).upper()
        quantity = float(risk["position_size"])
        stop_loss = float(risk["stop_loss"])
        order_side = final_action
        reason = str(decision["reasoning"])
        if position is not None:
            current_quantity = float(position.get("quantity", 0.0))
            if current_quantity > 0.0 and final_action == "BUY":
                self._metrics.increment_counter("orders_suppressed_same_side_position_total")
                self._logger.info("skipping symbol=%s because the account is already long", symbol)
                return
            if current_quantity < 0.0 and final_action == "SELL":
                self._metrics.increment_counter("orders_suppressed_same_side_position_total")
                self._logger.info("skipping symbol=%s because the account is already short", symbol)
                return
            if current_quantity != 0.0:
                quantity = abs(current_quantity)
                order_side = "SELL" if current_quantity > 0.0 else "BUY"
                stop_loss = 0.0
                reason = "close existing position before reversal"

        if quantity <= 0.0:
            self._metrics.increment_counter("orders_suppressed_zero_quantity_total")
            self._logger.info("skipping symbol=%s because quantity resolved to zero", symbol)
            return

        limit_price = float(event.last_price or 0.0)
        client_order_key = self._build_client_order_key(
            symbol=symbol,
            occurred_at=event.occurred_at,
            side=order_side,
            limit_price=limit_price,
        )
        order_status = "DRY_RUN" if self._safety_config.dry_run else "NEW"
        await self._engine.publish(
            OrderEvent(
                source=self._risk_agent.agent_id,
                order_id=client_order_key,
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                status=order_status,
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
        metric_name = "orders_dry_run_total" if self._safety_config.dry_run else "orders_published_total"
        self._metrics.increment_counter(metric_name)
        self._logger.info(
            "order step symbol=%s side=%s quantity=%.4f stop_loss=%.4f client_order_key=%s reason=%s status=%s",
            symbol,
            order_side,
            quantity,
            stop_loss,
            client_order_key,
            self._truncate_for_log(reason),
            order_status,
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

    def _schedule_news_refresh(self, *, now: datetime, correlation_id: Any | None) -> None:
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
            except Exception as exc:
                self._metrics.increment_counter("news_refresh_failures_total")
                await self._record_dead_letter(
                    stage="background_news_refresh",
                    event=None,
                    exc=exc,
                    payload={"occurred_at": refresh_at.isoformat()},
                )
                await self._emit_alert(
                    kind="news_refresh_failure",
                    severity="error",
                    message=str(exc),
                    payload={"occurred_at": refresh_at.isoformat()},
                )
                self._logger.exception("background news refresh failed strategy_id=%s", self._strategy_id)
            finally:
                if self._latency_recorder is not None:
                    self._latency_recorder.record_ns("news_agent_processing", perf_counter_ns() - started_at_ns)

            if self._pending_news_refresh_at is None:
                return

    async def _refresh_news_until(self, *, now: datetime, correlation_id: Any | None) -> None:
        """Analyze and publish all due articles up to the provided market timestamp."""

        due_articles = await self._collect_due_articles(now)
        for article in due_articles:
            published_at = article.published_at or now
            news_age_seconds = max((now - published_at).total_seconds(), 0.0)
            if news_age_seconds > self._news_max_age_seconds:
                self._suppressed_article_fingerprints.add(article.fingerprint)
                self._metrics.increment_counter("news_articles_stale_total")
                self._logger.info(
                    "dropping stale news article source=%s symbols=%s age_seconds=%.2f",
                    article.source,
                    ",".join(article.symbols or ("__GLOBAL__",)),
                    news_age_seconds,
                )
                continue

            analysis = normalize_news_analysis(await self._news_agent.analyze_article(article))
            self._metrics.increment_counter("news_articles_analyzed_total")
            if str(analysis["scope"]) == "market":
                targets = ("__GLOBAL__",)
            else:
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

    def _current_news_analysis(self, *, symbol: str, now: datetime) -> dict[str, Any]:
        """Return the freshest cached news context for a symbol or the global fallback."""

        selected = self._latest_news.get(symbol) or self._latest_news.get("__GLOBAL__")
        if selected is None or self._news_context_is_stale(selected, now=now):
            return dict(NEUTRAL_NEWS_ANALYSIS)
        decayed = apply_freshness_decay(
            NewsAnalysisSchema(**normalize_news_analysis(selected.analysis)),
            published_at=selected.published_at,
            now=now,
        )
        return decayed.to_dict()

    async def _account_state(self, *, now: datetime) -> dict[str, float]:
        """Return account state normalized for the risk layer."""

        snapshot = dict(await self._broker.get_account_snapshot())
        self._latest_account_snapshot = snapshot
        capital = float(snapshot.get("capital", snapshot.get("equity", 0.0)))
        if capital <= 0.0:
            raise ValueError("broker account snapshot must expose a positive capital value")
        self._daily_start_capital.setdefault(now.date(), capital)
        daily_loss = max(self._daily_start_capital[now.date()] - capital, 0.0)
        gross_exposure = float(snapshot.get("gross_exposure", snapshot.get("exposure", 0.0)))
        if self._daily_start_capital[now.date()] > 0.0:
            daily_loss_fraction = daily_loss / self._daily_start_capital[now.date()]
            if daily_loss_fraction >= self._safety_config.max_daily_loss_shutdown_fraction:
                await self.engage_kill_switch(
                    (
                        "daily loss shutdown breached "
                        f"fraction={daily_loss_fraction:.4f} "
                        f"threshold={self._safety_config.max_daily_loss_shutdown_fraction:.4f}"
                    ),
                    source="daily_loss",
                )
        return {
            "capital": capital,
            "daily_loss": daily_loss,
            "gross_exposure": gross_exposure,
        }

    async def _audit_event(self, event: BaseEvent) -> None:
        """Record observed events and enqueue audit records without blocking the hot path."""

        label = event.event_type if not isinstance(event, OrderEvent) else f"{event.event_type}:{event.status}"
        self._observed_events.append(label)
        self._metrics.increment_counter(f"events.{event.event_type}.total")
        if isinstance(event, OrderEvent):
            self._last_order_event_at = event.occurred_at
        if self._audit_journal is not None:
            self._audit_journal.append_nowait(event)

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

    def _augment_market_signal(self, *, event: MarketEvent, market_signal: Mapping[str, Any]) -> dict[str, Any]:
        """Attach lightweight session and spread context without recomputing indicators."""

        enriched = dict(market_signal)
        features = dict(market_signal["features"])
        features.setdefault("session_phase", self._session_phase(event.occurred_at))

        spread_bps = self._spread_bps(event)
        if spread_bps is not None:
            features.setdefault("spread_bps", spread_bps)

        if "dollar_volume" not in features:
            bar_volume = event.volume
            if bar_volume is None:
                payload = dict(event.payload or {})
                bar = payload.get("bar")
                if isinstance(bar, Mapping) and "volume" in bar:
                    try:
                        bar_volume = float(bar["volume"])
                    except (TypeError, ValueError):
                        bar_volume = None
            if event.last_price is not None and bar_volume is not None:
                features["dollar_volume"] = float(event.last_price) * float(bar_volume)

        if event.symbol:
            enriched["symbol"] = event.symbol.strip().upper()
        enriched["features"] = features
        return enriched

    def _session_phase(self, timestamp: datetime) -> str:
        """Return a coarse US cash-session phase."""

        eastern = timestamp.astimezone(_US_EASTERN)
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

    def _spread_bps(self, event: MarketEvent) -> float | None:
        """Return the quoted spread in basis points when the market event exposes one."""

        if event.bid is None or event.ask is None:
            return None
        bid = float(event.bid)
        ask = float(event.ask)
        mid = (bid + ask) / 2.0
        if bid <= 0.0 or ask <= 0.0 or mid <= 0.0 or ask < bid:
            return None
        return ((ask - bid) / mid) * 10_000.0

    async def _start_writers(self) -> None:
        """Start all configured asynchronous writers."""

        for writer in (self._audit_journal, self._dead_letter_journal, self._alert_journal):
            if writer is not None:
                await writer.start()

    async def _stop_writers(self) -> None:
        """Stop all configured asynchronous writers."""

        for writer in (self._audit_journal, self._dead_letter_journal, self._alert_journal):
            if writer is not None:
                await writer.stop()

    async def _cancel_ops_task(self) -> None:
        """Cancel the background operational task."""

        if self._ops_task is None:
            return
        self._ops_task.cancel()
        try:
            await self._ops_task
        except asyncio.CancelledError:
            pass
        finally:
            self._ops_task = None

    async def _handle_handler_failure(
        self,
        *,
        event: BaseEvent | None,
        stage: str,
        exc: Exception,
        engage_kill_switch: bool,
    ) -> None:
        """Record a runtime failure and optionally engage the kill switch."""

        self._metrics.increment_counter("dead_letters_total")
        await self._record_dead_letter(stage=stage, event=event, exc=exc)
        await self._emit_alert(
            kind="handler_failure",
            severity="critical" if engage_kill_switch else "error",
            message=f"{stage} failed: {exc}",
            payload={"event_type": None if event is None else event.event_type},
        )
        if engage_kill_switch:
            await self.engage_kill_switch(f"{stage} failed: {exc}", source=stage)

    async def _record_dead_letter(
        self,
        *,
        stage: str,
        exc: Exception,
        event: BaseEvent | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist a dead-letter record for failed runtime work."""

        if self._dead_letter_journal is None:
            return
        record = {
            "record_type": "dead_letter",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "strategy_id": self._strategy_id,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "event": None if event is None else serialize_event_record(
                event,
                replay_history_bars=self._persistence_config.replay_history_bars,
            ),
            "payload": payload or {},
        }
        self._dead_letter_journal.append_nowait(record)

    async def _emit_alert(
        self,
        *,
        kind: str,
        severity: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit an operational alert to the alert journal."""

        self._metrics.increment_counter("alerts_total")
        record = {
            "record_type": "alert",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "strategy_id": self._strategy_id,
            "kind": kind,
            "severity": severity,
            "message": message,
            "payload": payload or {},
        }
        self._logger.warning("alert kind=%s severity=%s message=%s", kind, severity, self._truncate_for_log(message))
        if self._alert_journal is not None:
            self._alert_journal.append_nowait(record)

    async def _ops_loop(self) -> None:
        """Run background maintenance, heartbeat persistence, and kill-switch polling."""

        next_heartbeat = monotonic() + self._observability_config.heartbeat_interval_seconds
        next_maintenance = monotonic() + self._observability_config.maintenance_interval_seconds
        next_kill_switch_poll = monotonic() + self._observability_config.kill_switch_poll_seconds
        next_trading_snapshot = monotonic() + self._persistence_config.snapshot_interval_seconds
        while True:
            try:
                now_tick = monotonic()
                if now_tick >= next_maintenance:
                    await self._run_maintenance()
                    next_maintenance = monotonic() + self._observability_config.maintenance_interval_seconds
                if now_tick >= next_kill_switch_poll:
                    await self._sync_persisted_kill_switch()
                    next_kill_switch_poll = monotonic() + self._observability_config.kill_switch_poll_seconds
                if now_tick >= next_heartbeat:
                    await self._persist_operational_state(reason="heartbeat", include_trading_state=False)
                    next_heartbeat = monotonic() + self._observability_config.heartbeat_interval_seconds
                if now_tick >= next_trading_snapshot:
                    await self._persist_operational_state(reason="periodic_snapshot", include_trading_state=True)
                    next_trading_snapshot = monotonic() + self._persistence_config.snapshot_interval_seconds
                sleep_seconds = min(
                    next_heartbeat,
                    next_maintenance,
                    next_kill_switch_poll,
                    next_trading_snapshot,
                ) - monotonic()
                await asyncio.sleep(max(min(sleep_seconds, 1.0), 0.1))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._handle_handler_failure(
                    event=None,
                    stage="ops_loop",
                    exc=exc,
                    engage_kill_switch=True,
                )
                await asyncio.sleep(1.0)

    async def _run_maintenance(self) -> None:
        """Run periodic broker and runtime maintenance work."""

        cancelled = await self._order_manager.cancel_stale_orders()
        if cancelled:
            self._metrics.increment_counter("stale_orders_cancelled_total", float(len(cancelled)))
        open_orders = await self._broker.get_open_orders()
        self._metrics.set_gauge("open_orders", float(len(open_orders)))
        positions = {
            symbol: dict(position)
            for symbol, position in (await self._broker.get_positions()).items()
        }
        self._latest_position_snapshot = positions
        self._metrics.set_gauge("positions_count", float(len(positions)))
        if self._latency_recorder is not None:
            await self._maybe_emit_latency_alert()

    async def _sync_persisted_kill_switch(self) -> None:
        """Apply externally persisted kill-switch state changes."""

        persisted = await self._state_store.read_kill_switch()
        if persisted.strategy_id != self._strategy_id:
            return
        if persisted.engaged and (
            not self._kill_switch.engaged
            or persisted.reason != self._kill_switch.reason
            or persisted.source != self._kill_switch.source
        ):
            await self.engage_kill_switch(
                persisted.reason or "external kill switch request",
                source=persisted.source or "operator",
                persist=False,
            )
        elif not persisted.engaged and self._kill_switch.engaged:
            await self.release_kill_switch(
                reason="external release request",
                source="operator",
                persist=False,
            )

    async def _persist_operational_state(self, *, reason: str, include_trading_state: bool) -> None:
        """Persist health, metrics, kill-switch, and trading-state snapshots."""

        health = await self.health_status()
        metrics = await self.metrics_snapshot()
        await self._state_store.write_health_snapshot(health)
        await self._state_store.write_metrics_snapshot(metrics)
        await self._state_store.write_kill_switch(self._kill_switch)
        if self._last_reconciliation is not None:
            await self._state_store.write_reconciliation(self._last_reconciliation)
        if include_trading_state:
            await self._state_store.write_trading_state(await self._build_trading_state(reason=reason))
        self._last_snapshot_at = datetime.now(timezone.utc)

    async def _build_trading_state(self, *, reason: str) -> dict[str, Any]:
        """Return a lightweight trading-state snapshot used for reconciliation."""

        account_snapshot = dict(await self._broker.get_account_snapshot())
        positions = {
            symbol: dict(position)
            for symbol, position in (await self._broker.get_positions()).items()
        }
        self._latest_account_snapshot = account_snapshot
        self._latest_position_snapshot = positions
        return {
            "strategy_id": self._strategy_id,
            "mode": str(getattr(self._runtime_config.mode, "value", self._runtime_config.mode)),
            "reason": reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "account": account_snapshot,
            "positions": positions,
            "executions": list(self.executions),
            "daily_start_capital": {key.isoformat(): value for key, value in self._daily_start_capital.items()},
            "kill_switch": self._kill_switch.to_dict(),
        }

    async def _build_snapshot(self) -> dict[str, Any]:
        """Return a serializable runtime snapshot."""

        account_snapshot = dict(await self._broker.get_account_snapshot())
        positions = {
            symbol: dict(position)
            for symbol, position in (await self._broker.get_positions()).items()
        }
        self._latest_account_snapshot = account_snapshot
        self._latest_position_snapshot = positions
        return {
            "strategy_id": self._strategy_id,
            "account": account_snapshot,
            "positions": positions,
            "executions": list(self.executions),
            "observed_events": list(self._observed_events),
            "kill_switch": self._kill_switch.to_dict(),
            "health": await self.health_status(),
            "metrics": await self.metrics_snapshot(),
            "reconciliation": self._last_reconciliation,
            "daily_start_capital": {key.isoformat(): value for key, value in self._daily_start_capital.items()},
            "market_agent": dict(await self._market_agent.snapshot_state()),
            "news_agent": dict(await self._news_agent.snapshot_state()),
            "decision_agent": dict(await self._decision_agent.snapshot_state()),
            "critic_agent": dict(await self._critic_agent.snapshot_state()),
            "risk_agent": dict(await self._risk_agent.snapshot_state()),
        }

    async def _restore_persisted_state(self) -> None:
        """Restore persisted safety and reconciliation state before startup."""

        persisted_kill_switch = await self._state_store.read_kill_switch()
        if persisted_kill_switch.strategy_id == self._strategy_id:
            self._kill_switch = persisted_kill_switch
        else:
            self._kill_switch = KillSwitchState.disengaged()
        persisted_trading_state = await self._state_store.read_trading_state()
        if isinstance(persisted_trading_state, dict):
            daily_start_capital = persisted_trading_state.get("daily_start_capital", {})
            if isinstance(daily_start_capital, Mapping):
                self._daily_start_capital = {
                    date.fromisoformat(str(day)): float(value)
                    for day, value in daily_start_capital.items()
                }
            account = persisted_trading_state.get("account")
            if isinstance(account, Mapping):
                self._latest_account_snapshot = dict(account)
            positions = persisted_trading_state.get("positions")
            if isinstance(positions, Mapping):
                self._latest_position_snapshot = {
                    str(symbol): dict(position)
                    for symbol, position in positions.items()
                    if isinstance(position, Mapping)
                }
        self._last_reconciliation = await self._state_store.read_reconciliation()

    async def _reconcile_startup_state(self) -> None:
        """Reconcile broker positions against the last persisted trading-state snapshot."""

        if not self._runtime_config.reconcile_on_start:
            self._last_reconciliation = {
                "status": "skipped",
                "reason": "runtime.reconcile_on_start disabled",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._state_store.write_reconciliation(self._last_reconciliation)
            return
        if not self._safety_config.startup_position_reconciliation:
            self._last_reconciliation = {
                "status": "skipped",
                "reason": "safety.startup_position_reconciliation disabled",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._state_store.write_reconciliation(self._last_reconciliation)
            return

        persisted = await self._state_store.read_trading_state()
        broker_account_snapshot = dict(await self._broker.get_account_snapshot())
        self._latest_account_snapshot = broker_account_snapshot
        broker_account = str(broker_account_snapshot.get("account", "")).strip()
        broker_positions = {
            symbol: dict(position)
            for symbol, position in (await self._broker.get_positions()).items()
        }
        self._latest_position_snapshot = broker_positions
        if not isinstance(persisted, Mapping):
            self._last_reconciliation = {
                "status": "ok",
                "reason": "no persisted trading state available",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "mismatches": [],
            }
            await self._state_store.write_reconciliation(self._last_reconciliation)
            return
        if str(persisted.get("strategy_id", "")).strip() != self._strategy_id:
            self._last_reconciliation = {
                "status": "skipped",
                "reason": "persisted trading state belongs to a different strategy_id",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._state_store.write_reconciliation(self._last_reconciliation)
            return
        persisted_account_raw = persisted.get("account", {})
        persisted_account = ""
        if isinstance(persisted_account_raw, Mapping):
            persisted_account = str(persisted_account_raw.get("account", "")).strip()
        if persisted_account and broker_account and persisted_account != broker_account:
            self._last_reconciliation = {
                "status": "skipped",
                "reason": "persisted trading state belongs to a different broker account",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._state_store.write_reconciliation(self._last_reconciliation)
            return

        persisted_positions_raw = persisted.get("positions", {})
        persisted_positions = (
            {
                str(symbol): dict(position)
                for symbol, position in persisted_positions_raw.items()
                if isinstance(position, Mapping)
            }
            if isinstance(persisted_positions_raw, Mapping)
            else {}
        )
        mismatches = self._position_mismatches(
            persisted_positions=persisted_positions,
            broker_positions=broker_positions,
            tolerance=self._safety_config.startup_position_quantity_tolerance,
        )
        status = "ok" if not mismatches else "mismatch"
        self._last_reconciliation = {
            "status": status,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "mismatches": mismatches,
        }
        await self._state_store.write_reconciliation(self._last_reconciliation)
        if mismatches:
            fail_closed = (
                self._safety_config.fail_closed_on_startup_reconciliation
                and str(getattr(self._runtime_config.mode, "value", self._runtime_config.mode)).lower() == "live"
            )
            await self._emit_alert(
                kind="startup_reconciliation_mismatch",
                severity="critical" if fail_closed else "error",
                message=f"startup reconciliation detected {len(mismatches)} position mismatches",
                payload={"mismatches": mismatches},
            )
            if fail_closed:
                await self.engage_kill_switch(
                    f"startup reconciliation mismatch count={len(mismatches)}",
                    source="startup_reconciliation",
                )
                raise RuntimeError("startup reconciliation failed closed because broker positions differ from persisted state")

    def _position_mismatches(
        self,
        *,
        persisted_positions: Mapping[str, Mapping[str, Any]],
        broker_positions: Mapping[str, Mapping[str, Any]],
        tolerance: float,
    ) -> list[dict[str, Any]]:
        """Return normalized position mismatches."""

        mismatches: list[dict[str, Any]] = []
        symbols = set(persisted_positions) | set(broker_positions)
        for symbol in sorted(symbols):
            persisted_quantity = float(persisted_positions.get(symbol, {}).get("quantity", 0.0))
            broker_quantity = float(broker_positions.get(symbol, {}).get("quantity", 0.0))
            if abs(persisted_quantity - broker_quantity) <= tolerance:
                continue
            mismatches.append(
                {
                    "symbol": symbol,
                    "persisted_quantity": persisted_quantity,
                    "broker_quantity": broker_quantity,
                }
            )
        return mismatches

    async def _maybe_emit_latency_alert(self) -> None:
        """Emit a latency alert when observed p95 latency crosses the configured threshold."""

        report = self.latency_report
        if report is None or not report.top_bottlenecks:
            return
        top = report.top_bottlenecks[0]
        signature = (top.name, round(top.p95_ms, 3))
        if top.p95_ms < self._observability_config.latency_alert_threshold_ms:
            self._last_latency_alert_signature = None
            return
        if self._last_latency_alert_signature == signature:
            return
        self._last_latency_alert_signature = signature
        await self._emit_alert(
            kind="latency_threshold_breach",
            severity="warning",
            message=(
                f"latency metric {top.name} crossed the alert threshold "
                f"p95_ms={top.p95_ms:.3f} threshold_ms={self._observability_config.latency_alert_threshold_ms:.3f}"
            ),
            payload=top.to_dict(),
        )
