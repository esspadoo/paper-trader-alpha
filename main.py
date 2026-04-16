"""Runnable entrypoint for the integrated intraday trading system."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.agents import CriticAgent, CriticPolicy, DecisionAgent, MarketAgent, NewsAgent, RiskAgent, RiskPolicy
from trading_system.data import JSONNewsSource, RSSNewsSource
from trading_system.execution import IBKRBroker, IBKRBrokerConfig, OrderJournal, OrderManager, PaperBroker, PaperBrokerConfig
from trading_system.infra import LatencyRecorder
from trading_system.models import (
    LlamaCppBackend,
    LocalNewsLLMAnalyzer,
    MarketFeatureConfig,
    VLLMBackend,
    XGBoostReturnModelConfig,
)
from trading_system.system import (
    DemoLocalLLMBackend,
    IntegratedTradingSystem,
    KillSwitchState,
    PersistenceConfig,
    RuntimeStateStore,
    build_demo_market_frame,
    build_market_events,
    load_replay_market_events,
    load_system_config,
)
from trading_system.system.config import NewsConfig, SystemConfig


class JsonLogFormatter(logging.Formatter):
    """Simple structured JSON formatter for operational logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON-encoded log record."""

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


async def async_main(config_path: str) -> int:
    """Run the integrated trading workflow from a TOML configuration file."""

    return await async_run(config_path)


async def async_run(config_path: str) -> int:
    """Run the configured demo or paper-trading workflow."""

    config, system, market_frame = await build_integrated_system(config_path)
    events = build_market_events(market_frame, warmup_bars=config.market.training_bars)

    await system.start()
    try:
        await system.run(events, delay_seconds=config.runtime.publish_delay_seconds)
        snapshot = await system.snapshot_state()
    finally:
        await system.stop()

    logging.getLogger("trading_system.main").info(
        "run complete capital=%.2f positions=%s executions=%s",
        float(snapshot["account"]["capital"]),
        snapshot["positions"],
        len(snapshot["executions"]),
    )
    print(json.dumps(snapshot, indent=2, default=str))
    return 0


async def async_replay(config_path: str, *, audit_path: str | None = None) -> int:
    """Replay persisted market events through the current runtime."""

    config, system, _market_frame = await build_integrated_system(config_path)
    replay_path = (
        resolve_config_path(config_path=config_path, target_path=config.persistence.audit_journal_path)
        if audit_path is None
        else resolve_config_path(config_path=config_path, target_path=audit_path)
    )
    events = load_replay_market_events(replay_path)
    if not events:
        raise ValueError(f"no replayable MarketEvent records were found in {replay_path}")

    await system.start()
    try:
        await system.run(events, delay_seconds=0.0)
        snapshot = await system.snapshot_state()
    finally:
        await system.stop()

    print(json.dumps({"replayed_events": len(events), "snapshot": snapshot}, indent=2, default=str))
    return 0


async def async_health(config_path: str) -> int:
    """Print the latest persisted runtime health snapshot."""

    config = load_system_config(config_path)
    store = RuntimeStateStore(resolve_config_path(config_path=config_path, target_path=config.persistence.state_store_path))
    health = await store.read_health_snapshot()
    reconciliation = await store.read_reconciliation()
    kill_switch = await store.read_kill_switch()
    payload = {
        "health": health,
        "reconciliation": reconciliation,
        "kill_switch": kill_switch.to_dict(),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


async def async_metrics(config_path: str) -> int:
    """Print the latest persisted runtime metrics snapshot."""

    config = load_system_config(config_path)
    store = RuntimeStateStore(resolve_config_path(config_path=config_path, target_path=config.persistence.state_store_path))
    metrics = await store.read_metrics_snapshot()
    print(json.dumps({"metrics": metrics}, indent=2, default=str))
    return 0


async def async_kill_switch(config_path: str, *, action: str, reason: str | None = None) -> int:
    """Manage the persisted kill-switch state."""

    config = load_system_config(config_path)
    store = RuntimeStateStore(resolve_config_path(config_path=config_path, target_path=config.persistence.state_store_path))
    normalized_action = action.strip().lower()
    if normalized_action == "status":
        print(json.dumps((await store.read_kill_switch()).to_dict(), indent=2, default=str))
        return 0
    if normalized_action == "engage":
        normalized_reason = (reason or "").strip()
        if not normalized_reason:
            raise ValueError("kill-switch engage requires a non-empty --reason")
        state = KillSwitchState(
            engaged=True,
            reason=normalized_reason,
            engaged_at=datetime.now(timezone.utc).isoformat(),
            source="operator",
            strategy_id=config.runtime.strategy_id,
        )
        await store.write_kill_switch(state)
        print(json.dumps(state.to_dict(), indent=2, default=str))
        return 0
    if normalized_action == "release":
        state = KillSwitchState.disengaged()
        await store.write_kill_switch(state)
        print(json.dumps(state.to_dict(), indent=2, default=str))
        return 0
    raise ValueError(f"unsupported kill-switch action: {action}")


async def build_integrated_system(config_path: str) -> tuple[SystemConfig, IntegratedTradingSystem, object]:
    """Build the integrated runtime and its dependencies from config."""

    config = load_system_config(config_path)
    configure_logging(level=config.logging.level, log_format=config.logging.format)

    market_frame = build_demo_market_frame(config.market)
    training_frame = market_frame.iloc[: config.market.training_bars].copy(deep=True)
    training_frame = training_frame.set_index(["symbol", "timestamp"]).sort_index()

    market_agent = MarketAgent(
        model_config=XGBoostReturnModelConfig(
            feature_config=MarketFeatureConfig(prediction_horizon_bars=config.market.prediction_horizon_bars),
            validation_fraction=config.market.validation_fraction,
            min_training_rows=config.market.min_training_rows,
            n_estimators=config.market.n_estimators,
            learning_rate=config.market.learning_rate,
            max_depth=config.market.max_depth,
            n_jobs=1,
        )
    )
    await market_agent.train(training_frame)

    news_source = build_news_source(
        config_path=config_path,
        source_path=config.news.source_path,
        source_type=config.news.source_type,
    )
    news_agent = NewsAgent(
        source=news_source,
        analyzer=LocalNewsLLMAnalyzer(
            backend=build_news_backend(config.news),
            timeout_seconds=config.news.timeout_seconds,
            max_retries=config.news.max_retries,
        ),
    )
    decision_agent = DecisionAgent(action_threshold=config.decision.action_threshold)
    critic_agent = CriticAgent(
        policy=CriticPolicy(
            max_volatility=config.critic.max_volatility,
            min_confidence=config.critic.min_confidence,
            conflict_threshold=config.critic.conflict_threshold,
        )
    )
    risk_agent = RiskAgent(
        policy=RiskPolicy(
            max_risk_per_trade_fraction=config.risk.max_risk_per_trade_fraction,
            atr_period=config.risk.atr_period,
            atr_multiplier=config.risk.atr_multiplier,
            max_daily_loss_fraction=config.risk.max_daily_loss_fraction,
            max_exposure_fraction=config.risk.max_exposure_fraction,
        )
    )
    broker = build_broker(config)
    journal = OrderJournal(resolve_config_path(config_path=config_path, target_path=config.execution.journal_path))
    order_manager = OrderManager(
        broker,
        stale_order_seconds=config.execution.stale_order_seconds,
        journal=journal,
        logger=logging.getLogger("trading_system.order_manager"),
    )
    resolved_persistence = PersistenceConfig(
        state_store_path=str(resolve_config_path(config_path=config_path, target_path=config.persistence.state_store_path)),
        audit_journal_path=str(resolve_config_path(config_path=config_path, target_path=config.persistence.audit_journal_path)),
        dead_letter_path=str(resolve_config_path(config_path=config_path, target_path=config.persistence.dead_letter_path)),
        alert_journal_path=str(resolve_config_path(config_path=config_path, target_path=config.persistence.alert_journal_path)),
        snapshot_interval_seconds=config.persistence.snapshot_interval_seconds,
        replay_history_bars=config.persistence.replay_history_bars,
        enable_event_audit=config.persistence.enable_event_audit,
        enable_dead_letter_journal=config.persistence.enable_dead_letter_journal,
        enable_alert_journal=config.persistence.enable_alert_journal,
    )
    system = IntegratedTradingSystem(
        market_agent=market_agent,
        news_agent=news_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        risk_agent=risk_agent,
        broker=broker,
        order_manager=order_manager,
        execution_config=config.execution,
        runtime_config=config.runtime,
        safety_config=config.safety,
        persistence_config=resolved_persistence,
        observability_config=config.observability,
        news_max_age_seconds=config.news.max_article_age_seconds,
        strategy_id=config.runtime.strategy_id,
        logger=logging.getLogger("trading_system.runtime"),
        latency_recorder=LatencyRecorder(),
    )
    return config, system, market_frame


def build_news_backend(config: NewsConfig) -> object:
    """Build the configured local LLM backend."""

    backend_name = str(config.backend).strip().lower()
    if backend_name == "demo":
        return DemoLocalLLMBackend()
    if backend_name == "vllm":
        return VLLMBackend(model=str(config.model), base_url=str(config.base_url))
    if backend_name == "llama_cpp":
        return LlamaCppBackend(base_url=str(config.base_url))
    raise ValueError(f"unsupported news backend: {config.backend}")


def build_news_source(*, config_path: str, source_path: str, source_type: str) -> object:
    """Build the configured local news source."""

    normalized_source_type = source_type.strip().lower()
    if normalized_source_type == "json":
        resolved = resolve_config_path(config_path=config_path, target_path=source_path)
        return JSONNewsSource(source_name="configured-news", endpoint=str(resolved))
    if normalized_source_type == "rss":
        feed_uri = source_path if "://" in source_path else str(resolve_config_path(config_path=config_path, target_path=source_path))
        return RSSNewsSource(source_name="configured-news", feed_uri=feed_uri)
    raise ValueError(f"unsupported news source type: {source_type}")


def build_broker(config: SystemConfig) -> object:
    """Build the configured broker implementation."""

    broker_kind = str(config.execution.broker).strip().lower()
    if broker_kind == "paper":
        return PaperBroker(
            config=PaperBrokerConfig(
                initial_capital=config.execution.initial_capital,
                account=config.execution.account or "DU-PAPER",
                fill_latency_seconds=config.execution.fill_latency_seconds,
            ),
            logger=logging.getLogger("trading_system.paper_broker"),
        )
    if broker_kind == "ibkr":
        return IBKRBroker(
            config=IBKRBrokerConfig(
                host=config.execution.host,
                port=config.execution.port,
                client_id=config.execution.client_id,
                account=config.execution.account,
                paper_trading=config.execution.paper_trading,
                read_only=config.execution.read_only,
                connection_timeout_seconds=config.execution.connection_timeout_seconds,
                request_timeout_seconds=config.execution.request_timeout_seconds,
                max_retries=config.execution.max_retries,
                retry_backoff_seconds=config.execution.retry_backoff_seconds,
            ),
            logger=logging.getLogger("trading_system.ibkr_broker"),
        )
    raise ValueError(f"unsupported broker type: {config.execution.broker}")


def resolve_config_path(*, config_path: str, target_path: str) -> Path:
    """Resolve a config-relative path to an absolute path."""

    resolved = Path(target_path)
    if resolved.is_absolute():
        return resolved
    config_dir = Path(config_path).resolve().parent
    candidates = (
        config_dir / resolved,
        config_dir.parent / resolved,
        Path.cwd() / resolved,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return config_dir.parent / resolved


def configure_logging(*, level: str, log_format: str) -> None:
    """Configure root logging for the runnable application."""

    normalized_level = getattr(logging, level.upper(), logging.INFO)
    normalized_format = log_format.strip().lower()
    if normalized_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logging.basicConfig(level=normalized_level, handlers=[handler], force=True)
        return
    logging.basicConfig(level=normalized_level, format=log_format, force=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the integrated intraday trading system.")
    parser.add_argument(
        "--config",
        default="config/trading_system.example.toml",
        help="Path to the TOML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="Run the configured trading workflow.")
    replay_parser = subparsers.add_parser("replay", help="Replay persisted market events from the audit journal.")
    replay_parser.add_argument("--audit-path", default=None, help="Optional path to an audit journal JSONL file.")
    subparsers.add_parser("health", help="Print the latest persisted runtime health snapshot.")
    subparsers.add_parser("metrics", help="Print the latest persisted runtime metrics snapshot.")
    kill_switch_parser = subparsers.add_parser("kill-switch", help="Inspect or update the persisted kill-switch state.")
    kill_switch_parser.add_argument(
        "kill_switch_action",
        choices=("status", "engage", "release"),
        help="Kill-switch operation.",
    )
    kill_switch_parser.add_argument("--reason", default=None, help="Reason for engaging or releasing the kill switch.")
    args = parser.parse_args()
    if args.command is None:
        args.command = "run"
    return args


async def dispatch_command(args: argparse.Namespace) -> int:
    """Dispatch the parsed command-line request."""

    if args.command == "run":
        return await async_run(args.config)
    if args.command == "replay":
        return await async_replay(args.config, audit_path=args.audit_path)
    if args.command == "health":
        return await async_health(args.config)
    if args.command == "metrics":
        return await async_metrics(args.config)
    if args.command == "kill-switch":
        return await async_kill_switch(args.config, action=args.kill_switch_action, reason=args.reason)
    raise ValueError(f"unsupported command: {args.command}")


def main() -> int:
    """Run the application synchronously."""

    args = parse_args()
    return asyncio.run(dispatch_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
