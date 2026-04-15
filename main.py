"""Runnable entrypoint for the integrated intraday trading system."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from trading_system.agents import CriticAgent, CriticPolicy, DecisionAgent, MarketAgent, NewsAgent, RiskAgent, RiskPolicy
from trading_system.data import JSONNewsSource, RSSNewsSource
from trading_system.execution import IBKRBroker, IBKRBrokerConfig, OrderJournal, OrderManager, PaperBroker, PaperBrokerConfig
from trading_system.models import (
    LlamaCppBackend,
    LocalNewsLLMAnalyzer,
    MarketFeatureConfig,
    VLLMBackend,
    XGBoostReturnModelConfig,
)
from trading_system.system import DemoLocalLLMBackend, IntegratedTradingSystem, build_demo_market_frame, build_market_events, load_system_config
from trading_system.system.config import NewsConfig, SystemConfig


async def async_main(config_path: str) -> int:
    """Run the integrated trading workflow from a TOML configuration file."""

    config = load_system_config(config_path)
    configure_logging(level=config.logging.level, log_format=config.logging.format)
    logger = logging.getLogger("trading_system.main")

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

    news_source = build_news_source(config_path=config_path, source_path=config.news.source_path, source_type=config.news.source_type)
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
        news_max_age_seconds=config.news.max_article_age_seconds,
        strategy_id=config.runtime.strategy_id,
        logger=logging.getLogger("trading_system.runtime"),
    )

    events = build_market_events(market_frame, warmup_bars=config.market.training_bars)
    await system.start()
    try:
        await system.run(events, delay_seconds=config.runtime.publish_delay_seconds)
        snapshot = await system.snapshot_state()
    finally:
        await system.stop()

    logger.info(
        "run complete capital=%.2f positions=%s executions=%s",
        float(snapshot["account"]["capital"]),
        snapshot["positions"],
        len(snapshot["executions"]),
    )
    print(json.dumps(snapshot, indent=2, default=str))
    return 0


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

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=log_format)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the integrated intraday trading system.")
    parser.add_argument(
        "--config",
        default="config/trading_system.example.toml",
        help="Path to the TOML configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the application synchronously."""

    args = parse_args()
    return asyncio.run(async_main(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
