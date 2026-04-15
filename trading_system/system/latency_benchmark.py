"""Reproducible latency benchmark for the integrated trading system."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_system.agents import CriticAgent, CriticPolicy, DecisionAgent, MarketAgent, NewsAgent, RiskAgent, RiskPolicy
from trading_system.data import JSONNewsSource
from trading_system.execution import OrderJournal, OrderManager, PaperBroker, PaperBrokerConfig
from trading_system.infra import LatencyRecorder, LatencyReport
from trading_system.models import (
    LocalNewsLLMAnalyzer,
    MarketFeatureConfig,
    XGBoostReturnModelConfig,
)
from trading_system.system.config import ExecutionConfig, MarketConfig, RuntimeConfig
from trading_system.system.demo import DemoLocalLLMBackend, build_demo_market_frame, build_market_events
from trading_system.system.runtime import IntegratedTradingSystem


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Input configuration for a reproducible latency benchmark."""

    symbol: str = "AAPL"
    training_bars: int = 180
    stream_bars: int = 240
    prediction_horizon_bars: int = 2
    news_max_age_seconds: float = 14_400.0
    latency_sample_limit: int = 8_192


async def run_latency_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Run a deterministic latency benchmark and return a structured report."""

    benchmark = config or BenchmarkConfig()
    logger = logging.getLogger("trading_system.latency_benchmark")
    logger.setLevel(logging.WARNING)

    recorder = LatencyRecorder(sample_limit=benchmark.latency_sample_limit)
    market_config = MarketConfig(
        symbol=benchmark.symbol,
        training_bars=benchmark.training_bars,
        stream_bars=benchmark.stream_bars,
        prediction_horizon_bars=benchmark.prediction_horizon_bars,
        min_training_rows=60,
        n_estimators=96,
    )
    market_frame = build_demo_market_frame(market_config)
    training_frame = market_frame.iloc[: market_config.training_bars].copy(deep=True).set_index(["symbol", "timestamp"]).sort_index()

    market_agent = MarketAgent(
        model_config=XGBoostReturnModelConfig(
            feature_config=MarketFeatureConfig(prediction_horizon_bars=benchmark.prediction_horizon_bars),
            min_training_rows=60,
            n_estimators=96,
            learning_rate=0.05,
            max_depth=4,
            n_jobs=1,
        ),
        latency_recorder=recorder,
    )
    await market_agent.train(training_frame)

    with tempfile.TemporaryDirectory() as tmpdir:
        news_path = Path(tmpdir) / "benchmark_news.json"
        news_path.write_text(json.dumps(_build_news_payload()), encoding="utf-8")
        broker = PaperBroker(config=PaperBrokerConfig(initial_capital=100_000.0))
        order_manager = OrderManager(
            broker,
            stale_order_seconds=15.0,
            journal=OrderJournal(Path(tmpdir) / "orders.json"),
            latency_recorder=recorder,
        )
        system = IntegratedTradingSystem(
            market_agent=market_agent,
            news_agent=NewsAgent(
                source=JSONNewsSource(source_name="benchmark-news", endpoint=str(news_path)),
                analyzer=LocalNewsLLMAnalyzer(
                    backend=DemoLocalLLMBackend(),
                    timeout_seconds=2.0,
                    max_retries=2,
                ),
            ),
            decision_agent=DecisionAgent(action_threshold=0.0),
            critic_agent=CriticAgent(
                policy=CriticPolicy(max_volatility=10.0, min_confidence=0.0, conflict_threshold=1.0)
            ),
            risk_agent=RiskAgent(
                policy=RiskPolicy(
                    max_risk_per_trade_fraction=0.001,
                    atr_period=5,
                    atr_multiplier=1.0,
                    max_daily_loss_fraction=1.0,
                    max_exposure_fraction=1.0,
                )
            ),
            broker=broker,
            order_manager=order_manager,
            execution_config=ExecutionConfig(initial_capital=100_000.0, journal_path=str(Path(tmpdir) / "orders.json")),
            runtime_config=RuntimeConfig(strategy_id="latency-benchmark"),
            strategy_id="latency-benchmark",
            logger=logger,
            latency_recorder=recorder,
            news_max_age_seconds=benchmark.news_max_age_seconds,
        )
        events = build_market_events(market_frame, warmup_bars=market_config.training_bars)
        await system.start()
        try:
            await system.run(events)
            snapshot = await system.snapshot_state()
        finally:
            await system.stop()

    report = system.latency_report or recorder.report()
    return {
        "config": {
            "symbol": benchmark.symbol,
            "training_bars": benchmark.training_bars,
            "stream_bars": benchmark.stream_bars,
            "prediction_horizon_bars": benchmark.prediction_horizon_bars,
            "news_max_age_seconds": benchmark.news_max_age_seconds,
        },
        "latency_report": report.to_dict(),
        "executions": len(snapshot["executions"]),
        "observed_events": len(snapshot["observed_events"]),
    }


def _build_news_payload() -> dict[str, Any]:
    """Return deterministic benchmark articles spread across the simulated session."""

    return {
        "articles": [
            {
                "id": "bench-1",
                "title": "AAPL launch drives strong demand",
                "url": "https://local.test/bench-1",
                "summary": "Launch and partnership commentary remain constructive.",
                "content": "Strong launch and partnership demand support upside sentiment.",
                "published_at": "2026-04-15T05:00:00+00:00",
                "symbols": ["AAPL"],
            },
            {
                "id": "bench-2",
                "title": "Sector update remains neutral",
                "url": "https://local.test/bench-2",
                "summary": "Macro conditions are mixed with limited immediate impact.",
                "content": "Macro conditions are neutral with limited near-term impact.",
                "published_at": "2026-04-15T05:40:00+00:00",
                "symbols": [],
            },
            {
                "id": "bench-3",
                "title": "AAPL lawsuit creates weak sentiment",
                "url": "https://local.test/bench-3",
                "summary": "Litigation and weak outlook create downside risk.",
                "content": "A new lawsuit and weak demand commentary weigh on sentiment.",
                "published_at": "2026-04-15T06:20:00+00:00",
                "symbols": ["AAPL"],
            },
        ]
    }


def render_latency_report(report: LatencyReport) -> str:
    """Render a Markdown latency summary."""

    lines = [
        "| Metric | Count | Mean ms | p95 ms | p99 ms | Worst ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in report.metrics:
        lines.append(
            f"| {metric.name} | {metric.count} | {metric.mean_ms:.3f} | {metric.p95_ms:.3f} | {metric.p99_ms:.3f} | {metric.worst_ms:.3f} |"
        )
    if report.top_bottlenecks:
        lines.append("")
        lines.append("Top bottlenecks:")
        for metric in report.top_bottlenecks:
            lines.append(
                f"- `{metric.name}`: mean={metric.mean_ms:.3f}ms p95={metric.p95_ms:.3f}ms p99={metric.p99_ms:.3f}ms worst={metric.worst_ms:.3f}ms"
            )
    return "\n".join(lines)
