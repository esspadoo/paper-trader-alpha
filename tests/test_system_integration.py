"""Integration tests for the runnable async trading-system pipeline."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

from trading_system.agents import CriticAgent, CriticPolicy, DecisionAgent, MarketAgent, NewsAgent, RiskAgent, RiskPolicy
from trading_system.data import JSONNewsSource
from trading_system.execution import OrderManager, PaperBroker, PaperBrokerConfig
from trading_system.models import LocalNewsLLMAnalyzer, MarketFeatureConfig, XGBoostReturnModelConfig
from trading_system.system import DemoLocalLLMBackend, IntegratedTradingSystem, MarketConfig, build_demo_market_frame, build_market_events
from trading_system.system.config import ExecutionConfig


@unittest.skipUnless(PANDAS_AVAILABLE and XGBOOST_AVAILABLE, "pandas and xgboost are required for system integration tests")
class IntegratedTradingSystemTests(unittest.IsolatedAsyncioTestCase):
    """Verify the end-to-end system can run from market event to execution."""

    async def test_runtime_processes_market_to_execution_end_to_end(self) -> None:
        """The integrated runtime should publish orders and update the paper account."""

        config = MarketConfig(training_bars=160, stream_bars=10, prediction_horizon_bars=2, min_training_rows=50, n_estimators=48)
        market_frame = build_demo_market_frame(config)
        training_frame = market_frame.iloc[: config.training_bars].copy(deep=True).set_index(["symbol", "timestamp"]).sort_index()
        market_agent = MarketAgent(
            model_config=XGBoostReturnModelConfig(
                feature_config=MarketFeatureConfig(prediction_horizon_bars=2),
                min_training_rows=50,
                n_estimators=48,
                max_depth=3,
                learning_rate=0.08,
                n_jobs=1,
            )
        )
        await market_agent.train(training_frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            news_path = Path(tmpdir) / "news.json"
            news_path.write_text(
                json.dumps(
                    {
                        "articles": [
                            {
                                "id": "a1",
                                "title": "AAPL launch drives strong demand",
                                "url": "https://local.test/a1",
                                "summary": "Strong launch and partnership demand update.",
                                "content": "The launch and partnership commentary highlight strong demand.",
                                "published_at": "2026-04-15T03:00:00+00:00",
                                "symbols": ["AAPL"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            news_agent = NewsAgent(
                source=JSONNewsSource(source_name="test-news", endpoint=str(news_path)),
                analyzer=LocalNewsLLMAnalyzer(backend=DemoLocalLLMBackend(), timeout_seconds=2.0, max_retries=2),
            )
            broker = PaperBroker(config=PaperBrokerConfig(initial_capital=100_000.0))
            system = IntegratedTradingSystem(
                market_agent=market_agent,
                news_agent=news_agent,
                decision_agent=DecisionAgent(action_threshold=0.0),
                critic_agent=CriticAgent(
                    policy=CriticPolicy(max_volatility=10.0, min_confidence=0.0, conflict_threshold=1.0)
                ),
                risk_agent=RiskAgent(
                    policy=RiskPolicy(
                        max_risk_per_trade_fraction=0.01,
                        atr_period=5,
                        atr_multiplier=1.0,
                        max_daily_loss_fraction=1.0,
                        max_exposure_fraction=1.0,
                    )
                ),
                broker=broker,
                order_manager=OrderManager(broker, stale_order_seconds=15.0),
                execution_config=ExecutionConfig(initial_capital=100_000.0),
                strategy_id="integration-test",
            )
            events = build_market_events(market_frame, warmup_bars=config.training_bars)
            await system.start()
            try:
                await system.run(events)
                snapshot = await system.snapshot_state()
            finally:
                await system.stop()

        self.assertGreaterEqual(len(snapshot["executions"]), 1)
        self.assertIn("OrderEvent:FILLED", snapshot["observed_events"])
        self.assertIn("NewsEvent", snapshot["observed_events"])
        self.assertGreater(float(snapshot["account"]["capital"]), 0.0)
