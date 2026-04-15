"""Safety and deployment-readiness tests for the integrated trading runtime."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from time import perf_counter

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

from trading_system.agents import BaseAgent, CriticAgent, CriticPolicy, DecisionAgent, NewsAgent, RiskAgent, RiskPolicy
from trading_system.core.events import BaseEvent, MarketEvent
from trading_system.data import JSONNewsSource
from trading_system.execution import OrderJournal, OrderManager, PaperBroker, PaperBrokerConfig
from trading_system.models import LocalNewsLLMAnalyzer
from trading_system.system import DemoLocalLLMBackend, IntegratedTradingSystem, RuntimeConfig, load_system_config
from trading_system.system.config import ExecutionConfig


class StaticMarketAgent(BaseAgent):
    """Minimal deterministic market agent used by runtime safety tests."""

    def __init__(self) -> None:
        """Initialize the static signal payload."""

        self._last_output = {
            "signal": 0.8,
            "confidence": 0.9,
            "features": {
                "volatility": 0.01,
                "ema_9": 101.0,
                "ema_21": 100.5,
                "ema_50": 100.0,
                "rsi": 58.0,
                "vwap": 100.8,
                "volume_spike": 1.1,
            },
        }

    @property
    def agent_id(self) -> str:
        """Return the stable agent id."""

        return "static-market-agent"

    @property
    def subscribed_topics(self) -> tuple[str, ...]:
        """Return the supported topic set."""

        return ("market",)

    async def start(self) -> None:
        """Start the agent."""

    async def stop(self) -> None:
        """Stop the agent."""

    async def on_event(self, event: BaseEvent) -> None:
        """Ignore market events because the signal is static."""

    async def snapshot_state(self) -> dict[str, object]:
        """Return the latest static output."""

        return {
            "agent_id": self.agent_id,
            "running": True,
            "last_output": dict(self._last_output),
        }


class SlowDemoLocalLLMBackend(DemoLocalLLMBackend):
    """Deterministic backend that delays analysis to test runtime decoupling."""

    def __init__(self, *, delay_seconds: float) -> None:
        """Initialize the backend with a fixed async delay."""

        self._delay_seconds = delay_seconds

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        timeout_seconds: float,
    ) -> str:
        """Delay before returning the deterministic demo response."""

        await asyncio.sleep(self._delay_seconds)
        return await super().generate(messages=messages, prompt=prompt, timeout_seconds=timeout_seconds)


class SystemConfigValidationTests(unittest.TestCase):
    """Verify the validated system configuration fails closed."""

    def test_live_mode_requires_explicit_account_confirmation(self) -> None:
        """Live deployment should be rejected without the explicit safety gate."""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "live.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[news]",
                        'source_type = "rss"',
                        'source_path = "https://example.com/feed.xml"',
                        'backend = "vllm"',
                        'model = "local-model"',
                        'base_url = "http://127.0.0.1:8000"',
                        "",
                        "[execution]",
                        'broker = "ibkr"',
                        "paper_trading = false",
                        'account = "U1234567"',
                        "",
                        "[runtime]",
                        'mode = "live"',
                        "allow_demo_components = false",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "allow_live_trading"):
                load_system_config(config_path)


class OrderJournalTests(unittest.IsolatedAsyncioTestCase):
    """Verify order idempotency survives process restarts."""

    async def test_order_manager_suppresses_duplicate_submission_across_restart(self) -> None:
        """A persisted client order key should prevent duplicate fills after restart."""

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = OrderJournal(Path(tmpdir) / "orders.json")
            broker = PaperBroker(config=PaperBrokerConfig(initial_capital=100_000.0))
            manager = OrderManager(broker, stale_order_seconds=10.0, journal=journal)

            await broker.connect()
            initial = await manager.submit_limit_order(
                symbol="AAPL",
                side="BUY",
                quantity=10.0,
                limit_price=100.0,
                order_ref="client-order:test-order-1",
                client_order_key="test-order-1",
            )
            await broker.disconnect()

            restarted_manager = OrderManager(
                broker,
                stale_order_seconds=10.0,
                journal=OrderJournal(Path(tmpdir) / "orders.json"),
            )
            await broker.connect()
            duplicate = await restarted_manager.submit_limit_order(
                symbol="AAPL",
                side="BUY",
                quantity=10.0,
                limit_price=100.0,
                order_ref="client-order:test-order-1",
                client_order_key="test-order-1",
            )
            account = await broker.get_account_snapshot()
            positions = await broker.get_positions()
            await broker.disconnect()

        self.assertEqual(initial["order_id"], duplicate["order_id"])
        self.assertEqual(positions["AAPL"]["quantity"], 10.0)
        self.assertEqual(account["cash"], 99_000.0)


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for runtime safety tests")
class IntegratedRuntimeSafetyTests(unittest.IsolatedAsyncioTestCase):
    """Verify the integrated runtime enforces stale-news and journal safeguards."""

    async def test_runtime_ignores_stale_news_and_executes_from_fresh_market_signal(self) -> None:
        """Articles older than the configured TTL should not influence decisions."""

        with tempfile.TemporaryDirectory() as tmpdir:
            news_path = Path(tmpdir) / "news.json"
            news_path.write_text(
                json.dumps(
                    {
                        "articles": [
                            {
                                "id": "n1",
                                "title": "AAPL lawsuit weak demand update",
                                "url": "https://local.test/stale-news",
                                "summary": "Weak guidance and lawsuit pressure sentiment lower.",
                                "content": "The company faces a lawsuit and weak demand conditions.",
                                "published_at": "2026-04-15T13:00:00+00:00",
                                "symbols": ["AAPL"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            timestamps = pd.date_range("2026-04-15T15:00:00+00:00", periods=8, freq="5min", tz="UTC")
            ohlcv = pd.DataFrame(
                {
                    "open": [100.0, 100.2, 100.5, 100.7, 100.9, 101.1, 101.2, 101.4],
                    "high": [100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.5, 101.7],
                    "low": [99.8, 100.0, 100.2, 100.5, 100.7, 100.9, 101.0, 101.2],
                    "close": [100.2, 100.5, 100.7, 100.9, 101.1, 101.2, 101.4, 101.6],
                    "volume": [1_200_000.0] * 8,
                },
                index=timestamps,
            )

            event = MarketEvent(
                source="test-market-feed",
                symbol="AAPL",
                venue="SIM",
                bid=101.6,
                ask=101.6,
                last_price=101.6,
                volume=1_200_000.0,
                payload={
                    "bar": {
                        "open": 101.4,
                        "high": 101.7,
                        "low": 101.2,
                        "close": 101.6,
                        "volume": 1_200_000.0,
                    },
                    "ohlcv": ohlcv,
                },
                occurred_at=pd.Timestamp("2026-04-15T15:00:00+00:00").to_pydatetime(),
            )

            broker = PaperBroker(config=PaperBrokerConfig(initial_capital=100_000.0))
            system = IntegratedTradingSystem(
                market_agent=StaticMarketAgent(),
                news_agent=NewsAgent(
                    source=JSONNewsSource(source_name="stale-news", endpoint=str(news_path)),
                    analyzer=LocalNewsLLMAnalyzer(backend=DemoLocalLLMBackend(), timeout_seconds=2.0, max_retries=2),
                ),
                decision_agent=DecisionAgent(action_threshold=0.5),
                critic_agent=CriticAgent(
                    policy=CriticPolicy(max_volatility=10.0, min_confidence=0.0, conflict_threshold=1.0)
                ),
                risk_agent=RiskAgent(
                    policy=RiskPolicy(
                        max_risk_per_trade_fraction=0.01,
                        atr_period=3,
                        atr_multiplier=1.0,
                        max_daily_loss_fraction=1.0,
                        max_exposure_fraction=1.0,
                    )
                ),
                broker=broker,
                order_manager=OrderManager(
                    broker,
                    stale_order_seconds=15.0,
                    journal=OrderJournal(Path(tmpdir) / "orders.json"),
                ),
                execution_config=ExecutionConfig(initial_capital=100_000.0, journal_path=str(Path(tmpdir) / "orders.json")),
                runtime_config=RuntimeConfig(strategy_id="stale-news-test", max_log_reason_length=96),
                news_max_age_seconds=60.0,
                strategy_id="stale-news-test",
            )

            await system.start()
            try:
                await system.run([event])
                snapshot = await system.snapshot_state()
            finally:
                await system.stop()

        self.assertGreaterEqual(len(snapshot["executions"]), 1)
        self.assertNotIn("NewsEvent", snapshot["observed_events"])

    async def test_runtime_does_not_block_market_decision_on_slow_news_analysis(self) -> None:
        """Market decisions should proceed from cached news state while slow news analysis runs in the background."""

        with tempfile.TemporaryDirectory() as tmpdir:
            news_path = Path(tmpdir) / "news.json"
            news_path.write_text(
                json.dumps(
                    {
                        "articles": [
                            {
                                "id": "n2",
                                "title": "AAPL lawsuit weak demand update",
                                "url": "https://local.test/fresh-slow-news",
                                "summary": "Weak guidance and lawsuit pressure sentiment lower.",
                                "content": "The company faces a lawsuit and weak demand conditions.",
                                "published_at": "2026-04-15T15:00:00+00:00",
                                "symbols": ["AAPL"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            timestamps = pd.date_range("2026-04-15T14:25:00+00:00", periods=8, freq="5min", tz="UTC")
            ohlcv = pd.DataFrame(
                {
                    "open": [100.0, 100.1, 100.3, 100.5, 100.7, 100.8, 101.0, 101.2],
                    "high": [100.3, 100.5, 100.7, 100.9, 101.0, 101.2, 101.4, 101.6],
                    "low": [99.8, 99.9, 100.1, 100.3, 100.5, 100.6, 100.8, 101.0],
                    "close": [100.1, 100.3, 100.5, 100.7, 100.8, 101.0, 101.2, 101.4],
                    "volume": [1_200_000.0] * 8,
                },
                index=timestamps,
            )
            event = MarketEvent(
                source="test-market-feed",
                symbol="AAPL",
                venue="SIM",
                bid=101.4,
                ask=101.4,
                last_price=101.4,
                volume=1_200_000.0,
                payload={
                    "bar": {
                        "open": 101.2,
                        "high": 101.6,
                        "low": 101.0,
                        "close": 101.4,
                        "volume": 1_200_000.0,
                    },
                    "ohlcv": ohlcv,
                },
                occurred_at=pd.Timestamp("2026-04-15T15:00:00+00:00").to_pydatetime(),
            )

            broker = PaperBroker(config=PaperBrokerConfig(initial_capital=100_000.0))
            system = IntegratedTradingSystem(
                market_agent=StaticMarketAgent(),
                news_agent=NewsAgent(
                    source=JSONNewsSource(source_name="slow-news", endpoint=str(news_path)),
                    analyzer=LocalNewsLLMAnalyzer(
                        backend=SlowDemoLocalLLMBackend(delay_seconds=0.25),
                        timeout_seconds=1.0,
                        max_retries=2,
                    ),
                ),
                decision_agent=DecisionAgent(action_threshold=0.5),
                critic_agent=CriticAgent(
                    policy=CriticPolicy(max_volatility=10.0, min_confidence=0.0, conflict_threshold=1.0)
                ),
                risk_agent=RiskAgent(
                    policy=RiskPolicy(
                        max_risk_per_trade_fraction=0.01,
                        atr_period=3,
                        atr_multiplier=1.0,
                        max_daily_loss_fraction=1.0,
                        max_exposure_fraction=1.0,
                    )
                ),
                broker=broker,
                order_manager=OrderManager(
                    broker,
                    stale_order_seconds=15.0,
                    journal=OrderJournal(Path(tmpdir) / "orders.db"),
                ),
                execution_config=ExecutionConfig(initial_capital=100_000.0, journal_path=str(Path(tmpdir) / "orders.db")),
                runtime_config=RuntimeConfig(strategy_id="slow-news-test", max_log_reason_length=96),
                news_max_age_seconds=3_600.0,
                strategy_id="slow-news-test",
            )

            await system.start()
            try:
                started_at = perf_counter()
                await system.run([event])
                run_elapsed = perf_counter() - started_at
                snapshot = await system.snapshot_state()
            finally:
                await system.stop()

        self.assertLess(run_elapsed, 0.20)
        self.assertGreaterEqual(len(snapshot["executions"]), 1)
        self.assertIn("NewsEvent", snapshot["observed_events"])
