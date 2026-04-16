"""Operational readiness tests for persisted safety and reconciliation controls."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

from trading_system.agents import BaseAgent, CriticAgent, CriticPolicy, DecisionAgent, NewsAgent, RiskAgent, RiskPolicy
from trading_system.core.events import BaseEvent, MarketEvent
from trading_system.data import JSONNewsSource
from trading_system.execution import OrderJournal, OrderManager, PaperBroker, PaperBrokerConfig
from trading_system.models import LocalNewsLLMAnalyzer
from trading_system.system import (
    DemoLocalLLMBackend,
    DeploymentMode,
    IntegratedTradingSystem,
    KillSwitchState,
    PersistenceConfig,
    RuntimeConfig,
    RuntimeStateStore,
)
from trading_system.system.config import ExecutionConfig


class StaticMarketAgent(BaseAgent):
    """Minimal deterministic market agent for operational tests."""

    def __init__(self) -> None:
        """Initialize a static long signal."""

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
        """Return the stable agent identifier."""

        return "static-market-agent"

    @property
    def subscribed_topics(self) -> tuple[str, ...]:
        """Return the supported topics."""

        return ("market",)

    async def start(self) -> None:
        """Start the agent."""

    async def stop(self) -> None:
        """Stop the agent."""

    async def on_event(self, event: BaseEvent) -> None:
        """Keep the static output unchanged."""

    async def snapshot_state(self) -> dict[str, object]:
        """Return the latest output."""

        return {
            "agent_id": self.agent_id,
            "running": True,
            "last_output": dict(self._last_output),
        }


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for operational readiness tests")
class OperationalReadinessTests(unittest.IsolatedAsyncioTestCase):
    """Verify persisted operational controls behave safely."""

    async def test_strategy_scoped_kill_switch_restores_and_blocks_order_flow(self) -> None:
        """A persisted kill switch for the matching strategy should suppress executions after restart."""

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStateStore(Path(tmpdir) / "runtime_state.db")
            await store.write_kill_switch(
                KillSwitchState(
                    engaged=True,
                    reason="manual halt",
                    engaged_at=datetime.now(timezone.utc).isoformat(),
                    source="operator",
                    strategy_id="ops-kill-switch",
                )
            )
            system, event = self._build_system(
                tmpdir=tmpdir,
                strategy_id="ops-kill-switch",
                persistence=PersistenceConfig(
                    state_store_path=str(Path(tmpdir) / "runtime_state.db"),
                    audit_journal_path=str(Path(tmpdir) / "events.jsonl"),
                    dead_letter_path=str(Path(tmpdir) / "dead_letters.jsonl"),
                    alert_journal_path=str(Path(tmpdir) / "alerts.jsonl"),
                ),
            )
            await system.start()
            try:
                await system.run([event])
                snapshot = await system.snapshot_state()
            finally:
                await system.stop()

        self.assertTrue(snapshot["kill_switch"]["engaged"])
        self.assertEqual(snapshot["kill_switch"]["strategy_id"], "ops-kill-switch")
        self.assertEqual(len(snapshot["executions"]), 0)

    async def test_live_mode_startup_reconciliation_fails_closed_on_position_mismatch(self) -> None:
        """Live-mode startup should fail closed when persisted positions differ from the broker."""

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStateStore(Path(tmpdir) / "runtime_state.db")
            await store.write_trading_state(
                {
                    "strategy_id": "live-reconcile",
                    "account": {"account": "DU-PAPER"},
                    "positions": {"AAPL": {"quantity": 25.0}},
                    "daily_start_capital": {},
                }
            )
            system, _event = self._build_system(
                tmpdir=tmpdir,
                strategy_id="live-reconcile",
                runtime_config=RuntimeConfig(strategy_id="live-reconcile", mode=DeploymentMode.LIVE),
                persistence=PersistenceConfig(
                    state_store_path=str(Path(tmpdir) / "runtime_state.db"),
                    audit_journal_path=str(Path(tmpdir) / "events.jsonl"),
                    dead_letter_path=str(Path(tmpdir) / "dead_letters.jsonl"),
                    alert_journal_path=str(Path(tmpdir) / "alerts.jsonl"),
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "startup reconciliation failed closed"):
                await system.start()

            reconciliation = await store.read_reconciliation()
            kill_switch = await store.read_kill_switch()

        self.assertEqual(reconciliation["status"], "mismatch")
        self.assertTrue(kill_switch.engaged)
        self.assertEqual(kill_switch.strategy_id, "live-reconcile")

    def _build_system(
        self,
        *,
        tmpdir: str,
        strategy_id: str,
        persistence: PersistenceConfig,
        runtime_config: RuntimeConfig | None = None,
    ) -> tuple[IntegratedTradingSystem, MarketEvent]:
        """Build a deterministic runtime and a single market event."""

        news_path = Path(tmpdir) / "news.json"
        news_path.write_text(json.dumps({"articles": []}), encoding="utf-8")
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
                source=JSONNewsSource(source_name="ops-news", endpoint=str(news_path)),
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
                journal=OrderJournal(Path(tmpdir) / "orders.db"),
            ),
            execution_config=ExecutionConfig(initial_capital=100_000.0, journal_path=str(Path(tmpdir) / "orders.db")),
            runtime_config=runtime_config or RuntimeConfig(strategy_id=strategy_id),
            persistence_config=persistence,
            news_max_age_seconds=3_600.0,
            strategy_id=strategy_id,
        )
        return system, event
