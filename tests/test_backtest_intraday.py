"""Tests for the event-driven intraday backtester."""

from __future__ import annotations

import importlib.util
import math
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

from trading_system.agents import CriticAgent, CriticPolicy, DecisionAgent, RiskAgent, RiskPolicy
from trading_system.agents.base import BaseAgent
from trading_system.backtest import BacktestConfig, IntradayBacktester
from trading_system.core import BaseEvent, MarketEvent, NewsEvent


class ScriptedMarketAgent(BaseAgent):
    """Deterministic market agent for backtest integration tests."""

    def __init__(self) -> None:
        """Initialize the scripted market output state."""

        self._running = False
        self._last_output: dict[str, Any] | None = None

    @property
    def agent_id(self) -> str:
        """Return the stable identifier."""

        return "scripted-market-agent"

    @property
    def subscribed_topics(self) -> Sequence[str]:
        """Return the market topic."""

        return ("market",)

    async def start(self) -> None:
        """Mark the agent as active."""

        self._running = True

    async def stop(self) -> None:
        """Mark the agent as inactive."""

        self._running = False

    async def on_event(self, event: BaseEvent) -> None:
        """Emit a scripted market signal from event extras."""

        if not isinstance(event, MarketEvent):
            return
        extras = dict(event.payload.get("extras") or {})
        self._last_output = {
            "signal": float(extras.get("planned_signal", 0.0)),
            "confidence": float(extras.get("planned_confidence", 0.0)),
            "predicted_return": float(
                extras.get(
                    "planned_predicted_return",
                    float(extras.get("planned_signal", 0.0)) * 0.0018,
                )
            ),
            "features": {
                "volatility_20": float(extras.get("planned_volatility", 0.0)),
                "volume_spike": float(extras.get("planned_volume_spike", 0.0)),
                "spread_bps": float(extras.get("planned_spread_bps", 2.0)),
                "dollar_volume": float(extras.get("planned_dollar_volume", 20_000_000.0)),
                "ema_gap_9_21": float(extras.get("planned_ema_gap_9_21", 0.0020)),
                "ema_gap_21_50": float(extras.get("planned_ema_gap_21_50", 0.0014)),
                "vwap_gap": float(extras.get("planned_vwap_gap", 0.0008)),
                "rsi_14": float(extras.get("planned_rsi_14", 58.0)),
            },
        }

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot."""

        return {
            "agent_id": self.agent_id,
            "running": self._running,
            "last_output": self._last_output,
        }


class ScriptedNewsAgent(BaseAgent):
    """Deterministic news agent for backtest integration tests."""

    def __init__(self) -> None:
        """Initialize the scripted news output state."""

        self._running = False
        self._last_output: dict[str, float | str] | None = None

    @property
    def agent_id(self) -> str:
        """Return the stable identifier."""

        return "scripted-news-agent"

    @property
    def subscribed_topics(self) -> Sequence[str]:
        """Return the news topic."""

        return ("news",)

    async def start(self) -> None:
        """Mark the agent as active."""

        self._running = True

    async def stop(self) -> None:
        """Mark the agent as inactive."""

        self._running = False

    async def on_event(self, event: BaseEvent) -> None:
        """Emit a scripted news analysis from the event payload."""

        if not isinstance(event, NewsEvent):
            return
        payload = dict(event.payload or {})
        self._last_output = {
            "sentiment": float(payload.get("sentiment", 0.0)),
            "impact": float(payload.get("impact", 0.0)),
            "event_type": str(payload.get("event_type", "other")),
            "summary": str(payload.get("summary", "scripted news")),
        }

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot."""

        return {
            "agent_id": self.agent_id,
            "running": self._running,
            "last_output": self._last_output,
        }


def build_partial_fill_market_frame() -> "pd.DataFrame":
    """Build intraday bars that produce partial limit fills and a profitable exit."""

    timestamps = pd.date_range("2026-04-14 13:30:00+00:00", periods=6, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "timestamp": timestamps,
            "open": [100.0, 100.0, 101.0, 102.0, 103.0, 105.0],
            "high": [101.0, 102.0, 103.0, 104.0, 106.0, 107.0],
            "low": [99.0, 99.0, 100.0, 101.4, 101.5, 101.6],
            "close": [100.0, 101.0, 102.0, 103.0, 105.0, 106.0],
            "volume": [300.0] * 6,
            "planned_signal": [0.0, 0.0, 0.9, 0.9, 0.9, 0.9],
            "planned_confidence": [0.8] * 6,
            "planned_volatility": [0.01] * 6,
            "planned_predicted_return": [0.0, 0.0, 0.0036, 0.0034, 0.0031, 0.0029],
            "planned_spread_bps": [2.0] * 6,
            "planned_dollar_volume": [20_000_000.0] * 6,
        }
    )
    return frame


def build_stop_loss_market_frame() -> "pd.DataFrame":
    """Build intraday bars that trigger a stop-loss exit."""

    timestamps = pd.date_range("2026-04-15 13:30:00+00:00", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": timestamps,
            "open": [100.0, 100.0, 101.0, 99.0],
            "high": [101.0, 102.0, 102.0, 100.0],
            "low": [99.0, 99.0, 100.0, 96.0],
            "close": [100.0, 101.0, 101.0, 97.0],
            "volume": [1_000.0] * 4,
            "planned_signal": [0.0, 0.9, 0.9, 0.0],
            "planned_confidence": [0.8] * 4,
            "planned_volatility": [0.01] * 4,
            "planned_predicted_return": [0.0, 0.0035, 0.0030, 0.0],
            "planned_spread_bps": [2.0] * 4,
            "planned_dollar_volume": [20_000_000.0] * 4,
        }
    )
    return frame


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for backtest tests")
class IntradayBacktesterTests(unittest.TestCase):
    """Verify the intraday backtester integrates with agents and simulates fills."""

    def test_backtester_runs_agent_pipeline_with_partial_fills_and_metrics(self) -> None:
        """The backtester should simulate latency, partial fills, commissions, and metrics."""

        backtester = IntradayBacktester(
            config=BacktestConfig(
                initial_capital=100_000.0,
                slippage_bps=5.0,
                latency_ms=250,
                commission_per_share=0.01,
                minimum_commission=1.0,
                max_participation_rate=0.50,
            ),
            market_agent=ScriptedMarketAgent(),
            news_agent=ScriptedNewsAgent(),
            decision_agent=DecisionAgent(action_threshold=0.05),
            critic_agent=CriticAgent(
                policy=CriticPolicy(max_volatility=1.0, min_confidence=0.0, conflict_threshold=1.0)
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
        )

        result = backtester.run(
            build_partial_fill_market_frame(),
            news_events=[
                NewsEvent(
                    source="news-wire",
                    headline="Positive demand update",
                    symbols=("AAPL",),
                    payload={
                        "sentiment": 0.6,
                        "impact": 0.7,
                        "event_type": "product_launch",
                        "summary": "Demand commentary is positive.",
                    },
                    occurred_at=pd.Timestamp("2026-04-14 13:34:00+00:00").to_pydatetime(),
                )
            ],
        )

        self.assertGreaterEqual(len(result.fills), 3)
        self.assertEqual(result.fills[0].timestamp, pd.Timestamp("2026-04-14 13:45:00+00:00").to_pydatetime())
        self.assertIn("OrderEvent:PARTIALLY_FILLED", result.observed_events)
        self.assertIn("OrderEvent:FILLED", result.observed_events)
        self.assertGreater(result.metrics.total_commissions, 0.0)
        self.assertGreater(result.metrics.total_trades, 0)
        self.assertGreaterEqual(result.metrics.max_drawdown, 0.0)
        self.assertTrue(math.isinf(result.metrics.profit_factor) or result.metrics.profit_factor > 1.0)
        self.assertEqual(result.metadata["open_positions"], {})

    def test_backtester_honors_stop_loss_and_records_drawdown(self) -> None:
        """The backtester should execute stop-loss exits and reflect the loss in metrics."""

        backtester = IntradayBacktester(
            config=BacktestConfig(
                initial_capital=100_000.0,
                slippage_bps=5.0,
                latency_ms=250,
                commission_per_share=0.01,
                minimum_commission=1.0,
                max_participation_rate=1.0,
            ),
            market_agent=ScriptedMarketAgent(),
            news_agent=ScriptedNewsAgent(),
            decision_agent=DecisionAgent(action_threshold=0.05),
            critic_agent=CriticAgent(
                policy=CriticPolicy(max_volatility=1.0, min_confidence=0.0, conflict_threshold=1.0)
            ),
            risk_agent=RiskAgent(
                policy=RiskPolicy(
                    max_risk_per_trade_fraction=0.01,
                    atr_period=2,
                    atr_multiplier=1.0,
                    max_daily_loss_fraction=1.0,
                    max_exposure_fraction=1.0,
                )
            ),
        )

        result = backtester.run(
            build_stop_loss_market_frame(),
            news_events=[
                NewsEvent(
                    source="news-wire",
                    headline="Neutral news backdrop",
                    symbols=("AAPL",),
                    payload={
                        "sentiment": 0.4,
                        "impact": 0.2,
                        "event_type": "other",
                        "summary": "Minor constructive backdrop.",
                    },
                    occurred_at=pd.Timestamp("2026-04-15 13:31:00+00:00").to_pydatetime(),
                )
            ],
        )

        self.assertTrue(any(fill.reason == "stop_loss_triggered" for fill in result.fills))
        self.assertGreater(result.metrics.max_drawdown, 0.0)
        self.assertGreaterEqual(result.metrics.total_trades, 1)
        self.assertLessEqual(result.metrics.profit_factor, 1.0)
