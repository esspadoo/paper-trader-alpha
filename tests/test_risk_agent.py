"""Tests for deterministic trade-risk sizing and overrides."""

from __future__ import annotations

import importlib.util
import unittest

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

from trading_system.agents import RiskAgent, RiskPolicy
from trading_system.core import MarketEvent, SignalEvent


def build_constant_range_ohlcv(symbol: str = "AAPL", bars: int = 20) -> "pd.DataFrame":
    """Build a simple intraday OHLCV frame with a stable ATR."""

    timestamps = pd.date_range("2026-04-14 13:30:00+00:00", periods=bars, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [101.0] * bars,
            "high": [102.0] * bars,
            "low": [100.0] * bars,
            "close": [101.0] * bars,
            "volume": [1_000_000.0] * bars,
        },
        index=timestamps,
    )
    frame["symbol"] = symbol
    return frame.reset_index(names="timestamp").set_index(["symbol", "timestamp"])


def build_variable_range_ohlcv(symbol: str = "AAPL", bars: int = 20) -> "pd.DataFrame":
    """Build intraday OHLCV bars with a changing true range for ATR tests."""

    frame = build_constant_range_ohlcv(symbol=symbol, bars=bars).copy(deep=True)
    for index in range(bars):
        frame.iloc[index, frame.columns.get_loc("open")] = 101.0 + (0.1 * index)
        frame.iloc[index, frame.columns.get_loc("high")] = 102.0 + (0.2 * index)
        frame.iloc[index, frame.columns.get_loc("low")] = 100.0 - (0.05 * index)
        frame.iloc[index, frame.columns.get_loc("close")] = 101.0 + (0.12 * index)
    return frame


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for risk-agent tests")
class RiskAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify deterministic sizing, stop loss, and hard overrides."""

    async def test_risk_agent_sizes_trade_from_one_percent_risk_budget(self) -> None:
        """The risk agent should size positions from 1 percent capital and ATR stop distance."""

        ohlcv = build_constant_range_ohlcv()
        agent = RiskAgent(
            policy=RiskPolicy(
                max_risk_per_trade_fraction=0.01,
                atr_period=14,
                atr_multiplier=2.0,
                max_daily_loss_fraction=0.03,
                max_exposure_fraction=1.0,
            )
        )

        output = await agent.assess(
            {"action": "BUY", "score": 0.42, "reasoning": "positive blended signal"},
            market_signal={"signal": 0.6, "confidence": 0.8, "features": {"volatility_20": 0.012}},
            account_state={"capital": 100_000.0, "daily_pnl": 0.0, "gross_exposure": 0.0},
            ohlcv=ohlcv,
            symbol="AAPL",
        )
        snapshot = await agent.snapshot_state()

        self.assertEqual(output["final_action"], "BUY")
        self.assertAlmostEqual(output["position_size"], 250.0)
        self.assertAlmostEqual(output["stop_loss"], 97.0)
        self.assertIn("risk-approved BUY", snapshot["last_reason"])

    async def test_risk_agent_overrides_to_hold_when_daily_loss_is_breached(self) -> None:
        """The risk layer should cancel new trades after the daily loss limit is hit."""

        ohlcv = build_constant_range_ohlcv()
        agent = RiskAgent(policy=RiskPolicy(max_daily_loss_fraction=0.03, max_exposure_fraction=1.0))

        output = await agent.assess(
            {"action": "BUY", "score": 0.38, "reasoning": "setup is valid before risk"},
            market_signal={"signal": 0.5, "confidence": 0.75, "features": {"volatility_20": 0.012}},
            account_state={"capital": 100_000.0, "daily_loss": 3_200.0, "gross_exposure": 0.0},
            ohlcv=ohlcv,
            symbol="AAPL",
        )

        self.assertEqual(output["final_action"], "HOLD")
        self.assertEqual(output["position_size"], 0.0)
        self.assertEqual(output["stop_loss"], 0.0)

    async def test_risk_agent_overrides_to_hold_when_exposure_limit_is_reached(self) -> None:
        """The risk layer should reject trades once maximum exposure is exhausted."""

        ohlcv = build_constant_range_ohlcv()
        agent = RiskAgent(policy=RiskPolicy(max_daily_loss_fraction=0.03, max_exposure_fraction=0.10))

        output = await agent.assess(
            {"action": "SELL", "score": -0.44, "reasoning": "negative blended signal"},
            market_signal={"signal": -0.7, "confidence": 0.82, "features": {"volatility_20": 0.018}},
            account_state={"capital": 100_000.0, "daily_pnl": 0.0, "gross_exposure": 10_000.0},
            ohlcv=ohlcv,
            symbol="AAPL",
        )

        self.assertEqual(output["final_action"], "HOLD")
        self.assertEqual(output["position_size"], 0.0)
        self.assertEqual(output["stop_loss"], 0.0)

    async def test_risk_agent_updates_from_events(self) -> None:
        """Decision and market events should trigger a risk assessment when context is complete."""

        ohlcv = build_constant_range_ohlcv(symbol="MSFT")
        agent = RiskAgent(policy=RiskPolicy(max_exposure_fraction=1.0))

        market_signal_event = SignalEvent(
            source="market-agent",
            symbol="MSFT",
            strategy_id="market-alpha",
            confidence=0.88,
            payload={
                "signal": -0.55,
                "confidence": 0.88,
                "features": {"volatility_20": 0.014},
            },
        )
        decision_event = SignalEvent(
            source="decision-agent",
            symbol="MSFT",
            strategy_id="blended-alpha",
            confidence=0.88,
            payload={
                "action": "SELL",
                "score": -0.41,
                "reasoning": "blended decision favors the short side",
            },
        )
        context_event = MarketEvent(
            source="market-data",
            symbol="MSFT",
            last_price=101.0,
            payload={
                "ohlcv": ohlcv,
                "account_state": {"capital": 100_000.0, "daily_pnl": 0.0, "gross_exposure": 0.0},
            },
        )

        await agent.start()
        await agent.on_event(market_signal_event)
        await agent.on_event(decision_event)
        interim = await agent.snapshot_state()
        await agent.on_event(context_event)
        snapshot = await agent.snapshot_state()
        await agent.stop()

        self.assertIsNone(interim["last_output"])
        self.assertEqual(snapshot["last_output"]["final_action"], "SELL")
        self.assertAlmostEqual(snapshot["last_output"]["position_size"], 250.0)
        self.assertAlmostEqual(snapshot["last_output"]["stop_loss"], 105.0)

    async def test_risk_agent_scales_down_lower_conviction_trade(self) -> None:
        """Rich decision payloads should reduce position size when conviction is weaker."""

        ohlcv = build_constant_range_ohlcv()
        agent = RiskAgent(
            policy=RiskPolicy(
                max_risk_per_trade_fraction=0.01,
                atr_period=14,
                atr_multiplier=2.0,
                max_daily_loss_fraction=0.03,
                max_exposure_fraction=1.0,
            )
        )

        strong = await agent.assess(
            {
                "action": "BUY",
                "side": "LONG",
                "score": 0.55,
                "confidence": 0.90,
                "expected_edge": 0.0016,
                "holding_horizon_estimate": 2,
                "rationale_codes": ("trade_candidate",),
                "blockers": (),
                "supporting_evidence": {"session_phase": "morning"},
                "size_multiplier": 0.90,
                "reasoning": "strong-long",
            },
            market_signal={
                "signal": 0.7,
                "confidence": 0.9,
                "features": {"volatility_20": 0.012, "spread_bps": 3.0, "dollar_volume": 15_000_000.0},
            },
            account_state={"capital": 100_000.0, "daily_loss": 0.0, "gross_exposure": 0.0},
            ohlcv=ohlcv,
            symbol="AAPL",
        )
        weaker = await agent.assess(
            {
                "action": "BUY",
                "side": "LONG",
                "score": 0.32,
                "confidence": 0.62,
                "expected_edge": 0.0007,
                "holding_horizon_estimate": 1,
                "rationale_codes": ("trade_candidate",),
                "blockers": (),
                "supporting_evidence": {"session_phase": "morning"},
                "size_multiplier": 0.35,
                "reasoning": "weaker-long",
            },
            market_signal={
                "signal": 0.45,
                "confidence": 0.62,
                "features": {"volatility_20": 0.012, "spread_bps": 3.0, "dollar_volume": 15_000_000.0},
            },
            account_state={"capital": 100_000.0, "daily_loss": 0.0, "gross_exposure": 0.0},
            ohlcv=ohlcv,
            symbol="AAPL",
        )

        self.assertEqual(strong["final_action"], "BUY")
        self.assertEqual(weaker["final_action"], "BUY")
        self.assertLess(float(weaker["position_size"]), float(strong["position_size"]))

    async def test_incremental_atr_matches_full_recompute(self) -> None:
        """Incremental ATR updates should match a fresh full-frame ATR recomputation."""

        ohlcv = build_variable_range_ohlcv(bars=18)
        seed_frame = ohlcv.iloc[:14]
        next_frame = ohlcv.iloc[:15]
        next_timestamp = next_frame.index.get_level_values("timestamp")[-1]
        next_bar = next_frame.iloc[-1].to_dict()
        policy = RiskPolicy(
            max_risk_per_trade_fraction=0.01,
            atr_period=14,
            atr_multiplier=2.0,
            max_daily_loss_fraction=1.0,
            max_exposure_fraction=1.0,
        )
        decision = {"action": "BUY", "score": 0.42, "reasoning": "positive blended signal"}
        market_signal = {"signal": 0.6, "confidence": 0.8, "features": {"volatility_20": 0.012}}
        account_state = {"capital": 100_000.0, "daily_pnl": 0.0, "gross_exposure": 0.0}

        incremental_agent = RiskAgent(policy=policy)
        await incremental_agent.assess(
            decision,
            market_signal=market_signal,
            account_state=account_state,
            ohlcv=seed_frame,
            symbol="AAPL",
            entry_price=float(seed_frame["close"].iloc[-1]),
            timestamp=seed_frame.index.get_level_values("timestamp")[-1],
            bar=seed_frame.iloc[-1].to_dict(),
        )
        incremental = await incremental_agent.assess(
            decision,
            market_signal=market_signal,
            account_state=account_state,
            ohlcv=next_frame,
            symbol="AAPL",
            entry_price=float(next_frame["close"].iloc[-1]),
            timestamp=next_timestamp,
            bar=next_bar,
        )

        full_agent = RiskAgent(policy=policy)
        full = await full_agent.assess(
            decision,
            market_signal=market_signal,
            account_state=account_state,
            ohlcv=next_frame,
            symbol="AAPL",
            entry_price=float(next_frame["close"].iloc[-1]),
        )

        self.assertEqual(incremental["final_action"], full["final_action"])
        self.assertAlmostEqual(incremental["position_size"], full["position_size"])
        self.assertAlmostEqual(incremental["stop_loss"], full["stop_loss"])
