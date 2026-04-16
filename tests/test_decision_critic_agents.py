"""Tests for deterministic decision and critic agents."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading_system.agents import CriticAgent, CriticPolicy, DecisionAgent
from trading_system.core import NewsEvent, SignalEvent


def build_company_news(**overrides: object) -> dict[str, object]:
    """Return a rich company-news payload for decision tests."""

    payload: dict[str, object] = {
        "sentiment": 0.6,
        "impact": 0.8,
        "event_type": "product_launch",
        "summary": "A major launch improved demand expectations.",
        "effective_sentiment": 0.42,
        "scope": "company",
        "scheduled_event": False,
        "relevance_score": 1.0,
        "novelty_score": 1.0,
        "credibility_score": 0.9,
        "freshness_score": 0.95,
        "catalyst_score": 0.70,
        "source": "reuters",
        "symbols": ("AAPL",),
        "primary_symbol": "AAPL",
        "rationale_codes": ("scope_company", "high_impact_catalyst"),
        "is_high_impact": True,
    }
    payload.update(overrides)
    return payload


def build_market_signal(**overrides: object) -> dict[str, object]:
    """Return a rich market-signal payload for decision tests."""

    payload: dict[str, object] = {
        "signal": 0.82,
        "confidence": 0.91,
        "predicted_return": 0.0016,
        "symbol": "AAPL",
        "features": {
            "volatility_20": 0.012,
            "volume_spike": 1.0,
            "ema_gap_9_21": 0.0021,
            "ema_gap_21_50": 0.0016,
            "vwap_gap": 0.0008,
            "rsi_14": 58.0,
            "spread_bps": 4.0,
            "dollar_volume": 12_000_000.0,
            "session_phase": "morning",
        },
    }
    payload.update(overrides)
    return payload


class DecisionAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify the richer deterministic decision policy."""

    async def test_decision_agent_emits_structured_long_trade(self) -> None:
        """Aligned trend and high-impact company news should produce a long trade."""

        agent = DecisionAgent(agent_id="decision-test", action_threshold=0.05)

        await agent.start()
        output = await agent.decide(
            market_signal=build_market_signal(),
            news_analysis=build_company_news(),
            symbol="AAPL",
            occurred_at=datetime(2026, 4, 15, 14, 35, tzinfo=timezone.utc),
        )
        snapshot = await agent.snapshot_state()
        await agent.stop()

        self.assertEqual(output["action"], "BUY")
        self.assertEqual(output["side"], "LONG")
        self.assertGreater(float(output["confidence"]), 0.5)
        self.assertGreater(float(output["expected_edge"]), 0.0)
        self.assertGreater(float(output["size_multiplier"]), 0.0)
        self.assertIn("regime_trend", output["rationale_codes"])
        self.assertIn("trade_candidate", output["rationale_codes"])
        self.assertEqual(tuple(output["blockers"]), ())
        self.assertIn("expected_edge_bps", output["reasoning"])
        self.assertEqual(snapshot["agent_id"], "decision-test")
        self.assertEqual(snapshot["last_output"], output)

    async def test_decision_agent_abstains_when_lunch_edge_is_weak(self) -> None:
        """Weak midday evidence should lead to abstention instead of a marginal trade."""

        agent = DecisionAgent(action_threshold=0.05)

        output = await agent.decide(
            market_signal=build_market_signal(
                signal=0.28,
                confidence=0.62,
                predicted_return=0.00020,
                features={
                    "volatility_20": 0.013,
                    "volume_spike": 0.0,
                    "ema_gap_9_21": 0.0002,
                    "ema_gap_21_50": 0.0001,
                    "vwap_gap": 0.0003,
                    "rsi_14": 53.0,
                    "spread_bps": 4.0,
                    "dollar_volume": 10_000_000.0,
                    "session_phase": "lunch",
                },
            ),
            news_analysis=build_company_news(
                sentiment=0.15,
                effective_sentiment=0.05,
                impact=0.20,
                catalyst_score=0.12,
                is_high_impact=False,
            ),
            symbol="AAPL",
            occurred_at=datetime(2026, 4, 15, 16, 15, tzinfo=timezone.utc),
        )

        self.assertEqual(output["action"], "HOLD")
        self.assertEqual(output["side"], "FLAT")
        self.assertEqual(float(output["size_multiplier"]), 0.0)
        self.assertIn("midday_low_edge", output["blockers"])

    async def test_decision_agent_applies_cooldown_to_repeat_same_side_signal(self) -> None:
        """Repeated same-side entries inside the cooldown window should be blocked unless clearly stronger."""

        agent = DecisionAgent(action_threshold=0.05)
        first_time = datetime(2026, 4, 15, 14, 35, tzinfo=timezone.utc)
        second_time = first_time + timedelta(minutes=5)

        first = await agent.decide(
            market_signal=build_market_signal(),
            news_analysis=build_company_news(),
            symbol="AAPL",
            occurred_at=first_time,
        )
        second = await agent.decide(
            market_signal=build_market_signal(signal=0.84, confidence=0.91, predicted_return=0.00162),
            news_analysis=build_company_news(),
            symbol="AAPL",
            occurred_at=second_time,
        )

        self.assertEqual(first["action"], "BUY")
        self.assertEqual(second["action"], "HOLD")
        self.assertIn("cooldown_active", second["blockers"])


class CriticAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify adversarial trade rejection guardrails."""

    async def test_critic_agent_rejects_high_volatility(self) -> None:
        """Trades above the configured volatility threshold should be rejected."""

        critic = CriticAgent(policy=CriticPolicy(max_volatility=0.03, min_confidence=0.55, conflict_threshold=0.10))

        review = await critic.review(
            {
                "action": "BUY",
                "side": "LONG",
                "score": 0.54,
                "confidence": 0.81,
                "expected_edge": 0.0012,
                "holding_horizon_estimate": 2,
                "rationale_codes": ("regime_trend", "trade_candidate"),
                "blockers": (),
                "supporting_evidence": {"session_phase": "morning"},
                "size_multiplier": 0.55,
                "reasoning": "structured-long",
            },
            market_signal=build_market_signal(features={**build_market_signal()["features"], "volatility_20": 0.051}),
            news_analysis=build_company_news(),
            symbol="AAPL",
            occurred_at=datetime(2026, 4, 15, 14, 35, tzinfo=timezone.utc),
        )

        self.assertFalse(review["approved"])
        self.assertIn("high_volatility", review["blocker_codes"])

    async def test_critic_agent_rejects_conflicting_high_impact_news(self) -> None:
        """Opposing market and high-impact news inputs should be rejected."""

        critic = CriticAgent(policy=CriticPolicy(max_volatility=0.03, min_confidence=0.55, conflict_threshold=0.10))

        review = await critic.review(
            {
                "action": "BUY",
                "side": "LONG",
                "score": 0.45,
                "confidence": 0.82,
                "expected_edge": 0.0014,
                "holding_horizon_estimate": 2,
                "rationale_codes": ("regime_trend", "trade_candidate"),
                "blockers": (),
                "supporting_evidence": {"session_phase": "morning"},
                "size_multiplier": 0.62,
                "reasoning": "structured-long",
            },
            market_signal=build_market_signal(),
            news_analysis=build_company_news(
                sentiment=-0.8,
                effective_sentiment=-0.55,
                impact=0.9,
                catalyst_score=0.8,
                summary="A severe legal catalyst cuts against the long thesis.",
                event_type="litigation",
            ),
            symbol="AAPL",
            occurred_at=datetime(2026, 4, 15, 14, 40, tzinfo=timezone.utc),
        )

        self.assertFalse(review["approved"])
        self.assertIn("conflicting_signals", review["blocker_codes"])

    async def test_critic_agent_rejects_stale_news_supported_trade(self) -> None:
        """Trades relying on stale confirming news should be rejected."""

        critic = CriticAgent(policy=CriticPolicy(max_volatility=0.03, min_confidence=0.55, conflict_threshold=0.10))

        review = await critic.review(
            {
                "action": "BUY",
                "side": "LONG",
                "score": 0.42,
                "confidence": 0.74,
                "expected_edge": 0.0011,
                "holding_horizon_estimate": 2,
                "rationale_codes": ("news_confirms_market", "trade_candidate"),
                "blockers": (),
                "supporting_evidence": {"session_phase": "morning"},
                "size_multiplier": 0.48,
                "reasoning": "structured-long",
            },
            market_signal=build_market_signal(),
            news_analysis=build_company_news(freshness_score=0.20, effective_sentiment=0.28),
            symbol="AAPL",
            occurred_at=datetime(2026, 4, 15, 14, 45, tzinfo=timezone.utc),
        )

        self.assertFalse(review["approved"])
        self.assertIn("stale_news_support", review["blocker_codes"])

    async def test_critic_agent_approves_aligned_trade_from_events(self) -> None:
        """Aligned event-driven trade context should pass review."""

        critic = CriticAgent()
        market_event = SignalEvent(
            source="market-agent",
            symbol="MSFT",
            strategy_id="market-alpha",
            confidence=0.88,
            occurred_at=datetime(2026, 4, 15, 14, 35, tzinfo=timezone.utc),
            payload={
                "signal": 0.55,
                "confidence": 0.88,
                "predicted_return": 0.0016,
                "features": {
                    "volatility_20": 0.013,
                    "spread_bps": 3.0,
                    "dollar_volume": 20_000_000.0,
                    "session_phase": "morning",
                },
            },
        )
        news_event = NewsEvent(
            source="news-agent",
            headline="Product launch gains traction",
            symbols=("MSFT",),
            occurred_at=datetime(2026, 4, 15, 14, 34, tzinfo=timezone.utc),
            payload=build_company_news(source="reuters", symbols=("MSFT",), primary_symbol="MSFT"),
        )
        decision_event = SignalEvent(
            source="decision-agent",
            symbol="MSFT",
            strategy_id="blended-alpha",
            confidence=0.88,
            occurred_at=datetime(2026, 4, 15, 14, 35, tzinfo=timezone.utc),
            payload={
                "action": "BUY",
                "side": "LONG",
                "score": 0.49,
                "confidence": 0.86,
                "expected_edge": 0.0015,
                "holding_horizon_estimate": 2,
                "rationale_codes": ("regime_trend", "trade_candidate"),
                "blockers": (),
                "supporting_evidence": {"session_phase": "morning"},
                "size_multiplier": 0.70,
                "reasoning": "structured-long",
            },
        )

        await critic.start()
        await critic.on_event(market_event)
        await critic.on_event(news_event)
        pre_decision = await critic.snapshot_state()
        await critic.on_event(decision_event)
        snapshot = await critic.snapshot_state()
        await critic.stop()

        self.assertIsNone(pre_decision["last_output"])
        self.assertTrue(snapshot["last_output"]["approved"])
        self.assertEqual(tuple(snapshot["last_output"]["blocker_codes"]), ())

    async def test_critic_agent_rejects_duplicate_same_side_signal(self) -> None:
        """Repeated same-side signals without enough improvement should be rejected."""

        critic = CriticAgent()
        first_time = datetime(2026, 4, 15, 14, 35, tzinfo=timezone.utc)
        second_time = first_time + timedelta(minutes=5)
        decision = {
            "action": "BUY",
            "side": "LONG",
            "score": 0.46,
            "confidence": 0.84,
            "expected_edge": 0.0013,
            "holding_horizon_estimate": 2,
            "rationale_codes": ("regime_trend", "trade_candidate"),
            "blockers": (),
            "supporting_evidence": {"session_phase": "morning"},
            "size_multiplier": 0.61,
            "reasoning": "structured-long",
        }
        market_signal = build_market_signal()
        news_analysis = build_company_news()

        first_review = await critic.review(
            decision,
            market_signal=market_signal,
            news_analysis=news_analysis,
            symbol="AAPL",
            occurred_at=first_time,
        )
        second_review = await critic.review(
            {**decision, "score": 0.49},
            market_signal=market_signal,
            news_analysis=news_analysis,
            symbol="AAPL",
            occurred_at=second_time,
        )

        self.assertTrue(first_review["approved"])
        self.assertFalse(second_review["approved"])
        self.assertIn("duplicate_signal", second_review["blocker_codes"])
