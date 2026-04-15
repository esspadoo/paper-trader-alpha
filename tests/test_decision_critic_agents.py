"""Tests for deterministic decision and critic agents."""

from __future__ import annotations

import unittest

from trading_system.agents import CriticAgent, CriticPolicy, DecisionAgent
from trading_system.core import NewsEvent, SignalEvent


class DecisionAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify deterministic market-plus-news decision logic."""

    async def test_decision_agent_combines_market_and_news_inputs(self) -> None:
        """The decision score should follow the required fixed weighting."""

        agent = DecisionAgent(agent_id="decision-test", action_threshold=0.05)

        await agent.start()
        output = await agent.decide(
            market_signal={
                "signal": 0.8,
                "confidence": 0.9,
                "features": {"volatility_20": 0.012, "volume_spike": 1.0},
            },
            news_analysis={
                "sentiment": -0.2,
                "impact": 0.4,
                "event_type": "guidance",
                "summary": "Guidance commentary softened the setup.",
            },
        )
        snapshot = await agent.snapshot_state()
        await agent.stop()

        self.assertEqual(output["action"], "BUY")
        self.assertAlmostEqual(output["score"], 0.5)
        self.assertIn("0.7*market_signal(0.8000)", output["reasoning"])
        self.assertIn("0.3*news_sentiment(-0.2000)", output["reasoning"])
        self.assertEqual(snapshot["agent_id"], "decision-test")
        self.assertEqual(snapshot["last_output"], output)

    async def test_decision_agent_updates_from_events(self) -> None:
        """Signal and news events should update state and trigger a decision when both are present."""

        agent = DecisionAgent(action_threshold=0.05)

        market_event = SignalEvent(
            source="market-agent",
            symbol="AAPL",
            strategy_id="market-alpha",
            confidence=0.83,
            payload={
                "signal": 0.4,
                "features": {"volatility_20": 0.014},
            },
        )
        news_event = NewsEvent(
            source="news-agent",
            headline="Analyst raises outlook",
            symbols=("AAPL",),
            payload={
                "sentiment": 0.2,
                "impact": 0.6,
                "event_type": "analyst_rating",
                "summary": "Sell-side sentiment improved modestly.",
            },
        )

        await agent.start()
        await agent.on_event(market_event)
        interim = await agent.snapshot_state()
        await agent.on_event(news_event)
        snapshot = await agent.snapshot_state()
        await agent.stop()

        self.assertIsNone(interim["last_output"])
        self.assertEqual(snapshot["last_output"]["action"], "BUY")
        self.assertAlmostEqual(snapshot["last_output"]["score"], 0.34)
        self.assertEqual(snapshot["last_market_signal"]["confidence"], 0.83)


class CriticAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify deterministic trade rejection guardrails."""

    async def test_critic_agent_rejects_high_volatility(self) -> None:
        """Trades above the configured volatility threshold should be rejected."""

        critic = CriticAgent(policy=CriticPolicy(max_volatility=0.03, min_confidence=0.55, conflict_threshold=0.10))

        review = await critic.review(
            {
                "action": "BUY",
                "score": 0.54,
                "reasoning": "aligned positive setup",
            },
            market_signal={
                "signal": 0.6,
                "confidence": 0.9,
                "features": {"volatility_20": 0.051},
            },
            news_analysis={
                "sentiment": 0.4,
                "impact": 0.7,
                "event_type": "earnings",
                "summary": "Earnings momentum remains constructive.",
            },
        )

        self.assertFalse(review["approved"])
        self.assertIn("high volatility", review["reason"])

    async def test_critic_agent_rejects_low_confidence(self) -> None:
        """Trades below the minimum confidence threshold should be rejected."""

        critic = CriticAgent(policy=CriticPolicy(max_volatility=0.03, min_confidence=0.60, conflict_threshold=0.10))

        review = await critic.review(
            {
                "action": "BUY",
                "score": 0.36,
                "reasoning": "positive but weaker setup",
            },
            market_signal={
                "signal": 0.4,
                "confidence": 0.45,
                "features": {"volatility_20": 0.012},
            },
            news_analysis={
                "sentiment": 0.25,
                "impact": 0.5,
                "event_type": "partnership",
                "summary": "Partnership news supports the long thesis.",
            },
        )

        self.assertFalse(review["approved"])
        self.assertIn("low confidence", review["reason"])

    async def test_critic_agent_rejects_conflicting_signals(self) -> None:
        """Opposing market and news signals should be rejected when both are material."""

        critic = CriticAgent(policy=CriticPolicy(max_volatility=0.03, min_confidence=0.55, conflict_threshold=0.10))

        review = await critic.review(
            {
                "action": "BUY",
                "score": 0.29,
                "reasoning": "market signal outweighs the headline",
            },
            market_signal={
                "signal": 0.7,
                "confidence": 0.84,
                "features": {"volatility_20": 0.011},
            },
            news_analysis={
                "sentiment": -0.6,
                "impact": 0.8,
                "event_type": "litigation",
                "summary": "Legal headline cuts against the market setup.",
            },
        )

        self.assertFalse(review["approved"])
        self.assertIn("conflicting signals", review["reason"])

    async def test_critic_agent_approves_aligned_trade_from_events(self) -> None:
        """The critic should approve aligned event-driven trade context."""

        critic = CriticAgent()
        market_event = SignalEvent(
            source="market-agent",
            symbol="MSFT",
            strategy_id="market-alpha",
            confidence=0.88,
            payload={
                "signal": 0.55,
                "confidence": 0.88,
                "features": {"volatility_20": 0.013},
            },
        )
        news_event = NewsEvent(
            source="news-agent",
            headline="Product launch gains traction",
            symbols=("MSFT",),
            payload={
                "sentiment": 0.35,
                "impact": 0.6,
                "event_type": "product_launch",
                "summary": "Product reception supports the bullish setup.",
            },
        )
        decision_event = SignalEvent(
            source="decision-agent",
            symbol="MSFT",
            strategy_id="blended-alpha",
            confidence=0.88,
            payload={
                "action": "BUY",
                "score": 0.49,
                "reasoning": "combined market and news inputs are aligned",
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
        self.assertIn("approved", snapshot["last_output"]["reason"])
