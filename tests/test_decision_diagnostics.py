"""Tests for decision and critic diagnostics helpers."""

from __future__ import annotations

import unittest

from trading_system.agents import summarize_decisions, summarize_reviews


class DecisionDiagnosticsTests(unittest.TestCase):
    """Verify lightweight decision attribution summaries."""

    def test_summarize_decisions_counts_actions_and_codes(self) -> None:
        """Decision summaries should count actions, rationale codes, and blockers."""

        summary = summarize_decisions(
            [
                {
                    "action": "BUY",
                    "side": "LONG",
                    "rationale_codes": ("regime_trend", "trade_candidate"),
                    "blockers": (),
                },
                {
                    "action": "HOLD",
                    "side": "FLAT",
                    "rationale_codes": ("abstain",),
                    "blockers": ("midday_low_edge",),
                },
            ]
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["actions"]["BUY"], 1)
        self.assertEqual(summary["actions"]["HOLD"], 1)
        self.assertEqual(summary["rationale_codes"]["regime_trend"], 1)
        self.assertEqual(summary["blockers"]["midday_low_edge"], 1)

    def test_summarize_reviews_counts_rejections_and_flags(self) -> None:
        """Review summaries should count approval outcomes and blocker codes."""

        summary = summarize_reviews(
            [
                {"approved": True, "blocker_codes": (), "risk_flags": ("high_impact_news",)},
                {"approved": False, "blocker_codes": ("duplicate_signal",), "risk_flags": ()},
            ]
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["approvals"]["approved"], 1)
        self.assertEqual(summary["approvals"]["rejected"], 1)
        self.assertEqual(summary["blocker_codes"]["duplicate_signal"], 1)
        self.assertEqual(summary["risk_flags"]["high_impact_news"], 1)
