"""Deterministic trade critic for volatility, confidence, and signal alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trading_system.agents._validation import (
    extract_volatility,
    normalize_decision_output,
    normalize_market_signal,
    normalize_news_analysis,
    signals_conflict,
)
from trading_system.agents.base import BaseAgent
from trading_system.core.events import BaseEvent, NewsEvent, SignalEvent


@dataclass(frozen=True, slots=True)
class CriticPolicy:
    """Thresholds controlling deterministic trade review."""

    max_volatility: float = 0.03
    min_confidence: float = 0.55
    conflict_threshold: float = 0.10

    def __post_init__(self) -> None:
        """Validate the critic thresholds."""

        if self.max_volatility < 0.0:
            raise ValueError("max_volatility must be greater than or equal to 0")
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.conflict_threshold < 0.0 or self.conflict_threshold > 1.0:
            raise ValueError("conflict_threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CriticReview:
    """Structured output from the deterministic trade critic."""

    approved: bool
    reason: str

    def __post_init__(self) -> None:
        """Validate the review payload."""

        if not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, bool | str]:
        """Return a JSON-serializable review payload."""

        return dict(asdict(self))


class CriticAgent(BaseAgent):
    """Reject deterministic trade decisions that violate guardrail policies."""

    def __init__(
        self,
        *,
        agent_id: str = "critic-agent",
        policy: CriticPolicy | None = None,
    ) -> None:
        """Initialize the critic policy and latest observed state."""

        self._agent_id = agent_id
        self._policy = policy or CriticPolicy()
        self._running = False
        self._latest_market_signal: dict[str, Any] | None = None
        self._latest_news_analysis: dict[str, float | str] | None = None
        self._latest_decision: dict[str, float | str] | None = None
        self._last_output: dict[str, bool | str] | None = None

    @property
    def agent_id(self) -> str:
        """Return the stable identifier for this agent."""

        return self._agent_id

    @property
    def subscribed_topics(self) -> Sequence[str]:
        """Return the event topics this agent can consume."""

        return ("signal", "news")

    async def start(self) -> None:
        """Mark the critic agent as active."""

        self._running = True

    async def stop(self) -> None:
        """Mark the critic agent as inactive."""

        self._running = False

    async def on_event(self, event: BaseEvent) -> None:
        """Consume decision, market-signal, and news-analysis events."""

        if isinstance(event, SignalEvent):
            payload = dict(event.payload or {})
            if {"action", "score", "reasoning"} <= set(payload):
                self._latest_decision = normalize_decision_output(payload)
            elif "signal" in payload:
                payload.setdefault("confidence", event.confidence)
                if event.symbol:
                    payload.setdefault("symbol", event.symbol)
                self._latest_market_signal = normalize_market_signal(payload, default_confidence=event.confidence)
            else:
                return
        elif isinstance(event, NewsEvent):
            payload = dict(event.payload or {})
            if "sentiment" not in payload:
                return
            self._latest_news_analysis = normalize_news_analysis(payload)
        else:
            return

        if (
            self._latest_decision is not None
            and self._latest_market_signal is not None
            and self._latest_news_analysis is not None
        ):
            self._last_output = (await self.review()).copy()

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of agent state."""

        return {
            "agent_id": self._agent_id,
            "running": self._running,
            "policy": dict(asdict(self._policy)),
            "last_market_signal": self._latest_market_signal,
            "last_news_analysis": self._latest_news_analysis,
            "last_decision": self._latest_decision,
            "last_output": self._last_output,
        }

    async def review(
        self,
        decision: Mapping[str, Any] | None = None,
        *,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
    ) -> dict[str, bool | str]:
        """Return a deterministic approval decision for a proposed trade."""

        if decision is not None:
            self._latest_decision = normalize_decision_output(decision)
        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
        if news_analysis is not None:
            self._latest_news_analysis = normalize_news_analysis(news_analysis)

        return self._review_current_state()

    def review_sync(
        self,
        decision: Mapping[str, Any] | None = None,
        *,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
    ) -> dict[str, bool | str]:
        """Synchronously return a deterministic approval decision."""

        if decision is not None:
            self._latest_decision = normalize_decision_output(decision)
        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
        if news_analysis is not None:
            self._latest_news_analysis = normalize_news_analysis(news_analysis)

        return self._review_current_state()

    def _review_current_state(self) -> dict[str, bool | str]:
        """Evaluate the currently stored decision context."""

        if self._latest_decision is None:
            raise ValueError("decision is required before review")
        if self._latest_market_signal is None:
            raise ValueError("market_signal is required before review")
        if self._latest_news_analysis is None:
            raise ValueError("news_analysis is required before review")

        decision = self._latest_decision
        market_signal = self._latest_market_signal
        news_analysis = self._latest_news_analysis

        if str(decision["action"]).upper() == "HOLD":
            output = CriticReview(approved=False, reason="decision action is HOLD; no trade to approve").to_dict()
            self._last_output = output
            return output.copy()

        reasons: list[str] = []
        volatility = extract_volatility(market_signal["features"])
        if volatility is not None and volatility > self._policy.max_volatility:
            reasons.append(
                f"high volatility: {volatility:.4f} exceeds max_volatility {self._policy.max_volatility:.4f}"
            )

        confidence = float(market_signal["confidence"])
        if confidence < self._policy.min_confidence:
            reasons.append(
                f"low confidence: {confidence:.4f} is below min_confidence {self._policy.min_confidence:.4f}"
            )

        signal_value = float(market_signal["signal"])
        news_sentiment = float(news_analysis["sentiment"])
        if signals_conflict(signal_value, news_sentiment, threshold=self._policy.conflict_threshold):
            reasons.append(
                f"conflicting signals: market_signal {signal_value:.4f} opposes news_sentiment {news_sentiment:.4f}"
            )

        if reasons:
            output = CriticReview(approved=False, reason="; ".join(reasons)).to_dict()
        else:
            output = CriticReview(
                approved=True,
                reason=(
                    "approved: volatility, confidence, and signal alignment are within deterministic guardrails"
                ),
            ).to_dict()

        self._last_output = output
        return output.copy()
