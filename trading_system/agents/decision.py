"""Deterministic decision agent combining market and news signals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trading_system.agents._validation import normalize_market_signal, normalize_news_analysis
from trading_system.agents.base import BaseAgent
from trading_system.core.events import BaseEvent, NewsEvent, SignalEvent


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Structured output from the deterministic decision layer."""

    action: str
    score: float
    reasoning: str

    def __post_init__(self) -> None:
        """Validate the structured decision payload."""

        if self.action not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("action must be one of BUY, SELL, HOLD")
        if not math.isfinite(self.score) or not -1.0 <= self.score <= 1.0:
            raise ValueError("score must be a finite float between -1 and 1")
        if not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string")

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serializable decision payload."""

        return dict(asdict(self))


class DecisionAgent(BaseAgent):
    """Combine market and news outputs into a deterministic trading decision."""

    MARKET_WEIGHT = 0.7
    NEWS_WEIGHT = 0.3

    def __init__(
        self,
        *,
        agent_id: str = "decision-agent",
        action_threshold: float = 0.05,
    ) -> None:
        """Initialize the decision policy and latest observed inputs."""

        if action_threshold < 0.0 or action_threshold > 1.0:
            raise ValueError("action_threshold must be between 0 and 1")

        self._agent_id = agent_id
        self._action_threshold = float(action_threshold)
        self._running = False
        self._latest_market_signal: dict[str, Any] | None = None
        self._latest_news_analysis: dict[str, float | str] | None = None
        self._last_output: dict[str, float | str] | None = None

    @property
    def agent_id(self) -> str:
        """Return the stable identifier for this agent."""

        return self._agent_id

    @property
    def subscribed_topics(self) -> Sequence[str]:
        """Return the event topics this agent can consume."""

        return ("signal", "news")

    async def start(self) -> None:
        """Mark the decision agent as active."""

        self._running = True

    async def stop(self) -> None:
        """Mark the decision agent as inactive."""

        self._running = False

    async def on_event(self, event: BaseEvent) -> None:
        """Consume market-signal and news-analysis events."""

        if isinstance(event, SignalEvent):
            market_payload = self._market_payload_from_event(event)
            if market_payload is None:
                return
            self._latest_market_signal = market_payload
        elif isinstance(event, NewsEvent):
            news_payload = self._news_payload_from_event(event)
            if news_payload is None:
                return
            self._latest_news_analysis = news_payload
        else:
            return

        if self._latest_market_signal is not None and self._latest_news_analysis is not None:
            self._last_output = (await self.decide()).copy()

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of agent state."""

        return {
            "agent_id": self._agent_id,
            "running": self._running,
            "action_threshold": self._action_threshold,
            "last_market_signal": self._latest_market_signal,
            "last_news_analysis": self._latest_news_analysis,
            "last_output": self._last_output,
        }

    async def decide(
        self,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
    ) -> dict[str, float | str]:
        """Return a deterministic action from market and news inputs."""

        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
        if news_analysis is not None:
            self._latest_news_analysis = normalize_news_analysis(news_analysis)

        if self._latest_market_signal is None:
            raise ValueError("market_signal is required before making a decision")
        if self._latest_news_analysis is None:
            raise ValueError("news_analysis is required before making a decision")

        market_value = float(self._latest_market_signal["signal"])
        news_value = float(self._latest_news_analysis["sentiment"])
        final_score = (self.MARKET_WEIGHT * market_value) + (self.NEWS_WEIGHT * news_value)
        action = self._action_from_score(final_score)
        reasoning = self._build_reasoning(
            market_signal=market_value,
            news_sentiment=news_value,
            score=final_score,
            action=action,
        )
        output = DecisionResult(action=action, score=float(final_score), reasoning=reasoning).to_dict()
        self._last_output = output
        return output.copy()

    def decide_sync(
        self,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
    ) -> dict[str, float | str]:
        """Synchronously return a deterministic action from market and news inputs."""

        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
        if news_analysis is not None:
            self._latest_news_analysis = normalize_news_analysis(news_analysis)

        if self._latest_market_signal is None:
            raise ValueError("market_signal is required before making a decision")
        if self._latest_news_analysis is None:
            raise ValueError("news_analysis is required before making a decision")

        market_value = float(self._latest_market_signal["signal"])
        news_value = float(self._latest_news_analysis["sentiment"])
        final_score = (self.MARKET_WEIGHT * market_value) + (self.NEWS_WEIGHT * news_value)
        action = self._action_from_score(final_score)
        reasoning = self._build_reasoning(
            market_signal=market_value,
            news_sentiment=news_value,
            score=final_score,
            action=action,
        )
        output = DecisionResult(action=action, score=float(final_score), reasoning=reasoning).to_dict()
        self._last_output = output
        return output.copy()

    def _action_from_score(self, score: float) -> str:
        """Map the combined score into a deterministic action."""

        if score >= self._action_threshold:
            return "BUY"
        if score <= -self._action_threshold:
            return "SELL"
        return "HOLD"

    def _build_reasoning(
        self,
        *,
        market_signal: float,
        news_sentiment: float,
        score: float,
        action: str,
    ) -> str:
        """Return a deterministic audit string for the decision."""

        return (
            f"score={score:.4f} from "
            f"{self.MARKET_WEIGHT:.1f}*market_signal({market_signal:.4f}) + "
            f"{self.NEWS_WEIGHT:.1f}*news_sentiment({news_sentiment:.4f}); "
            f"threshold={self._action_threshold:.4f}; action={action}"
        )

    def _market_payload_from_event(self, event: SignalEvent) -> dict[str, Any] | None:
        """Return a normalized market-signal payload from a signal event."""

        payload = dict(event.payload or {})
        if "signal" not in payload:
            return None

        payload.setdefault("confidence", event.confidence)
        if event.symbol:
            payload.setdefault("symbol", event.symbol)
        return normalize_market_signal(payload, default_confidence=event.confidence)

    def _news_payload_from_event(self, event: NewsEvent) -> dict[str, float | str] | None:
        """Return a normalized news-analysis payload from a news event."""

        payload = dict(event.payload or {})
        if "sentiment" not in payload:
            return None
        return normalize_news_analysis(payload)
