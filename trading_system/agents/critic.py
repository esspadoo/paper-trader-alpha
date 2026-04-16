"""Deterministic adversarial trade critic for execution-quality guardrails."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from trading_system.agents._validation import (
    extract_dollar_volume,
    extract_expected_edge_bps,
    extract_news_strength,
    extract_session_phase,
    extract_spread_bps,
    extract_volatility,
    normalize_critic_review,
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
    max_spread_bps: float = 16.0
    min_dollar_volume: float = 5_000_000.0
    min_expected_edge_bps: float = 4.0
    min_news_freshness_for_boost: float = 0.35
    duplicate_signal_cooldown_seconds: float = 900.0
    min_score_improvement: float = 0.08
    lunch_min_expected_edge_bps: float = 6.0
    open_close_min_expected_edge_bps: float = 5.0

    def __post_init__(self) -> None:
        """Validate the critic thresholds."""

        unit_interval = {
            "min_confidence": self.min_confidence,
            "conflict_threshold": self.conflict_threshold,
            "min_news_freshness_for_boost": self.min_news_freshness_for_boost,
        }
        for field_name, value in unit_interval.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        positive = {
            "max_volatility": self.max_volatility,
            "max_spread_bps": self.max_spread_bps,
            "min_dollar_volume": self.min_dollar_volume,
            "min_expected_edge_bps": self.min_expected_edge_bps,
            "duplicate_signal_cooldown_seconds": self.duplicate_signal_cooldown_seconds,
            "min_score_improvement": self.min_score_improvement,
            "lunch_min_expected_edge_bps": self.lunch_min_expected_edge_bps,
            "open_close_min_expected_edge_bps": self.open_close_min_expected_edge_bps,
        }
        for field_name, value in positive.items():
            if value < 0.0:
                raise ValueError(f"{field_name} must be greater than or equal to 0")


@dataclass(frozen=True, slots=True)
class CriticReview:
    """Structured output from the deterministic trade critic."""

    approved: bool
    reason: str
    blocker_codes: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the review payload via the shared schema."""

        normalize_critic_review(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable review payload."""

        return normalize_critic_review(asdict(self))


@dataclass(frozen=True, slots=True)
class _ApprovedSignal:
    """Recently approved actionable trade used for duplicate-signal suppression."""

    action: str
    score: float
    occurred_at: datetime


class CriticAgent(BaseAgent):
    """Reject deterministic trade decisions that violate execution and signal-quality guardrails."""

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
        self._latest_news_analysis: dict[str, Any] | None = None
        self._latest_decision: dict[str, Any] | None = None
        self._latest_symbol: str | None = None
        self._latest_occurred_at: datetime | None = None
        self._last_output: dict[str, Any] | None = None
        self._last_approved_signal_by_symbol: dict[str, _ApprovedSignal] = {}

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
            if event.symbol:
                self._latest_symbol = event.symbol.strip().upper()
            self._latest_occurred_at = event.occurred_at
        elif isinstance(event, NewsEvent):
            payload = dict(event.payload or {})
            if "sentiment" not in payload:
                return
            self._latest_news_analysis = normalize_news_analysis(payload)
            if event.symbols:
                self._latest_symbol = str(event.symbols[0]).strip().upper()
            self._latest_occurred_at = event.occurred_at
        else:
            return

        if (
            self._latest_decision is not None
            and self._latest_market_signal is not None
            and self._latest_news_analysis is not None
        ):
            self._last_output = (
                await self.review(symbol=self._latest_symbol, occurred_at=self._latest_occurred_at)
            ).copy()

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of agent state."""

        return {
            "agent_id": self._agent_id,
            "running": self._running,
            "policy": dict(asdict(self._policy)),
            "last_market_signal": self._latest_market_signal,
            "last_news_analysis": self._latest_news_analysis,
            "last_decision": self._latest_decision,
            "last_symbol": self._latest_symbol,
            "last_occurred_at": self._latest_occurred_at.isoformat() if self._latest_occurred_at else None,
            "last_output": self._last_output,
        }

    async def review(
        self,
        decision: Mapping[str, Any] | None = None,
        *,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
        symbol: str | None = None,
        occurred_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic approval decision for a proposed trade."""

        if decision is not None:
            self._latest_decision = normalize_decision_output(decision)
        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
            maybe_symbol = self._latest_market_signal.get("symbol")
            if isinstance(maybe_symbol, str) and maybe_symbol:
                self._latest_symbol = maybe_symbol
        if news_analysis is not None:
            self._latest_news_analysis = normalize_news_analysis(news_analysis)
        if symbol is not None:
            normalized_symbol = symbol.strip().upper()
            if not normalized_symbol:
                raise ValueError("symbol must be a non-empty string when provided")
            self._latest_symbol = normalized_symbol
        if occurred_at is not None:
            self._latest_occurred_at = occurred_at

        return self._review_current_state()

    def review_sync(
        self,
        decision: Mapping[str, Any] | None = None,
        *,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
        symbol: str | None = None,
        occurred_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Synchronously return a deterministic approval decision."""

        if decision is not None:
            self._latest_decision = normalize_decision_output(decision)
        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
            maybe_symbol = self._latest_market_signal.get("symbol")
            if isinstance(maybe_symbol, str) and maybe_symbol:
                self._latest_symbol = maybe_symbol
        if news_analysis is not None:
            self._latest_news_analysis = normalize_news_analysis(news_analysis)
        if symbol is not None:
            normalized_symbol = symbol.strip().upper()
            if not normalized_symbol:
                raise ValueError("symbol must be a non-empty string when provided")
            self._latest_symbol = normalized_symbol
        if occurred_at is not None:
            self._latest_occurred_at = occurred_at

        return self._review_current_state()

    def _review_current_state(self) -> dict[str, Any]:
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
        symbol = self._latest_symbol
        occurred_at = self._latest_occurred_at

        action = str(decision["action"]).upper()
        if action == "HOLD":
            output = CriticReview(
                approved=False,
                reason="blockers=no_trade; risk_flags=none",
                blocker_codes=("no_trade",),
                risk_flags=(),
            ).to_dict()
            self._last_output = output
            return output.copy()

        blocker_codes: list[str] = []
        risk_flags: list[str] = []
        features = dict(market_signal["features"])
        volatility = extract_volatility(features)
        if volatility is not None and volatility > self._policy.max_volatility:
            blocker_codes.append("high_volatility")

        confidence = max(float(decision["confidence"]), float(market_signal["confidence"]))
        if confidence < self._policy.min_confidence:
            blocker_codes.append("low_confidence")

        expected_edge_bps = self._effective_expected_edge_bps(decision=decision, market_signal=market_signal)
        if expected_edge_bps < self._policy.min_expected_edge_bps:
            blocker_codes.append("low_expected_edge_after_costs")

        spread_bps = extract_spread_bps(features)
        if spread_bps is not None:
            if spread_bps > self._policy.max_spread_bps:
                blocker_codes.append("wide_spread")
            elif spread_bps > self._policy.max_spread_bps * 0.75:
                risk_flags.append("elevated_spread")

        dollar_volume = extract_dollar_volume(features)
        if dollar_volume is not None:
            if dollar_volume < self._policy.min_dollar_volume:
                blocker_codes.append("thin_liquidity")
            elif dollar_volume < self._policy.min_dollar_volume * 1.5:
                risk_flags.append("moderate_liquidity")

        session_phase = extract_session_phase(features)
        if session_phase == "lunch" and expected_edge_bps < self._policy.lunch_min_expected_edge_bps:
            blocker_codes.append("midday_edge_too_small")
        if session_phase in {"open", "close"} and expected_edge_bps < self._policy.open_close_min_expected_edge_bps:
            blocker_codes.append("session_edge_too_small")

        news_strength = extract_news_strength(news_analysis)
        news_catalyst = float(news_analysis["catalyst_score"])
        news_freshness = float(news_analysis["freshness_score"])
        if signals_conflict(float(market_signal["signal"]), news_strength, threshold=self._policy.conflict_threshold) and news_catalyst >= 0.20:
            blocker_codes.append("conflicting_signals")

        if "news_confirms_market" in tuple(decision["rationale_codes"]) and news_freshness < self._policy.min_news_freshness_for_boost:
            blocker_codes.append("stale_news_support")
        if news_freshness < self._policy.min_news_freshness_for_boost and news_catalyst >= 0.20:
            risk_flags.append("decayed_news_context")
        if bool(news_analysis["scheduled_event"]):
            risk_flags.append("scheduled_event")
        if bool(news_analysis["is_high_impact"]):
            risk_flags.append("high_impact_news")
        if tuple(decision["blockers"]):
            blocker_codes.extend(str(code) for code in tuple(decision["blockers"]))

        duplicate_blocker = self._duplicate_signal_blocker(
            symbol=symbol,
            occurred_at=occurred_at,
            action=action,
            score=float(decision["score"]),
        )
        if duplicate_blocker is not None:
            blocker_codes.append(duplicate_blocker)

        blocker_codes = list(dict.fromkeys(blocker_codes))
        risk_flags = list(dict.fromkeys(risk_flags))
        approved = not blocker_codes
        reason = self._build_reason(blocker_codes=blocker_codes, risk_flags=risk_flags)
        output = CriticReview(
            approved=approved,
            reason=reason,
            blocker_codes=tuple(blocker_codes),
            risk_flags=tuple(risk_flags),
        ).to_dict()

        if approved and symbol is not None and occurred_at is not None:
            self._last_approved_signal_by_symbol[symbol] = _ApprovedSignal(
                action=action,
                score=float(decision["score"]),
                occurred_at=occurred_at,
            )

        self._last_output = output
        return output.copy()

    def _duplicate_signal_blocker(
        self,
        *,
        symbol: str | None,
        occurred_at: datetime | None,
        action: str,
        score: float,
    ) -> str | None:
        """Return a blocker when the proposed signal duplicates a recent approval."""

        if symbol is None or occurred_at is None:
            return None
        previous = self._last_approved_signal_by_symbol.get(symbol)
        if previous is None or previous.action != action:
            return None
        if occurred_at - previous.occurred_at > timedelta(seconds=self._policy.duplicate_signal_cooldown_seconds):
            return None
        if abs(score) <= abs(previous.score) + self._policy.min_score_improvement:
            return "duplicate_signal"
        return None

    def _effective_expected_edge_bps(
        self,
        *,
        decision: Mapping[str, Any],
        market_signal: Mapping[str, Any],
    ) -> float:
        """Return expected edge in basis points with backward-compatible fallback estimation."""

        expected_edge_bps = extract_expected_edge_bps(decision)
        rationale_codes = tuple(decision.get("rationale_codes", ()) or ())
        blockers = tuple(decision.get("blockers", ()) or ())
        is_legacy_payload = (
            float(decision.get("confidence", 0.0)) == 0.0
            and float(decision.get("expected_edge", 0.0)) == 0.0
            and float(decision.get("size_multiplier", 0.0)) == 0.0
            and not rationale_codes
            and not blockers
        )
        if not is_legacy_payload:
            return expected_edge_bps
        market_signal_value = abs(float(market_signal["signal"]))
        market_confidence = float(market_signal["confidence"])
        return market_signal_value * max(market_confidence, 0.35) * 12.0

    def _build_reason(self, *, blocker_codes: list[str], risk_flags: list[str]) -> str:
        """Return a deterministic reason string."""

        blockers = ",".join(blocker_codes) or "none"
        flags = ",".join(risk_flags) or "none"
        return f"blockers={blockers}; risk_flags={flags}"
