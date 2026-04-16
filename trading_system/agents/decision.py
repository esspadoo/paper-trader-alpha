"""Deterministic professional-grade fusion of market and news signals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from trading_system.agents._validation import (
    extract_dollar_volume,
    extract_feature,
    extract_news_strength,
    extract_rsi,
    extract_session_phase,
    extract_spread_bps,
    extract_volatility,
    normalize_decision_output,
    normalize_market_signal,
    normalize_news_analysis,
)
from trading_system.agents.base import BaseAgent
from trading_system.core.events import BaseEvent, NewsEvent, SignalEvent


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Deterministic thresholds controlling decision fusion."""

    action_threshold: float = 0.10
    min_signal_strength: float = 0.18
    min_confidence: float = 0.35
    min_expected_edge: float = 0.00035
    strong_expected_edge: float = 0.00120
    trend_gap_threshold: float = 0.0008
    mean_reversion_vwap_gap: float = 0.0020
    mean_reversion_rsi_low: float = 36.0
    mean_reversion_rsi_high: float = 64.0
    high_volatility_cutoff: float = 0.028
    strong_news_catalyst: float = 0.30
    news_veto_catalyst: float = 0.45
    max_soft_spread_bps: float = 15.0
    min_dollar_volume: float = 5_000_000.0
    open_close_min_edge: float = 0.00055
    lunch_min_edge: float = 0.00065
    cooldown_minutes: float = 15.0
    min_score_improvement_for_reentry: float = 0.08

    def __post_init__(self) -> None:
        """Validate policy thresholds."""

        bounded = {
            "action_threshold": self.action_threshold,
            "min_signal_strength": self.min_signal_strength,
            "min_confidence": self.min_confidence,
        }
        for field_name, value in bounded.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        positive = {
            "min_expected_edge": self.min_expected_edge,
            "strong_expected_edge": self.strong_expected_edge,
            "trend_gap_threshold": self.trend_gap_threshold,
            "mean_reversion_vwap_gap": self.mean_reversion_vwap_gap,
            "high_volatility_cutoff": self.high_volatility_cutoff,
            "strong_news_catalyst": self.strong_news_catalyst,
            "news_veto_catalyst": self.news_veto_catalyst,
            "max_soft_spread_bps": self.max_soft_spread_bps,
            "min_dollar_volume": self.min_dollar_volume,
            "open_close_min_edge": self.open_close_min_edge,
            "lunch_min_edge": self.lunch_min_edge,
            "cooldown_minutes": self.cooldown_minutes,
            "min_score_improvement_for_reentry": self.min_score_improvement_for_reentry,
        }
        for field_name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{field_name} must be greater than 0")
        if self.mean_reversion_rsi_low <= 0.0 or self.mean_reversion_rsi_low >= 50.0:
            raise ValueError("mean_reversion_rsi_low must be between 0 and 50")
        if self.mean_reversion_rsi_high <= 50.0 or self.mean_reversion_rsi_high >= 100.0:
            raise ValueError("mean_reversion_rsi_high must be between 50 and 100")


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Structured output from the deterministic decision policy."""

    action: str
    side: str
    score: float
    confidence: float
    expected_edge: float
    holding_horizon_estimate: int
    rationale_codes: tuple[str, ...]
    blockers: tuple[str, ...]
    supporting_evidence: dict[str, bool | int | float | str | None]
    size_multiplier: float
    reasoning: str

    def __post_init__(self) -> None:
        """Validate the decision payload via the shared schema."""

        normalize_decision_output(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision payload."""

        return normalize_decision_output(asdict(self))


@dataclass(frozen=True, slots=True)
class _PreviousDecision:
    """Recent actionable decision retained for cooldown control."""

    action: str
    score: float
    occurred_at: datetime


class DecisionAgent(BaseAgent):
    """Fuse market and news inputs with deterministic, abstention-first policy logic."""

    def __init__(
        self,
        *,
        agent_id: str = "decision-agent",
        action_threshold: float = 0.05,
        policy: DecisionPolicy | None = None,
    ) -> None:
        """Initialize the decision policy and retained symbol state."""

        self._agent_id = agent_id
        self._policy = policy or DecisionPolicy(action_threshold=float(action_threshold))
        self._running = False
        self._latest_market_signal: dict[str, Any] | None = None
        self._latest_news_analysis: dict[str, Any] | None = None
        self._latest_symbol: str | None = None
        self._latest_occurred_at: datetime | None = None
        self._last_output: dict[str, Any] | None = None
        self._last_trade_decision_by_symbol: dict[str, _PreviousDecision] = {}

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
            if event.symbol:
                self._latest_symbol = event.symbol.strip().upper()
            self._latest_occurred_at = event.occurred_at
        elif isinstance(event, NewsEvent):
            news_payload = self._news_payload_from_event(event)
            if news_payload is None:
                return
            self._latest_news_analysis = news_payload
            if event.symbols:
                self._latest_symbol = str(event.symbols[0]).strip().upper()
            self._latest_occurred_at = event.occurred_at
        else:
            return

        if self._latest_market_signal is not None and self._latest_news_analysis is not None:
            self._last_output = (
                await self.decide(symbol=self._latest_symbol, occurred_at=self._latest_occurred_at)
            ).copy()

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of agent state."""

        return {
            "agent_id": self._agent_id,
            "running": self._running,
            "policy": dict(asdict(self._policy)),
            "last_market_signal": self._latest_market_signal,
            "last_news_analysis": self._latest_news_analysis,
            "last_symbol": self._latest_symbol,
            "last_occurred_at": self._latest_occurred_at.isoformat() if self._latest_occurred_at else None,
            "last_output": self._last_output,
        }

    async def decide(
        self,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
        *,
        symbol: str | None = None,
        occurred_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic action from market and news inputs."""

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

        return self._decide_current_state()

    def decide_sync(
        self,
        market_signal: Mapping[str, Any] | None = None,
        news_analysis: Mapping[str, Any] | None = None,
        *,
        symbol: str | None = None,
        occurred_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Synchronously return a deterministic action from market and news inputs."""

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

        return self._decide_current_state()

    def _decide_current_state(self) -> dict[str, Any]:
        """Evaluate the currently stored decision context."""

        if self._latest_market_signal is None:
            raise ValueError("market_signal is required before making a decision")
        if self._latest_news_analysis is None:
            raise ValueError("news_analysis is required before making a decision")

        market_signal = self._latest_market_signal
        news_analysis = self._latest_news_analysis
        symbol = self._resolve_symbol(market_signal)
        occurred_at = self._latest_occurred_at
        features = dict(market_signal["features"])

        signal_value = float(market_signal["signal"])
        model_confidence = float(market_signal["confidence"])
        predicted_return = float(market_signal.get("predicted_return", self._fallback_expected_edge(signal_value, model_confidence)))
        volatility = extract_volatility(features)
        rsi = extract_rsi(features)
        spread_bps = extract_spread_bps(features)
        dollar_volume = extract_dollar_volume(features)
        session_phase = extract_session_phase(features)
        vwap_gap = extract_feature(features, "vwap_gap") or 0.0
        ema_fast_mid = extract_feature(features, "ema_gap_9_21") or 0.0
        ema_mid_slow = extract_feature(features, "ema_gap_21_50") or 0.0
        volume_spike = extract_feature(features, "volume_spike") or 0.0

        rationale_codes: list[str] = []
        blockers: list[str] = []
        supporting_evidence: dict[str, bool | int | float | str | None] = {
            "signal": round(signal_value, 6),
            "model_confidence": round(model_confidence, 6),
            "predicted_return_bps": round(predicted_return * 10_000.0, 4),
            "session_phase": session_phase,
        }
        if volatility is not None:
            supporting_evidence["volatility"] = round(volatility, 6)
        if spread_bps is not None:
            supporting_evidence["spread_bps"] = round(spread_bps, 4)
        if dollar_volume is not None:
            supporting_evidence["dollar_volume"] = round(dollar_volume, 2)

        if abs(signal_value) < self._policy.min_signal_strength:
            blockers.append("weak_market_signal")
        if model_confidence < self._policy.min_confidence:
            blockers.append("low_model_confidence")
        if spread_bps is not None and spread_bps > self._policy.max_soft_spread_bps:
            blockers.append("wide_spread")
        if dollar_volume is not None and dollar_volume < self._policy.min_dollar_volume:
            blockers.append("thin_liquidity")

        regime = self._determine_regime(
            signal_value=signal_value,
            ema_fast_mid=ema_fast_mid,
            ema_mid_slow=ema_mid_slow,
            vwap_gap=vwap_gap,
            rsi=rsi,
        )
        rationale_codes.append(f"regime_{regime}")
        if volume_spike >= 1.0:
            rationale_codes.append("volume_spike_confirmation")
        if volatility is not None and volatility >= self._policy.high_volatility_cutoff:
            rationale_codes.append("volatility_shock")

        news_strength = extract_news_strength(news_analysis)
        news_catalyst = float(news_analysis["catalyst_score"])
        news_freshness = float(news_analysis["freshness_score"])
        news_scope = str(news_analysis["scope"])
        news_event_type = str(news_analysis["event_type"])
        is_high_impact_news = bool(news_analysis["is_high_impact"])
        supporting_evidence["news_effective_sentiment"] = round(news_strength, 6)
        supporting_evidence["news_catalyst_score"] = round(news_catalyst, 6)
        supporting_evidence["news_freshness_score"] = round(news_freshness, 6)
        supporting_evidence["news_scope"] = news_scope

        if news_catalyst > 0.0:
            rationale_codes.append(f"news_scope_{news_scope}")
            rationale_codes.append(f"news_event_{news_event_type}")
        else:
            rationale_codes.append("news_neutral")

        raw_alpha = predicted_return
        news_edge = self._news_edge_adjustment(
            signal_value=signal_value,
            news_strength=news_strength,
            news_catalyst=news_catalyst,
            news_scope=news_scope,
            is_high_impact_news=is_high_impact_news,
        )
        raw_alpha += news_edge

        regime_bonus = self._regime_bonus(
            regime=regime,
            signal_value=signal_value,
            vwap_gap=vwap_gap,
            ema_fast_mid=ema_fast_mid,
            ema_mid_slow=ema_mid_slow,
        )
        news_score_component = self._news_score_component(news_strength=news_strength, news_catalyst=news_catalyst, news_scope=news_scope)
        raw_score = (signal_value * model_confidence * 0.75) + regime_bonus + news_score_component
        score = float(max(min(raw_score, 1.0), -1.0))
        confidence = self._calibrated_confidence(
            model_confidence=model_confidence,
            news_strength=news_strength,
            news_catalyst=news_catalyst,
            regime=regime,
            volatility=volatility,
            volume_spike=volume_spike,
        )

        if abs(news_strength) >= self._policy.min_signal_strength and (signal_value * news_strength) < 0.0:
            rationale_codes.append("mixed_market_news")
            if news_catalyst >= self._policy.news_veto_catalyst and news_freshness >= 0.40:
                blockers.append("conflicting_high_impact_news")
        elif news_catalyst >= self._policy.strong_news_catalyst and abs(news_strength) >= self._policy.min_signal_strength:
            rationale_codes.append("news_confirms_market")

        if volatility is not None and volatility >= self._policy.high_volatility_cutoff and news_catalyst < self._policy.strong_news_catalyst:
            blockers.append("volatility_regime_without_catalyst")

        expected_edge = self._apply_costs(
            raw_alpha=raw_alpha,
            spread_bps=spread_bps,
            session_phase=session_phase,
        )
        supporting_evidence["expected_edge_bps"] = round(expected_edge * 10_000.0, 4)

        if session_phase in {"open", "close"} and abs(expected_edge) < self._policy.open_close_min_edge:
            blockers.append("insufficient_edge_for_session")
        if session_phase == "lunch" and news_catalyst < self._policy.strong_news_catalyst and abs(expected_edge) < self._policy.lunch_min_edge:
            blockers.append("midday_low_edge")
        if abs(expected_edge) < self._policy.min_expected_edge:
            blockers.append("expected_edge_too_small")

        cooldown_blocker = self._cooldown_blocker(
            symbol=symbol,
            occurred_at=occurred_at,
            proposed_action=self._action_from_score(score=score, expected_edge=expected_edge),
            proposed_score=score,
        )
        if cooldown_blocker is not None:
            blockers.append(cooldown_blocker)

        if regime == "neutral" and news_catalyst < self._policy.strong_news_catalyst and abs(signal_value) < 0.35:
            blockers.append("no_clear_regime_edge")

        action = self._action_from_score(score=score, expected_edge=expected_edge)
        if blockers and action != "HOLD":
            action = "HOLD"
        side = {"BUY": "LONG", "SELL": "SHORT", "HOLD": "FLAT"}[action]
        if action == "HOLD" and not blockers:
            blockers.append("abstain_mixed_or_weak_evidence")

        if action != "HOLD":
            rationale_codes.append("trade_candidate")
        else:
            rationale_codes.append("abstain")

        size_multiplier = self._size_multiplier(
            action=action,
            confidence=confidence,
            expected_edge=expected_edge,
            news_catalyst=news_catalyst,
            session_phase=session_phase,
            volatility=volatility,
        )
        holding_horizon_estimate = self._holding_horizon_estimate(
            action=action,
            regime=regime,
            news_catalyst=news_catalyst,
            session_phase=session_phase,
        )
        reasoning = self._build_reasoning(
            score=score,
            confidence=confidence,
            expected_edge=expected_edge,
            rationale_codes=rationale_codes,
            blockers=blockers,
        )
        result = DecisionResult(
            action=action,
            side=side,
            score=float(score),
            confidence=float(confidence),
            expected_edge=float(expected_edge),
            holding_horizon_estimate=holding_horizon_estimate,
            rationale_codes=tuple(dict.fromkeys(rationale_codes)),
            blockers=tuple(dict.fromkeys(blockers)),
            supporting_evidence=supporting_evidence,
            size_multiplier=float(size_multiplier),
            reasoning=reasoning,
        ).to_dict()

        if action != "HOLD" and symbol is not None and occurred_at is not None:
            self._last_trade_decision_by_symbol[symbol] = _PreviousDecision(
                action=action,
                score=float(score),
                occurred_at=occurred_at,
            )

        self._last_output = result
        return result.copy()

    def _resolve_symbol(self, market_signal: Mapping[str, Any]) -> str | None:
        """Return the active symbol when available."""

        candidate = market_signal.get("symbol", self._latest_symbol)
        if not isinstance(candidate, str):
            return None
        normalized = candidate.strip().upper()
        return normalized or None

    def _fallback_expected_edge(self, signal_value: float, confidence: float) -> float:
        """Return a conservative signed edge estimate when the model payload omits one."""

        return signal_value * max(0.00060, confidence * 0.00120)

    def _determine_regime(
        self,
        *,
        signal_value: float,
        ema_fast_mid: float,
        ema_mid_slow: float,
        vwap_gap: float,
        rsi: float | None,
    ) -> str:
        """Return the dominant trading regime implied by the features."""

        aligned_trend = (
            abs(ema_fast_mid) >= self._policy.trend_gap_threshold
            and abs(ema_mid_slow) >= self._policy.trend_gap_threshold
            and (ema_fast_mid * ema_mid_slow) > 0.0
            and (ema_fast_mid * signal_value) > 0.0
        )
        if aligned_trend:
            return "trend"

        if rsi is not None and abs(vwap_gap) >= self._policy.mean_reversion_vwap_gap:
            if signal_value > 0.0 and vwap_gap < 0.0 and rsi <= self._policy.mean_reversion_rsi_low:
                return "mean_reversion"
            if signal_value < 0.0 and vwap_gap > 0.0 and rsi >= self._policy.mean_reversion_rsi_high:
                return "mean_reversion"
        return "neutral"

    def _regime_bonus(
        self,
        *,
        regime: str,
        signal_value: float,
        vwap_gap: float,
        ema_fast_mid: float,
        ema_mid_slow: float,
    ) -> float:
        """Return a bounded regime contribution to the decision score."""

        if regime == "trend":
            trend_strength = min(abs(ema_fast_mid) + abs(ema_mid_slow), 0.010)
            return math.copysign(min(0.12, 12.0 * trend_strength), signal_value)
        if regime == "mean_reversion":
            return math.copysign(min(0.10, abs(vwap_gap) * 12.0), signal_value)
        return 0.0

    def _news_edge_adjustment(
        self,
        *,
        signal_value: float,
        news_strength: float,
        news_catalyst: float,
        news_scope: str,
        is_high_impact_news: bool,
    ) -> float:
        """Return a signed news edge adjustment in return space."""

        if news_catalyst <= 0.0 or abs(news_strength) < 0.05:
            return 0.0
        scope_multiplier = {"company": 1.0, "sector": 0.65, "market": 0.45}.get(news_scope, 0.35)
        impact_multiplier = 1.15 if is_high_impact_news else 0.75
        directional_alignment = 1.0 if (signal_value == 0.0 or (signal_value * news_strength) >= 0.0) else 0.85
        return news_strength * news_catalyst * scope_multiplier * impact_multiplier * directional_alignment * 0.00120

    def _news_score_component(self, *, news_strength: float, news_catalyst: float, news_scope: str) -> float:
        """Return the bounded news contribution to the score, keeping news subordinate to price."""

        if news_catalyst <= 0.0:
            return 0.0
        scope_weight = {"company": 0.18, "sector": 0.12, "market": 0.08}.get(news_scope, 0.06)
        return float(max(min(news_strength * news_catalyst * scope_weight, 0.22), -0.22))

    def _calibrated_confidence(
        self,
        *,
        model_confidence: float,
        news_strength: float,
        news_catalyst: float,
        regime: str,
        volatility: float | None,
        volume_spike: float,
    ) -> float:
        """Return a calibrated decision confidence score."""

        confidence = model_confidence * 0.75
        if regime != "neutral":
            confidence += 0.08
        if abs(news_strength) >= 0.10 and news_catalyst >= self._policy.strong_news_catalyst:
            confidence += 0.08
        if volume_spike >= 1.0:
            confidence += 0.04
        if volatility is not None and volatility >= self._policy.high_volatility_cutoff:
            confidence -= 0.12
        return float(max(min(confidence, 1.0), 0.0))

    def _apply_costs(self, *, raw_alpha: float, spread_bps: float | None, session_phase: str) -> float:
        """Return expected edge after conservative cost deductions."""

        if raw_alpha == 0.0:
            return 0.0
        spread_cost = 0.00010 if spread_bps is None else max(spread_bps, 0.0) / 10_000.0
        session_cost = 0.00018 if session_phase in {"open", "close"} else 0.00008
        total_cost = spread_cost + session_cost
        if abs(raw_alpha) <= total_cost:
            return 0.0
        return raw_alpha - math.copysign(total_cost, raw_alpha)

    def _action_from_score(self, *, score: float, expected_edge: float) -> str:
        """Map score and expected edge into an action."""

        if score >= self._policy.action_threshold and expected_edge > 0.0:
            return "BUY"
        if score <= -self._policy.action_threshold and expected_edge < 0.0:
            return "SELL"
        return "HOLD"

    def _cooldown_blocker(
        self,
        *,
        symbol: str | None,
        occurred_at: datetime | None,
        proposed_action: str,
        proposed_score: float,
    ) -> str | None:
        """Return a cooldown blocker for repeated same-side entries when applicable."""

        if symbol is None or occurred_at is None or proposed_action == "HOLD":
            return None
        previous = self._last_trade_decision_by_symbol.get(symbol)
        if previous is None or previous.action != proposed_action:
            return None
        elapsed = occurred_at - previous.occurred_at
        if elapsed > timedelta(minutes=self._policy.cooldown_minutes):
            return None
        if abs(proposed_score) <= abs(previous.score) + self._policy.min_score_improvement_for_reentry:
            return "cooldown_active"
        return None

    def _size_multiplier(
        self,
        *,
        action: str,
        confidence: float,
        expected_edge: float,
        news_catalyst: float,
        session_phase: str,
        volatility: float | None,
    ) -> float:
        """Return a deterministic participation multiplier for the risk layer."""

        if action == "HOLD":
            return 0.0
        edge_scale = min(abs(expected_edge) / self._policy.strong_expected_edge, 1.0)
        multiplier = edge_scale * (0.45 + (0.55 * confidence))
        if news_catalyst >= self._policy.strong_news_catalyst:
            multiplier += 0.05
        if session_phase in {"open", "close"}:
            multiplier *= 0.85
        if volatility is not None and volatility >= self._policy.high_volatility_cutoff:
            multiplier *= 0.70
        return float(max(min(multiplier, 1.0), 0.10))

    def _holding_horizon_estimate(
        self,
        *,
        action: str,
        regime: str,
        news_catalyst: float,
        session_phase: str,
    ) -> int:
        """Return the estimated holding horizon in bars."""

        if action == "HOLD":
            return 0
        if regime == "trend" and news_catalyst >= self._policy.strong_news_catalyst:
            return 3
        if regime == "trend":
            return 2
        if session_phase in {"open", "close"}:
            return 1
        return 2

    def _build_reasoning(
        self,
        *,
        score: float,
        confidence: float,
        expected_edge: float,
        rationale_codes: list[str],
        blockers: list[str],
    ) -> str:
        """Return a deterministic, auditable reasoning string."""

        rationale = ",".join(dict.fromkeys(rationale_codes)) or "none"
        blocker_text = ",".join(dict.fromkeys(blockers)) or "none"
        return (
            f"score={score:.4f};confidence={confidence:.4f};expected_edge_bps={expected_edge * 10000.0:.2f};"
            f"rationale={rationale};blockers={blocker_text}"
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

    def _news_payload_from_event(self, event: NewsEvent) -> dict[str, Any] | None:
        """Return a normalized news-analysis payload from a news event."""

        payload = dict(event.payload or {})
        if "sentiment" not in payload:
            return None
        return normalize_news_analysis(payload)
