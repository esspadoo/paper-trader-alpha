"""Shared deterministic payload validation and extraction helpers for trading agents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from trading_system.agents.schemas import CriticReviewSchema, DecisionSchema, NewsAnalysisSchema


def coerce_bounded_float(
    value: Any,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Return a validated finite float within the requested bounds."""

    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")

    numeric = float(value)
    if numeric < minimum or numeric > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return numeric


def coerce_unbounded_float(value: Any, *, field_name: str) -> float:
    """Return a validated finite float without range constraints."""

    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def normalize_features(features: Any) -> dict[str, Any]:
    """Return a sanitized feature mapping."""

    if features is None:
        return {}
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping when provided")

    normalized: dict[str, Any] = {}
    for key, value in features.items():
        feature_name = str(key).strip()
        if not feature_name:
            raise ValueError("feature names must be non-empty strings")
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise ValueError(f"feature {feature_name} must be finite")
            normalized[feature_name] = float(value)
            continue
        if isinstance(value, str):
            normalized[feature_name] = value.strip()
            continue
        normalized[feature_name] = value
    return normalized


def normalize_market_signal(
    payload: Mapping[str, Any],
    *,
    default_confidence: float | None = None,
) -> dict[str, Any]:
    """Return a validated market-signal payload."""

    if "signal" not in payload:
        raise ValueError("market signal payload must include signal")

    confidence_value = payload.get("confidence", default_confidence if default_confidence is not None else 0.0)
    normalized = {
        "signal": coerce_bounded_float(payload["signal"], field_name="signal", minimum=-1.0, maximum=1.0),
        "confidence": coerce_bounded_float(
            confidence_value,
            field_name="confidence",
            minimum=0.0,
            maximum=1.0,
        ),
        "features": normalize_features(payload.get("features")),
    }

    if "predicted_return" in payload:
        normalized["predicted_return"] = coerce_unbounded_float(payload["predicted_return"], field_name="predicted_return")
    if "symbol" in payload and payload["symbol"] is not None:
        symbol = str(payload["symbol"]).strip().upper()
        if symbol:
            normalized["symbol"] = symbol
    return normalized


def normalize_news_analysis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated rich news-analysis payload."""

    defaults = {
        "effective_sentiment": float(payload.get("effective_sentiment", payload.get("sentiment", 0.0))),
        "scope": payload.get("scope", "market"),
        "scheduled_event": bool(payload.get("scheduled_event", False)),
        "relevance_score": float(payload.get("relevance_score", 0.0)),
        "novelty_score": float(payload.get("novelty_score", 1.0)),
        "credibility_score": float(payload.get("credibility_score", 0.5)),
        "freshness_score": float(payload.get("freshness_score", 1.0)),
        "catalyst_score": float(payload.get("catalyst_score", payload.get("impact", 0.0))),
        "source": str(payload.get("source", "unknown")),
        "symbols": tuple(payload.get("symbols", ()) or ()),
        "primary_symbol": payload.get("primary_symbol"),
        "rationale_codes": tuple(payload.get("rationale_codes", ()) or ()),
        "is_high_impact": bool(payload.get("is_high_impact", False)),
    }
    schema = NewsAnalysisSchema(
        sentiment=float(payload.get("sentiment", 0.0)),
        impact=float(payload.get("impact", 0.0)),
        event_type=str(payload.get("event_type", "other")),
        summary=str(payload.get("summary", "No material news available.")),
        **defaults,
    )
    return schema.to_dict()


def normalize_decision_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated rich decision payload with backward-compatible defaults."""

    action = str(payload.get("action", "HOLD")).strip().upper() or "HOLD"
    side = str(payload.get("side", {"BUY": "LONG", "SELL": "SHORT", "HOLD": "FLAT"}.get(action, "FLAT"))).strip().upper()
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("reasoning must be a non-empty string")

    schema = DecisionSchema(
        action=action,
        side=side,
        score=float(payload.get("score", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        expected_edge=float(payload.get("expected_edge", 0.0)),
        holding_horizon_estimate=int(payload.get("holding_horizon_estimate", 0)),
        rationale_codes=tuple(payload.get("rationale_codes", ()) or ()),
        blockers=tuple(payload.get("blockers", ()) or ()),
        supporting_evidence=dict(payload.get("supporting_evidence", {}) or {}),
        size_multiplier=float(payload.get("size_multiplier", 0.0)),
        reasoning=reasoning,
    )
    return schema.to_dict()


def normalize_critic_review(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated critic review payload."""

    schema = CriticReviewSchema(
        approved=bool(payload.get("approved", False)),
        reason=str(payload.get("reason", "")).strip(),
        blocker_codes=tuple(payload.get("blocker_codes", ()) or ()),
        risk_flags=tuple(payload.get("risk_flags", ()) or ()),
    )
    return schema.to_dict()


def extract_feature(features: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first finite numeric feature for the provided keys."""

    for key in keys:
        if key not in features:
            continue
        try:
            return coerce_unbounded_float(features[key], field_name=key)
        except ValueError:
            continue
    return None


def extract_volatility(features: Mapping[str, Any]) -> float | None:
    """Return the first available normalized volatility feature when present."""

    if "volatility" in features:
        return coerce_unbounded_float(features["volatility"], field_name="volatility")

    volatility_keys = sorted(key for key in features if str(key).startswith("volatility_"))
    if not volatility_keys:
        return None
    key = volatility_keys[0]
    return coerce_unbounded_float(features[key], field_name=str(key))


def extract_rsi(features: Mapping[str, Any]) -> float | None:
    """Return the first available RSI feature when present."""

    if "rsi" in features:
        return coerce_unbounded_float(features["rsi"], field_name="rsi")
    rsi_keys = sorted(key for key in features if str(key).startswith("rsi_"))
    if not rsi_keys:
        return None
    key = rsi_keys[0]
    return coerce_unbounded_float(features[key], field_name=str(key))


def extract_spread_bps(features: Mapping[str, Any]) -> float | None:
    """Return the current spread in basis points when present."""

    return extract_feature(features, "spread_bps")


def extract_dollar_volume(features: Mapping[str, Any]) -> float | None:
    """Return the estimated current dollar volume when present."""

    return extract_feature(features, "dollar_volume")


def extract_session_phase(features: Mapping[str, Any]) -> str:
    """Return the named session phase when present."""

    candidate = features.get("session_phase")
    if not isinstance(candidate, str):
        return "unknown"
    normalized = candidate.strip().lower()
    return normalized or "unknown"


def extract_expected_edge_bps(decision: Mapping[str, Any]) -> float:
    """Return the decision expected edge in basis points."""

    expected_edge = float(decision.get("expected_edge", 0.0))
    return expected_edge * 10_000.0


def extract_news_strength(news_analysis: Mapping[str, Any]) -> float:
    """Return the decayed effective news sentiment used by the fusion policy."""

    candidate = news_analysis.get("effective_sentiment", news_analysis.get("sentiment", 0.0))
    return coerce_bounded_float(candidate, field_name="effective_sentiment", minimum=-1.0, maximum=1.0)


def signals_conflict(market_signal: float, news_sentiment: float, *, threshold: float) -> bool:
    """Return whether market and news inputs disagree materially."""

    return abs(market_signal) >= threshold and abs(news_sentiment) >= threshold and (market_signal * news_sentiment) < 0.0
