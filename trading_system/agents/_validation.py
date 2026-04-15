"""Shared deterministic payload validation helpers for trading agents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


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


def normalize_news_analysis(payload: Mapping[str, Any]) -> dict[str, float | str]:
    """Return a validated news-analysis payload."""

    if "sentiment" not in payload:
        raise ValueError("news analysis payload must include sentiment")

    impact_value = payload.get("impact", 0.0)
    event_type = str(payload.get("event_type", "other")).strip()
    summary = str(payload.get("summary", "")).strip()
    if not event_type:
        raise ValueError("event_type must be a non-empty string")
    if not summary:
        raise ValueError("summary must be a non-empty string")

    return {
        "sentiment": coerce_bounded_float(payload["sentiment"], field_name="sentiment", minimum=-1.0, maximum=1.0),
        "impact": coerce_bounded_float(impact_value, field_name="impact", minimum=0.0, maximum=1.0),
        "event_type": event_type,
        "summary": summary,
    }


def normalize_decision_output(payload: Mapping[str, Any]) -> dict[str, float | str]:
    """Return a validated decision payload."""

    action = str(payload.get("action", "")).strip().upper()
    reasoning = str(payload.get("reasoning", "")).strip()
    if action not in {"BUY", "SELL", "HOLD"}:
        raise ValueError("action must be one of BUY, SELL, HOLD")
    if not reasoning:
        raise ValueError("reasoning must be a non-empty string")

    return {
        "action": action,
        "score": coerce_bounded_float(payload.get("score"), field_name="score", minimum=-1.0, maximum=1.0),
        "reasoning": reasoning,
    }


def coerce_unbounded_float(value: Any, *, field_name: str) -> float:
    """Return a validated finite float without range constraints."""

    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def extract_volatility(features: Mapping[str, Any]) -> float | None:
    """Return the first available normalized volatility feature when present."""

    if "volatility" in features:
        return coerce_unbounded_float(features["volatility"], field_name="volatility")

    volatility_keys = sorted(key for key in features if str(key).startswith("volatility_"))
    if not volatility_keys:
        return None
    key = volatility_keys[0]
    return coerce_unbounded_float(features[key], field_name=str(key))


def signals_conflict(market_signal: float, news_sentiment: float, *, threshold: float) -> bool:
    """Return whether market and news inputs disagree materially."""

    return abs(market_signal) >= threshold and abs(news_sentiment) >= threshold and (market_signal * news_sentiment) < 0.0
