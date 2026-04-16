"""Deterministic enrichment and scoring rules for intraday news signals."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from trading_system.agents.schemas import NewsAnalysisSchema
from trading_system.data.news import NewsArticle


_HIGH_CREDIBILITY_KEYWORDS = ("reuters", "dowjones", "bloomberg", "wsj", "financial-times", "sec", "federalreserve")
_LOW_CREDIBILITY_KEYWORDS = ("social", "blog", "forum", "rumor", "anon")
_MACRO_KEYWORDS = ("fed", "cpi", "ppi", "payrolls", "macro", "fomc", "treasury", "rates", "economy")
_SCHEDULED_KEYWORDS = ("earnings", "guidance", "conference call", "cfo", "ceo", "investor day", "dividend", "split")
_SECTOR_KEYWORDS = ("sector", "industry", "peers", "semiconductor", "banking", "energy")
_TOKEN_NORMALIZER = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class NewsScoringConfig:
    """Deterministic weights used for news enrichment."""

    freshness_half_life_minutes: float = 120.0
    market_freshness_half_life_minutes: float = 90.0
    novelty_floor: float = 0.15
    default_credibility: float = 0.55
    high_impact_threshold: float = 0.35


def canonicalize_article_text(article: NewsArticle) -> str:
    """Return a canonical article text string for near-duplicate detection."""

    combined = " ".join(
        part for part in (article.title, article.summary, article.content) if part.strip()
    ).lower()
    normalized = _TOKEN_NORMALIZER.sub(" ", combined)
    return " ".join(token for token in normalized.split() if len(token) > 2)


def score_article_analysis(
    *,
    article: NewsArticle,
    sentiment: float,
    impact: float,
    event_type: str,
    summary: str,
    novelty_score: float,
    config: NewsScoringConfig | None = None,
) -> NewsAnalysisSchema:
    """Return a deterministic enriched news signal."""

    scoring = config or NewsScoringConfig()
    scope = _infer_scope(article=article, event_type=event_type)
    scheduled_event = _infer_scheduled_event(article=article, event_type=event_type)
    relevance_score = _infer_relevance(article=article, scope=scope)
    credibility_score = _infer_credibility(article=article, default=scoring.default_credibility)
    freshness_score = 1.0
    base_catalyst = impact * relevance_score * novelty_score * credibility_score
    catalyst_score = float(min(max(base_catalyst, 0.0), 1.0))
    effective_sentiment = float(max(min(sentiment * catalyst_score, 1.0), -1.0))
    rationale_codes = _build_rationale_codes(
        article=article,
        event_type=event_type,
        scope=scope,
        scheduled_event=scheduled_event,
        catalyst_score=catalyst_score,
        novelty_score=novelty_score,
        credibility_score=credibility_score,
    )
    primary_symbol = article.symbols[0] if len(article.symbols) == 1 else None
    return NewsAnalysisSchema(
        sentiment=sentiment,
        impact=impact,
        event_type=event_type,
        summary=summary,
        effective_sentiment=effective_sentiment,
        scope=scope,
        scheduled_event=scheduled_event,
        relevance_score=relevance_score,
        novelty_score=novelty_score,
        credibility_score=credibility_score,
        freshness_score=freshness_score,
        catalyst_score=catalyst_score,
        source=article.source,
        symbols=article.symbols,
        primary_symbol=primary_symbol,
        rationale_codes=rationale_codes,
        is_high_impact=catalyst_score >= scoring.high_impact_threshold and impact >= 0.6,
    )


def apply_freshness_decay(
    analysis: NewsAnalysisSchema,
    *,
    published_at: datetime,
    now: datetime,
    config: NewsScoringConfig | None = None,
) -> NewsAnalysisSchema:
    """Return a copy of an analysis payload with freshness-decayed fields."""

    scoring = config or NewsScoringConfig()
    published = published_at.astimezone(timezone.utc)
    current = now.astimezone(timezone.utc)
    age_minutes = max((current - published).total_seconds(), 0.0) / 60.0
    half_life = scoring.market_freshness_half_life_minutes if analysis.scope == "market" else scoring.freshness_half_life_minutes
    freshness_score = math.exp(-math.log(2.0) * (age_minutes / max(half_life, 1.0)))
    freshness_score = float(min(max(freshness_score, 0.0), 1.0))
    catalyst_score = float(min(max(analysis.catalyst_score * freshness_score, 0.0), 1.0))
    effective_sentiment = float(max(min(analysis.sentiment * catalyst_score, 1.0), -1.0))
    updated_rationale = tuple(dict.fromkeys((*analysis.rationale_codes, "fresh_news" if freshness_score >= 0.5 else "decayed_news")))
    return analysis.model_copy(
        update={
            "freshness_score": freshness_score,
            "catalyst_score": catalyst_score,
            "effective_sentiment": effective_sentiment,
            "rationale_codes": updated_rationale,
            "is_high_impact": analysis.is_high_impact and freshness_score >= 0.35,
        }
    )


def novelty_from_seen_state(*, seen_exact: bool, seen_canonical: bool, config: NewsScoringConfig | None = None) -> float:
    """Return a deterministic novelty score from deduplication state."""

    scoring = config or NewsScoringConfig()
    if seen_exact:
        return 0.0
    if seen_canonical:
        return scoring.novelty_floor
    return 1.0


def _infer_scope(*, article: NewsArticle, event_type: str) -> str:
    """Return whether the article is company, sector, or market scoped."""

    text = canonicalize_article_text(article)
    if article.symbols:
        return "company" if len(article.symbols) == 1 else "sector"
    if event_type in {"macro", "regulation"} or any(keyword in text for keyword in _MACRO_KEYWORDS):
        return "market"
    if any(keyword in text for keyword in _SECTOR_KEYWORDS):
        return "sector"
    return "market"


def _infer_scheduled_event(*, article: NewsArticle, event_type: str) -> bool:
    """Return whether the catalyst is likely scheduled."""

    text = canonicalize_article_text(article)
    return event_type in {"earnings", "guidance", "analyst_rating", "management_change", "financing"} or any(
        keyword in text for keyword in _SCHEDULED_KEYWORDS
    )


def _infer_relevance(*, article: NewsArticle, scope: str) -> float:
    """Return a deterministic relevance score for trading decisions."""

    if article.symbols:
        return 1.0 if len(article.symbols) == 1 else 0.8
    if scope == "sector":
        return 0.55
    return 0.35


def _infer_credibility(*, article: NewsArticle, default: float) -> float:
    """Return a source credibility score from deterministic source rules."""

    source_key = f"{article.source} {urlparse(article.url).netloc}".lower()
    if any(keyword in source_key for keyword in _HIGH_CREDIBILITY_KEYWORDS):
        return 0.9
    if any(keyword in source_key for keyword in _LOW_CREDIBILITY_KEYWORDS):
        return 0.25
    if urlparse(article.url).scheme in {"http", "https"} and "." in urlparse(article.url).netloc:
        return min(max(default + 0.05, 0.0), 1.0)
    return default


def _build_rationale_codes(
    *,
    article: NewsArticle,
    event_type: str,
    scope: str,
    scheduled_event: bool,
    catalyst_score: float,
    novelty_score: float,
    credibility_score: float,
) -> tuple[str, ...]:
    """Return deterministic rationale codes for the news payload."""

    codes = [
        f"scope_{scope}",
        f"event_{event_type}",
        "scheduled_event" if scheduled_event else "unscheduled_event",
    ]
    if article.symbols:
        codes.append("explicit_ticker_mapping")
    else:
        codes.append("broad_market_mapping")
    if novelty_score < 0.5:
        codes.append("low_novelty")
    else:
        codes.append("novel_catalyst")
    if credibility_score >= 0.8:
        codes.append("high_credibility_source")
    if catalyst_score >= 0.35:
        codes.append("high_impact_catalyst")
    return tuple(dict.fromkeys(codes))
