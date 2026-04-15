"""News agent backed by local LLM analysis and source deduplication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trading_system.agents.base import BaseAgent
from trading_system.core.events import BaseEvent, NewsEvent
from trading_system.data import BaseNewsSource, NewsArticle, NewsDeduplicator
from trading_system.infra import TTLCache
from trading_system.models import LocalNewsLLMAnalyzer


class NewsAgent(BaseAgent):
    """Ingest news, deduplicate articles, and analyze them with a local LLM."""

    def __init__(
        self,
        *,
        agent_id: str = "news-agent",
        source: BaseNewsSource | None = None,
        analyzer: LocalNewsLLMAnalyzer,
        deduplicator: NewsDeduplicator | None = None,
        result_cache: TTLCache[str, dict[str, float | str]] | None = None,
        result_ttl_seconds: float = 43_200.0,
    ) -> None:
        """Initialize the NewsAgent and its local analysis pipeline."""

        if result_ttl_seconds <= 0:
            raise ValueError("result_ttl_seconds must be greater than 0")

        self._agent_id = agent_id
        self._source = source
        self._analyzer = analyzer
        self._deduplicator = deduplicator or NewsDeduplicator()
        self._result_cache = result_cache or TTLCache[str, dict[str, float | str]](
            default_ttl_seconds=result_ttl_seconds
        )
        self._result_ttl_seconds = result_ttl_seconds
        self._running = False
        self._last_output: dict[str, float | str] | None = None
        self._last_article_fingerprint: str | None = None

    @property
    def agent_id(self) -> str:
        """Return the stable identifier for the agent."""

        return self._agent_id

    @property
    def subscribed_topics(self) -> Sequence[str]:
        """Return the event topics this agent can consume."""

        return ("news",)

    @property
    def source_name(self) -> str | None:
        """Return the configured source identifier when available."""

        return self._source.source_name if self._source is not None else None

    async def start(self) -> None:
        """Warm the local analyzer and activate the agent."""

        await self._analyzer.warmup()
        self._running = True

    async def stop(self) -> None:
        """Deactivate the agent."""

        self._running = False

    async def ingest(self) -> list[NewsArticle]:
        """Fetch new, not-yet-seen articles from the configured source."""

        if self._source is None:
            raise ValueError("NewsAgent ingest requires a configured news source")

        fetched = await self._source.fetch_articles()
        batch_fingerprints: set[str] = set()
        unique_articles: list[NewsArticle] = []
        for article in fetched:
            fingerprint = article.fingerprint
            if fingerprint in batch_fingerprints or self._deduplicator.has_seen(article):
                continue
            batch_fingerprints.add(fingerprint)
            unique_articles.append(article)
        return unique_articles

    async def analyze_article(self, article: NewsArticle) -> dict[str, float | str]:
        """Analyze a single article with deduplication and cached results."""

        cached = self._result_cache.get(article.fingerprint)
        if cached is not None:
            self._last_output = cached
            self._last_article_fingerprint = article.fingerprint
            return cached

        analysis = (await self._analyzer.analyze(article)).to_dict()
        self._deduplicator.mark_seen(article)
        self._result_cache.set(article.fingerprint, analysis, ttl_seconds=self._result_ttl_seconds)
        self._last_output = analysis
        self._last_article_fingerprint = article.fingerprint
        return analysis

    async def poll(self) -> list[dict[str, float | str]]:
        """Fetch, deduplicate, and analyze the latest articles from the source."""

        results: list[dict[str, float | str]] = []
        for article in await self.ingest():
            results.append(await self.analyze_article(article))
        return results

    async def on_event(self, event: BaseEvent) -> None:
        """Consume a NewsEvent and analyze its normalized article payload."""

        if not isinstance(event, NewsEvent):
            return

        article = self._article_from_event(event)
        self._last_output = await self.analyze_article(article)

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of agent state."""

        return {
            "agent_id": self._agent_id,
            "running": self._running,
            "source_name": self.source_name,
            "backend_name": self._analyzer.backend_name,
            "last_article_fingerprint": self._last_article_fingerprint,
            "last_output": self._last_output,
        }

    def _article_from_event(self, event: NewsEvent) -> NewsArticle:
        """Normalize a NewsEvent into a NewsArticle for analysis."""

        payload = dict(event.payload or {})
        payload.setdefault("source", event.source)
        payload.setdefault("headline", event.headline)
        payload.setdefault("summary", event.body or "")
        payload.setdefault("content", event.body or "")
        payload.setdefault("published_at", event.occurred_at.isoformat())
        payload.setdefault("symbols", list(event.symbols))
        payload.setdefault("url", payload.get("url") or f"event://{event.event_id}")
        return NewsArticle.from_mapping(event.source, payload)
