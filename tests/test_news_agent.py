"""Offline tests for news ingestion and local-LLM analysis."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from trading_system.agents import NewsAgent
from trading_system.core import NewsEvent
from trading_system.data import JSONNewsSource, NewsArticle, RSSNewsSource
from trading_system.models import LocalLLMBackend, LocalNewsLLMAnalyzer, StrictJSONParseError


class FakeLocalLLMBackend(LocalLLMBackend):
    """Deterministic backend stub used by the news-agent tests."""

    def __init__(self, responses: list[str]) -> None:
        """Store the response sequence emitted by the fake backend."""

        self._responses = list(responses)
        self.calls = 0

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""

        return "fake-local-llm"

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        timeout_seconds: float,
    ) -> str:
        """Return the next canned response."""

        self.calls += 1
        if not self._responses:
            raise RuntimeError("no fake responses remaining")
        return self._responses.pop(0)


class NewsAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify RSS/API ingestion, retry logic, and strict JSON analysis."""

    async def test_rss_source_and_news_agent_deduplicate_and_cache(self) -> None:
        """The agent should deduplicate repeated RSS items and cache analysis results."""

        rss_payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Local Feed</title>
    <item>
      <guid>article-1</guid>
      <title>Acme beats earnings estimates</title>
      <link>https://local.test/acme-earnings</link>
      <description>Revenue and guidance both moved higher.</description>
      <pubDate>Mon, 14 Apr 2026 09:30:00 GMT</pubDate>
    </item>
    <item>
      <guid>article-1</guid>
      <title>Acme beats earnings estimates</title>
      <link>https://local.test/acme-earnings</link>
      <description>Revenue and guidance both moved higher.</description>
      <pubDate>Mon, 14 Apr 2026 09:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

        source = RSSNewsSource(
            source_name="rss-local",
            feed_uri="memory://rss",
            fetcher=lambda target, timeout: rss_payload,
        )
        backend = FakeLocalLLMBackend(
            [
                "not-json",
                '{"sentiment": 0.7, "impact": 0.8, "event_type": "earnings", "summary": "Strong earnings and higher guidance."}',
            ]
        )
        agent = NewsAgent(source=source, analyzer=LocalNewsLLMAnalyzer(backend, max_retries=2))

        await agent.start()
        first_pass = await agent.poll()
        second_pass = await agent.poll()
        await agent.stop()

        self.assertEqual(len(first_pass), 1)
        self.assertEqual(second_pass, [])
        self.assertEqual(backend.calls, 2)
        self.assertEqual(first_pass[0]["event_type"], "earnings")
        self.assertAlmostEqual(first_pass[0]["sentiment"], 0.7)

    async def test_json_source_parses_articles(self) -> None:
        """The JSON source should normalize article mappings into NewsArticle records."""

        payload = b"""
{
  "articles": [
    {
      "id": "json-1",
      "headline": "Factory outage affects supply",
      "link": "https://local.test/factory",
      "description": "Production disruption could pressure margins.",
      "body": "A major plant outage is expected to last one week.",
      "published_at": "2026-04-14T10:00:00Z",
      "tickers": ["ACME"]
    }
  ]
}
"""
        source = JSONNewsSource(
            source_name="json-local",
            endpoint="memory://json",
            fetcher=lambda target, timeout: payload,
        )

        articles = await source.fetch_articles()
        self.assertEqual(len(articles), 1)
        self.assertIsInstance(articles[0], NewsArticle)
        self.assertEqual(articles[0].symbols, ("ACME",))
        self.assertEqual(articles[0].published_at, datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc))

    async def test_news_event_is_analyzed_and_cached(self) -> None:
        """NewsEvent payloads should be converted into articles and analyzed once."""

        backend = FakeLocalLLMBackend(
            ['{"sentiment": -0.4, "impact": 0.6, "event_type": "regulation", "summary": "Regulatory pressure may hurt operations."}']
        )
        agent = NewsAgent(analyzer=LocalNewsLLMAnalyzer(backend))

        event = NewsEvent(
            source="news-wire",
            headline="Regulator opens investigation",
            body="A regulator opened a new investigation into the company.",
            symbols=("ACME",),
            payload={"url": "https://local.test/regulation"},
        )

        await agent.start()
        await agent.on_event(event)
        first_snapshot = await agent.snapshot_state()
        await agent.on_event(event)
        second_snapshot = await agent.snapshot_state()
        await agent.stop()

        self.assertEqual(backend.calls, 1)
        self.assertEqual(first_snapshot["last_output"]["event_type"], "regulation")
        self.assertEqual(first_snapshot["last_output"], second_snapshot["last_output"])

    async def test_news_agent_enriches_company_news_with_deterministic_scores(self) -> None:
        """Structured LLM output should be enriched with catalyst, scope, and credibility scores."""

        backend = FakeLocalLLMBackend(
            ['{"sentiment": 0.8, "impact": 0.9, "event_type": "earnings", "summary": "Strong beat and raised guidance."}']
        )
        agent = NewsAgent(analyzer=LocalNewsLLMAnalyzer(backend))
        event = NewsEvent(
            source="reuters",
            headline="ACME beats and raises guidance",
            body="ACME posted a strong beat and raised full-year guidance.",
            symbols=("ACME",),
            payload={"url": "https://www.reuters.com/local/acme"},
            occurred_at=datetime(2026, 4, 15, 13, 45, tzinfo=timezone.utc),
        )

        await agent.start()
        await agent.on_event(event)
        snapshot = await agent.snapshot_state()
        await agent.stop()

        output = snapshot["last_output"]
        self.assertEqual(output["scope"], "company")
        self.assertGreater(float(output["credibility_score"]), 0.8)
        self.assertGreater(float(output["catalyst_score"]), 0.5)
        self.assertGreater(float(output["effective_sentiment"]), 0.0)
        self.assertTrue(bool(output["is_high_impact"]))

    async def test_news_agent_deduplicates_near_duplicate_articles_in_same_batch(self) -> None:
        """Near-duplicate articles should only be analyzed once per batch."""

        payload = b"""
{
  "articles": [
    {
      "id": "dup-1",
      "headline": "ACME launches new platform",
      "link": "https://wire.local/acme-1",
      "description": "ACME launches a new platform for enterprise clients.",
      "body": "The company launches a new platform for enterprise clients.",
      "published_at": "2026-04-15T10:00:00Z",
      "tickers": ["ACME"]
    },
    {
      "id": "dup-2",
      "headline": "ACME launches new platform",
      "link": "https://wire.local/acme-2",
      "description": "ACME launches a new platform for enterprise clients.",
      "body": "The company launches a new platform for enterprise clients.",
      "published_at": "2026-04-15T10:01:00Z",
      "tickers": ["ACME"]
    }
  ]
}
"""
        source = JSONNewsSource(
            source_name="json-local",
            endpoint="memory://json",
            fetcher=lambda target, timeout: payload,
        )
        backend = FakeLocalLLMBackend(
            ['{"sentiment": 0.4, "impact": 0.5, "event_type": "product_launch", "summary": "Launch may support intraday demand."}']
        )
        agent = NewsAgent(source=source, analyzer=LocalNewsLLMAnalyzer(backend))

        await agent.start()
        results = await agent.poll()
        await agent.stop()

        self.assertEqual(len(results), 1)
        self.assertEqual(backend.calls, 1)


class StrictJSONTests(unittest.TestCase):
    """Verify strict JSON parsing rejects malformed local-LLM output."""

    def test_extra_text_is_rejected(self) -> None:
        """Responses with surrounding prose must fail strict parsing."""

        analyzer = LocalNewsLLMAnalyzer(FakeLocalLLMBackend([]))
        with self.assertRaises(StrictJSONParseError):
            analyzer.parse_strict_json('Here is the result: {"sentiment": 0.1, "impact": 0.2, "event_type": "other", "summary": "ok"}')
