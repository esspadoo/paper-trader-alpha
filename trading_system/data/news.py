"""News ingestion, normalization, and deduplication utilities."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from trading_system.data.exceptions import NewsParsingError, NewsSourceError, NewsTimeoutError
from trading_system.infra import TTLCache

ContentFetcher = Callable[[str, float], bytes]

_CONTENT_NAMESPACE = {"content": "http://purl.org/rss/1.0/modules/content/"}


def _normalize_text(value: str | None) -> str:
    """Return a stripped string without surrounding whitespace."""

    return (value or "").strip()


def _normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize and deduplicate ticker symbols."""

    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols or ():
        candidate = str(symbol).strip().upper()
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return tuple(normalized)


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a timestamp to UTC when present."""

    if not value:
        return None

    candidate = value.strip()
    if not candidate:
        return None

    parsers = (
        lambda raw: parsedate_to_datetime(raw),
        lambda raw: datetime.fromisoformat(raw.replace("Z", "+00:00")),
    )
    for parser in parsers:
        try:
            parsed = parser(candidate)
        except (TypeError, ValueError):
            continue

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    raise NewsParsingError(f"unable to parse news timestamp: {candidate}")


def _default_content_fetcher(target: str, timeout_seconds: float) -> bytes:
    """Fetch bytes from a local file path or HTTP endpoint."""

    parsed = urlparse(target)
    if parsed.scheme in {"http", "https"}:
        request = Request(target, headers={"User-Agent": "trading-system-news-agent/1.0"})
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()

    path = Path(target)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise NewsSourceError(f"unable to read news source from {target}") from exc


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """Normalized news article record used by the NewsAgent."""

    source: str
    title: str
    url: str
    summary: str = ""
    content: str = ""
    published_at: datetime | None = None
    article_id: str | None = None
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the article fields and validate required values."""

        source = _normalize_text(self.source)
        title = _normalize_text(self.title)
        url = _normalize_text(self.url)
        summary = _normalize_text(self.summary)
        content = _normalize_text(self.content)
        article_id = _normalize_text(self.article_id) or None
        published_at = self.published_at

        if not source:
            raise ValueError("source must be a non-empty string")
        if not title:
            raise ValueError("title must be a non-empty string")
        if not url:
            raise ValueError("url must be a non-empty string")
        if published_at is not None and published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        if published_at is not None:
            published_at = published_at.astimezone(timezone.utc)

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "article_id", article_id)
        object.__setattr__(self, "published_at", published_at)
        object.__setattr__(self, "symbols", _normalize_symbols(self.symbols))

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for deduplication and caching."""

        payload = "|".join(
            [
                self.source.lower(),
                (self.article_id or "").lower(),
                self.url.lower(),
                self.title.lower(),
                self.summary.lower(),
                self.published_at.isoformat() if self.published_at else "",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_prompt_context(self) -> dict[str, Any]:
        """Return a serializable article payload for prompt rendering."""

        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "content": self.content,
            "published_at": self.published_at.isoformat() if self.published_at else "",
            "symbols": list(self.symbols),
        }

    @classmethod
    def from_mapping(cls, source_name: str, payload: dict[str, Any]) -> "NewsArticle":
        """Build a normalized article from a generic RSS or API mapping."""

        title = payload.get("title") or payload.get("headline") or ""
        url = payload.get("url") or payload.get("link") or payload.get("href") or ""
        summary = payload.get("summary") or payload.get("description") or payload.get("teaser") or ""
        content = payload.get("content") or payload.get("body") or payload.get("article") or ""
        published_raw = (
            payload.get("published_at")
            or payload.get("published")
            or payload.get("updated")
            or payload.get("pubDate")
            or payload.get("date")
        )
        symbols = payload.get("symbols") or payload.get("tickers") or ()
        if isinstance(symbols, str):
            symbols = [token.strip() for token in symbols.split(",")]

        return cls(
            source=_normalize_text(payload.get("source") or source_name),
            title=str(title),
            url=str(url),
            summary=str(summary),
            content=str(content),
            published_at=_parse_timestamp(str(published_raw)) if published_raw else None,
            article_id=str(payload.get("article_id") or payload.get("id") or payload.get("guid") or "") or None,
            symbols=tuple(symbols),
        )


class BaseNewsSource(ABC):
    """Abstract asynchronous source of normalized news articles."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the stable source identifier."""

    @abstractmethod
    async def fetch_articles(self) -> list[NewsArticle]:
        """Return the latest available normalized articles."""


class RSSNewsSource(BaseNewsSource):
    """Read and normalize articles from an RSS or Atom feed."""

    def __init__(
        self,
        *,
        source_name: str,
        feed_uri: str,
        timeout_seconds: float = 5.0,
        fetcher: ContentFetcher | None = None,
    ) -> None:
        """Initialize the source with a feed URI and optional fetcher."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self._source_name = source_name.strip()
        self._feed_uri = feed_uri.strip()
        self._timeout_seconds = timeout_seconds
        self._fetcher = fetcher or _default_content_fetcher

    @property
    def source_name(self) -> str:
        """Return the configured source identifier."""

        return self._source_name

    async def fetch_articles(self) -> list[NewsArticle]:
        """Fetch and parse articles from the RSS or Atom feed."""

        payload = await self._fetch_bytes(self._feed_uri)
        return self._parse_feed(payload)

    async def _fetch_bytes(self, target: str) -> bytes:
        """Fetch raw source bytes with timeout handling."""

        try:
            return self._fetcher(target, self._timeout_seconds)
        except TimeoutError as exc:
            raise NewsTimeoutError(f"news source {self._source_name} exceeded {self._timeout_seconds:.2f}s") from exc
        except (OSError, URLError, NewsSourceError) as exc:
            raise NewsSourceError(f"unable to fetch RSS feed for {self._source_name}") from exc

    def _parse_feed(self, payload: bytes) -> list[NewsArticle]:
        """Parse an RSS or Atom feed payload into normalized articles."""

        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise NewsParsingError(f"invalid XML feed for {self._source_name}") from exc

        if root.tag.endswith("feed"):
            return self._parse_atom(root)
        return self._parse_rss(root)

    def _parse_rss(self, root: ElementTree.Element) -> list[NewsArticle]:
        """Parse an RSS 2.0 feed."""

        items = root.findall("./channel/item")
        articles: list[NewsArticle] = []
        for item in items:
            title = _normalize_text(item.findtext("title"))
            link = _normalize_text(item.findtext("link"))
            summary = _normalize_text(item.findtext("description"))
            content = _normalize_text(item.findtext("content:encoded", default="", namespaces=_CONTENT_NAMESPACE))
            published_at = _parse_timestamp(item.findtext("pubDate"))
            article_id = _normalize_text(item.findtext("guid")) or None

            if not title or not link:
                continue

            articles.append(
                NewsArticle(
                    source=self._source_name,
                    title=title,
                    url=link,
                    summary=summary,
                    content=content,
                    published_at=published_at,
                    article_id=article_id,
                )
            )
        return articles

    def _parse_atom(self, root: ElementTree.Element) -> list[NewsArticle]:
        """Parse an Atom feed."""

        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}", 1)[0] + "}"

        articles: list[NewsArticle] = []
        for entry in root.findall(f"./{namespace}entry"):
            title = _normalize_text(entry.findtext(f"{namespace}title"))
            link_element = entry.find(f"{namespace}link")
            link = _normalize_text(link_element.get("href") if link_element is not None else "")
            summary = _normalize_text(entry.findtext(f"{namespace}summary"))
            content = _normalize_text(entry.findtext(f"{namespace}content"))
            published_at = _parse_timestamp(
                entry.findtext(f"{namespace}updated") or entry.findtext(f"{namespace}published")
            )
            article_id = _normalize_text(entry.findtext(f"{namespace}id")) or None

            if not title or not link:
                continue

            articles.append(
                NewsArticle(
                    source=self._source_name,
                    title=title,
                    url=link,
                    summary=summary,
                    content=content,
                    published_at=published_at,
                    article_id=article_id,
                )
            )
        return articles


class JSONNewsSource(BaseNewsSource):
    """Read and normalize articles from a local JSON endpoint or file."""

    def __init__(
        self,
        *,
        source_name: str,
        endpoint: str,
        timeout_seconds: float = 5.0,
        fetcher: ContentFetcher | None = None,
        articles_key: str = "articles",
    ) -> None:
        """Initialize the source with a JSON endpoint and response key."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if not articles_key:
            raise ValueError("articles_key must be a non-empty string")

        self._source_name = source_name.strip()
        self._endpoint = endpoint.strip()
        self._timeout_seconds = timeout_seconds
        self._fetcher = fetcher or _default_content_fetcher
        self._articles_key = articles_key

    @property
    def source_name(self) -> str:
        """Return the configured source identifier."""

        return self._source_name

    async def fetch_articles(self) -> list[NewsArticle]:
        """Fetch and parse articles from a JSON payload."""

        try:
            payload = self._fetcher(self._endpoint, self._timeout_seconds)
        except TimeoutError as exc:
            raise NewsTimeoutError(f"news source {self._source_name} exceeded {self._timeout_seconds:.2f}s") from exc
        except (OSError, URLError, NewsSourceError) as exc:
            raise NewsSourceError(f"unable to fetch JSON feed for {self._source_name}") from exc

        return self._parse_payload(payload)

    def _parse_payload(self, payload: bytes) -> list[NewsArticle]:
        """Parse a JSON article payload."""

        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NewsParsingError(f"invalid JSON feed for {self._source_name}") from exc

        if isinstance(decoded, list):
            records = decoded
        elif isinstance(decoded, dict):
            records = decoded.get(self._articles_key)
        else:
            raise NewsParsingError(f"unexpected JSON payload type for {self._source_name}")

        if not isinstance(records, list):
            raise NewsParsingError(f"payload for {self._source_name} does not contain a list of articles")

        articles: list[NewsArticle] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            article = NewsArticle.from_mapping(self._source_name, record)
            articles.append(article)

        return articles


class NewsDeduplicator:
    """Deduplicate normalized articles using a TTL-backed fingerprint cache."""

    def __init__(self, *, cache: TTLCache[str, bool] | None = None, ttl_seconds: float = 86_400.0) -> None:
        """Initialize the deduplicator with a cache and TTL."""

        self._cache = cache or TTLCache[str, bool](default_ttl_seconds=ttl_seconds)
        self._ttl_seconds = ttl_seconds

    def has_seen(self, article: NewsArticle) -> bool:
        """Return whether the article fingerprint is already present."""

        return self._cache.contains(article.fingerprint)

    def mark_seen(self, article: NewsArticle) -> None:
        """Mark an article as processed."""

        self._cache.set(article.fingerprint, True, ttl_seconds=self._ttl_seconds)

    def is_duplicate(self, article: NewsArticle) -> bool:
        """Return whether the article fingerprint has already been seen."""

        if self.has_seen(article):
            return True

        self.mark_seen(article)
        return False
