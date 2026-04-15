"""Local-LLM news analysis with strict JSON output and retries."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from trading_system.data.news import NewsArticle
from trading_system.models.exceptions import LLMBackendError, LLMTimeoutError, StrictJSONParseError


def _normalize_base_url(value: str) -> str:
    """Return a base URL without a trailing slash."""

    return value.rstrip("/")


def _http_post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    """Send a JSON POST request and return the decoded JSON response."""

    request = Request(
        url=url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise LLMTimeoutError(f"local LLM request to {url} exceeded {timeout_seconds:.2f}s") from exc
    except URLError as exc:
        raise LLMBackendError(f"local LLM request to {url} failed") from exc

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMBackendError(f"local LLM backend at {url} returned invalid JSON") from exc

    if not isinstance(decoded, dict):
        raise LLMBackendError(f"local LLM backend at {url} returned an unexpected payload type")
    return decoded


@dataclass(frozen=True, slots=True)
class NewsAnalysis:
    """Normalized structured output for a single article."""

    sentiment: float
    impact: float
    event_type: str
    summary: str

    def __post_init__(self) -> None:
        """Validate structured-news output."""

        if not math.isfinite(self.sentiment) or not -1.0 <= self.sentiment <= 1.0:
            raise ValueError("sentiment must be a finite float between -1 and 1")
        if not math.isfinite(self.impact) or not 0.0 <= self.impact <= 1.0:
            raise ValueError("impact must be a finite float between 0 and 1")
        if not self.event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not self.summary.strip():
            raise ValueError("summary must be a non-empty string")

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serializable analysis payload."""

        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class NewsPromptTemplate:
    """Prompt template for deterministic financial-news classification."""

    max_summary_words: int = 40

    @property
    def system_message(self) -> str:
        """Return the system instruction for the local model."""

        return (
            "You are a financial news analysis engine. "
            "Return ONLY one JSON object with exactly these keys: "
            "sentiment, impact, event_type, summary. "
            "sentiment must be a float in [-1, 1]. "
            "impact must be a float in [0, 1]. "
            "event_type must be a lowercase snake_case string. "
            f"summary must be a concise plain-English string under {self.max_summary_words} words. "
            "Do not include markdown, prose, explanations, or extra keys."
        )

    def render_messages(self, article: NewsArticle, *, retry_reason: str | None = None) -> list[dict[str, str]]:
        """Render chat-style messages for OpenAI-compatible local backends."""

        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.render_user_prompt(article, retry_reason=retry_reason)},
        ]

    def render_completion_prompt(self, article: NewsArticle, *, retry_reason: str | None = None) -> str:
        """Render a single prompt for completion-style local backends."""

        return (
            f"System:\n{self.system_message}\n\n"
            f"User:\n{self.render_user_prompt(article, retry_reason=retry_reason)}\n\n"
            "Assistant:\n"
        )

    def render_user_prompt(self, article: NewsArticle, *, retry_reason: str | None = None) -> str:
        """Render the article payload and explicit JSON schema."""

        context = article.to_prompt_context()
        retry_note = (
            ""
            if retry_reason is None
            else (
                "The previous response was invalid. "
                f"Reason: {retry_reason}. "
                "Return ONLY a valid JSON object.\n\n"
            )
        )
        return (
            f"{retry_note}"
            "Analyze the following news article for market relevance.\n"
            "Classify the dominant event type using one of these families when possible: "
            "earnings, guidance, merger_acquisition, regulation, litigation, analyst_rating, "
            "product_launch, macro, management_change, financing, operations, partnership, other.\n\n"
            f"Source: {context['source']}\n"
            f"Title: {context['title']}\n"
            f"URL: {context['url']}\n"
            f"Published At: {context['published_at']}\n"
            f"Symbols: {', '.join(context['symbols']) if context['symbols'] else 'None'}\n"
            f"Summary: {context['summary']}\n"
            f"Content: {context['content']}\n\n"
            'Return this exact schema:\n'
            '{"sentiment": 0.0, "impact": 0.0, "event_type": "other", "summary": "..." }\n'
        )


class LocalLLMBackend(ABC):
    """Abstract interface for local LLM runtimes."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier."""

    async def warmup(self) -> None:
        """Optionally warm backend resources before the first request."""

    @abstractmethod
    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        timeout_seconds: float,
    ) -> str:
        """Generate text from a prompt under a timeout budget."""


class VLLMBackend(LocalLLMBackend):
    """OpenAI-compatible local backend for vLLM servers."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:8000",
        max_tokens: int = 256,
        seed: int = 42,
    ) -> None:
        """Configure the local vLLM endpoint."""

        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        self._model = model.strip()
        self._base_url = _normalize_base_url(base_url)
        self._max_tokens = max_tokens
        self._seed = seed

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""

        return "vllm"

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        timeout_seconds: float,
    ) -> str:
        """Call the local vLLM chat-completions endpoint."""

        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        decoded = _http_post_json(url, payload, timeout_seconds)

        try:
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMBackendError("vLLM response did not contain chat completion content") from exc
        if not isinstance(content, str):
            raise LLMBackendError("vLLM response content must be a string")
        return content


class LlamaCppBackend(LocalLLMBackend):
    """Local backend for llama.cpp HTTP servers."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        n_predict: int = 256,
        seed: int = 42,
    ) -> None:
        """Configure the local llama.cpp endpoint."""

        if n_predict <= 0:
            raise ValueError("n_predict must be greater than 0")

        self._base_url = _normalize_base_url(base_url)
        self._n_predict = n_predict
        self._seed = seed

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""

        return "llama_cpp"

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        timeout_seconds: float,
    ) -> str:
        """Call the local llama.cpp completion endpoint."""

        url = f"{self._base_url}/completion"
        payload = {
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "cache_prompt": True,
            "n_predict": self._n_predict,
            "seed": self._seed,
        }
        decoded = _http_post_json(url, payload, timeout_seconds)

        content = decoded.get("content")
        if not isinstance(content, str):
            raise LLMBackendError("llama.cpp response did not contain completion content")
        return content


class LocalNewsLLMAnalyzer:
    """Analyze articles with a local LLM and strict JSON validation."""

    def __init__(
        self,
        backend: LocalLLMBackend,
        *,
        prompt_template: NewsPromptTemplate | None = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the analyzer with a backend and retry policy."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self._backend = backend
        self._prompt_template = prompt_template or NewsPromptTemplate()
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    @property
    def backend_name(self) -> str:
        """Return the underlying backend identifier."""

        return self._backend.backend_name

    async def warmup(self) -> None:
        """Warm the local backend if supported."""

        await self._backend.warmup()

    async def analyze(self, article: NewsArticle) -> NewsAnalysis:
        """Analyze a news article and return strict structured output."""

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            retry_reason = None if last_error is None else str(last_error)
            try:
                response = await self._backend.generate(
                    messages=self._prompt_template.render_messages(article, retry_reason=retry_reason),
                    prompt=self._prompt_template.render_completion_prompt(article, retry_reason=retry_reason),
                    timeout_seconds=self._timeout_seconds,
                )
                return self.parse_strict_json(response)
            except (LLMBackendError, LLMTimeoutError, StrictJSONParseError) as exc:
                last_error = exc

        raise LLMBackendError(
            f"local news analysis failed after {self._max_retries} attempt(s): {last_error}"
        ) from last_error

    def parse_strict_json(self, raw_response: str) -> NewsAnalysis:
        """Parse and validate a strict JSON response from the local model."""

        payload = raw_response.strip()
        if not payload.startswith("{") or not payload.endswith("}"):
            raise StrictJSONParseError("response must be a single JSON object with no surrounding text")

        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StrictJSONParseError("response is not valid JSON") from exc

        if not isinstance(decoded, dict):
            raise StrictJSONParseError("response must decode to a JSON object")

        expected_keys = {"sentiment", "impact", "event_type", "summary"}
        if set(decoded.keys()) != expected_keys:
            raise StrictJSONParseError(f"response keys must be exactly {sorted(expected_keys)}")

        sentiment = decoded["sentiment"]
        impact = decoded["impact"]
        event_type = decoded["event_type"]
        summary = decoded["summary"]

        if not isinstance(sentiment, (int, float)) or not math.isfinite(float(sentiment)):
            raise StrictJSONParseError("sentiment must be a finite number")
        if not isinstance(impact, (int, float)) or not math.isfinite(float(impact)):
            raise StrictJSONParseError("impact must be a finite number")
        if not isinstance(event_type, str) or not event_type.strip():
            raise StrictJSONParseError("event_type must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise StrictJSONParseError("summary must be a non-empty string")

        event_type_value = event_type.strip()
        if any(character.isupper() for character in event_type_value) or " " in event_type_value or "-" in event_type_value:
            raise StrictJSONParseError("event_type must be lowercase snake_case")

        try:
            return NewsAnalysis(
                sentiment=float(sentiment),
                impact=float(impact),
                event_type=event_type_value,
                summary=summary.strip(),
            )
        except ValueError as exc:
            raise StrictJSONParseError(str(exc)) from exc
