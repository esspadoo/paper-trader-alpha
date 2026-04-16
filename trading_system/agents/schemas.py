"""Strongly typed schemas for decision, critic, and news payloads."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ModuleNotFoundError:  # pragma: no cover - exercised on lean interpreters
    BaseModel = None
    ConfigDict = None
    Field = None
    field_validator = None
    model_validator = None


def _coerce_tuple_strings(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Return a deduplicated tuple of non-empty strings."""

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values or ():
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return tuple(normalized)


if BaseModel is not None:

    class _FrozenSchema(BaseModel):
        """Shared pydantic configuration for immutable agent payloads."""

        model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


    class NewsAnalysisSchema(_FrozenSchema):
        """Normalized structured news payload used across the decision stack."""

        sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
        impact: float = Field(default=0.0, ge=0.0, le=1.0)
        event_type: str = "other"
        summary: str = "No material news available."
        effective_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
        scope: Literal["company", "sector", "market"] = "market"
        scheduled_event: bool = False
        relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
        novelty_score: float = Field(default=1.0, ge=0.0, le=1.0)
        credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
        freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)
        catalyst_score: float = Field(default=0.0, ge=0.0, le=1.0)
        source: str = "unknown"
        symbols: tuple[str, ...] = ()
        primary_symbol: str | None = None
        rationale_codes: tuple[str, ...] = ()
        is_high_impact: bool = False

        @field_validator("event_type", "summary", "source")
        @classmethod
        def _validate_non_empty_strings(cls, value: str, info: Any) -> str:
            if not value.strip():
                raise ValueError(f"{info.field_name} must be a non-empty string")
            return value

        @field_validator("symbols", "rationale_codes", mode="before")
        @classmethod
        def _normalize_string_collections(cls, value: Any) -> tuple[str, ...]:
            return _coerce_tuple_strings(value)

        @field_validator("primary_symbol")
        @classmethod
        def _normalize_primary_symbol(cls, value: str | None) -> str | None:
            if value is None:
                return None
            normalized = value.strip().upper()
            return normalized or None

        @model_validator(mode="after")
        def _validate_primary_symbol(self) -> "NewsAnalysisSchema":
            if self.primary_symbol is not None and self.symbols and self.primary_symbol not in self.symbols:
                raise ValueError("primary_symbol must be present in symbols when both are provided")
            return self

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()


    class DecisionSchema(_FrozenSchema):
        """Typed decision output used by the fusion layer and downstream agents."""

        action: Literal["BUY", "SELL", "HOLD"]
        side: Literal["LONG", "SHORT", "FLAT"]
        score: float = Field(ge=-1.0, le=1.0)
        confidence: float = Field(ge=0.0, le=1.0)
        expected_edge: float
        holding_horizon_estimate: int = Field(ge=0)
        rationale_codes: tuple[str, ...] = ()
        blockers: tuple[str, ...] = ()
        supporting_evidence: dict[str, bool | int | float | str | None] = Field(default_factory=dict)
        size_multiplier: float = Field(default=0.0, ge=0.0, le=1.0)
        reasoning: str

        @field_validator("rationale_codes", "blockers", mode="before")
        @classmethod
        def _normalize_collections(cls, value: Any) -> tuple[str, ...]:
            return _coerce_tuple_strings(value)

        @field_validator("reasoning")
        @classmethod
        def _validate_reasoning(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("reasoning must be a non-empty string")
            return value

        @field_validator("expected_edge")
        @classmethod
        def _validate_expected_edge(cls, value: float) -> float:
            if not math.isfinite(float(value)):
                raise ValueError("expected_edge must be a finite number")
            return float(value)

        @field_validator("supporting_evidence")
        @classmethod
        def _validate_supporting_evidence(cls, value: dict[str, Any]) -> dict[str, Any]:
            normalized: dict[str, bool | int | float | str | None] = {}
            for key, candidate in dict(value).items():
                name = str(key).strip()
                if not name:
                    raise ValueError("supporting_evidence keys must be non-empty strings")
                if isinstance(candidate, float) and not math.isfinite(candidate):
                    raise ValueError(f"supporting_evidence value for {name} must be finite")
                normalized[name] = candidate
            return normalized

        @model_validator(mode="after")
        def _validate_action_side_pair(self) -> "DecisionSchema":
            expected_side = {"BUY": "LONG", "SELL": "SHORT", "HOLD": "FLAT"}[self.action]
            if self.side != expected_side:
                raise ValueError("side must match action")
            if self.action == "HOLD" and self.size_multiplier != 0.0:
                raise ValueError("size_multiplier must be zero for HOLD decisions")
            return self

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()


    class CriticReviewSchema(_FrozenSchema):
        """Typed critic review output."""

        approved: bool
        reason: str
        blocker_codes: tuple[str, ...] = ()
        risk_flags: tuple[str, ...] = ()

        @field_validator("reason")
        @classmethod
        def _validate_reason(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("reason must be a non-empty string")
            return value

        @field_validator("blocker_codes", "risk_flags", mode="before")
        @classmethod
        def _normalize_collections(cls, value: Any) -> tuple[str, ...]:
            return _coerce_tuple_strings(value)

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()

else:

    class _FrozenSchema:
        """Lean schema base used when `pydantic` is unavailable."""

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

        def model_copy(self, *, update: dict[str, Any] | None = None) -> Any:
            payload = self.model_dump()
            if update:
                payload.update(update)
            return type(self)(**payload)


    @dataclass(frozen=True, slots=True)
    class NewsAnalysisSchema(_FrozenSchema):
        """Normalized structured news payload used across the decision stack."""

        sentiment: float = 0.0
        impact: float = 0.0
        event_type: str = "other"
        summary: str = "No material news available."
        effective_sentiment: float = 0.0
        scope: Literal["company", "sector", "market"] = "market"
        scheduled_event: bool = False
        relevance_score: float = 0.0
        novelty_score: float = 1.0
        credibility_score: float = 0.5
        freshness_score: float = 1.0
        catalyst_score: float = 0.0
        source: str = "unknown"
        symbols: tuple[str, ...] = ()
        primary_symbol: str | None = None
        rationale_codes: tuple[str, ...] = ()
        is_high_impact: bool = False

        def __post_init__(self) -> None:
            object.__setattr__(self, "symbols", _coerce_tuple_strings(self.symbols))
            object.__setattr__(self, "rationale_codes", _coerce_tuple_strings(self.rationale_codes))
            if self.primary_symbol is not None:
                primary_symbol = self.primary_symbol.strip().upper()
                object.__setattr__(self, "primary_symbol", primary_symbol or None)
            self._validate_unit("sentiment", self.sentiment, -1.0, 1.0)
            self._validate_unit("impact", self.impact, 0.0, 1.0)
            self._validate_unit("effective_sentiment", self.effective_sentiment, -1.0, 1.0)
            self._validate_unit("relevance_score", self.relevance_score, 0.0, 1.0)
            self._validate_unit("novelty_score", self.novelty_score, 0.0, 1.0)
            self._validate_unit("credibility_score", self.credibility_score, 0.0, 1.0)
            self._validate_unit("freshness_score", self.freshness_score, 0.0, 1.0)
            self._validate_unit("catalyst_score", self.catalyst_score, 0.0, 1.0)
            if self.scope not in {"company", "sector", "market"}:
                raise ValueError("scope must be company, sector, or market")
            for field_name in ("event_type", "summary", "source"):
                if not str(getattr(self, field_name)).strip():
                    raise ValueError(f"{field_name} must be a non-empty string")
            if self.primary_symbol is not None and self.symbols and self.primary_symbol not in self.symbols:
                raise ValueError("primary_symbol must be present in symbols when both are provided")

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()

        @staticmethod
        def _validate_unit(field_name: str, value: float, minimum: float, maximum: float) -> None:
            if not math.isfinite(float(value)) or float(value) < minimum or float(value) > maximum:
                raise ValueError(f"{field_name} must be between {minimum} and {maximum}")


    @dataclass(frozen=True, slots=True)
    class DecisionSchema(_FrozenSchema):
        """Typed decision output used by the fusion layer and downstream agents."""

        action: Literal["BUY", "SELL", "HOLD"]
        side: Literal["LONG", "SHORT", "FLAT"]
        score: float = 0.0
        confidence: float = 0.0
        expected_edge: float = 0.0
        holding_horizon_estimate: int = 0
        rationale_codes: tuple[str, ...] = ()
        blockers: tuple[str, ...] = ()
        supporting_evidence: dict[str, bool | int | float | str | None] = field(default_factory=dict)
        size_multiplier: float = 0.0
        reasoning: str = ""

        def __post_init__(self) -> None:
            object.__setattr__(self, "rationale_codes", _coerce_tuple_strings(self.rationale_codes))
            object.__setattr__(self, "blockers", _coerce_tuple_strings(self.blockers))
            if self.action not in {"BUY", "SELL", "HOLD"}:
                raise ValueError("action must be BUY, SELL, or HOLD")
            if self.side not in {"LONG", "SHORT", "FLAT"}:
                raise ValueError("side must be LONG, SHORT, or FLAT")
            expected_side = {"BUY": "LONG", "SELL": "SHORT", "HOLD": "FLAT"}[self.action]
            if self.side != expected_side:
                raise ValueError("side must match action")
            if not math.isfinite(float(self.score)) or not -1.0 <= float(self.score) <= 1.0:
                raise ValueError("score must be between -1 and 1")
            if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            if not math.isfinite(float(self.expected_edge)):
                raise ValueError("expected_edge must be finite")
            if self.holding_horizon_estimate < 0:
                raise ValueError("holding_horizon_estimate must be greater than or equal to 0")
            if not math.isfinite(float(self.size_multiplier)) or not 0.0 <= float(self.size_multiplier) <= 1.0:
                raise ValueError("size_multiplier must be between 0 and 1")
            if self.action == "HOLD" and self.size_multiplier != 0.0:
                raise ValueError("size_multiplier must be zero for HOLD decisions")
            if not self.reasoning.strip():
                raise ValueError("reasoning must be a non-empty string")
            normalized_evidence: dict[str, bool | int | float | str | None] = {}
            for key, candidate in dict(self.supporting_evidence).items():
                name = str(key).strip()
                if not name:
                    raise ValueError("supporting_evidence keys must be non-empty strings")
                if isinstance(candidate, float) and not math.isfinite(candidate):
                    raise ValueError(f"supporting_evidence value for {name} must be finite")
                normalized_evidence[name] = candidate
            object.__setattr__(self, "supporting_evidence", normalized_evidence)

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()


    @dataclass(frozen=True, slots=True)
    class CriticReviewSchema(_FrozenSchema):
        """Typed critic review output."""

        approved: bool
        reason: str
        blocker_codes: tuple[str, ...] = ()
        risk_flags: tuple[str, ...] = ()

        def __post_init__(self) -> None:
            if not self.reason.strip():
                raise ValueError("reason must be a non-empty string")
            object.__setattr__(self, "blocker_codes", _coerce_tuple_strings(self.blocker_codes))
            object.__setattr__(self, "risk_flags", _coerce_tuple_strings(self.risk_flags))

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()
