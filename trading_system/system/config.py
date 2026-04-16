"""Validated configuration models and TOML loading for the integrated trading system."""

from __future__ import annotations

import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class DeploymentMode(str, Enum):
    """Supported deployment modes for the integrated runtime."""

    DEV = "dev"
    PAPER = "paper"
    LIVE = "live"


class _BaseConfigModel(BaseModel):
    """Shared pydantic configuration for system config models."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class LoggingConfig(_BaseConfigModel):
    """Logging configuration for the integrated runtime."""

    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)s %(name)s %(message)s"

    @field_validator("level")
    @classmethod
    def _normalize_level(cls, value: str) -> str:
        """Validate and normalize the configured log level."""

        normalized = value.strip().upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("level must be one of CRITICAL, ERROR, WARNING, INFO, or DEBUG")
        return normalized

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        """Require a non-empty log format string."""

        if not value.strip():
            raise ValueError("format must be a non-empty string")
        return value


class MarketConfig(_BaseConfigModel):
    """Synthetic market-feed and model-training configuration."""

    symbol: str = "AAPL"
    start_timestamp: str = "2026-04-14T13:30:00+00:00"
    training_bars: int = 180
    stream_bars: int = 24
    bar_interval_minutes: int = 5
    base_price: float = 100.0
    trend_per_bar: float = 0.08
    amplitude: float = 1.2
    base_volume: float = 1_250_000.0
    validation_fraction: float = 0.2
    min_training_rows: int = 60
    n_estimators: int = 96
    learning_rate: float = 0.05
    max_depth: int = 4
    prediction_horizon_bars: int = 2

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        """Require a non-empty ticker symbol."""

        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol must be a non-empty string")
        return normalized

    @field_validator("training_bars", "stream_bars", "bar_interval_minutes", "min_training_rows", "n_estimators", "max_depth", "prediction_horizon_bars")
    @classmethod
    def _validate_positive_ints(cls, value: int, info: Any) -> int:
        """Require positive integer market configuration values."""

        if value <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return value

    @field_validator("base_price", "base_volume")
    @classmethod
    def _validate_positive_floats(cls, value: float, info: Any) -> float:
        """Require positive float market configuration values."""

        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return float(value)

    @field_validator("validation_fraction")
    @classmethod
    def _validate_validation_fraction(cls, value: float) -> float:
        """Require a sane time-split validation fraction."""

        if not 0.0 < value < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        return float(value)

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        """Require a positive learning rate."""

        if value <= 0.0:
            raise ValueError("learning_rate must be greater than 0")
        return float(value)


class NewsConfig(_BaseConfigModel):
    """News-source and local-LLM backend configuration."""

    source_type: Literal["json", "rss"] = "json"
    source_path: str = "examples/demo_news.json"
    backend: Literal["demo", "vllm", "llama_cpp"] = "demo"
    model: str = "local-demo"
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 5.0
    max_retries: int = 2
    max_article_age_seconds: float = 14_400.0

    @field_validator("source_path", "model", "base_url")
    @classmethod
    def _validate_non_empty_strings(cls, value: str, info: Any) -> str:
        """Require non-empty strings for news configuration values."""

        if not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value

    @field_validator("timeout_seconds", "max_article_age_seconds")
    @classmethod
    def _validate_positive_timeout(cls, value: float, info: Any) -> float:
        """Require positive timeout and recency windows."""

        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return float(value)

    @field_validator("max_retries")
    @classmethod
    def _validate_retries(cls, value: int) -> int:
        """Require at least one retry attempt."""

        if value < 1:
            raise ValueError("max_retries must be at least 1")
        return value


class DecisionConfig(_BaseConfigModel):
    """Decision-agent configuration."""

    action_threshold: float = 0.05

    @field_validator("action_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        """Require a bounded decision threshold."""

        if value < 0.0 or value > 1.0:
            raise ValueError("action_threshold must be between 0 and 1")
        return float(value)


class CriticConfig(_BaseConfigModel):
    """Critic-agent policy configuration."""

    max_volatility: float = 0.08
    min_confidence: float = 0.10
    conflict_threshold: float = 0.75

    @field_validator("max_volatility")
    @classmethod
    def _validate_max_volatility(cls, value: float) -> float:
        """Require a non-negative volatility threshold."""

        if value < 0.0:
            raise ValueError("max_volatility must be greater than or equal to 0")
        return float(value)

    @field_validator("min_confidence", "conflict_threshold")
    @classmethod
    def _validate_unit_interval(cls, value: float, info: Any) -> float:
        """Require bounded critic thresholds."""

        if value < 0.0 or value > 1.0:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return float(value)


class RiskConfig(_BaseConfigModel):
    """Risk-agent policy configuration."""

    max_risk_per_trade_fraction: float = 0.01
    atr_period: int = 14
    atr_multiplier: float = 2.0
    max_daily_loss_fraction: float = 0.03
    max_exposure_fraction: float = 0.50

    @field_validator("max_risk_per_trade_fraction", "max_daily_loss_fraction", "max_exposure_fraction")
    @classmethod
    def _validate_fraction(cls, value: float, info: Any) -> float:
        """Require bounded risk fractions."""

        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return float(value)

    @field_validator("atr_period")
    @classmethod
    def _validate_atr_period(cls, value: int) -> int:
        """Require a meaningful ATR lookback."""

        if value < 2:
            raise ValueError("atr_period must be at least 2")
        return value

    @field_validator("atr_multiplier")
    @classmethod
    def _validate_atr_multiplier(cls, value: float) -> float:
        """Require a positive ATR multiple."""

        if value <= 0.0:
            raise ValueError("atr_multiplier must be greater than 0")
        return float(value)


class ExecutionConfig(_BaseConfigModel):
    """Execution and broker routing configuration."""

    broker: Literal["paper", "ibkr"] = "paper"
    initial_capital: float = 100_000.0
    stale_order_seconds: float = 30.0
    exchange: str = "SMART"
    currency: str = "USD"
    tif: str = "DAY"
    smart_routing: bool = True
    outside_rth: bool = False
    host: str = "127.0.0.1"
    port: int | None = None
    client_id: int = 11
    account: str | None = None
    paper_trading: bool = True
    fill_latency_seconds: float = 0.0
    read_only: bool = False
    connection_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    allow_live_trading: bool = False
    confirm_live_account: str | None = None
    journal_path: str = "state/orders.json"

    @field_validator("initial_capital", "stale_order_seconds", "connection_timeout_seconds", "request_timeout_seconds")
    @classmethod
    def _validate_positive_execution_floats(cls, value: float, info: Any) -> float:
        """Require positive execution timeouts and capital values."""

        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return float(value)

    @field_validator("fill_latency_seconds", "retry_backoff_seconds")
    @classmethod
    def _validate_non_negative_execution_floats(cls, value: float, info: Any) -> float:
        """Require non-negative latency and retry backoff values."""

        if value < 0.0:
            raise ValueError(f"{info.field_name} must be greater than or equal to 0")
        return float(value)

    @field_validator("exchange", "currency", "tif", "host", "journal_path")
    @classmethod
    def _validate_execution_strings(cls, value: str, info: Any) -> str:
        """Require non-empty execution strings."""

        if not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value

    @field_validator("client_id")
    @classmethod
    def _validate_client_id(cls, value: int) -> int:
        """Require a non-negative IB client id."""

        if value < 0:
            raise ValueError("client_id must be greater than or equal to 0")
        return value

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int | None) -> int | None:
        """Require a positive explicit port when provided."""

        if value is not None and value <= 0:
            raise ValueError("port must be greater than 0 when provided")
        return value

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, value: int) -> int:
        """Require at least one retry attempt."""

        if value < 1:
            raise ValueError("max_retries must be at least 1")
        return value

    @field_validator("account", "confirm_live_account")
    @classmethod
    def _normalize_optional_account(cls, value: str | None) -> str | None:
        """Normalize optional account identifiers."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class RuntimeConfig(_BaseConfigModel):
    """Top-level runtime behavior configuration."""

    strategy_id: str = "integrated-intraday"
    mode: DeploymentMode = DeploymentMode.PAPER
    publish_delay_seconds: float = 0.0
    allow_demo_components: bool = True
    reconcile_on_start: bool = True
    max_log_reason_length: int = 160

    @field_validator("strategy_id")
    @classmethod
    def _validate_strategy_id(cls, value: str) -> str:
        """Require a non-empty strategy identifier."""

        if not value.strip():
            raise ValueError("strategy_id must be a non-empty string")
        return value

    @field_validator("publish_delay_seconds")
    @classmethod
    def _validate_publish_delay(cls, value: float) -> float:
        """Require a non-negative publish delay."""

        if value < 0.0:
            raise ValueError("publish_delay_seconds must be greater than or equal to 0")
        return float(value)

    @field_validator("max_log_reason_length")
    @classmethod
    def _validate_reason_length(cls, value: int) -> int:
        """Require a practical positive log truncation limit."""

        if value < 32:
            raise ValueError("max_log_reason_length must be at least 32")
        return value


class SafetyConfig(_BaseConfigModel):
    """Hard safety controls for live and paper runtime behavior."""

    kill_switch_enabled: bool = True
    cancel_open_orders_on_kill_switch: bool = True
    fail_closed_on_startup_reconciliation: bool = True
    startup_position_reconciliation: bool = True
    startup_position_quantity_tolerance: float = 1e-6
    max_daily_loss_shutdown_fraction: float = 0.05
    dry_run: bool = False

    @field_validator("startup_position_quantity_tolerance")
    @classmethod
    def _validate_startup_tolerance(cls, value: float) -> float:
        """Require a non-negative reconciliation tolerance."""

        if value < 0.0:
            raise ValueError("startup_position_quantity_tolerance must be greater than or equal to 0")
        return float(value)

    @field_validator("max_daily_loss_shutdown_fraction")
    @classmethod
    def _validate_shutdown_fraction(cls, value: float) -> float:
        """Require a bounded daily-loss shutdown fraction."""

        if value <= 0.0 or value > 1.0:
            raise ValueError("max_daily_loss_shutdown_fraction must be between 0 and 1")
        return float(value)


class PersistenceConfig(_BaseConfigModel):
    """Persistence and journaling configuration."""

    state_store_path: str = "state/runtime_state.db"
    audit_journal_path: str = "state/events.jsonl"
    dead_letter_path: str = "state/dead_letters.jsonl"
    alert_journal_path: str = "state/alerts.jsonl"
    snapshot_interval_seconds: float = 10.0
    replay_history_bars: int = 64
    enable_event_audit: bool = True
    enable_dead_letter_journal: bool = True
    enable_alert_journal: bool = True

    @field_validator("state_store_path", "audit_journal_path", "dead_letter_path", "alert_journal_path")
    @classmethod
    def _validate_paths(cls, value: str, info: Any) -> str:
        """Require non-empty persistence paths."""

        if not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value

    @field_validator("snapshot_interval_seconds")
    @classmethod
    def _validate_snapshot_interval(cls, value: float) -> float:
        """Require a positive snapshot interval."""

        if value <= 0.0:
            raise ValueError("snapshot_interval_seconds must be greater than 0")
        return float(value)

    @field_validator("replay_history_bars")
    @classmethod
    def _validate_replay_history_bars(cls, value: int) -> int:
        """Require a useful replay history window."""

        if value < 16:
            raise ValueError("replay_history_bars must be at least 16")
        return value


class ObservabilityConfig(_BaseConfigModel):
    """Operational heartbeat, maintenance, and latency-observability settings."""

    heartbeat_interval_seconds: float = 5.0
    maintenance_interval_seconds: float = 5.0
    kill_switch_poll_seconds: float = 2.0
    latency_alert_threshold_ms: float = 100.0

    @field_validator("heartbeat_interval_seconds", "maintenance_interval_seconds", "kill_switch_poll_seconds")
    @classmethod
    def _validate_positive_intervals(cls, value: float, info: Any) -> float:
        """Require positive operational intervals."""

        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return float(value)

    @field_validator("latency_alert_threshold_ms")
    @classmethod
    def _validate_latency_alert_threshold(cls, value: float) -> float:
        """Require a positive alert threshold."""

        if value <= 0.0:
            raise ValueError("latency_alert_threshold_ms must be greater than 0")
        return float(value)


class SystemConfig(_BaseConfigModel):
    """Complete application configuration."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @model_validator(mode="after")
    def _validate_cross_section_constraints(self) -> "SystemConfig":
        """Fail closed for unsafe deployment combinations."""

        if self.execution.broker == "paper" and not self.execution.paper_trading:
            raise ValueError("paper broker requires execution.paper_trading = true")

        if self.execution.allow_live_trading and self.runtime.mode is not DeploymentMode.LIVE:
            raise ValueError("execution.allow_live_trading can only be enabled when runtime.mode = 'live'")

        if not self.runtime.allow_demo_components and self.news.backend == "demo":
            raise ValueError("demo news backend is disabled when runtime.allow_demo_components = false")

        if self.safety.dry_run and self.runtime.mode is DeploymentMode.LIVE:
            raise ValueError("runtime.mode = 'live' cannot be combined with safety.dry_run = true")

        if self.runtime.mode is DeploymentMode.LIVE:
            if self.execution.broker != "ibkr":
                raise ValueError("live mode requires execution.broker = 'ibkr'")
            if self.execution.paper_trading:
                raise ValueError("live mode requires execution.paper_trading = false")
            if not self.execution.allow_live_trading:
                raise ValueError("live mode requires execution.allow_live_trading = true")
            if self.execution.account is None:
                raise ValueError("live mode requires execution.account to be configured")
            if self.execution.confirm_live_account != self.execution.account:
                raise ValueError("live mode requires execution.confirm_live_account to exactly match execution.account")
            if self.runtime.allow_demo_components:
                raise ValueError("live mode requires runtime.allow_demo_components = false")
            if self.news.backend == "demo":
                raise ValueError("live mode cannot use the demo local-LLM backend")
            if self.news.source_type == "json":
                raise ValueError("live mode cannot use a static json news source")
            if self.execution.read_only:
                raise ValueError("live mode requires execution.read_only = false")
        return self


def load_system_config(path: str | Path) -> SystemConfig:
    """Load the integrated trading-system configuration from a TOML file."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    merged = _apply_environment_overrides(payload)
    try:
        return SystemConfig.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(f"invalid trading-system configuration in {config_path}: {exc}") from exc


def _apply_environment_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay well-scoped environment variables onto the parsed TOML payload."""

    merged = dict(payload)
    for section in (
        "logging",
        "market",
        "news",
        "decision",
        "critic",
        "risk",
        "execution",
        "runtime",
        "safety",
        "persistence",
        "observability",
    ):
        existing = merged.get(section)
        if existing is None:
            merged[section] = {}
        elif not isinstance(existing, dict):
            raise ValueError(f"configuration section '{section}' must be a mapping")
        else:
            merged[section] = dict(existing)

    env_map: dict[str, tuple[str, str, str]] = {
        "TRADING_SYSTEM_RUNTIME_MODE": ("runtime", "mode", "str"),
        "TRADING_SYSTEM_ALLOW_DEMO_COMPONENTS": ("runtime", "allow_demo_components", "bool"),
        "TRADING_SYSTEM_EXECUTION_BROKER": ("execution", "broker", "str"),
        "TRADING_SYSTEM_EXECUTION_ACCOUNT": ("execution", "account", "str"),
        "TRADING_SYSTEM_ALLOW_LIVE_TRADING": ("execution", "allow_live_trading", "bool"),
        "TRADING_SYSTEM_CONFIRM_LIVE_ACCOUNT": ("execution", "confirm_live_account", "str"),
        "TRADING_SYSTEM_NEWS_BACKEND": ("news", "backend", "str"),
        "TRADING_SYSTEM_NEWS_BASE_URL": ("news", "base_url", "str"),
        "TRADING_SYSTEM_DRY_RUN": ("safety", "dry_run", "bool"),
        "TRADING_SYSTEM_STATE_STORE_PATH": ("persistence", "state_store_path", "str"),
    }

    for env_name, (section, field_name, value_type) in env_map.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        if value_type == "bool":
            merged[section][field_name] = _parse_bool(raw_value, env_name)
        else:
            merged[section][field_name] = raw_value

    return merged


def _parse_bool(value: str, env_name: str) -> bool:
    """Parse a boolean environment variable."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"environment variable {env_name} must be a boolean-like value")
