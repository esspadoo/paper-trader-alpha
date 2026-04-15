"""Deterministic risk agent for sizing and hard trade guardrails."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from trading_system.agents._validation import normalize_decision_output, normalize_market_signal
from trading_system.agents.base import BaseAgent
from trading_system.core.events import BaseEvent, MarketEvent, SignalEvent
from trading_system.models.dependencies import require_pandas

if TYPE_CHECKING:
    import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    """Deterministic limits used by the risk agent."""

    max_risk_per_trade_fraction: float = 0.01
    atr_period: int = 14
    atr_multiplier: float = 2.0
    max_daily_loss_fraction: float = 0.03
    max_exposure_fraction: float = 0.25

    def __post_init__(self) -> None:
        """Validate the configured guardrails."""

        if self.max_risk_per_trade_fraction <= 0.0 or self.max_risk_per_trade_fraction > 1.0:
            raise ValueError("max_risk_per_trade_fraction must be between 0 and 1")
        if self.atr_period < 2:
            raise ValueError("atr_period must be at least 2")
        if self.atr_multiplier <= 0.0:
            raise ValueError("atr_multiplier must be greater than 0")
        if self.max_daily_loss_fraction < 0.0 or self.max_daily_loss_fraction > 1.0:
            raise ValueError("max_daily_loss_fraction must be between 0 and 1")
        if self.max_exposure_fraction <= 0.0 or self.max_exposure_fraction > 1.0:
            raise ValueError("max_exposure_fraction must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Structured output from the risk layer."""

    final_action: str
    position_size: float
    stop_loss: float
    reason: str

    def __post_init__(self) -> None:
        """Validate the risk output."""

        if self.final_action not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("final_action must be one of BUY, SELL, HOLD")
        if not math.isfinite(self.position_size) or self.position_size < 0.0:
            raise ValueError("position_size must be a finite number greater than or equal to 0")
        if not math.isfinite(self.stop_loss) or self.stop_loss < 0.0:
            raise ValueError("stop_loss must be a finite number greater than or equal to 0")
        if not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, float | str]:
        """Return the user-facing risk payload."""

        return {
            "final_action": self.final_action,
            "position_size": float(self.position_size),
            "stop_loss": float(self.stop_loss),
        }


class RiskAgent(BaseAgent):
    """Apply deterministic sizing, stop-loss, and portfolio guardrails."""

    def __init__(
        self,
        *,
        agent_id: str = "risk-agent",
        policy: RiskPolicy | None = None,
    ) -> None:
        """Initialize the risk agent and its retained state."""

        self._agent_id = agent_id
        self._policy = policy or RiskPolicy()
        self._running = False
        self._latest_decision: dict[str, float | str] | None = None
        self._latest_market_signal: dict[str, Any] | None = None
        self._latest_account_state: dict[str, float] | None = None
        self._latest_symbol: str | None = None
        self._latest_entry_price: float | None = None
        self._latest_ohlcv: Any = None
        self._latest_bar: Mapping[str, Any] | None = None
        self._latest_timestamp: Any | None = None
        self._last_output: dict[str, float | str] | None = None
        self._last_reason: str | None = None
        self._atr_states: dict[str, _IncrementalATRState] = {}

    @property
    def agent_id(self) -> str:
        """Return the stable identifier for this agent."""

        return self._agent_id

    @property
    def subscribed_topics(self) -> Sequence[str]:
        """Return the event topics this agent can consume."""

        return ("signal", "market")

    async def start(self) -> None:
        """Mark the risk agent as active."""

        self._running = True

    async def stop(self) -> None:
        """Mark the risk agent as inactive."""

        self._running = False

    async def on_event(self, event: BaseEvent) -> None:
        """Consume decision, market-signal, and market-context events."""

        if isinstance(event, SignalEvent):
            payload = dict(event.payload or {})
            if {"action", "score", "reasoning"} <= set(payload):
                self._latest_decision = normalize_decision_output(payload)
            elif "signal" in payload:
                payload.setdefault("confidence", event.confidence)
                if event.symbol:
                    payload.setdefault("symbol", event.symbol)
                self._latest_market_signal = normalize_market_signal(payload, default_confidence=event.confidence)
            else:
                return

            if event.symbol:
                self._latest_symbol = event.symbol.strip().upper()
        elif isinstance(event, MarketEvent):
            payload = dict(event.payload or {})
            if "ohlcv" in payload:
                self._latest_ohlcv = payload["ohlcv"]
            if isinstance(payload.get("bar"), Mapping):
                self._latest_bar = payload["bar"]
                self._latest_timestamp = event.occurred_at
            if "account_state" in payload and isinstance(payload["account_state"], Mapping):
                self._latest_account_state = self._normalize_account_state(payload["account_state"])
            if "entry_price" in payload:
                self._latest_entry_price = self._coerce_positive_float(payload["entry_price"], field_name="entry_price")
            elif event.last_price is not None:
                self._latest_entry_price = self._coerce_positive_float(event.last_price, field_name="last_price")
            if event.symbol:
                self._latest_symbol = event.symbol.strip().upper()
        else:
            return

        if self._can_assess():
            assessment = await self.assess()
            self._last_output = assessment.copy()

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of the risk layer state."""

        return {
            "agent_id": self._agent_id,
            "running": self._running,
            "policy": dict(asdict(self._policy)),
            "last_decision": self._latest_decision,
            "last_market_signal": self._latest_market_signal,
            "last_account_state": self._latest_account_state,
            "last_symbol": self._latest_symbol,
            "last_entry_price": self._latest_entry_price,
            "last_timestamp": self._latest_timestamp,
            "last_output": self._last_output,
            "last_reason": self._last_reason,
        }

    async def assess(
        self,
        decision: Mapping[str, Any] | None = None,
        *,
        market_signal: Mapping[str, Any] | None = None,
        account_state: Mapping[str, Any] | None = None,
        ohlcv: "pd.DataFrame | None" = None,
        symbol: str | None = None,
        entry_price: float | None = None,
        timestamp: Any | None = None,
        bar: Mapping[str, Any] | None = None,
    ) -> dict[str, float | str]:
        """Return a deterministic post-risk trade decision."""

        if decision is not None:
            self._latest_decision = normalize_decision_output(decision)
        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
            maybe_symbol = self._latest_market_signal.get("symbol")
            if isinstance(maybe_symbol, str) and maybe_symbol:
                self._latest_symbol = maybe_symbol
        if account_state is not None:
            self._latest_account_state = self._normalize_account_state(account_state)
        if ohlcv is not None:
            self._latest_ohlcv = ohlcv
        if symbol is not None:
            normalized_symbol = symbol.strip().upper()
            if not normalized_symbol:
                raise ValueError("symbol must be a non-empty string when provided")
            self._latest_symbol = normalized_symbol
        if entry_price is not None:
            self._latest_entry_price = self._coerce_positive_float(entry_price, field_name="entry_price")
        if timestamp is not None:
            self._latest_timestamp = timestamp
        if bar is not None:
            self._latest_bar = dict(bar)

        assessment = self._assess_current_state()
        self._last_output = assessment.to_dict()
        self._last_reason = assessment.reason
        return self._last_output.copy()

    def assess_sync(
        self,
        decision: Mapping[str, Any] | None = None,
        *,
        market_signal: Mapping[str, Any] | None = None,
        account_state: Mapping[str, Any] | None = None,
        ohlcv: "pd.DataFrame | None" = None,
        symbol: str | None = None,
        entry_price: float | None = None,
        timestamp: Any | None = None,
        bar: Mapping[str, Any] | None = None,
    ) -> dict[str, float | str]:
        """Synchronously return a deterministic post-risk trade decision."""

        if decision is not None:
            self._latest_decision = normalize_decision_output(decision)
        if market_signal is not None:
            self._latest_market_signal = normalize_market_signal(market_signal)
            maybe_symbol = self._latest_market_signal.get("symbol")
            if isinstance(maybe_symbol, str) and maybe_symbol:
                self._latest_symbol = maybe_symbol
        if account_state is not None:
            self._latest_account_state = self._normalize_account_state(account_state)
        if ohlcv is not None:
            self._latest_ohlcv = ohlcv
        if symbol is not None:
            normalized_symbol = symbol.strip().upper()
            if not normalized_symbol:
                raise ValueError("symbol must be a non-empty string when provided")
            self._latest_symbol = normalized_symbol
        if entry_price is not None:
            self._latest_entry_price = self._coerce_positive_float(entry_price, field_name="entry_price")
        if timestamp is not None:
            self._latest_timestamp = timestamp
        if bar is not None:
            self._latest_bar = dict(bar)

        assessment = self._assess_current_state()
        self._last_output = assessment.to_dict()
        self._last_reason = assessment.reason
        return self._last_output.copy()

    def _assess_current_state(self) -> RiskAssessment:
        """Evaluate the currently stored decision context against hard limits."""

        if self._latest_decision is None:
            raise ValueError("decision is required before risk assessment")
        if self._latest_market_signal is None:
            raise ValueError("market_signal is required before risk assessment")
        if self._latest_account_state is None:
            raise ValueError("account_state is required before risk assessment")
        if self._latest_ohlcv is None:
            raise ValueError("ohlcv is required before risk assessment")

        decision = self._latest_decision
        account_state = self._latest_account_state
        action = str(decision["action"]).upper()

        entry_price, symbol = self._resolve_entry_price_and_symbol(self._latest_ohlcv)
        if action == "HOLD":
            return RiskAssessment(
                final_action="HOLD",
                position_size=0.0,
                stop_loss=0.0,
                reason="decision action is HOLD; risk layer will not allocate capital",
            )

        daily_loss_limit = account_state["capital"] * self._policy.max_daily_loss_fraction
        if account_state["daily_loss"] >= daily_loss_limit:
            return RiskAssessment(
                final_action="HOLD",
                position_size=0.0,
                stop_loss=0.0,
                reason=(
                    f"daily loss limit breached: {account_state['daily_loss']:.2f} "
                    f">= {daily_loss_limit:.2f}"
                ),
            )

        remaining_exposure = (account_state["capital"] * self._policy.max_exposure_fraction) - account_state["gross_exposure"]
        if remaining_exposure <= 0.0:
            return RiskAssessment(
                final_action="HOLD",
                position_size=0.0,
                stop_loss=0.0,
                reason="max exposure reached; no additional exposure is allowed",
            )

        atr_value = self._compute_atr(
            self._latest_ohlcv,
            symbol=symbol,
            timestamp=self._latest_timestamp,
            bar=self._latest_bar,
        )
        if not math.isfinite(atr_value) or atr_value <= 0.0:
            return RiskAssessment(
                final_action="HOLD",
                position_size=0.0,
                stop_loss=0.0,
                reason="ATR is unavailable or non-positive; refusing to size the trade",
            )

        stop_distance = atr_value * self._policy.atr_multiplier
        if stop_distance <= 0.0:
            return RiskAssessment(
                final_action="HOLD",
                position_size=0.0,
                stop_loss=0.0,
                reason="stop distance is non-positive; refusing to size the trade",
            )

        risk_budget = account_state["capital"] * self._policy.max_risk_per_trade_fraction
        size_from_risk = risk_budget / stop_distance
        size_from_exposure = remaining_exposure / entry_price
        position_size = min(size_from_risk, size_from_exposure)
        if not math.isfinite(position_size) or position_size <= 0.0:
            return RiskAssessment(
                final_action="HOLD",
                position_size=0.0,
                stop_loss=0.0,
                reason="position size resolved to zero after applying risk and exposure caps",
            )

        stop_loss = entry_price - stop_distance if action == "BUY" else entry_price + stop_distance
        if action == "BUY" and stop_loss <= 0.0:
            return RiskAssessment(
                final_action="HOLD",
                position_size=0.0,
                stop_loss=0.0,
                reason="ATR stop would place the long stop at or below zero; refusing the trade",
            )

        return RiskAssessment(
            final_action=action,
            position_size=float(position_size),
            stop_loss=float(stop_loss),
            reason=(
                f"risk-approved {action} on {symbol}: "
                f"atr={atr_value:.4f}, stop_distance={stop_distance:.4f}, "
                f"risk_budget={risk_budget:.2f}, remaining_exposure={remaining_exposure:.2f}"
            ),
        )

    def _resolve_entry_price_and_symbol(self, ohlcv: "pd.DataFrame") -> tuple[float, str]:
        """Resolve the active symbol and entry price from cached state."""

        symbol = self._latest_symbol or self._infer_symbol_from_ohlcv(ohlcv)
        if self._latest_entry_price is not None:
            return self._latest_entry_price, symbol

        pd = require_pandas()
        frame = self._normalize_ohlcv_frame(ohlcv, pd)
        symbol_frame = frame.xs(symbol, level="symbol")
        close = self._coerce_positive_float(symbol_frame["close"].iloc[-1], field_name="close")
        self._latest_entry_price = close
        return close, symbol

    def _compute_atr(
        self,
        ohlcv: "pd.DataFrame",
        *,
        symbol: str,
        timestamp: Any | None = None,
        bar: Mapping[str, Any] | None = None,
    ) -> float:
        """Compute the latest ATR value for the requested symbol."""

        pd = require_pandas()
        if bar is not None and timestamp is not None:
            normalized_symbol = symbol.strip().upper()
            normalized_timestamp = self._normalize_timestamp(timestamp, pd)
            normalized_bar = self._normalize_bar(bar)
            state = self._atr_states.get(normalized_symbol)
            if state is None or state.last_timestamp is None or normalized_timestamp <= state.last_timestamp:
                state = self._prime_atr_state(ohlcv, symbol=normalized_symbol, pd=pd)
                self._atr_states[normalized_symbol] = state
                atr_value = state.latest_atr()
            else:
                atr_value = state.update(timestamp=normalized_timestamp, bar=normalized_bar)
            return self._coerce_positive_float(atr_value, field_name="atr")

        frame = self._normalize_ohlcv_frame(ohlcv, pd)
        symbol_frame = frame.xs(symbol, level="symbol")
        if len(symbol_frame) < self._policy.atr_period:
            raise ValueError(
                f"at least {self._policy.atr_period} bars are required to compute ATR for {symbol}"
            )

        high = symbol_frame["high"]
        low = symbol_frame["low"]
        close = symbol_frame["close"]
        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(window=self._policy.atr_period, min_periods=self._policy.atr_period).mean()
        atr_value = atr.iloc[-1]
        return self._coerce_positive_float(atr_value, field_name="atr")

    def _prime_atr_state(self, ohlcv: "pd.DataFrame", *, symbol: str, pd: Any) -> "_IncrementalATRState":
        """Build an incremental ATR state from historical OHLCV bars."""

        frame = self._normalize_ohlcv_frame(ohlcv, pd)
        symbol_frame = frame.xs(symbol, level="symbol")
        if len(symbol_frame) < self._policy.atr_period:
            raise ValueError(
                f"at least {self._policy.atr_period} bars are required to compute ATR for {symbol}"
            )

        state = _IncrementalATRState(symbol=symbol, period=self._policy.atr_period)
        for row_timestamp, row in symbol_frame.iterrows():
            state.update(
                timestamp=self._normalize_timestamp(row_timestamp, pd),
                bar={
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                },
            )
        return state

    def _normalize_ohlcv_frame(self, ohlcv: "pd.DataFrame", pd: Any) -> "pd.DataFrame":
        """Normalize an OHLCV frame to a sorted symbol-timestamp MultiIndex."""

        if not isinstance(ohlcv, pd.DataFrame):
            raise ValueError("ohlcv must be a pandas DataFrame")
        if ohlcv.empty:
            raise ValueError("ohlcv DataFrame cannot be empty")

        frame = ohlcv.copy(deep=True)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"missing required OHLCV columns: {missing_columns}")

        frame = frame.loc[:, list(REQUIRED_OHLCV_COLUMNS)]
        if isinstance(frame.index, pd.MultiIndex):
            if frame.index.nlevels != 2:
                raise ValueError("MultiIndex OHLCV frames must contain symbol and timestamp levels")
            frame.index = frame.index.set_names(["symbol", "timestamp"])
        else:
            timestamps = pd.to_datetime(frame.index, utc=True)
            frame["symbol"] = self._latest_symbol or "__default__"
            frame = frame.reset_index(drop=True)
            frame["timestamp"] = timestamps
            frame = frame.set_index(["symbol", "timestamp"])

        symbols = frame.index.get_level_values("symbol").map(lambda value: str(value).strip().upper())
        timestamps = pd.to_datetime(frame.index.get_level_values("timestamp"), utc=True)
        frame.index = pd.MultiIndex.from_arrays([symbols, timestamps], names=["symbol", "timestamp"])
        return frame.sort_index()

    def _infer_symbol_from_ohlcv(self, ohlcv: "pd.DataFrame") -> str:
        """Infer the active symbol from the OHLCV frame when unambiguous."""

        pd = require_pandas()
        frame = self._normalize_ohlcv_frame(ohlcv, pd)
        symbols = frame.index.get_level_values("symbol").unique()
        if len(symbols) != 1:
            raise ValueError("symbol must be provided when OHLCV contains multiple symbols")
        return str(symbols[0]).strip().upper()

    def _normalize_timestamp(self, timestamp: Any, pd: Any) -> Any:
        """Return a UTC-normalized timestamp value."""

        normalized = pd.Timestamp(timestamp)
        return normalized.tz_convert("UTC") if normalized.tzinfo is not None else normalized.tz_localize("UTC")

    def _normalize_bar(self, bar: Mapping[str, Any]) -> dict[str, float]:
        """Validate and normalize a single OHLCV bar payload."""

        normalized: dict[str, float] = {}
        for field_name in REQUIRED_OHLCV_COLUMNS:
            if field_name not in bar:
                raise ValueError(f"bar is missing required field {field_name}")
            normalized[field_name] = self._coerce_finite_float(bar[field_name], field_name=field_name)
        return normalized

    def _normalize_account_state(self, account_state: Mapping[str, Any]) -> dict[str, float]:
        """Return a normalized account snapshot used by the risk rules."""

        capital_value = None
        for field_name in ("capital", "equity", "net_liquidation"):
            if field_name in account_state:
                capital_value = self._coerce_positive_float(account_state[field_name], field_name=field_name)
                break
        if capital_value is None:
            raise ValueError("account_state must include capital, equity, or net_liquidation")

        if "daily_loss" in account_state:
            daily_loss = self._coerce_non_negative_float(account_state["daily_loss"], field_name="daily_loss")
        else:
            daily_pnl = self._coerce_finite_float(account_state.get("daily_pnl", 0.0), field_name="daily_pnl")
            daily_loss = max(-daily_pnl, 0.0)

        exposure_key = "gross_exposure" if "gross_exposure" in account_state else "exposure"
        gross_exposure = self._coerce_non_negative_float(account_state.get(exposure_key, 0.0), field_name=exposure_key)

        return {
            "capital": capital_value,
            "daily_loss": daily_loss,
            "gross_exposure": gross_exposure,
        }

    def _can_assess(self) -> bool:
        """Return whether the cached state is sufficient for risk evaluation."""

        return (
            self._latest_decision is not None
            and self._latest_market_signal is not None
            and self._latest_account_state is not None
            and self._latest_ohlcv is not None
        )

    def _coerce_finite_float(self, value: Any, *, field_name: str) -> float:
        """Return a validated finite float."""

        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must be a finite number")
        return float(value)

    def _coerce_positive_float(self, value: Any, *, field_name: str) -> float:
        """Return a validated positive float."""

        numeric = self._coerce_finite_float(value, field_name=field_name)
        if numeric <= 0.0:
            raise ValueError(f"{field_name} must be greater than 0")
        return numeric

    def _coerce_non_negative_float(self, value: Any, *, field_name: str) -> float:
        """Return a validated non-negative float."""

        numeric = self._coerce_finite_float(value, field_name=field_name)
        if numeric < 0.0:
            raise ValueError(f"{field_name} must be greater than or equal to 0")
        return numeric


class _IncrementalATRState:
    """Per-symbol rolling ATR state for constant-time updates."""

    def __init__(self, *, symbol: str, period: int) -> None:
        """Initialize empty ATR state for a symbol."""

        self.symbol = symbol
        self.period = period
        self.last_timestamp: Any | None = None
        self._prev_close: float | None = None
        self._true_ranges: deque[float] = deque(maxlen=period)
        self._true_range_sum = 0.0

    def latest_atr(self) -> float:
        """Return the latest ATR value when the window is complete."""

        if len(self._true_ranges) < self.period:
            raise ValueError(f"at least {self.period} bars are required to compute ATR for {self.symbol}")
        return self._true_range_sum / self.period

    def update(self, *, timestamp: Any, bar: Mapping[str, float]) -> float:
        """Advance the rolling ATR state with one additional bar."""

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        true_range = high - low
        if self._prev_close is not None:
            true_range = max(
                true_range,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )

        if len(self._true_ranges) == self._true_ranges.maxlen:
            assert self._true_ranges.maxlen is not None
            self._true_range_sum -= self._true_ranges[0]
        self._true_ranges.append(true_range)
        self._true_range_sum += true_range
        self._prev_close = close
        self.last_timestamp = timestamp
        if len(self._true_ranges) < self.period:
            return float("nan")
        return self.latest_atr()
