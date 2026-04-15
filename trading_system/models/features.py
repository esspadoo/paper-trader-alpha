"""Vectorized intraday feature engineering for market agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Any

from trading_system.models.dependencies import require_numpy, require_pandas
from trading_system.models.exceptions import FeatureEngineeringError, TrainingDataError

if TYPE_CHECKING:
    import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
DEFAULT_SINGLE_SYMBOL = "__default__"


@dataclass(frozen=True, slots=True)
class MarketFeatureConfig:
    """Configuration for vectorized feature engineering."""

    ema_periods: tuple[int, int, int] = (9, 21, 50)
    rsi_period: int = 14
    volatility_window: int = 20
    volume_window: int = 20
    volume_spike_threshold: float = 2.0
    prediction_horizon_bars: int = 3

    def __post_init__(self) -> None:
        """Validate feature engineering parameters."""

        if len(self.ema_periods) != 3:
            raise ValueError("ema_periods must contain exactly three spans")
        if sorted(self.ema_periods) != list(self.ema_periods):
            raise ValueError("ema_periods must be ordered from shortest to longest")
        if self.rsi_period < 2:
            raise ValueError("rsi_period must be at least 2")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if self.volume_spike_threshold <= 0:
            raise ValueError("volume_spike_threshold must be greater than 0")
        if self.prediction_horizon_bars not in (1, 2, 3):
            raise ValueError("prediction_horizon_bars must be 1, 2, or 3")


class MarketFeatureEngineer:
    """Create no-leakage features from intraday OHLCV bars."""

    def __init__(self, config: MarketFeatureConfig | None = None) -> None:
        """Initialize the engineer with a validated feature config."""

        self._config = config or MarketFeatureConfig()
        self._incremental_states: dict[str, _IncrementalFeatureState] = {}

    @property
    def config(self) -> MarketFeatureConfig:
        """Return the active feature configuration."""

        return self._config

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the ordered feature column names."""

        ema_short, ema_medium, ema_long = self._config.ema_periods
        rsi_period = self._config.rsi_period
        volatility_window = self._config.volatility_window
        volume_window = self._config.volume_window
        return (
            "vwap",
            "vwap_gap",
            f"ema_{ema_short}",
            f"ema_{ema_medium}",
            f"ema_{ema_long}",
            f"ema_gap_{ema_short}_{ema_medium}",
            f"ema_gap_{ema_medium}_{ema_long}",
            f"rsi_{rsi_period}",
            f"volatility_{volatility_window}",
            "return_1",
            "return_3",
            "return_5",
            "log_return_1",
            "high_low_range",
            "close_open_range",
            f"volume_mean_{volume_window}",
            f"volume_ratio_{volume_window}",
            "volume_spike",
            "dollar_volume",
        )

    def transform(self, ohlcv: "pd.DataFrame") -> "pd.DataFrame":
        """Return a vectorized feature frame aligned to the input bars."""

        np = require_numpy()
        pd = require_pandas()
        frame = self._normalize_ohlcv_frame(ohlcv, pd)

        close = frame["close"]
        volume = frame["volume"]
        open_ = frame["open"]
        high = frame["high"]
        low = frame["low"]

        return_1 = close.groupby(level="symbol", sort=False).pct_change()
        return_3 = close.groupby(level="symbol", sort=False).pct_change(3)
        return_5 = close.groupby(level="symbol", sort=False).pct_change(5)
        log_return_1 = np.log1p(return_1)

        typical_price = (high + low + close) / 3.0
        timestamp_index = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
        session_index = timestamp_index.normalize()
        symbol_index = frame.index.get_level_values("symbol")
        cumulative_turnover = (typical_price * volume).groupby([symbol_index, session_index], sort=False).cumsum()
        cumulative_volume = volume.groupby([symbol_index, session_index], sort=False).cumsum().replace(0.0, np.nan)
        vwap = cumulative_turnover / cumulative_volume
        vwap_gap = close / vwap - 1.0

        ema_short = self._grouped_ewm(close, span=self._config.ema_periods[0], pd=pd)
        ema_medium = self._grouped_ewm(close, span=self._config.ema_periods[1], pd=pd)
        ema_long = self._grouped_ewm(close, span=self._config.ema_periods[2], pd=pd)

        ema_gap_9_21 = ema_short / ema_medium - 1.0
        ema_gap_21_50 = ema_medium / ema_long - 1.0
        rsi = self._compute_rsi(close, pd=pd)
        volatility = self._grouped_rolling_std(return_1, window=self._config.volatility_window, pd=pd)

        volume_mean = self._grouped_rolling_mean(volume, window=self._config.volume_window, pd=pd)
        volume_ratio = volume / volume_mean.replace(0.0, np.nan)
        volume_spike = (volume_ratio >= self._config.volume_spike_threshold).astype(float)

        high_low_range = (high - low) / close.replace(0.0, np.nan)
        close_open_range = (close - open_) / open_.replace(0.0, np.nan)
        dollar_volume = close * volume

        ema_short_span, ema_medium_span, ema_long_span = self._config.ema_periods
        features = pd.DataFrame(index=frame.index)
        features["vwap"] = vwap
        features["vwap_gap"] = vwap_gap
        features[f"ema_{ema_short_span}"] = ema_short
        features[f"ema_{ema_medium_span}"] = ema_medium
        features[f"ema_{ema_long_span}"] = ema_long
        features[f"ema_gap_{ema_short_span}_{ema_medium_span}"] = ema_gap_9_21
        features[f"ema_gap_{ema_medium_span}_{ema_long_span}"] = ema_gap_21_50
        features[f"rsi_{self._config.rsi_period}"] = rsi
        features[f"volatility_{self._config.volatility_window}"] = volatility
        features["return_1"] = return_1
        features["return_3"] = return_3
        features["return_5"] = return_5
        features["log_return_1"] = log_return_1
        features["high_low_range"] = high_low_range
        features["close_open_range"] = close_open_range
        features[f"volume_mean_{self._config.volume_window}"] = volume_mean
        features[f"volume_ratio_{self._config.volume_window}"] = volume_ratio
        features["volume_spike"] = volume_spike
        features["dollar_volume"] = dollar_volume
        return features.sort_index()

    def prepare_training_frame(self, ohlcv: "pd.DataFrame") -> "pd.DataFrame":
        """Return the feature frame augmented with a no-leakage future-return target."""

        pd = require_pandas()
        frame = self._normalize_ohlcv_frame(ohlcv, pd)
        features = self.transform(frame)
        close = frame["close"]
        target = close.groupby(level="symbol", sort=False).shift(-self._config.prediction_horizon_bars) / close - 1.0

        training_frame = features.copy()
        training_frame["target_return"] = target
        training_frame = training_frame.dropna(subset=[*self.feature_columns, "target_return"])
        if training_frame.empty:
            raise TrainingDataError("training frame is empty after feature engineering and target alignment")

        return training_frame.sort_index()

    def latest_feature_row(self, ohlcv: "pd.DataFrame", *, symbol: str | None = None) -> "pd.Series":
        """Return the most recent complete feature row for the requested symbol."""

        feature_frame = self.transform(ohlcv).dropna(subset=self.feature_columns)
        if feature_frame.empty:
            raise TrainingDataError("insufficient bars to compute a complete feature row")

        if symbol is None:
            symbols = feature_frame.index.get_level_values("symbol").unique()
            if len(symbols) != 1:
                raise FeatureEngineeringError("symbol must be provided when the OHLCV frame contains multiple symbols")
            symbol = str(symbols[0])

        symbol_key = symbol.strip().upper()
        try:
            symbol_frame = feature_frame.xs(symbol_key, level="symbol")
        except KeyError as exc:
            available_symbols = feature_frame.index.get_level_values("symbol").unique()
            if len(available_symbols) == 1:
                symbol_frame = feature_frame.xs(str(available_symbols[0]), level="symbol")
            else:
                raise FeatureEngineeringError(f"symbol {symbol_key} not found in the OHLCV frame") from exc

        if symbol_frame.empty:
            raise TrainingDataError(f"no complete feature rows available for {symbol_key}")

        latest = symbol_frame.iloc[-1].copy()
        latest.name = symbol_frame.index[-1]
        return latest

    def latest_feature_row_incremental(
        self,
        *,
        symbol: str,
        timestamp: Any,
        bar: dict[str, Any] | Any,
        ohlcv: "pd.DataFrame | None" = None,
    ) -> "pd.Series":
        """Return the latest feature row using an incremental per-symbol state cache."""

        pd = require_pandas()
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise FeatureEngineeringError("symbol must be a non-empty string")
        normalized_timestamp = self._normalize_timestamp(timestamp, pd)
        normalized_bar = self._normalize_bar(bar)

        state = self._incremental_states.get(normalized_symbol)
        if state is None or state.last_timestamp is None or normalized_timestamp <= state.last_timestamp:
            if ohlcv is None:
                raise TrainingDataError("ohlcv is required to prime or rebuild incremental feature state")
            state = self._prime_incremental_state(ohlcv, symbol=normalized_symbol, pd=pd)
            self._incremental_states[normalized_symbol] = state
            values = state.latest_values()
        else:
            values = state.update(timestamp=normalized_timestamp, bar=normalized_bar)

        series = pd.Series(values)
        series.name = normalized_timestamp
        return series

    def reset_incremental_state(self, *, symbol: str | None = None) -> None:
        """Clear incremental feature state for one or all symbols."""

        if symbol is None:
            self._incremental_states.clear()
            return
        self._incremental_states.pop(symbol.strip().upper(), None)

    def _normalize_ohlcv_frame(self, ohlcv: "pd.DataFrame", pd: Any) -> "pd.DataFrame":
        """Normalize a raw OHLCV frame into a sorted `symbol,timestamp` MultiIndex."""

        if not isinstance(ohlcv, pd.DataFrame):
            raise FeatureEngineeringError("ohlcv must be a pandas DataFrame")
        if ohlcv.empty:
            raise FeatureEngineeringError("ohlcv DataFrame cannot be empty")

        frame = ohlcv.copy(deep=True)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise FeatureEngineeringError(f"missing required OHLCV columns: {missing_columns}")

        frame = frame.loc[:, list(REQUIRED_OHLCV_COLUMNS)]
        if isinstance(frame.index, pd.MultiIndex):
            if frame.index.nlevels != 2:
                raise FeatureEngineeringError("MultiIndex OHLCV frames must contain exactly two levels: symbol and timestamp")

            frame.index = frame.index.set_names(["symbol", "timestamp"])
        else:
            try:
                timestamps = pd.to_datetime(frame.index, utc=True)
            except Exception as exc:
                raise FeatureEngineeringError("OHLCV index must be datetime-like") from exc

            frame["symbol"] = DEFAULT_SINGLE_SYMBOL
            frame = frame.reset_index(drop=True)
            frame["timestamp"] = timestamps
            frame = frame.set_index(["symbol", "timestamp"])

        symbols = frame.index.get_level_values("symbol").map(lambda value: str(value).strip().upper())
        timestamps = pd.to_datetime(frame.index.get_level_values("timestamp"), utc=True)
        frame.index = pd.MultiIndex.from_arrays([symbols, timestamps], names=["symbol", "timestamp"])
        frame = frame.sort_index()

        for column in REQUIRED_OHLCV_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=list(REQUIRED_OHLCV_COLUMNS))
        if frame.empty:
            raise FeatureEngineeringError("OHLCV frame is empty after numeric normalization")

        return frame

    def _normalize_bar(self, bar: dict[str, Any] | Any) -> dict[str, float]:
        """Validate and normalize a single OHLCV bar payload."""

        if hasattr(bar, "to_dict"):
            bar = bar.to_dict()
        if not isinstance(bar, dict):
            raise FeatureEngineeringError("bar must be a mapping")
        normalized: dict[str, float] = {}
        for field_name in REQUIRED_OHLCV_COLUMNS:
            if field_name not in bar:
                raise FeatureEngineeringError(f"bar is missing required field {field_name}")
            try:
                numeric = float(bar[field_name])
            except (TypeError, ValueError) as exc:
                raise FeatureEngineeringError(f"bar field {field_name} must be numeric") from exc
            normalized[field_name] = numeric
        return normalized

    def _prime_incremental_state(self, ohlcv: "pd.DataFrame", *, symbol: str, pd: Any) -> "_IncrementalFeatureState":
        """Build a fresh incremental state from historical OHLCV bars."""

        frame = self._normalize_ohlcv_frame(ohlcv, pd)
        try:
            symbol_frame = frame.xs(symbol, level="symbol")
        except KeyError as exc:
            available_symbols = frame.index.get_level_values("symbol").unique()
            if len(available_symbols) == 1:
                symbol_frame = frame.xs(str(available_symbols[0]), level="symbol")
            else:
                raise FeatureEngineeringError(f"symbol {symbol} not found in the OHLCV frame") from exc
        state = _IncrementalFeatureState(symbol=symbol, config=self._config)
        for timestamp, row in symbol_frame.iterrows():
            state.update(
                timestamp=self._normalize_timestamp(timestamp, pd),
                bar={
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                },
            )
        return state

    def _normalize_timestamp(self, timestamp: Any, pd: Any) -> Any:
        """Return a UTC-normalized timestamp value."""

        normalized = pd.Timestamp(timestamp)
        return normalized.tz_convert("UTC") if normalized.tzinfo is not None else normalized.tz_localize("UTC")

    def _grouped_ewm(self, series: "pd.Series", *, span: int, pd: Any) -> "pd.Series":
        """Return a per-symbol exponentially weighted moving average."""

        return series.groupby(level="symbol", group_keys=False, sort=False).apply(
            lambda values: values.ewm(span=span, adjust=False, min_periods=span).mean()
        )

    def _grouped_rolling_mean(self, series: "pd.Series", *, window: int, pd: Any) -> "pd.Series":
        """Return a per-symbol rolling mean aligned to the original index."""

        return series.groupby(level="symbol", sort=False).rolling(window=window, min_periods=window).mean().reset_index(
            level=0,
            drop=True,
        )

    def _grouped_rolling_std(self, series: "pd.Series", *, window: int, pd: Any) -> "pd.Series":
        """Return a per-symbol rolling standard deviation aligned to the original index."""

        return series.groupby(level="symbol", sort=False).rolling(window=window, min_periods=window).std(ddof=0).reset_index(
            level=0,
            drop=True,
        )

    def _compute_rsi(self, close: "pd.Series", *, pd: Any) -> "pd.Series":
        """Compute per-symbol RSI using Wilder-style smoothing."""

        np = require_numpy()
        delta = close.groupby(level="symbol", sort=False).diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)

        average_gain = gains.groupby(level="symbol", group_keys=False, sort=False).apply(
            lambda values: values.ewm(alpha=1 / self._config.rsi_period, adjust=False, min_periods=self._config.rsi_period).mean()
        )
        average_loss = losses.groupby(level="symbol", group_keys=False, sort=False).apply(
            lambda values: values.ewm(alpha=1 / self._config.rsi_period, adjust=False, min_periods=self._config.rsi_period).mean()
        )

        relative_strength = average_gain / average_loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.where(average_loss != 0.0, 100.0)
        rsi = rsi.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)
        return rsi


class _IncrementalFeatureState:
    """Per-symbol rolling state for constant-time incremental feature updates."""

    def __init__(self, *, symbol: str, config: MarketFeatureConfig) -> None:
        """Initialize empty incremental state for a symbol."""

        self.symbol = symbol
        self.config = config
        self.last_timestamp: Any | None = None
        self._bar_count = 0
        self._delta_count = 0
        self._prev_close: float | None = None
        self._current_session: Any | None = None
        self._cumulative_turnover = 0.0
        self._cumulative_volume = 0.0
        self._close_history: deque[float] = deque(maxlen=5)
        self._volume_window: deque[float] = deque(maxlen=self.config.volume_window)
        self._return_window: deque[float] = deque(maxlen=self.config.volatility_window)
        self._volume_sum = 0.0
        self._return_sum = 0.0
        self._return_sum_sq = 0.0
        self._ema_values: dict[int, float] = {}
        self._ema_alphas = {span: 2.0 / (span + 1.0) for span in self.config.ema_periods}
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self._latest_values = self._nan_feature_payload()

    def latest_values(self) -> dict[str, float]:
        """Return the most recently computed feature payload."""

        return dict(self._latest_values)

    def update(self, *, timestamp: Any, bar: dict[str, float]) -> dict[str, float]:
        """Advance the state with one new bar and return the latest features."""

        session_key = getattr(timestamp, "date", lambda: timestamp)()
        if self._current_session != session_key:
            self._current_session = session_key
            self._cumulative_turnover = 0.0
            self._cumulative_volume = 0.0

        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = float(bar["volume"])
        typical_price = (high + low + close) / 3.0

        self._cumulative_turnover += typical_price * volume
        self._cumulative_volume += volume
        vwap = self._safe_ratio(self._cumulative_turnover, self._cumulative_volume)
        vwap_gap = self._safe_ratio(close, vwap) - 1.0 if vwap == vwap and vwap != 0.0 else float("nan")

        self._bar_count += 1
        for span in self.config.ema_periods:
            previous = self._ema_values.get(span)
            if previous is None:
                self._ema_values[span] = close
            else:
                alpha = self._ema_alphas[span]
                self._ema_values[span] = (alpha * close) + ((1.0 - alpha) * previous)

        return_1 = float("nan")
        log_return_1 = float("nan")
        rsi_value = float("nan")
        if self._prev_close is not None and self._prev_close != 0.0:
            return_1 = (close / self._prev_close) - 1.0
            log_return_1 = float(require_numpy().log1p(return_1))
            self._append_return(return_1)

            delta = close - self._prev_close
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            alpha = 1.0 / self.config.rsi_period
            if self._avg_gain is None or self._avg_loss is None:
                self._avg_gain = gain
                self._avg_loss = loss
            else:
                self._avg_gain = (alpha * gain) + ((1.0 - alpha) * self._avg_gain)
                self._avg_loss = (alpha * loss) + ((1.0 - alpha) * self._avg_loss)
            self._delta_count += 1
            if self._delta_count >= self.config.rsi_period:
                rsi_value = self._compute_rsi_value()

        return_3 = self._safe_return(close, lag=3)
        return_5 = self._safe_return(close, lag=5)

        volume_mean = self._append_volume(volume)
        volume_ratio = volume / volume_mean if volume_mean == volume_mean and volume_mean != 0.0 else float("nan")
        volume_spike = 1.0 if volume_ratio == volume_ratio and volume_ratio >= self.config.volume_spike_threshold else 0.0
        volatility = self._compute_volatility()

        ema_short_span, ema_medium_span, ema_long_span = self.config.ema_periods
        ema_short = self._ema_or_nan(ema_short_span)
        ema_medium = self._ema_or_nan(ema_medium_span)
        ema_long = self._ema_or_nan(ema_long_span)
        ema_gap_short_medium = self._safe_ratio(ema_short, ema_medium) - 1.0 if ema_short == ema_short and ema_medium == ema_medium and ema_medium != 0.0 else float("nan")
        ema_gap_medium_long = self._safe_ratio(ema_medium, ema_long) - 1.0 if ema_medium == ema_medium and ema_long == ema_long and ema_long != 0.0 else float("nan")

        self._latest_values = {
            "vwap": vwap,
            "vwap_gap": vwap_gap,
            f"ema_{ema_short_span}": ema_short,
            f"ema_{ema_medium_span}": ema_medium,
            f"ema_{ema_long_span}": ema_long,
            f"ema_gap_{ema_short_span}_{ema_medium_span}": ema_gap_short_medium,
            f"ema_gap_{ema_medium_span}_{ema_long_span}": ema_gap_medium_long,
            f"rsi_{self.config.rsi_period}": rsi_value,
            f"volatility_{self.config.volatility_window}": volatility,
            "return_1": return_1,
            "return_3": return_3,
            "return_5": return_5,
            "log_return_1": log_return_1,
            "high_low_range": self._safe_ratio(high - low, close),
            "close_open_range": self._safe_ratio(close - open_price, open_price),
            f"volume_mean_{self.config.volume_window}": volume_mean,
            f"volume_ratio_{self.config.volume_window}": volume_ratio,
            "volume_spike": volume_spike,
            "dollar_volume": close * volume,
        }

        self._close_history.append(close)
        self._prev_close = close
        self.last_timestamp = timestamp
        return dict(self._latest_values)

    def _append_volume(self, volume: float) -> float:
        """Append a volume observation and return the rolling mean."""

        if len(self._volume_window) == self._volume_window.maxlen:
            assert self._volume_window.maxlen is not None
            self._volume_sum -= self._volume_window[0]
        self._volume_window.append(volume)
        self._volume_sum += volume
        if len(self._volume_window) < self.config.volume_window:
            return float("nan")
        return self._volume_sum / self.config.volume_window

    def _append_return(self, return_1: float) -> None:
        """Append a return observation to the rolling volatility window."""

        if len(self._return_window) == self._return_window.maxlen:
            assert self._return_window.maxlen is not None
            outgoing = self._return_window[0]
            self._return_sum -= outgoing
            self._return_sum_sq -= outgoing * outgoing
        self._return_window.append(return_1)
        self._return_sum += return_1
        self._return_sum_sq += return_1 * return_1

    def _compute_volatility(self) -> float:
        """Return the rolling standard deviation of 1-bar returns."""

        if len(self._return_window) < self.config.volatility_window:
            return float("nan")
        window = self.config.volatility_window
        mean_return = self._return_sum / window
        variance = max((self._return_sum_sq / window) - (mean_return * mean_return), 0.0)
        return sqrt(variance)

    def _safe_return(self, close: float, *, lag: int) -> float:
        """Return a lagged percent return when enough history exists."""

        if len(self._close_history) < lag:
            return float("nan")
        reference = self._close_history[-lag]
        return self._safe_ratio(close, reference) - 1.0 if reference != 0.0 else float("nan")

    def _compute_rsi_value(self) -> float:
        """Return the Wilder RSI value from the smoothed gain/loss state."""

        assert self._avg_gain is not None and self._avg_loss is not None
        if self._avg_loss == 0.0 and self._avg_gain == 0.0:
            return 50.0
        if self._avg_loss == 0.0:
            return 100.0
        relative_strength = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + relative_strength))

    def _ema_or_nan(self, span: int) -> float:
        """Return the EMA when enough bars exist, otherwise NaN."""

        if self._bar_count < span:
            return float("nan")
        return self._ema_values[span]

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Return a division result or NaN for zero denominators."""

        if denominator == 0.0:
            return float("nan")
        return numerator / denominator

    def _nan_feature_payload(self) -> dict[str, float]:
        """Return an all-NaN payload used before the first update."""

        ema_short_span, ema_medium_span, ema_long_span = self.config.ema_periods
        return {
            "vwap": float("nan"),
            "vwap_gap": float("nan"),
            f"ema_{ema_short_span}": float("nan"),
            f"ema_{ema_medium_span}": float("nan"),
            f"ema_{ema_long_span}": float("nan"),
            f"ema_gap_{ema_short_span}_{ema_medium_span}": float("nan"),
            f"ema_gap_{ema_medium_span}_{ema_long_span}": float("nan"),
            f"rsi_{self.config.rsi_period}": float("nan"),
            f"volatility_{self.config.volatility_window}": float("nan"),
            "return_1": float("nan"),
            "return_3": float("nan"),
            "return_5": float("nan"),
            "log_return_1": float("nan"),
            "high_low_range": float("nan"),
            "close_open_range": float("nan"),
            f"volume_mean_{self.config.volume_window}": float("nan"),
            f"volume_ratio_{self.config.volume_window}": float("nan"),
            "volume_spike": 0.0,
            "dollar_volume": float("nan"),
        }
