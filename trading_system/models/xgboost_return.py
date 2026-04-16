"""XGBoost return-forecast model for market agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

from trading_system.core.events import BaseEvent
from trading_system.infra import LatencyRecorder
from trading_system.models.base import BaseModel
from trading_system.models.dependencies import require_numpy, require_pandas, require_xgboost
from trading_system.models.exceptions import ModelNotFittedError, TrainingDataError
from trading_system.models.features import MarketFeatureConfig, MarketFeatureEngineer

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class XGBoostReturnModelConfig:
    """Configuration for the return-forecasting XGBoost model."""

    feature_config: MarketFeatureConfig = field(default_factory=MarketFeatureConfig)
    validation_fraction: float = 0.2
    min_training_rows: int = 200
    n_estimators: int = 250
    learning_rate: float = 0.05
    max_depth: int = 4
    min_child_weight: float = 8.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = 1
    signal_scale_floor: float = 0.0025

    def __post_init__(self) -> None:
        """Validate model hyperparameters and split settings."""

        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.min_training_rows < 50:
            raise ValueError("min_training_rows must be at least 50")
        if self.n_estimators < 10:
            raise ValueError("n_estimators must be at least 10")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.min_child_weight < 0.0:
            raise ValueError("min_child_weight cannot be negative")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be between 0 and 1")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be between 0 and 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if self.signal_scale_floor <= 0.0:
            raise ValueError("signal_scale_floor must be greater than 0")


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Immutable training summary returned by the model pipeline."""

    training_rows: int
    validation_rows: int
    feature_count: int
    target_horizon_bars: int
    rmse: float
    mae: float
    target_scale: float
    residual_scale: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable training summary."""

        return dict(asdict(self))


class XGBoostReturnModel(BaseModel):
    """Predict short-horizon returns from vectorized intraday features."""

    def __init__(
        self,
        config: XGBoostReturnModelConfig | None = None,
        *,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        """Initialize the model and its feature engineering pipeline."""

        self._config = config or XGBoostReturnModelConfig()
        self._feature_engineer = MarketFeatureEngineer(self._config.feature_config)
        self._latency_recorder = latency_recorder
        self._model: Any | None = None
        self._feature_names = self._feature_engineer.feature_columns
        self._summary: TrainingSummary | None = None
        self._target_scale = self._config.signal_scale_floor
        self._residual_scale = self._config.signal_scale_floor
        self._last_event_type: str | None = None

    @property
    def model_name(self) -> str:
        """Return the stable model identifier."""

        return "xgboost-return-model"

    @property
    def feature_engineer(self) -> MarketFeatureEngineer:
        """Return the feature engineering pipeline used by the model."""

        return self._feature_engineer

    @property
    def training_summary(self) -> TrainingSummary | None:
        """Return the latest training summary, if available."""

        return self._summary

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""

        return self._model is not None

    async def warmup(self) -> None:
        """Validate model dependencies without training."""

        require_numpy()
        require_pandas()
        require_xgboost()

    def fit(self, ohlcv: "pd.DataFrame") -> TrainingSummary:
        """Fit the XGBoost model on a no-leakage intraday dataset."""

        np = require_numpy()
        xgb = require_xgboost()

        dataset = self._feature_engineer.prepare_training_frame(ohlcv)
        if len(dataset) <= self._config.min_training_rows:
            raise TrainingDataError(
                f"training dataset must contain more than {self._config.min_training_rows} rows after feature engineering"
            )

        split_index = int(len(dataset) * (1.0 - self._config.validation_fraction))
        minimum_split_index = max(self._config.min_training_rows, 1)
        maximum_split_index = len(dataset) - 1
        split_index = min(max(split_index, minimum_split_index), maximum_split_index)
        if split_index >= len(dataset) or split_index < self._config.min_training_rows:
            raise TrainingDataError("validation split is empty; provide more bars or reduce min_training_rows")

        training_set = dataset.iloc[:split_index]
        validation_set = dataset.iloc[split_index:]
        if validation_set.empty:
            raise TrainingDataError("validation split is empty after time-based partitioning")

        x_train = training_set.loc[:, self._feature_names].to_numpy(dtype=float)
        y_train = training_set["target_return"].to_numpy(dtype=float)
        x_validation = validation_set.loc[:, self._feature_names].to_numpy(dtype=float)
        y_validation = validation_set["target_return"].to_numpy(dtype=float)

        dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(self._feature_names))
        dvalidation = xgb.DMatrix(x_validation, label=y_validation, feature_names=list(self._feature_names))
        booster = xgb.train(
            params={
                "objective": "reg:squarederror",
                "eta": self._config.learning_rate,
                "max_depth": self._config.max_depth,
                "min_child_weight": self._config.min_child_weight,
                "subsample": self._config.subsample,
                "colsample_bytree": self._config.colsample_bytree,
                "alpha": self._config.reg_alpha,
                "lambda": self._config.reg_lambda,
                "seed": self._config.random_state,
                "tree_method": "hist",
                "nthread": self._config.n_jobs,
                "verbosity": 0,
            },
            dtrain=dtrain,
            num_boost_round=self._config.n_estimators,
            evals=[(dvalidation, "validation")],
            verbose_eval=False,
        )

        validation_predictions = booster.predict(dvalidation)
        validation_errors = y_validation - validation_predictions.astype(float)
        rmse = float(sqrt(float(np.mean(np.square(validation_errors)))))
        mae = float(np.mean(np.abs(validation_errors)))
        target_scale = float(max(abs(float(np.std(y_train, ddof=0))), self._config.signal_scale_floor))
        residual_scale = float(max(abs(float(np.std(validation_errors, ddof=0))), self._config.signal_scale_floor))

        self._model = booster
        self._target_scale = target_scale
        self._residual_scale = residual_scale
        self._summary = TrainingSummary(
            training_rows=len(training_set),
            validation_rows=len(validation_set),
            feature_count=len(self._feature_names),
            target_horizon_bars=self._config.feature_config.prediction_horizon_bars,
            rmse=rmse,
            mae=mae,
            target_scale=target_scale,
            residual_scale=residual_scale,
        )
        return self._summary

    def predict(self, features: dict[str, Any] | Any) -> dict[str, float]:
        """Predict the next short-horizon return from a feature mapping."""

        np = require_numpy()
        xgb = require_xgboost()
        self._ensure_fitted()
        feature_row = self._coerce_feature_row(features)
        feature_matrix = self._build_feature_matrix([feature_row])
        started_at_ns = perf_counter_ns()
        predicted_return = self._predict_return_from_matrix(feature_matrix, xgb=xgb)
        if self._latency_recorder is not None:
            self._latency_recorder.record_ns("market_model_inference", perf_counter_ns() - started_at_ns)
        signal = float(np.tanh(predicted_return / max(self._target_scale, self._config.signal_scale_floor)))
        confidence = float(np.clip(np.tanh(abs(predicted_return) / max(self._residual_scale, self._config.signal_scale_floor)), 0.0, 1.0))
        return {
            "predicted_return": predicted_return,
            "signal": signal,
            "confidence": confidence,
        }

    def predict_frame(self, feature_frame: "pd.DataFrame") -> "pd.DataFrame":
        """Return vectorized predictions for a full feature frame."""

        np = require_numpy()
        pd = require_pandas()
        xgb = require_xgboost()
        self._ensure_fitted()

        candidate_frame = feature_frame.loc[:, self._feature_names].dropna(subset=self._feature_names)
        if candidate_frame.empty:
            return pd.DataFrame(columns=["predicted_return", "signal", "confidence"], index=feature_frame.index)

        prediction_data = xgb.DMatrix(candidate_frame.to_numpy(dtype=float), feature_names=list(self._feature_names))
        predictions = self._model.predict(prediction_data)
        signal = np.tanh(predictions / max(self._target_scale, self._config.signal_scale_floor))
        confidence = np.clip(
            np.tanh(np.abs(predictions) / max(self._residual_scale, self._config.signal_scale_floor)),
            0.0,
            1.0,
        )
        prediction_frame = pd.DataFrame(
            {
                "predicted_return": predictions.astype(float),
                "signal": signal.astype(float),
                "confidence": confidence.astype(float),
            },
            index=candidate_frame.index,
        )
        return prediction_frame.reindex(feature_frame.index)

    def infer_from_ohlcv(self, ohlcv: "pd.DataFrame", *, symbol: str | None = None) -> dict[str, Any]:
        """Engineer features from OHLCV bars and return the latest prediction payload."""

        features_started_at_ns = perf_counter_ns()
        latest_features = self._feature_engineer.latest_feature_row(ohlcv, symbol=symbol)
        if self._latency_recorder is not None:
            self._latency_recorder.record_ns("market_feature_compute", perf_counter_ns() - features_started_at_ns)
        prediction = self.predict(latest_features.to_dict())
        feature_payload = {
            key: float(value) if value is not None else value
            for key, value in latest_features.loc[list(self._feature_names)].to_dict().items()
        }
        return {
            "predicted_return": float(prediction["predicted_return"]),
            "signal": float(prediction["signal"]),
            "confidence": float(prediction["confidence"]),
            "features": feature_payload,
        }

    def infer_from_latest_bar(
        self,
        *,
        ohlcv: "pd.DataFrame",
        symbol: str,
        timestamp: Any,
        bar: dict[str, Any] | Any,
    ) -> dict[str, Any]:
        """Return a prediction payload from an incremental one-bar feature update."""

        np = require_numpy()
        features_started_at_ns = perf_counter_ns()
        latest_features = self._feature_engineer.latest_feature_row_incremental(
            symbol=symbol,
            timestamp=timestamp,
            bar=bar,
            ohlcv=ohlcv,
        )
        feature_payload = {
            key: float(value) if value is not None else value
            for key, value in latest_features.loc[list(self._feature_names)].to_dict().items()
        }
        if not bool(np.isfinite(list(feature_payload.values())).all()):
            raise TrainingDataError("insufficient bars to compute a complete feature row")
        if self._latency_recorder is not None:
            self._latency_recorder.record_ns("market_feature_compute", perf_counter_ns() - features_started_at_ns)
        prediction = self.predict(feature_payload)
        return {
            "predicted_return": float(prediction["predicted_return"]),
            "signal": float(prediction["signal"]),
            "confidence": float(prediction["confidence"]),
            "features": feature_payload,
        }

    def update(self, event: BaseEvent) -> None:
        """Track the last event type observed by the model."""

        self._last_event_type = event.event_type

    def reset(self) -> None:
        """Reset the trained model and calibration state."""

        self._model = None
        self._summary = None
        self._target_scale = self._config.signal_scale_floor
        self._residual_scale = self._config.signal_scale_floor
        self._last_event_type = None

    def _coerce_feature_row(self, features: dict[str, Any] | Any) -> dict[str, float]:
        """Coerce an arbitrary feature mapping into an ordered numeric row."""

        if hasattr(features, "to_dict"):
            features = features.to_dict()
        if not isinstance(features, dict):
            raise TrainingDataError("features must be a mapping or pandas Series")

        row: dict[str, float] = {}
        missing_columns = [column for column in self._feature_names if column not in features]
        if missing_columns:
            raise TrainingDataError(f"missing feature columns for inference: {missing_columns}")

        for column in self._feature_names:
            value = features[column]
            if value is None:
                raise TrainingDataError(f"feature {column} is None and cannot be used for inference")
            row[column] = float(value)

        return row

    def _build_feature_matrix(self, rows: list[dict[str, float]]) -> Any:
        """Build an ordered NumPy feature matrix for XGBoost."""

        np = require_numpy()
        return np.array([[row[column] for column in self._feature_names] for row in rows], dtype=float)

    def _predict_return_from_matrix(self, feature_matrix: Any, *, xgb: Any) -> float:
        """Return a scalar prediction from a one-row feature matrix."""

        inplace_predict = getattr(self._model, "inplace_predict", None)
        if callable(inplace_predict):
            try:
                predictions = inplace_predict(feature_matrix)
                return float(predictions[0])
            except Exception:
                pass

        prediction_data = xgb.DMatrix(feature_matrix, feature_names=list(self._feature_names))
        return float(self._model.predict(prediction_data)[0])

    def _ensure_fitted(self) -> None:
        """Ensure the model has been fitted before inference."""

        if self._model is None:
            raise ModelNotFittedError("model must be fitted before inference")
