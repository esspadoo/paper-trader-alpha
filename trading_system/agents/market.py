"""Concrete market agent backed by vectorized features and XGBoost inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from trading_system.agents.base import BaseAgent
from trading_system.core.events import BaseEvent, MarketEvent
from trading_system.infra import LatencyRecorder
from trading_system.models.features import MarketFeatureConfig
from trading_system.models.xgboost_return import TrainingSummary, XGBoostReturnModel, XGBoostReturnModelConfig

if TYPE_CHECKING:
    import pandas as pd


class MarketAgent(BaseAgent):
    """Train and serve a short-horizon intraday return model from OHLCV bars."""

    def __init__(
        self,
        *,
        agent_id: str = "market-agent",
        model: XGBoostReturnModel | None = None,
        model_config: XGBoostReturnModelConfig | None = None,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        """Initialize the market agent and its model pipeline."""

        self._agent_id = agent_id
        self._model = model or XGBoostReturnModel(model_config, latency_recorder=latency_recorder)
        self._running = False
        self._last_output: dict[str, Any] | None = None
        self._last_training_summary: TrainingSummary | None = None

    @property
    def agent_id(self) -> str:
        """Return the stable identifier for this agent."""

        return self._agent_id

    @property
    def subscribed_topics(self) -> Sequence[str]:
        """Return the event topics this agent can consume."""

        return ("market",)

    @property
    def model(self) -> XGBoostReturnModel:
        """Return the underlying return-forecast model."""

        return self._model

    @property
    def feature_config(self) -> MarketFeatureConfig:
        """Return the active feature engineering configuration."""

        return self._model.feature_engineer.config

    @property
    def last_output(self) -> dict[str, Any] | None:
        """Return the latest normalized market-signal payload."""

        return None if self._last_output is None else dict(self._last_output)

    async def start(self) -> None:
        """Warm model dependencies and mark the agent as active."""

        await self._model.warmup()
        self._running = True

    async def stop(self) -> None:
        """Mark the agent as inactive."""

        self._running = False

    async def on_event(self, event: BaseEvent) -> None:
        """Consume market events whose payload contains an OHLCV frame."""

        self._model.update(event)
        if not isinstance(event, MarketEvent):
            return

        payload = event.payload
        ohlcv = payload.get("ohlcv")
        if ohlcv is None:
            return

        symbol = event.symbol.strip().upper() if event.symbol else None
        bar = payload.get("bar")
        if symbol and bar is not None and self._model.is_fitted:
            self._last_output = self._model.infer_from_latest_bar(
                ohlcv=ohlcv,
                symbol=symbol,
                timestamp=event.occurred_at,
                bar=bar,
            )
            return

        self._last_output = await self.infer(ohlcv, symbol=symbol)

    async def snapshot_state(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of agent state."""

        return {
            "agent_id": self._agent_id,
            "running": self._running,
            "model_name": self._model.model_name,
            "is_fitted": self._model.is_fitted,
            "last_training_summary": self._last_training_summary.to_dict() if self._last_training_summary else None,
            "last_output": self._last_output,
        }

    async def train(self, ohlcv: "pd.DataFrame") -> dict[str, float | int]:
        """Fit the underlying model on OHLCV bars."""

        summary = self._model.fit(ohlcv)
        self._last_training_summary = summary
        return summary.to_dict()

    async def infer(self, ohlcv: "pd.DataFrame", *, symbol: str | None = None) -> dict[str, Any]:
        """Return the current signal payload for the latest complete feature row."""

        output = self._model.infer_from_ohlcv(ohlcv, symbol=symbol)
        self._last_output = output
        return output

    def train_sync(self, ohlcv: "pd.DataFrame") -> dict[str, float | int]:
        """Synchronously fit the underlying model."""

        summary = self._model.fit(ohlcv)
        self._last_training_summary = summary
        return summary.to_dict()

    def infer_sync(self, ohlcv: "pd.DataFrame", *, symbol: str | None = None) -> dict[str, Any]:
        """Synchronously return a signal payload for the latest feature row."""

        output = self._model.infer_from_ohlcv(ohlcv, symbol=symbol)
        self._last_output = output
        return output
