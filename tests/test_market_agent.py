"""Tests for the market-agent feature and inference pipeline."""

from __future__ import annotations

import importlib.util
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None

if NUMPY_AVAILABLE:
    import numpy as np

if PANDAS_AVAILABLE:
    import pandas as pd

from trading_system.agents import MarketAgent
from trading_system.models import MarketFeatureConfig, MarketFeatureEngineer, XGBoostReturnModelConfig


def build_intraday_ohlcv(symbols: tuple[str, ...] = ("AAPL", "MSFT"), sessions: int = 4) -> "pd.DataFrame":
    """Build a synthetic but structured intraday OHLCV frame."""

    timestamps = pd.DatetimeIndex([])
    for day_offset in range(sessions):
        session = pd.date_range(
            f"2026-04-{13 + day_offset:02d} 13:30:00+00:00",
            periods=78,
            freq="5min",
            tz="UTC",
        )
        timestamps = timestamps.append(session)

    rows: list[tuple[str, pd.Timestamp, float, float, float, float, float]] = []
    rng = np.random.default_rng(42)

    for symbol_index, symbol in enumerate(symbols):
        phase = symbol_index * 0.7
        base_price = 100.0 + symbol_index * 35.0
        latent_return = np.zeros(len(timestamps))
        for index in range(1, len(timestamps)):
            seasonal = 0.0009 * np.sin(index / 7.0 + phase)
            momentum = 0.45 * latent_return[index - 1]
            shock = 0.00025 * rng.standard_normal()
            latent_return[index] = momentum + seasonal + shock

        close = base_price * np.exp(np.cumsum(latent_return))
        open_ = np.concatenate(([base_price], close[:-1]))
        spread = 0.001 + 0.0004 * rng.random(len(timestamps))
        high = np.maximum(open_, close) * (1.0 + spread)
        low = np.minimum(open_, close) * (1.0 - spread)
        volume = 1_500_000.0 * (1.0 + 0.35 * np.sin(np.arange(len(timestamps)) / 8.0 + phase))
        volume = volume + 120_000.0 * rng.random(len(timestamps))
        volume = np.maximum(volume, 250_000.0)

        rows.extend(
            zip(
                [symbol] * len(timestamps),
                timestamps,
                open_.tolist(),
                high.tolist(),
                low.tolist(),
                close.tolist(),
                volume.tolist(),
                strict=True,
            )
        )

    frame = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    return frame.set_index(["symbol", "timestamp"]).sort_index()


@unittest.skipUnless(NUMPY_AVAILABLE and PANDAS_AVAILABLE, "numpy and pandas are required for market-agent tests")
class MarketFeatureEngineerTests(unittest.TestCase):
    """Verify vectorized features are stable and no-leakage."""

    def test_future_bars_do_not_change_past_features(self) -> None:
        """Mutating future bars must not alter previously engineered features."""

        engineer = MarketFeatureEngineer()
        frame = build_intraday_ohlcv(symbols=("AAPL",)).xs("AAPL", level="symbol")
        original = engineer.transform(frame)

        mutated = frame.copy(deep=True)
        mutated.iloc[-12:, mutated.columns.get_loc("close")] *= 1.20
        mutated.iloc[-12:, mutated.columns.get_loc("high")] *= 1.20
        mutated.iloc[-12:, mutated.columns.get_loc("low")] *= 1.20
        mutated.iloc[-12:, mutated.columns.get_loc("open")] *= 1.20

        mutated_features = engineer.transform(mutated)
        pd.testing.assert_frame_equal(
            original.iloc[:-12],
            mutated_features.iloc[:-12],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_incremental_latest_feature_row_matches_vectorized_latest_row(self) -> None:
        """Incremental feature updates should match the fully recomputed latest row."""

        engineer = MarketFeatureEngineer()
        frame = build_intraday_ohlcv(symbols=("AAPL",), sessions=2)
        symbol_frame = frame.xs("AAPL", level="symbol")
        latest_incremental = None

        for index, (timestamp, row) in enumerate(symbol_frame.iterrows()):
            if index == 0:
                partial = symbol_frame.iloc[:1].copy(deep=True)
                partial["symbol"] = "AAPL"
                partial = partial.reset_index(names="timestamp").set_index(["symbol", "timestamp"])
                latest_incremental = engineer.latest_feature_row_incremental(
                    symbol="AAPL",
                    timestamp=timestamp,
                    bar=row.to_dict(),
                    ohlcv=partial,
                )
            else:
                latest_incremental = engineer.latest_feature_row_incremental(
                    symbol="AAPL",
                    timestamp=timestamp,
                    bar=row.to_dict(),
                )

        assert latest_incremental is not None
        latest_vectorized = engineer.latest_feature_row(frame, symbol="AAPL")
        pd.testing.assert_series_equal(
            latest_incremental.loc[list(engineer.feature_columns)],
            latest_vectorized.loc[list(engineer.feature_columns)],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )


@unittest.skipUnless(
    NUMPY_AVAILABLE and PANDAS_AVAILABLE and XGBOOST_AVAILABLE,
    "numpy, pandas, and xgboost are required for full market-agent tests",
)
class MarketAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify the market agent can train and infer on intraday OHLCV bars."""

    async def test_market_agent_trains_and_infers(self) -> None:
        """The agent should train on synthetic bars and emit a bounded signal payload."""

        frame = build_intraday_ohlcv()
        agent = MarketAgent(
            agent_id="test-market-agent",
            model_config=XGBoostReturnModelConfig(
                feature_config=MarketFeatureConfig(prediction_horizon_bars=2),
                min_training_rows=150,
                n_estimators=80,
                max_depth=3,
                learning_rate=0.08,
                n_jobs=1,
            ),
        )

        await agent.start()
        summary = await agent.train(frame)
        output = await agent.infer(frame, symbol="AAPL")
        snapshot = await agent.snapshot_state()
        await agent.stop()

        self.assertGreater(summary["training_rows"], 150)
        self.assertGreater(summary["validation_rows"], 0)
        self.assertEqual(set(output.keys()), {"predicted_return", "signal", "confidence", "features"})
        self.assertTrue(np.isfinite(output["predicted_return"]))
        self.assertGreaterEqual(output["signal"], -1.0)
        self.assertLessEqual(output["signal"], 1.0)
        self.assertGreaterEqual(output["confidence"], 0.0)
        self.assertLessEqual(output["confidence"], 1.0)
        self.assertIn("vwap", output["features"])
        self.assertIn("volume_spike", output["features"])
        self.assertTrue(all(np.isfinite(list(output["features"].values()))))
        self.assertTrue(snapshot["is_fitted"])
        self.assertEqual(snapshot["agent_id"], "test-market-agent")
