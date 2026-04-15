"""Smoke tests for the runnable end-to-end entrypoint."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None

from main import async_main


@unittest.skipUnless(PANDAS_AVAILABLE and XGBOOST_AVAILABLE, "pandas and xgboost are required for main smoke tests")
class MainSmokeTests(unittest.IsolatedAsyncioTestCase):
    """Verify the packaged entrypoint runs end to end with the example config."""

    async def test_async_main_runs_with_example_config(self) -> None:
        """The example config should run end to end and persist the order journal."""

        example_path = Path("config/trading_system.example.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "smoke.toml"
            config_text = example_path.read_text(encoding="utf-8").replace(
                'journal_path = "state/orders.json"',
                f'journal_path = "{tmpdir}/orders.json"',
            )
            config_path.write_text(config_text, encoding="utf-8")

            exit_code = await async_main(str(config_path))

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmpdir) / "orders.json").exists())

