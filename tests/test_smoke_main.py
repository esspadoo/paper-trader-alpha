"""Smoke tests for the runnable end-to-end entrypoint."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None

from main import async_health, async_main, async_metrics, async_replay


@unittest.skipUnless(PANDAS_AVAILABLE and XGBOOST_AVAILABLE, "pandas and xgboost are required for main smoke tests")
class MainSmokeTests(unittest.IsolatedAsyncioTestCase):
    """Verify the packaged entrypoint runs end to end with the example config."""

    async def test_async_main_runs_with_example_config(self) -> None:
        """The example config should run end to end and persist the order journal."""

        example_path = Path("config/trading_system.example.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "smoke.toml"
            config_text = example_path.read_text(encoding="utf-8")
            replacements = {
                'journal_path = "state/orders.json"': f'journal_path = "{tmpdir}/orders.json"',
                'state_store_path = "state/runtime_state.db"': f'state_store_path = "{tmpdir}/runtime_state.db"',
                'audit_journal_path = "state/events.jsonl"': f'audit_journal_path = "{tmpdir}/events.jsonl"',
                'dead_letter_path = "state/dead_letters.jsonl"': f'dead_letter_path = "{tmpdir}/dead_letters.jsonl"',
                'alert_journal_path = "state/alerts.jsonl"': f'alert_journal_path = "{tmpdir}/alerts.jsonl"',
            }
            for source, target in replacements.items():
                config_text = config_text.replace(source, target)
            config_path.write_text(config_text, encoding="utf-8")

            exit_code = await async_main(str(config_path))

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmpdir) / "orders.json").exists())

    async def test_operational_commands_read_persisted_state_and_support_replay(self) -> None:
        """Health, metrics, and replay commands should work against persisted runtime state."""

        example_path = Path("config/trading_system.example.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ops.toml"
            config_text = example_path.read_text(encoding="utf-8")
            replacements = {
                'journal_path = "state/orders.json"': f'journal_path = "{tmpdir}/orders.json"',
                'state_store_path = "state/runtime_state.db"': f'state_store_path = "{tmpdir}/runtime_state.db"',
                'audit_journal_path = "state/events.jsonl"': f'audit_journal_path = "{tmpdir}/events.jsonl"',
                'dead_letter_path = "state/dead_letters.jsonl"': f'dead_letter_path = "{tmpdir}/dead_letters.jsonl"',
                'alert_journal_path = "state/alerts.jsonl"': f'alert_journal_path = "{tmpdir}/alerts.jsonl"',
            }
            for source, target in replacements.items():
                config_text = config_text.replace(source, target)
            config_path.write_text(config_text, encoding="utf-8")

            self.assertEqual(await async_main(str(config_path)), 0)

            with io.StringIO() as buffer, redirect_stdout(buffer):
                self.assertEqual(await async_health(str(config_path)), 0)
                health_payload = json.loads(buffer.getvalue())
            self.assertEqual(health_payload["health"]["strategy_id"], "integrated-intraday-demo")

            with io.StringIO() as buffer, redirect_stdout(buffer):
                self.assertEqual(await async_metrics(str(config_path)), 0)
                metrics_payload = json.loads(buffer.getvalue())
            self.assertIn("metrics", metrics_payload)
            self.assertIn("counters", metrics_payload["metrics"])

            self.assertEqual(await async_replay(str(config_path), audit_path=f"{tmpdir}/events.jsonl"), 0)
