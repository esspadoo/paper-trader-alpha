# Latency Report

## Scope

This report compares the integrated intraday runtime before and after the hot-path latency changes made on 2026-04-15.

Safety invariants preserved throughout:

- live-trading fail-closed gates
- duplicate-order suppression
- stale-news expiry
- startup reconciliation
- smoke-test coverage

## Benchmark Method

Command:

```bash
.venv/bin/python - <<'PY'
import asyncio, json
from trading_system.system.latency_benchmark import run_latency_benchmark

async def main():
    report = await run_latency_benchmark()
    print(json.dumps(report, indent=2))

asyncio.run(main())
PY
```

Benchmark config:

- symbol: `AAPL`
- training bars: `180`
- streamed bars: `240`
- bar interval: `5m`
- prediction horizon: `2` bars
- news max age: `14_400s`

## Before vs After

Baseline was captured after instrumentation was added and before the market/risk/news hot-path redesign. Post-change numbers are from the current codebase.

| Metric | Before Mean ms | After Mean ms | Change | Before p95 ms | After p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `market_feature_compute` | 97.048 | 1.017 | -98.95% | 123.062 | 1.027 |
| `market_model_inference` | 11.736 | 0.398 | -96.61% | 13.868 | 0.452 |
| `market_agent_total` | 110.853 | 1.553 | -98.60% | 137.477 | 1.627 |
| `risk_step` | 20.707 | 0.167 | -99.19% | 27.694 | 0.094 |
| `decision_critic_risk_path` | 21.197 | 0.318 | -98.50% | 28.116 | 0.272 |
| `order_manager_dispatch` | 10.531 | 0.307 | -97.08% | 16.681 | 0.390 |
| `order_submission_path` | 10.565 | 0.316 | -97.01% | 16.715 | 0.405 |
| `event_queue_latency` | 13.418 | 0.237 | -98.24% | 25.495 | 0.344 |
| `signal_to_order_latency` | 142.227 | 3.897 | -97.26% | 163.470 | 2.511 |
| `news_agent_processing` | 0.819 | 0.253 | -69.11% | 0.898 | 0.318 |

## After: Full Breakdown

| Metric | Count | Mean ms | p95 ms | p99 ms | Worst ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `critic_step` | 240 | 0.040 | 0.051 | 0.068 | 0.095 |
| `decision_critic_risk_path` | 240 | 0.318 | 0.272 | 0.375 | 21.037 |
| `decision_step` | 240 | 0.045 | 0.057 | 0.084 | 0.134 |
| `event_queue_latency` | 789 | 0.237 | 0.344 | 0.408 | 21.356 |
| `market_agent_total` | 240 | 1.553 | 1.627 | 1.961 | 37.611 |
| `market_event_receive_latency` | 240 | 0.029 | 0.037 | 0.063 | 0.147 |
| `market_feature_compute` | 240 | 1.017 | 1.027 | 1.195 | 36.452 |
| `market_model_inference` | 240 | 0.398 | 0.452 | 0.631 | 0.790 |
| `news_agent_processing` | 240 | 0.253 | 0.318 | 0.515 | 0.620 |
| `order_manager_dispatch` | 33 | 0.307 | 0.390 | 0.659 | 0.659 |
| `order_submission_path` | 33 | 0.316 | 0.405 | 0.693 | 0.693 |
| `risk_step` | 240 | 0.167 | 0.094 | 0.104 | 20.696 |
| `signal_to_order_latency` | 33 | 3.897 | 2.511 | 59.784 | 59.784 |

## Implemented Optimizations

### 1. Incremental market features

- Reused per-symbol rolling feature state instead of recomputing indicators across the full OHLCV frame on every bar.
- Kept the existing vectorized path as the fallback for rebuild and out-of-order recovery.
- Preserved no-leakage semantics.

Files:

- `trading_system/models/features.py`
- `trading_system/models/xgboost_return.py`
- `trading_system/agents/market.py`

### 2. Faster single-row model inference

- Replaced the always-allocate `DMatrix` single-row path with a direct matrix prediction path that uses `Booster.inplace_predict` when available.
- Reduced feature-row object churn in the market path.

Files:

- `trading_system/models/xgboost_return.py`

### 3. Incremental ATR in the risk layer

- Replaced full-frame ATR recomputation on each bar with a rolling per-symbol ATR state.
- Kept full recompute as the rebuild path.

Files:

- `trading_system/agents/risk.py`

### 4. News removed from the blocking decision path

- Market decisions now read the latest non-stale cached news state immediately.
- News ingestion and article analysis run in the background and update state asynchronously.
- Stale-news expiry and deduplication behavior were preserved.

Files:

- `trading_system/system/runtime.py`

### 5. Lower-overhead state access and persistence

- Runtime reads the market agent’s latest output directly instead of snapshotting the entire agent state in the hot path.
- Order persistence uses the existing SQLite journal path, keeping idempotency and restart safety while reducing dispatch overhead.

Files:

- `trading_system/system/runtime.py`
- `trading_system/execution/journal.py`
- `trading_system/execution/order_manager.py`

## Validation

Verification commands:

```bash
.venv/bin/python -m unittest -v
python3 -m compileall trading_system tests main.py
```

Both passed after the latency changes.

## Residual Latency Risks

The system is materially faster, but these latency risks remain:

- The first streamed bar still pays a rebuild cost when the incremental market and ATR states are primed from full history.
- News source fetches and local-LLM HTTP calls are still synchronous internally; they are no longer on the blocking market decision path, but they can still consume event-loop time while background refresh is running.
- Logging is still standard-library synchronous logging.
- Live broker account and position reads remain synchronous request/response calls on the risk and order path.
- The `signal_to_order_latency` p99 and worst-case still show first-order warm-up and cold-path spikes.

## Recommended Next Steps

1. Pre-prime market and ATR state before the first live decision bar.
2. Move broker account and position state to explicit cached subscriptions or bounded refresh tasks.
3. Replace the remaining synchronous news/backend I/O with a proven non-blocking transport that does not hang in the deployment environment.
4. Add asynchronous metrics export and queue-backed logging if the runtime is promoted beyond local/paper use.
