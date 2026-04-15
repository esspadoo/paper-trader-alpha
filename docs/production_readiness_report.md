# Production Readiness Report

## Executive Summary

This repository is now structurally sound for controlled paper trading, but it is **not yet live-trading deployable**.

The most important remediations in this change set are:

- live trading now fails closed behind explicit config gates
- duplicate order submission is suppressed across restarts with a persistent journal
- stale news is expired before it can influence market decisions
- config loading is validated with `pydantic` and environment overrides
- an end-to-end smoke test now exercises the runnable entrypoint

The highest remaining blockers are:

- synchronous network and model calls still execute on the event loop
- restart recovery is only partial; session equity baselines and broker reconciliation are incomplete
- the backtest still does not model spread, borrow, market calendar, or corporate actions closely enough for live parity
- observability is still log-centric and lacks metrics, health checks, and alerting

## A. System Map

### Actual Current Architecture

- `trading_system/core`: immutable events, async engine, queue-based bus
- `trading_system/data`: market data adapters, news source adapters, universe selection
- `trading_system/models`: feature engineering, XGBoost market model, local LLM news analysis
- `trading_system/agents`: market, news, decision, critic, risk agents
- `trading_system/execution`: broker abstraction, IBKR adapter, paper broker, order manager, order journal
- `trading_system/backtest`: event-driven intraday simulator, strategy coordinator, portfolio ledger
- `trading_system/system`: integrated runtime, validated config, demo fixtures
- `main.py`: runnable orchestration entrypoint

### Critical Execution Paths

1. `main.py` loads validated config, builds demo or local components, trains the market model, creates broker and order manager, and starts `IntegratedTradingSystem`.
2. `MarketEvent` enters `Engine` and is routed to the execution mark updater and the integrated runtime.
3. `IntegratedTradingSystem` runs:
   `MarketAgent -> NewsAgent -> DecisionAgent -> CriticAgent -> RiskAgent -> OrderEvent`.
4. `ExecutionService` converts `OrderEvent` into broker submissions through `OrderManager`.
5. `OrderManager` persists idempotency state in `OrderJournal` and reconciles broker state on startup.
6. Broker fills produce normalized `OrderEvent` execution updates and final account snapshots.

### Runtime Dependencies Between Modules

- `system/runtime.py` depends on `agents/*`, `core/*`, `execution/*`, and validated config in `system/config.py`
- `agents/market.py` depends on `models/features.py` and `models/xgboost_return.py`
- `agents/news.py` depends on `data/news.py`, `models/news_llm.py`, and `infra/cache.py`
- `execution/order_manager.py` depends on the broker contract plus `execution/journal.py`
- `backtest/strategy.py` mirrors the live agent flow and depends on `backtest/execution.py`

## B-F. Severity-Ranked Findings

### Critical

#### 1. Async runtime still contains blocking I/O and model calls

- Files:
  - `trading_system/data/news.py`
  - `trading_system/models/news_llm.py`
  - `trading_system/agents/market.py`
  - `main.py`
- Issue:
  - RSS/JSON fetches and local-LLM HTTP calls are synchronous under async methods, and market-model train/infer calls are still local blocking CPU work.
- Why it matters:
  - Under real market load this can stall the event loop, delay risk checks, and widen signal-to-order latency.
- Exact recommended fix:
  - move blocking fetch/inference/train work behind bounded executors or dedicated worker processes
  - keep runtime inference preloaded and short-lived
  - forbid in-process model training during a live session

#### 2. Backtest/live microstructure assumptions still diverge materially

- Files:
  - `trading_system/backtest/execution.py`
  - `trading_system/backtest/types.py`
  - `trading_system/backtest/intraday.py`
- Issue:
  - the simulator models slippage, latency, and commissions, but it still lacks spread, short borrow, auction behavior, exchange calendar/session rules, and corporate actions handling.
- Why it matters:
  - strategy performance can be overstated and execution behavior may not match production routing.
- Exact recommended fix:
  - add explicit spread and queue-position models
  - gate shorting behind borrow availability
  - enforce market sessions with an exchange calendar
  - normalize historical data for splits/dividends or reject unadjusted data explicitly

### High

#### 3. Restart recovery is only partially implemented

- Files:
  - `trading_system/system/runtime.py`
  - `trading_system/execution/order_manager.py`
  - `trading_system/execution/ibkr.py`
- Issue:
  - persistent order idempotency is now present, but the runtime still does not persist daily equity baselines or fully reconcile broker fills, open positions, and prior strategy state after restart.
- Why it matters:
  - a restart during a session can restore order submission state without restoring full risk state.
- Exact recommended fix:
  - persist session baseline equity and strategy state
  - reconcile positions, open orders, and recent executions at startup before accepting new signals
  - reject startup if broker state cannot be reconciled cleanly

#### 4. Observability is not production-grade yet

- Files:
  - `trading_system/system/runtime.py`
  - `main.py`
  - `trading_system/execution/*`
- Issue:
  - logs are present at each step, but there are no structured metrics, health checks, alert channels, or runbook automation hooks.
- Why it matters:
  - failures will be harder to detect, triage, and recover under paper/live operations.
- Exact recommended fix:
  - add metrics for event lag, handler latency, queue depth, order rejects, reconnects, and risk vetoes
  - expose health and readiness checks
  - add alert routing and documented operational procedures

#### 5. Secrets and environment separation remain basic

- Files:
  - `trading_system/system/config.py`
  - `config/trading_system.example.toml`
- Issue:
  - environment overrides exist, but there is no dedicated secrets provider, no config encryption, and no deployment-specific secret scoping.
- Why it matters:
  - operators can still accidentally manage live credentials through plain files or broad shell environments.
- Exact recommended fix:
  - move live credentials to a dedicated secret store
  - scope secrets by environment
  - add CI/CD checks preventing live config from shipping with demo defaults

### Medium

#### 6. Accidental live trading was previously insufficiently gated

- Files:
  - `trading_system/system/config.py`
  - `main.py`
- Issue:
  - live deployment previously depended on loosely validated config and did not require an explicit account confirmation handshake.
- Why it matters:
  - a misconfigured broker section could route live by mistake.
- Exact recommended fix:
  - implemented:
    - `DeploymentMode`
    - `allow_live_trading`
    - `confirm_live_account`
    - demo-component bans in live mode
    - mode-aware validation through `pydantic`

#### 7. Duplicate order submission across restart was possible

- Files:
  - `trading_system/execution/order_manager.py`
  - `trading_system/execution/journal.py`
  - `trading_system/system/runtime.py`
- Issue:
  - order intent was previously transient and restart-unsafe.
- Why it matters:
  - strategy replay could create duplicate live or paper orders.
- Exact recommended fix:
  - implemented:
    - deterministic client order keys
    - persistent order journal
    - startup reconciliation
    - duplicate suppression in `OrderManager`

#### 8. Stale news could affect decisions long after publication

- Files:
  - `trading_system/system/runtime.py`
- Issue:
  - latest analyzed news state was previously reused without expiry.
- Why it matters:
  - old catalysts could continue biasing trading decisions incorrectly.
- Exact recommended fix:
  - implemented:
    - `news.max_article_age_seconds`
    - stale-news suppression
    - deduplicated due-article activation

#### 9. Config path resolution was brittle

- Files:
  - `main.py`
- Issue:
  - config-relative file resolution assumed a single directory layout.
- Why it matters:
  - smoke runs and alternative deployment layouts could silently break news/journal paths.
- Exact recommended fix:
  - implemented:
    - path resolution now checks config directory, parent, and current working directory

### Low

#### 10. Universe selection still carries survivorship-bias risk unless the caller provides a historical universe

- Files:
  - `trading_system/data/universe.py`
- Issue:
  - the selector operates on caller-provided candidates rather than a historical constituent source.
- Why it matters:
  - backtests can be biased if the candidate list is not historically accurate.
- Exact recommended fix:
  - add a historical universe provider interface and reject production backtests that use static modern universes

## Final Verdict

**paper-trading deployable with fixes**

The codebase is now acceptable for controlled local and paper-trading use, but the open critical and high-severity items still block live deployment.

## Prioritized Remediation Roadmap

1. Remove blocking I/O from the async runtime and isolate model inference/training off the event loop.
2. Add full restart reconciliation:
   persisted session state, broker executions, open orders, and risk baselines.
3. Bring backtest/live assumptions closer:
   spread, borrow, market calendar, and corporate actions.
4. Add production observability:
   metrics, health checks, alerts, and structured logging sinks.
5. Move live secrets and environment management into a dedicated secret/config system.
