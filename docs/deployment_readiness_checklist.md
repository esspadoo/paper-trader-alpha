# Deployment Readiness Checklist

## Current State

- [x] Validated config loading with `pydantic`
- [x] Explicit deployment mode: `dev`, `paper`, `live`
- [x] Live trading requires `allow_live_trading = true`
- [x] Live trading requires `confirm_live_account == execution.account`
- [x] Demo news components are blocked in live mode
- [x] Persistent order journal prevents duplicate submission across restart
- [x] Startup reconciliation refreshes broker order state
- [x] Stale news expiry is enforced before decisions are made
- [x] End-to-end smoke test covers `main.py`

## Paper Trading Gate

- [x] Use `runtime.mode = "paper"` or `runtime.mode = "dev"`
- [x] Use `execution.broker = "paper"` or `execution.paper_trading = true`
- [x] Confirm `execution.journal_path` points to writable durable storage
- [x] Confirm `news.max_article_age_seconds` is set for the intended workflow
- [x] Run `.venv/bin/python -m unittest -v`
- [x] Run `.venv/bin/python -u main.py --config config/trading_system.example.toml`

## Live Trading Gate

- [ ] Replace synchronous local fetch/inference paths with bounded executors or sidecar services
- [ ] Persist and reconcile session equity baselines and broker executions on restart
- [ ] Add health checks, metrics, and alerting
- [ ] Add market calendar enforcement and live session guards
- [ ] Add structured secret management for live credentials
- [ ] Validate production runbooks for broker disconnect, reconnect, and stale-order cleanup
- [ ] Validate live market-data and news providers under production load
- [ ] Add explicit broker/account whitelisting in deployment automation

## Operational Preflight

- [ ] Confirm the exact broker account matches the intended environment
- [ ] Confirm live config does not reference demo or static JSON news sources
- [ ] Confirm log retention and redaction policies are in place
- [ ] Confirm order journal storage is backed up and rotated appropriately
- [ ] Confirm production host time is synchronized and monitored

## Backtest Parity Before Live

- [ ] Add spread modeling
- [ ] Add short borrow constraints if shorting is enabled
- [ ] Add exchange calendar/session handling
- [ ] Add auction/open-close behavior
- [ ] Add corporate actions normalization or hard rejection
