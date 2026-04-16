# Deployment Guide

## Prerequisites

- Python `3.11`
- local virtualenv with `pip install -e .[system]`
- writable `state/` directory or explicit paths in `[persistence]`
- for IBKR:
  TWS or IB Gateway running on the configured host/port

## Local Run

```bash
python main.py --config config/trading_system.example.toml
```

## Operational Commands

Health:

```bash
python main.py --config config/trading_system.example.toml health
```

Metrics:

```bash
python main.py --config config/trading_system.example.toml metrics
```

Kill switch:

```bash
python main.py --config config/trading_system.example.toml kill-switch status
python main.py --config config/trading_system.example.toml kill-switch engage --reason "manual halt"
python main.py --config config/trading_system.example.toml kill-switch release --reason "operator release"
```

Replay:

```bash
python main.py --config config/trading_system.example.toml replay
```

## Docker

Build:

```bash
docker build -t trading-system .
```

Run:

```bash
docker run --rm -it \
  -v "$(pwd)/state:/app/state" \
  trading-system
```

## Config Guidance

- keep `[runtime].mode = "paper"` until live promotion is explicitly approved
- keep `[execution].allow_live_trading = false` unless the live checklist is complete
- set explicit `[persistence]` paths per environment to avoid state cross-contamination
- use unique `strategy_id` values per deployment

## CI

The repository includes:

- compile verification via `python -m compileall`
- unit/integration tests via `python -m unittest -v`
- lint/type hooks through pre-commit
