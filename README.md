# Trading System

Production-grade Python package skeleton for an intraday multi-agent trading system.

## Package Layout

```text
trading_system/
├── __init__.py
├── agents/
│   ├── __init__.py
│   └── base.py
├── backtest/
│   └── __init__.py
├── core/
│   ├── __init__.py
│   ├── bus.py
│   ├── engine.py
│   ├── events.py
│   ├── example.py
│   └── exceptions.py
├── data/
│   ├── __init__.py
│   ├── base.py
│   ├── cache.py
│   ├── example.py
│   ├── exceptions.py
│   ├── market_data.py
│   ├── schemas.py
│   ├── universe.py
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── ibkr.py
│       ├── polygon.py
│       └── yfinance.py
├── execution/
│   ├── __init__.py
│   └── broker.py
├── infra/
│   └── __init__.py
└── models/
    ├── __init__.py
    └── base.py
```

## Core Runtime

The `trading_system.core` package now includes:

- Immutable event types: `MarketEvent`, `NewsEvent`, `SignalEvent`, `OrderEvent`
- A queue-backed `EventBus` built on `asyncio`
- An `Engine` lifecycle wrapper for startup, shutdown, and idle waits
- A runnable example flow in `trading_system/core/example.py`

## Market Data Layer

The `trading_system.data` package now includes:

- `MarketDataStream` for async OHLCV retrieval with transparent caching
- `UniverseSelector` for daily top-stock selection using price, volume, and volatility filters
- `YFinanceMarketDataProvider` for development use
- `PolygonMarketDataProvider` and `IBKRMarketDataProvider` scaffolds for production adapters
- Optional dependencies under the `market-data` extra: `pandas` and `yfinance`

## Integrated Runtime

The repo now includes an end-to-end async runtime under `trading_system.system` and a runnable root entrypoint in `main.py`.

Run the demo pipeline with:

```bash
python3 main.py --config config/trading_system.example.toml
```

The example config uses:

- Synthetic 5-minute market bars generated locally
- A local JSON news feed at `examples/demo_news.json`
- A deterministic local demo backend for news analysis
- The in-memory paper broker for execution

## Operations

Health:

```bash
python main.py --config config/trading_system.example.toml health
```

Metrics:

```bash
python main.py --config config/trading_system.example.toml metrics
```

Replay:

```bash
python main.py --config config/trading_system.example.toml replay
```

Kill switch:

```bash
python main.py --config config/trading_system.example.toml kill-switch status
python main.py --config config/trading_system.example.toml kill-switch engage --reason "manual halt"
python main.py --config config/trading_system.example.toml kill-switch release --reason "operator release"
```

## Documentation

- [Deployment Guide](docs/deployment_guide.md)
- [Runbook](docs/runbook.md)
- [Paper Trading Checklist](docs/paper_trading_checklist.md)
- [Live Trading Checklist](docs/live_trading_checklist.md)
- [Platform Gap Assessment](docs/platform_gap_assessment.md)
