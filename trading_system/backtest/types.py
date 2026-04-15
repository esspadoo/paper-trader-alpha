"""Typed models for event-driven intraday backtesting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution and metric configuration for intraday simulation."""

    initial_capital: float = 100_000.0
    slippage_bps: float = 3.0
    latency_ms: int = 250
    commission_per_share: float = 0.005
    minimum_commission: float = 1.0
    max_participation_rate: float = 0.10
    bars_per_year: int = 252 * 78

    def __post_init__(self) -> None:
        """Validate the simulation configuration."""

        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be a finite number greater than 0")
        if not math.isfinite(self.slippage_bps) or self.slippage_bps < 0.0:
            raise ValueError("slippage_bps must be a finite number greater than or equal to 0")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be greater than or equal to 0")
        if not math.isfinite(self.commission_per_share) or self.commission_per_share < 0.0:
            raise ValueError("commission_per_share must be a finite number greater than or equal to 0")
        if not math.isfinite(self.minimum_commission) or self.minimum_commission < 0.0:
            raise ValueError("minimum_commission must be a finite number greater than or equal to 0")
        if not math.isfinite(self.max_participation_rate) or not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be between 0 and 1")
        if self.bars_per_year <= 0:
            raise ValueError("bars_per_year must be greater than 0")


@dataclass(slots=True)
class SimulatedOrder:
    """Internal order state tracked by the execution simulator."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    remaining_quantity: float
    limit_price: float
    submitted_at: datetime
    eligible_after: datetime
    stop_loss: float | None = None
    strategy_id: str = ""
    reason: str = ""
    source: str = ""
    correlation_id: Any = None


@dataclass(slots=True)
class OpenPosition:
    """Open position tracked by the portfolio ledger."""

    symbol: str
    quantity: float
    average_entry_price: float
    stop_loss: float | None
    entry_time: datetime
    entry_commission: float = 0.0


@dataclass(frozen=True, slots=True)
class FillRecord:
    """Execution fill emitted by the simulator."""

    timestamp: datetime
    symbol: str
    order_id: str
    side: str
    quantity: float
    price: float
    commission: float
    reason: str
    correlation_id: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fill payload."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "order_id": self.order_id,
            "side": self.side,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "commission": float(self.commission),
            "reason": self.reason,
            "correlation_id": None if self.correlation_id is None else str(self.correlation_id),
        }


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """Closed trade record used for trade-level performance metrics."""

    symbol: str
    side: str
    quantity: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable closed-trade payload."""

        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "gross_pnl": float(self.gross_pnl),
            "net_pnl": float(self.net_pnl),
        }


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Single timestamped equity observation."""

    timestamp: datetime
    equity: float
    cash: float
    gross_exposure: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable equity observation."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": float(self.equity),
            "cash": float(self.cash),
            "gross_exposure": float(self.gross_exposure),
        }


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Performance metrics computed from a completed backtest."""

    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_return: float
    total_commissions: float
    total_trades: int
    win_rate: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable metrics payload."""

        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Completed backtest output with metrics, trades, fills, and equity curve."""

    metrics: BacktestMetrics
    equity_curve: tuple[EquityPoint, ...] = ()
    fills: tuple[FillRecord, ...] = ()
    trades: tuple[ClosedTrade, ...] = ()
    observed_events: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable backtest summary."""

        return {
            "metrics": self.metrics.to_dict(),
            "equity_curve": [point.to_dict() for point in self.equity_curve],
            "fills": [fill.to_dict() for fill in self.fills],
            "trades": [trade.to_dict() for trade in self.trades],
            "observed_events": list(self.observed_events),
            "metadata": dict(self.metadata),
        }
