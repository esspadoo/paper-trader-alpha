"""Portfolio accounting and performance metrics for intraday backtests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime
from typing import TYPE_CHECKING

from trading_system.backtest.types import BacktestConfig, BacktestMetrics, ClosedTrade, EquityPoint, FillRecord, OpenPosition
from trading_system.models.dependencies import require_pandas

if TYPE_CHECKING:
    import pandas as pd


class PortfolioLedger:
    """Track positions, cash, fills, trades, and marked-to-market equity."""

    def __init__(self, *, initial_capital: float) -> None:
        """Initialize the ledger with starting cash and no positions."""

        if not math.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("initial_capital must be a finite number greater than 0")

        self._cash = float(initial_capital)
        self._positions: dict[str, OpenPosition] = {}
        self._fills: list[FillRecord] = []
        self._trades: list[ClosedTrade] = []
        self._equity_curve: list[EquityPoint] = []
        self._current_prices: dict[str, float] = {}
        self._daily_start_equity: dict[date, float] = {}
        self._total_commissions = 0.0
        self._realized_pnl = 0.0

    @property
    def cash(self) -> float:
        """Return the current cash balance."""

        return self._cash

    @property
    def total_commissions(self) -> float:
        """Return the cumulative commissions paid."""

        return self._total_commissions

    @property
    def realized_pnl(self) -> float:
        """Return cumulative realized PnL from closed trades."""

        return self._realized_pnl

    @property
    def fills(self) -> tuple[FillRecord, ...]:
        """Return the immutable fill history."""

        return tuple(self._fills)

    @property
    def trades(self) -> tuple[ClosedTrade, ...]:
        """Return the immutable closed-trade history."""

        return tuple(self._trades)

    @property
    def equity_curve(self) -> tuple[EquityPoint, ...]:
        """Return the immutable equity curve."""

        return tuple(self._equity_curve)

    def positions(self) -> Mapping[str, OpenPosition]:
        """Return the current open positions keyed by symbol."""

        return dict(self._positions)

    def current_marks(self) -> Mapping[str, float]:
        """Return the latest marked prices keyed by symbol."""

        return dict(self._current_prices)

    def current_price(self, symbol: str) -> float:
        """Return the latest marked price for a symbol."""

        return self._price_for(symbol, self._current_prices)

    def position_for(self, symbol: str) -> OpenPosition | None:
        """Return the current open position for a symbol when present."""

        return self._positions.get(symbol.strip().upper())

    def has_open_position(self, symbol: str) -> bool:
        """Return whether a symbol currently has open exposure."""

        return self.position_for(symbol) is not None

    def record_fill(self, fill: FillRecord, *, stop_loss: float | None = None) -> None:
        """Apply a fill to cash, positions, and realized PnL."""

        symbol = fill.symbol.strip().upper()
        side = fill.side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("fill.side must be BUY or SELL")

        quantity = float(fill.quantity)
        price = float(fill.price)
        commission = float(fill.commission)
        if not math.isfinite(quantity) or quantity <= 0.0:
            raise ValueError("fill.quantity must be greater than 0")
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError("fill.price must be greater than 0")
        if not math.isfinite(commission) or commission < 0.0:
            raise ValueError("fill.commission must be greater than or equal to 0")

        signed_quantity = quantity if side == "BUY" else -quantity
        notional = quantity * price
        if side == "BUY":
            self._cash -= notional + commission
        else:
            self._cash += notional - commission

        self._total_commissions += commission
        self._current_prices[symbol] = price
        self._fills.append(fill)

        current = self._positions.get(symbol)
        if current is None:
            self._positions[symbol] = OpenPosition(
                symbol=symbol,
                quantity=signed_quantity,
                average_entry_price=price,
                stop_loss=stop_loss,
                entry_time=fill.timestamp,
                entry_commission=commission,
            )
            return

        current_sign = 1.0 if current.quantity > 0.0 else -1.0
        fill_sign = 1.0 if signed_quantity > 0.0 else -1.0
        if current_sign == fill_sign:
            total_quantity = abs(current.quantity) + quantity
            average_entry_price = (
                (abs(current.quantity) * current.average_entry_price) + (quantity * price)
            ) / total_quantity
            self._positions[symbol] = replace(
                current,
                quantity=current.quantity + signed_quantity,
                average_entry_price=average_entry_price,
                stop_loss=stop_loss if stop_loss is not None else current.stop_loss,
                entry_commission=current.entry_commission + commission,
            )
            return

        self._apply_offsetting_fill(current, fill=fill, signed_quantity=signed_quantity, stop_loss=stop_loss)

    def mark_to_market(self, timestamp: datetime, prices: Mapping[str, float]) -> EquityPoint:
        """Update marks and append a new equity observation."""

        for symbol, price in prices.items():
            symbol_key = str(symbol).strip().upper()
            numeric_price = float(price)
            if not math.isfinite(numeric_price) or numeric_price <= 0.0:
                raise ValueError(f"price for {symbol_key} must be a finite number greater than 0")
            self._current_prices[symbol_key] = numeric_price

        equity = self.equity(prices=self._current_prices)
        gross_exposure = self.gross_exposure(prices=self._current_prices)
        point = EquityPoint(timestamp=timestamp, equity=equity, cash=self._cash, gross_exposure=gross_exposure)
        self._equity_curve.append(point)
        self._daily_start_equity.setdefault(timestamp.date(), equity)
        return point

    def equity(self, *, prices: Mapping[str, float] | None = None) -> float:
        """Return the marked-to-market equity."""

        marks = self._current_prices if prices is None else prices
        position_value = 0.0
        for symbol, position in self._positions.items():
            position_value += position.quantity * self._price_for(symbol, marks)
        return self._cash + position_value

    def gross_exposure(self, *, prices: Mapping[str, float] | None = None) -> float:
        """Return the current absolute gross exposure."""

        marks = self._current_prices if prices is None else prices
        return sum(abs(position.quantity * self._price_for(symbol, marks)) for symbol, position in self._positions.items())

    def account_state(self, *, timestamp: datetime) -> dict[str, float]:
        """Return the current normalized account state for risk management."""

        equity = self.equity()
        start_equity = self._daily_start_equity.get(timestamp.date(), equity)
        daily_loss = max(start_equity - equity, 0.0)
        return {
            "capital": equity,
            "daily_loss": daily_loss,
            "gross_exposure": self.gross_exposure(),
        }

    def equity_frame(self) -> "pd.DataFrame":
        """Return the equity curve as a pandas DataFrame indexed by timestamp."""

        pd = require_pandas()
        records = [point.to_dict() for point in self._equity_curve]
        if not records:
            return pd.DataFrame(columns=["equity", "cash", "gross_exposure"])
        frame = pd.DataFrame(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()
        if frame.index.has_duplicates:
            frame = frame.groupby(level=0, sort=True).last()
        return frame

    def compute_metrics(self, *, config: BacktestConfig) -> BacktestMetrics:
        """Compute Sharpe, drawdown, and profit factor from ledger history."""

        equity_frame = self.equity_frame()
        if equity_frame.empty:
            return BacktestMetrics(
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                total_return=0.0,
                total_commissions=self._total_commissions,
                total_trades=0,
                win_rate=0.0,
            )

        returns = equity_frame["equity"].pct_change().fillna(0.0)
        volatility = float(returns.std(ddof=0))
        sharpe_ratio = 0.0 if volatility <= 0.0 else float(math.sqrt(config.bars_per_year) * returns.mean() / volatility)

        running_max = equity_frame["equity"].cummax()
        drawdowns = (equity_frame["equity"] / running_max) - 1.0
        max_drawdown = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0

        gross_profit = sum(max(trade.net_pnl, 0.0) for trade in self._trades)
        gross_loss = sum(min(trade.net_pnl, 0.0) for trade in self._trades)
        if gross_loss == 0.0:
            profit_factor = float("inf") if gross_profit > 0.0 else 0.0
        else:
            profit_factor = float(gross_profit / abs(gross_loss))

        ending_equity = float(equity_frame["equity"].iloc[-1])
        starting_equity = float(equity_frame["equity"].iloc[0])
        total_return = (ending_equity / starting_equity) - 1.0 if starting_equity > 0.0 else 0.0
        winning_trades = sum(1 for trade in self._trades if trade.net_pnl > 0.0)
        total_trades = len(self._trades)
        win_rate = 0.0 if total_trades == 0 else float(winning_trades / total_trades)

        return BacktestMetrics(
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            total_return=float(total_return),
            total_commissions=float(self._total_commissions),
            total_trades=total_trades,
            win_rate=win_rate,
        )

    def _apply_offsetting_fill(
        self,
        current: OpenPosition,
        *,
        fill: FillRecord,
        signed_quantity: float,
        stop_loss: float | None,
    ) -> None:
        """Apply a fill that reduces, closes, or reverses an existing position."""

        symbol = current.symbol
        current_quantity = abs(current.quantity)
        fill_quantity = abs(signed_quantity)
        closing_quantity = min(current_quantity, fill_quantity)

        entry_commission_alloc = current.entry_commission * (closing_quantity / current_quantity)
        exit_commission_alloc = fill.commission * (closing_quantity / fill_quantity)
        direction = 1.0 if current.quantity > 0.0 else -1.0
        gross_pnl = (fill.price - current.average_entry_price) * closing_quantity * direction
        net_pnl = gross_pnl - entry_commission_alloc - exit_commission_alloc

        self._realized_pnl += net_pnl
        self._trades.append(
            ClosedTrade(
                symbol=symbol,
                side="LONG" if current.quantity > 0.0 else "SHORT",
                quantity=closing_quantity,
                entry_time=current.entry_time,
                exit_time=fill.timestamp,
                entry_price=current.average_entry_price,
                exit_price=fill.price,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
            )
        )

        residual_quantity = current.quantity + signed_quantity
        residual_entry_commission = current.entry_commission - entry_commission_alloc
        opening_commission = fill.commission - exit_commission_alloc

        if residual_quantity == 0.0:
            self._positions.pop(symbol, None)
            return

        if (current.quantity > 0.0 and residual_quantity > 0.0) or (current.quantity < 0.0 and residual_quantity < 0.0):
            self._positions[symbol] = replace(
                current,
                quantity=residual_quantity,
                entry_commission=residual_entry_commission,
            )
            return

        self._positions[symbol] = OpenPosition(
            symbol=symbol,
            quantity=residual_quantity,
            average_entry_price=fill.price,
            stop_loss=stop_loss,
            entry_time=fill.timestamp,
            entry_commission=opening_commission,
        )

    def _price_for(self, symbol: str, prices: Mapping[str, float]) -> float:
        """Return the current mark price for a symbol."""

        symbol_key = symbol.strip().upper()
        if symbol_key not in prices:
            raise KeyError(f"missing mark price for {symbol_key}")
        return float(prices[symbol_key])
