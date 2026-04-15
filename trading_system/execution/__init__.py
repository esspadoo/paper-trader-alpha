"""Execution-layer contracts for broker and exchange connectivity."""

from trading_system.execution.broker import BrokerInterface
from trading_system.execution.exceptions import (
    BrokerConnectionError,
    ExecutionDependencyError,
    ExecutionError,
    OrderCancellationError,
    OrderNotFoundError,
    OrderSubmissionError,
)
from trading_system.execution.ibkr import IBKRBroker, IBKRBrokerConfig
from trading_system.execution.journal import OrderJournal, OrderJournalRecord
from trading_system.execution.order_manager import OrderManager
from trading_system.execution.paper import PaperBroker, PaperBrokerConfig

__all__ = [
    "BrokerInterface",
    "ExecutionError",
    "ExecutionDependencyError",
    "BrokerConnectionError",
    "OrderSubmissionError",
    "OrderCancellationError",
    "OrderNotFoundError",
    "IBKRBrokerConfig",
    "IBKRBroker",
    "OrderJournalRecord",
    "OrderJournal",
    "PaperBrokerConfig",
    "PaperBroker",
    "OrderManager",
]
