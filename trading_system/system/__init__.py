"""Integrated application runtime and configuration for the trading system."""

from trading_system.system.config import (
    CriticConfig,
    DecisionConfig,
    DeploymentMode,
    ExecutionConfig,
    LoggingConfig,
    MarketConfig,
    NewsConfig,
    ObservabilityConfig,
    PersistenceConfig,
    RiskConfig,
    RuntimeConfig,
    SafetyConfig,
    SystemConfig,
    load_system_config,
)
from trading_system.system.demo import DemoLocalLLMBackend, build_demo_market_frame, build_market_events
from trading_system.system.replay import load_replay_market_events
from trading_system.system.runtime import IntegratedTradingSystem
from trading_system.system.state import KillSwitchState, RuntimeStateStore

__all__ = [
    "LoggingConfig",
    "MarketConfig",
    "NewsConfig",
    "DecisionConfig",
    "CriticConfig",
    "RiskConfig",
    "DeploymentMode",
    "ExecutionConfig",
    "RuntimeConfig",
    "SafetyConfig",
    "PersistenceConfig",
    "ObservabilityConfig",
    "SystemConfig",
    "load_system_config",
    "DemoLocalLLMBackend",
    "build_demo_market_frame",
    "build_market_events",
    "load_replay_market_events",
    "KillSwitchState",
    "RuntimeStateStore",
    "IntegratedTradingSystem",
]
