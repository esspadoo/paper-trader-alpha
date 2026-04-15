"""Optional dependency helpers for ML models."""

from __future__ import annotations

from typing import Any

from trading_system.models.exceptions import ModelDependencyError


def require_numpy() -> Any:
    """Return the numpy module or raise a dependency error."""

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModelDependencyError("numpy is required for the model layer. Install the 'ml' extra.") from exc

    return np


def require_pandas() -> Any:
    """Return the pandas module or raise a dependency error."""

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModelDependencyError("pandas is required for the model layer. Install the 'ml' extra.") from exc

    return pd


def require_xgboost() -> Any:
    """Return the xgboost module or raise a dependency error."""

    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModelDependencyError("xgboost is required for the market agent. Install the 'ml' extra.") from exc

    return xgb
