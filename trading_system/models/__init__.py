"""Model contracts, feature engineering, and ML implementations."""

from trading_system.models.base import BaseModel
from trading_system.models.exceptions import (
    FeatureEngineeringError,
    LLMBackendError,
    LLMTimeoutError,
    ModelDependencyError,
    ModelLayerError,
    ModelNotFittedError,
    StrictJSONParseError,
    TrainingDataError,
)
from trading_system.models.features import MarketFeatureConfig, MarketFeatureEngineer
from trading_system.models.news_llm import (
    LlamaCppBackend,
    LocalLLMBackend,
    LocalNewsLLMAnalyzer,
    NewsAnalysis,
    NewsPromptTemplate,
    VLLMBackend,
)
from trading_system.models.xgboost_return import TrainingSummary, XGBoostReturnModel, XGBoostReturnModelConfig

__all__ = [
    "BaseModel",
    "ModelLayerError",
    "ModelDependencyError",
    "ModelNotFittedError",
    "FeatureEngineeringError",
    "TrainingDataError",
    "LLMBackendError",
    "LLMTimeoutError",
    "StrictJSONParseError",
    "MarketFeatureConfig",
    "MarketFeatureEngineer",
    "NewsAnalysis",
    "NewsPromptTemplate",
    "LocalLLMBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    "LocalNewsLLMAnalyzer",
    "TrainingSummary",
    "XGBoostReturnModelConfig",
    "XGBoostReturnModel",
]
