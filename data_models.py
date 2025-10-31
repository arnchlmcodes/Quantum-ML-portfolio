"""
Data models and containers for the Quantum-ML Portfolio Optimizer.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class MarketData:
    """Container for market data and processed features"""
    symbols: List[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    technical_indicators: Optional[pd.DataFrame] = None
    features: Optional[np.ndarray] = None
    data_quality_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Portfolio:
    """Container for portfolio information"""
    assets: List[str]
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    creation_timestamp: datetime = field(default_factory=datetime.now)
    performance_comparison: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    quantum_portfolio: Optional[Portfolio]
    ml_predictions: Optional[np.ndarray] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, Any] = field(default_factory=dict)
    performance_comparison: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    method_used: str = "unknown"