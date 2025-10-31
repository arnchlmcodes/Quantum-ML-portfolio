"""
Configuration classes and system configuration management for the Quantum-ML Portfolio Optimizer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class QuantumConfig:
    """Configuration for quantum optimization parameters"""
    qaoa_p_layers: int = 2
    n_assets_select: int = 5
    risk_aversion: float = 1.0
    vqe_ansatz_reps: int = 2
    optimizer: str = 'SPSA'
    maxiter: int = 100
    backend: str = 'qasm_simulator'


@dataclass
class MLConfig:
    """Configuration for machine learning parameters"""
    sequence_length: int = 60
    prediction_horizon: int = 30
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.2
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation parameters"""
    n_simulations: int = 10000
    scenario_length: int = 252
    market_regimes: Dict[str, float] = field(default_factory=lambda: {
        'normal': 0.7, 'bear': 0.2, 'crisis': 0.1
    })
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])


@dataclass
class SystemConfig:
    """Main system configuration with hardcoded parameters"""
    # NSE stock symbols for optimization
    nse_symbols: List[str] = field(default_factory=lambda: [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
        'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS'
    ])
    
    # Data fetching parameters
    data_period: str = '2y'  # 2 years of historical data
    data_interval: str = '1d'  # Daily data
    
    # Risk-free rate (10-year Indian Government Bond yield approximation)
    risk_free_rate: float = 0.07
    
    # Transaction cost assumptions
    transaction_cost: float = 0.001  # 0.1% per transaction
    
    # Rebalancing frequency
    rebalancing_frequency: int = 30  # days
    
    # Configuration objects
    quantum_config: QuantumConfig = field(default_factory=QuantumConfig)
    ml_config: MLConfig = field(default_factory=MLConfig)
    monte_carlo_config: MonteCarloConfig = field(default_factory=MonteCarloConfig)


class ConfigurationManager:
    """Manages system configuration and validation"""
    
    def __init__(self):
        self.config = SystemConfig()
    
    def get_config(self) -> SystemConfig:
        """Get the current system configuration"""
        return self.config
    
    def validate_configuration(self) -> bool:
        """Validate the current configuration"""
        # Validate quantum config
        assert self.config.quantum_config.qaoa_p_layers >= 1
        assert self.config.quantum_config.n_assets_select >= 1
        assert self.config.quantum_config.risk_aversion > 0
        
        # Validate ML config
        assert self.config.ml_config.sequence_length >= 1
        assert self.config.ml_config.prediction_horizon >= 1
        assert 0 < self.config.ml_config.dropout_rate < 1
        
        # Validate Monte Carlo config
        assert self.config.monte_carlo_config.n_simulations >= 1000
        assert self.config.monte_carlo_config.scenario_length >= 1
        
        # Validate market regime probabilities sum to 1
        regime_sum = sum(self.config.monte_carlo_config.market_regimes.values())
        assert 0.99 <= regime_sum <= 1.01
        
        # Validate NSE symbols
        assert len(self.config.nse_symbols) >= self.config.quantum_config.n_assets_select
        
        return True