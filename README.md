# Quantum-ML Hybrid Portfolio Optimization System

A **Quantum Machine Learning (QML)** hybrid system that combines Qiskit's **QAOA/VQE quantum algorithms** with **TensorFlow deep learning** to optimize and diversify NSE stock portfolios.

## What It Does

This project implements a **production-grade quantum-classical hybrid system** that:

### Quantum Optimization Layer
- **QAOA (Quantum Approximate Optimization Algorithm)** for combinatorial asset selection
- **VQE (Variational Quantum Eigensolver)** for continuous portfolio weight optimization  
- **Quantum circuits** with parameterized gates for portfolio optimization problems
- **Fallback mechanisms** to classical optimization when quantum fails

### Machine Learning Integration
- **Enhanced LSTM networks** with attention mechanisms for return prediction
- **TensorFlow deep learning** models for market regime detection
- **Multi-scale feature processing** with technical indicators
- **Time series forecasting** for portfolio rebalancing signals

### Risk Management & Stress Testing
- **Monte Carlo simulations** (10,000+ scenarios) for comprehensive stress testing
- **Multiple market regimes**: Normal (70%), Bear (20%), Crisis (10%) scenarios
- **Advanced risk metrics**: VaR, CVaR, Maximum Drawdown, Sharpe ratios
- **Stress test reporting** with probability distributions and tail risk analysis



## Project Structure

The system is now organized into modular components for better maintainability:

```
quantum-portfolio-optimizer/
├── main.py                    # Main entry point
├── config.py                  # Configuration classes and management
├── data_models.py             # Data containers and models
├── exceptions.py              # Custom exception classes
├── data_fetcher.py            # NSE data fetching and preprocessing
├── quantum_optimization.py    # Quantum algorithms (QAOA/VQE)
├── ml_predictor.py           # LSTM with attention mechanism
├── risk_analysis.py          # Risk metrics and Monte Carlo testing
├── baseline_comparator.py    # Baseline portfolio strategies
├── portfolio_optimizer.py   # Main orchestration class
├── test_integration.py       # Integration tests
└── README.md                 # This file
```

##  Installation

```bash
# Create virtual environment
python -m venv qiskit_env
source qiskit_env/bin/activate  # On Windows: qiskit_env\Scripts\activate

# Install quantum computing packages
pip install qiskit qiskit-algorithms qiskit-optimization

# Install machine learning packages
pip install tensorflow pandas numpy scikit-learn

# Install data and visualization packages
pip install yfinance matplotlib seaborn plotly cvxpy scipy

# Run the system
python main.py
```

##  Quick Start

```python
# Run the complete quantum-ML optimization system
python main.py

# The system will:
# 1. Fetch NSE stock data (RELIANCE, TCS, HDFCBANK, etc.)
# 2. Run QAOA for quantum asset selection
# 3. Use VQE for optimal weight determination
# 4. Execute Monte Carlo stress testing (10,000+ scenarios)
# 5. Compare performance vs classical baselines
# 6. Generate comprehensive performance reports
```

##  System Architecture

### 1. Main Portfolio Optimizer
```python
from portfolio_optimizer import QuantumMLPortfolioOptimizer

# Initialize the system
optimizer = QuantumMLPortfolioOptimizer()

# Run complete optimization pipeline
result = optimizer.run_integrated_optimization_pipeline()
```

### 2. Quantum Asset Selection
```python
from quantum_optimization import QuantumAssetSelector
from config import QuantumConfig

# QAOA for asset selection
selector = QuantumAssetSelector(QuantumConfig())
selected_assets, probs = selector.run_qaoa_optimization(returns_data)
```

### 3.LSTM Predictor
```python
from ml_predictor import EnhancedLSTMPredictor
from config import MLConfig

# LSTM with attention mechanism
predictor = EnhancedLSTMPredictor(MLConfig())
history = predictor.train(returns_data)
predictions = predictor.predict(recent_data)
```

### 4. Monte Carlo Stress Testing
```python
from risk_analysis import MonteCarloStressTester
from config import MonteCarloConfig

# Comprehensive stress testing
tester = MonteCarloStressTester(MonteCarloConfig())
results = tester.stress_test_portfolio(assets, weights, returns_data)
```

