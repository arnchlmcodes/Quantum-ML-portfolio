# QML Stock Predictor

A **Quantum Machine Learning (QML)** based stock price prediction system that combines quantum computing principles with classical machine learning to forecast NSE (National Stock Exchange) stock prices.

##  What It Does

This project implements a **hybrid quantum-classical model** that:

-  **Fetches real stock data** from NSE using Yahoo Finance API
-  **Calculates technical indicators** (RSI, MACD, Bollinger Bands, Moving Averages, etc.)
-  **Uses quantum circuits** for feature processing with variational quantum algorithms
-  **Combines quantum and classical layers** for hybrid machine learning
-  **Predicts future stock prices** with comprehensive performance metrics
-  **Provides detailed visualizations** of predictions vs actual prices


## Installation

```bash
# Install required packages
pip install pennylane yfinance pandas numpy scikit-learn matplotlib seaborn

# Clone or download the script
# Run the main function
python qml_stock_predictor.py
```

## Quick Start

```python
# Basic usage
python qml_stock_predictor.py

# The script will:
# 1. Fetch RELIANCE stock data from 2020-2024
# 2. Calculate technical indicators
# 3. Train quantum model for 30 epochs
# 4. Show predictions and performance metrics
# 5. Generate comprehensive visualizations
```

## Configuration

### Stock Selection
```python
symbol = "RELIANCE"  # Change to any NSE stock
# Examples: "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"
```

### Model Parameters
```python
n_qubits = 6        # Number of quantum qubits (2-8 recommended)
n_layers = 4        # Quantum circuit depth (3-6 recommended)  
epochs = 30         # Training epochs (20-100 recommended)
learning_rate = 0.01 # Learning rate (0.001-0.1 recommended)
```

### Date Range
```python
start_date = datetime(2020, 1, 1)  # Training start date
end_date = datetime(2024, 1, 1)    # Training end date
```


