"""
Main entry point for the Quantum-ML Hybrid Portfolio Optimization System.

A production-grade system that combines quantum computing algorithms (QAOA/VQE) 
with machine learning (TensorFlow LSTM) to optimize NSE stock portfolios.
"""

from portfolio_optimizer import QuantumMLPortfolioOptimizer


def main():
    """Main entry point for the application"""
    # Initialize the optimizer
    optimizer = QuantumMLPortfolioOptimizer()
    
    # Display system information
    system_info = optimizer.get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Use a subset of symbols for demonstration
    test_symbols = optimizer.config.nse_symbols[:8]  # First 8 symbols for faster execution
    print(f"Using symbols: {test_symbols}")
    
    # Execute the complete integrated pipeline
    optimization_result = optimizer.run_integrated_optimization_pipeline(test_symbols)
    
    # Display final results
    portfolio = optimization_result.quantum_portfolio
    print("\n" + "=" * 50)
    print("FINAL OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Selected Assets: {portfolio.assets}")
    print(f"Portfolio Weights:")
    for asset, weight in zip(portfolio.assets, portfolio.weights):
        print(f"  {asset}: {weight:.1%}")
    print(f"Expected Return: {portfolio.expected_return:.4f}")
    print(f"Volatility: {portfolio.volatility:.4f}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio:.4f}")
    print(f"Total Execution Time: {optimization_result.execution_time:.2f} seconds")
    
    # Display performance comparison if available
    if optimization_result.performance_comparison:
        print("\nPerformance vs Baselines:")
        for baseline, value in optimization_result.performance_comparison.items():
            if isinstance(value, (int, float)) and 'improvement' in baseline:
                print(f"  {baseline}: {value:+.2%}")
            else:
                print(f"  {baseline}: {value}")


if __name__ == "__main__":
    main()