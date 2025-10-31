#!/usr/bin/env python3
"""
Test script for the integrated Quantum-ML Portfolio Optimization Pipeline
"""

import logging
import quantum_portfolio_optimizer as qpo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_integration_pipeline():
    """Test the complete integration pipeline"""
    try:
        print("=" * 60)
        print("TESTING QUANTUM-ML INTEGRATION PIPELINE")
        print("=" * 60)
        
        # Initialize optimizer
        optimizer = qpo.QuantumMLPortfolioOptimizer()
        
        # Test with a small subset for faster execution
        test_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS']
        
        print(f"\nTesting with symbols: {test_symbols}")
        
        # Run the integrated optimization pipeline
        result = optimizer.run_integrated_optimization_pipeline(test_symbols)
        
        # Display results
        if result.quantum_portfolio:
            portfolio = result.quantum_portfolio
            print("\n" + "=" * 50)
            print("INTEGRATION TEST RESULTS")
            print("=" * 50)
            print(f"✓ Pipeline completed successfully")
            print(f"✓ Selected Assets: {portfolio.assets}")
            print(f"✓ Portfolio Weights:")
            for asset, weight in zip(portfolio.assets, portfolio.weights):
                print(f"    {asset}: {weight:.1%}")
            print(f"✓ Expected Return: {portfolio.expected_return:.4f}")
            print(f"✓ Volatility: {portfolio.volatility:.4f}")
            print(f"✓ Sharpe Ratio: {portfolio.sharpe_ratio:.4f}")
            print(f"✓ Execution Time: {result.execution_time:.2f} seconds")
            print(f"✓ Method Used: {result.method_used}")
            
            # Display risk metrics if available
            if result.risk_metrics:
                print(f"\n✓ Risk Metrics:")
                for metric, value in result.risk_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")
            
            # Display performance comparison if available
            if result.performance_comparison:
                print(f"\n✓ Performance vs Baselines:")
                for baseline, metrics in result.performance_comparison.items():
                    if isinstance(metrics, dict) and 'sharpe_improvement' in metrics:
                        improvement = metrics['sharpe_improvement']
                        print(f"    vs {baseline}: {improvement:+.2%} Sharpe improvement")
            
            print("\n✓ Integration pipeline test completed successfully!")
            return True
            
        else:
            print("✗ No portfolio generated from integration pipeline")
            return False
            
    except Exception as e:
        print(f"✗ Integration pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_integration_pipeline()
    if success:
        print("\n🎉 All integration tests passed!")
    else:
        print("\n❌ Integration tests failed!")