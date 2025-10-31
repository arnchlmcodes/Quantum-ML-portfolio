"""
Main Quantum-ML Portfolio Optimizer class that orchestrates all components.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import ConfigurationManager
from data_models import MarketData, Portfolio, OptimizationResult
from data_fetcher import DataFetcher
from quantum_optimization import QuantumAssetSelector, QuantumWeightOptimizer
from ml_predictor import EnhancedLSTMPredictor
from risk_analysis import MonteCarloStressTester
from baseline_comparator import BaselinePortfolioComparator
from exceptions import (
    DataFetchError, QuantumExecutionError, MLTrainingError, 
    QuantumMLOptimizationError
)

logger = logging.getLogger(__name__)


class QuantumMLPortfolioOptimizer:
    """Main class for the Quantum-ML Hybrid Portfolio Optimization System"""
    
    def __init__(self):
        """Initialize the portfolio optimizer with default configuration"""
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_config()
        
        # Validate configuration on initialization
        if not self.config_manager.validate_configuration():
            raise ValueError("Invalid configuration detected")
        
        # Initialize components
        self.data_fetcher = DataFetcher(self.config)
        self.quantum_asset_selector = QuantumAssetSelector(self.config.quantum_config)
        self.quantum_weight_optimizer = QuantumWeightOptimizer(self.config.quantum_config)
        self.lstm_predictor = EnhancedLSTMPredictor(self.config.ml_config)
        self.monte_carlo_tester = MonteCarloStressTester(self.config.monte_carlo_config)
        self.baseline_comparator = BaselinePortfolioComparator(self.config)
        self.market_data: Optional[MarketData] = None
        
        logger.info("Quantum-ML Portfolio Optimizer initialized successfully")
        self.config_manager.log_configuration()
    
    def fetch_market_data(self, symbols: List[str] = None) -> MarketData:
        """
        Fetch and preprocess market data for specified symbols
        
        Args:
            symbols: List of symbols to fetch (default: use config symbols)
            
        Returns:
            MarketData object with processed data
        """
        if symbols is None:
            symbols = self.config.nse_symbols
        
        logger.info(f"Fetching market data for {len(symbols)} symbols")
        
        try:
            self.market_data = self.data_fetcher.preprocess_data(symbols)
            logger.info("Market data fetched and processed successfully")
            return self.market_data
            
        except DataFetchError as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise
    
    def get_market_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the current market data
        
        Returns:
            Dictionary with market data summary
        """
        if self.market_data is None:
            return {'status': 'No market data available'}
        
        returns = self.market_data.returns
        
        summary = {
            'status': 'Available',
            'symbols': self.market_data.symbols,
            'n_symbols': len(self.market_data.symbols),
            'data_period': f"{returns.index[0].date()} to {returns.index[-1].date()}",
            'n_observations': len(returns),
            'data_quality_score': f"{self.market_data.data_quality_score:.2%}",
            'last_updated': self.market_data.last_updated.isoformat(),
            'returns_summary': {
                'mean_daily_return': returns.mean().mean(),
                'volatility': returns.std().mean(),
                'min_return': returns.min().min(),
                'max_return': returns.max().max()
            }
        }
        
        return summary
    
    def run_asset_selection(self, symbols: List[str] = None) -> Tuple[List[str], np.ndarray]:
        """
        Run quantum asset selection on available market data
        
        Args:
            symbols: List of symbols to select from (default: use current market data)
            
        Returns:
            Tuple of (selected_assets, selection_probabilities)
        """
        if symbols is not None:
            # Fetch data for specified symbols
            market_data = self.fetch_market_data(symbols)
        elif self.market_data is not None:
            market_data = self.market_data
        else:
            raise ValueError("No market data available. Fetch data first or provide symbols.")
        
        logger.info(f"Running asset selection on {len(market_data.returns.columns)} assets")
        
        try:
            selected_assets, selection_probs = self.quantum_asset_selector.run_qaoa_optimization(
                market_data.returns
            )
            
            logger.info(f"Asset selection completed. Selected {len(selected_assets)} assets.")
            return selected_assets, selection_probs
            
        except Exception as e:
            logger.error(f"Asset selection failed: {e}")
            raise QuantumExecutionError(f"Asset selection failed: {e}")
    
    def run_weight_optimization(self, selected_assets: List[str]) -> np.ndarray:
        """
        Run quantum weight optimization for selected assets
        
        Args:
            selected_assets: List of selected asset symbols
            
        Returns:
            Optimized portfolio weights
        """
        if self.market_data is None:
            raise ValueError("No market data available. Fetch data first.")
        
        logger.info(f"Running weight optimization for {len(selected_assets)} assets")
        
        try:
            weights = self.quantum_weight_optimizer.run_vqe_optimization(
                self.market_data.returns, selected_assets
            )
            
            logger.info(f"Weight optimization completed")
            return weights
            
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            raise QuantumExecutionError(f"Weight optimization failed: {e}")
    
    def run_quantum_optimization(self, symbols: List[str] = None) -> OptimizationResult:
        """
        Run complete quantum optimization pipeline (asset selection + weight optimization)
        
        Args:
            symbols: List of symbols to optimize (default: use current market data)
            
        Returns:
            OptimizationResult with quantum portfolio and metrics
        """
        start_time = time.time()
        logger.info("Starting complete quantum optimization pipeline")
        
        try:
            # Step 1: Asset Selection using QAOA
            selected_assets, selection_probs = self.run_asset_selection(symbols)
            
            # Step 2: Weight Optimization using VQE
            weights = self.run_weight_optimization(selected_assets)
            
            # Step 3: Calculate portfolio metrics
            asset_returns = self.market_data.returns[selected_assets]
            expected_return = np.dot(weights, asset_returns.mean())
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(asset_returns.cov(), weights)))
            
            # Calculate Sharpe ratio
            excess_return = expected_return - self.config.risk_free_rate / 252  # Daily risk-free rate
            sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Create portfolio object
            portfolio = Portfolio(
                assets=selected_assets,
                weights=weights,
                expected_return=expected_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio
            )
            
            # Run Monte Carlo stress testing on the optimized portfolio
            logger.info("Running Monte Carlo stress testing on optimized portfolio...")
            try:
                stress_test_results = self.monte_carlo_tester.stress_test_portfolio(
                    selected_assets, weights, self.market_data.returns
                )
            except Exception as e:
                logger.warning(f"Stress testing failed: {e}")
                stress_test_results = {}
            
            # Create optimization result
            execution_time = time.time() - start_time
            result = OptimizationResult(
                quantum_portfolio=portfolio,
                stress_test_results=stress_test_results,
                execution_time=execution_time,
                method_used="quantum_qaoa_vqe"
            )
            
            logger.info(f"Quantum optimization completed in {execution_time:.2f} seconds")
            logger.info(f"Portfolio Sharpe ratio: {sharpe_ratio:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Quantum optimization pipeline failed: {e}")
            raise QuantumExecutionError(f"Quantum optimization pipeline failed: {e}")
    
    def run_baseline_comparison(self, quantum_portfolio: Portfolio = None, 
                              symbols: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive baseline comparison for a quantum portfolio
        
        Args:
            quantum_portfolio: Portfolio to compare (if None, runs optimization first)
            symbols: List of symbols to use (default: use config symbols)
            
        Returns:
            Comprehensive comparison results
        """
        logger.info("Starting baseline portfolio comparison")
        
        try:
            # Ensure we have market data
            if self.market_data is None:
                if symbols is None:
                    symbols = self.config.nse_symbols
                self.fetch_market_data(symbols)
            
            # If no portfolio provided, run optimization first
            if quantum_portfolio is None:
                logger.info("No portfolio provided, running quantum optimization first...")
                optimization_result = self.run_quantum_optimization(symbols)
                if optimization_result.quantum_portfolio is None:
                    raise QuantumMLOptimizationError("Failed to generate quantum portfolio for comparison")
                quantum_portfolio = optimization_result.quantum_portfolio
            
            # Run baseline comparison
            comparison_results = self.baseline_comparator.compare_portfolios(
                quantum_portfolio, 
                self.market_data.returns
            )
            
            # Add comparison results to quantum portfolio's performance_comparison field
            if 'summary' in comparison_results:
                summary = comparison_results['summary']
                quantum_portfolio.performance_comparison.update({
                    'sharpe_improvement_vs_best': summary.get('sharpe_improvement_percentage', 0),
                    'meets_target_improvement': summary.get('meets_target_improvement', False),
                    'best_baseline': summary.get('best_baseline', 'unknown'),
                    'successful_baselines': summary.get('successful_baselines', 0)
                })
            
            # Log key comparison results
            logger.info("=== Baseline Comparison Results ===")
            if 'summary' in comparison_results:
                summary = comparison_results['summary']
                if 'sharpe_improvement_percentage' in summary:
                    improvement = summary['sharpe_improvement_percentage']
                    target_met = summary.get('meets_target_improvement', False)
                    logger.info(f"Sharpe Ratio Improvement: {improvement:.1f}%")
                    logger.info(f"Meets 44% Target: {'YES' if target_met else 'NO'}")
                
                successful = summary.get('successful_baselines', 0)
                total = summary.get('total_baselines', 0)
                logger.info(f"Successful Baselines: {successful}/{total}")
            
            logger.info("Baseline comparison completed successfully")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Baseline comparison failed: {e}")
            raise QuantumMLOptimizationError(f"Baseline comparison failed: {e}")
    
    def run_integrated_optimization_pipeline(self, symbols: List[str] = None) -> OptimizationResult:
        """
        Run the complete integrated optimization pipeline
        
        Args:
            symbols: List of symbols to optimize (default: use config symbols)
            
        Returns:
            Complete optimization result with all analyses
        """
        start_time = time.time()
        logger.info("Starting integrated quantum-ML optimization pipeline")
        
        try:
            # Step 1: Fetch market data
            if symbols is None:
                symbols = self.config.nse_symbols
            
            logger.info("Step 1: Fetching and preprocessing market data...")
            self.fetch_market_data(symbols)
            
            # Step 2: Run quantum optimization
            logger.info("Step 2: Running quantum portfolio optimization...")
            optimization_result = self.run_quantum_optimization(symbols)
            
            if optimization_result.quantum_portfolio is None:
                raise QuantumMLOptimizationError("Quantum optimization failed to produce a portfolio")
            
            # Step 3: Run baseline comparison
            logger.info("Step 3: Running baseline portfolio comparison...")
            try:
                comparison_results = self.run_baseline_comparison(optimization_result.quantum_portfolio)
                optimization_result.performance_comparison.update(comparison_results.get('summary', {}))
            except Exception as e:
                logger.warning(f"Baseline comparison failed: {e}")
            
            # Step 4: Generate comprehensive report
            logger.info("Step 4: Generating comprehensive analysis report...")
            
            # Update execution time
            total_execution_time = time.time() - start_time
            optimization_result.execution_time = total_execution_time
            
            logger.info(f"Integrated optimization pipeline completed in {total_execution_time:.2f} seconds")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Integrated optimization pipeline failed: {e}")
            raise QuantumMLOptimizationError(f"Integrated optimization pipeline failed: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status"""
        # Import here to avoid circular imports
        try:
            from quantum_optimization import QUANTUM_AVAILABLE
        except ImportError:
            QUANTUM_AVAILABLE = False
        
        return {
            'system_name': 'Quantum-ML Hybrid Portfolio Optimization System',
            'version': '1.0.0',
            'initialized': True,
            'config_valid': self.config_manager.validate_configuration(),
            'n_symbols': len(self.config.nse_symbols),
            'quantum_backend': self.config.quantum_config.backend,
            'quantum_available': QUANTUM_AVAILABLE,
            'market_data_available': self.market_data is not None,
            'timestamp': datetime.now().isoformat()
        }