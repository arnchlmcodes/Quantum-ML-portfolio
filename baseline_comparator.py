"""
Baseline portfolio strategies for performance comparison.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime

from config import SystemConfig
from data_models import Portfolio
from exceptions import QuantumMLOptimizationError

logger = logging.getLogger(__name__)


class BaselinePortfolioComparator:
    """Implements baseline portfolio strategies for performance comparison"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.risk_free_rate = config.risk_free_rate
        self.transaction_cost = config.transaction_cost
        logger.info("BaselinePortfolioComparator initialized")
    
    def create_equal_weight_portfolio(self, assets: List[str], returns_data: pd.DataFrame) -> Portfolio:
        """
        Create equal-weight portfolio baseline
        
        Args:
            assets: List of asset symbols
            returns_data: Historical returns data
            
        Returns:
            Portfolio with equal weights
        """
        try:
            logger.info(f"Creating equal-weight portfolio with {len(assets)} assets")
            
            # Equal weights for all assets
            n_assets = len(assets)
            weights = np.ones(n_assets) / n_assets
            
            # Calculate portfolio metrics
            asset_returns = returns_data[assets]
            portfolio_returns = (asset_returns * weights).sum(axis=1)
            
            expected_return = portfolio_returns.mean() * 252  # Annualized
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            portfolio = Portfolio(
                assets=assets,
                weights=weights,
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                creation_timestamp=datetime.now()
            )
            
            logger.info(f"Equal-weight portfolio: Return={expected_return:.4f}, Volatility={volatility:.4f}, Sharpe={sharpe_ratio:.4f}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to create equal-weight portfolio: {e}")
            raise QuantumMLOptimizationError(f"Equal-weight portfolio creation failed: {e}")
    
    def create_market_cap_weighted_portfolio(self, assets: List[str], returns_data: pd.DataFrame) -> Portfolio:
        """
        Create market-cap weighted portfolio baseline
        
        Args:
            assets: List of asset symbols
            returns_data: Historical returns data
            
        Returns:
            Portfolio with market-cap weights
        """
        try:
            logger.info(f"Creating market-cap weighted portfolio with {len(assets)} assets")
            
            # Fetch market cap data (using price as proxy since we don't have shares outstanding)
            # In practice, this would use actual market cap data
            market_caps = {}
            
            for asset in assets:
                try:
                    # Use average price over the period as market cap proxy
                    # This is a simplification - real implementation would use actual market cap
                    ticker = yf.Ticker(asset)
                    info = ticker.info
                    market_cap = info.get('marketCap', None)
                    
                    if market_cap is None:
                        # Fallback: use average price * volume as proxy
                        hist_data = ticker.history(period='1mo')
                        if not hist_data.empty:
                            avg_price = hist_data['Close'].mean()
                            avg_volume = hist_data['Volume'].mean()
                            market_cap = avg_price * avg_volume
                        else:
                            market_cap = 1.0  # Equal weight fallback
                    
                    market_caps[asset] = market_cap
                    
                except Exception as e:
                    logger.warning(f"Could not fetch market cap for {asset}: {e}")
                    market_caps[asset] = 1.0  # Equal weight fallback
            
            # Calculate market-cap weights
            total_market_cap = sum(market_caps.values())
            weights = np.array([market_caps[asset] / total_market_cap for asset in assets])
            
            # Ensure weights sum to 1 (handle any numerical errors)
            weights = weights / weights.sum()
            
            # Calculate portfolio metrics
            asset_returns = returns_data[assets]
            portfolio_returns = (asset_returns * weights).sum(axis=1)
            
            expected_return = portfolio_returns.mean() * 252  # Annualized
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            portfolio = Portfolio(
                assets=assets,
                weights=weights,
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                creation_timestamp=datetime.now()
            )
            
            logger.info(f"Market-cap weighted portfolio: Return={expected_return:.4f}, Volatility={volatility:.4f}, Sharpe={sharpe_ratio:.4f}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to create market-cap weighted portfolio: {e}")
            # Fallback to equal-weight
            logger.info("Falling back to equal-weight portfolio")
            return self.create_equal_weight_portfolio(assets, returns_data)
    
    def create_markowitz_portfolio(self, assets: List[str], returns_data: pd.DataFrame) -> Portfolio:
        """
        Create Markowitz mean-variance optimized portfolio baseline
        
        Args:
            assets: List of asset symbols
            returns_data: Historical returns data
            
        Returns:
            Portfolio with Markowitz optimal weights
        """
        try:
            logger.info(f"Creating Markowitz portfolio with {len(assets)} assets")
            
            # Calculate expected returns and covariance matrix
            asset_returns = returns_data[assets]
            expected_returns = asset_returns.mean().values * 252  # Annualized
            cov_matrix = asset_returns.cov().values * 252  # Annualized
            
            # Add small regularization to covariance matrix for numerical stability
            cov_matrix += np.eye(len(assets)) * 1e-8
            
            # Solve for maximum Sharpe ratio portfolio
            # This is equivalent to solving: max (w^T * mu - rf) / sqrt(w^T * Sigma * w)
            # Subject to: sum(w) = 1, w >= 0
            
            try:
                # Calculate inverse covariance matrix
                inv_cov = np.linalg.inv(cov_matrix)
                
                # Calculate optimal weights for maximum Sharpe ratio
                excess_returns = expected_returns - self.risk_free_rate
                numerator = inv_cov @ excess_returns
                weights = numerator / numerator.sum()
                
                # Ensure non-negative weights (long-only constraint)
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()  # Renormalize
                
            except np.linalg.LinAlgError:
                logger.warning("Covariance matrix is singular, using regularized version")
                # Add more regularization
                cov_matrix += np.eye(len(assets)) * 0.01
                inv_cov = np.linalg.inv(cov_matrix)
                excess_returns = expected_returns - self.risk_free_rate
                numerator = inv_cov @ excess_returns
                weights = numerator / numerator.sum()
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            portfolio = Portfolio(
                assets=assets,
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                creation_timestamp=datetime.now()
            )
            
            logger.info(f"Markowitz portfolio: Return={portfolio_return:.4f}, Volatility={portfolio_volatility:.4f}, Sharpe={sharpe_ratio:.4f}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to create Markowitz portfolio: {e}")
            # Fallback to equal-weight
            logger.info("Falling back to equal-weight portfolio")
            return self.create_equal_weight_portfolio(assets, returns_data)
    
    def calculate_performance_metrics(self, portfolio: Portfolio, returns_data: pd.DataFrame, 
                                    include_transaction_costs: bool = True) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a portfolio
        
        Args:
            portfolio: Portfolio to analyze
            returns_data: Historical returns data
            include_transaction_costs: Whether to include transaction costs
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Calculate portfolio returns
            asset_returns = returns_data[portfolio.assets]
            portfolio_returns = (asset_returns * portfolio.weights).sum(axis=1)
            
            # Apply transaction costs if requested
            if include_transaction_costs:
                # Assume rebalancing every rebalancing_frequency days
                rebalancing_days = self.config.rebalancing_frequency
                n_rebalances = len(portfolio_returns) // rebalancing_days
                total_transaction_cost = n_rebalances * self.transaction_cost
                
                # Subtract annualized transaction costs
                annual_transaction_cost = total_transaction_cost * (252 / len(portfolio_returns))
                portfolio_returns = portfolio_returns - (annual_transaction_cost / 252)
            
            # Calculate metrics
            metrics = {}
            
            # Basic return metrics
            metrics['total_return'] = (1 + portfolio_returns).prod() - 1
            metrics['annualized_return'] = portfolio_returns.mean() * 252
            metrics['annualized_volatility'] = portfolio_returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            if metrics['annualized_volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / metrics['annualized_volatility']
            else:
                metrics['sharpe_ratio'] = 0
            
            # Downside risk metrics
            negative_returns = portfolio_returns[portfolio_returns < 0]
            if len(negative_returns) > 0:
                metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252)
                if metrics['downside_deviation'] > 0:
                    metrics['sortino_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / metrics['downside_deviation']
                else:
                    metrics['sortino_ratio'] = 0
            else:
                metrics['downside_deviation'] = 0
                metrics['sortino_ratio'] = float('inf')
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdowns.min()
            
            # Value at Risk (95% and 99%)
            metrics['var_95'] = np.percentile(portfolio_returns, 5)
            metrics['var_99'] = np.percentile(portfolio_returns, 1)
            
            # Conditional Value at Risk
            var_95_threshold = metrics['var_95']
            var_99_threshold = metrics['var_99']
            metrics['cvar_95'] = portfolio_returns[portfolio_returns <= var_95_threshold].mean()
            metrics['cvar_99'] = portfolio_returns[portfolio_returns <= var_99_threshold].mean()
            
            # Win rate
            metrics['win_rate'] = (portfolio_returns > 0).mean()
            
            # Calmar ratio (return / max drawdown)
            if metrics['max_drawdown'] < 0:
                metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = float('inf')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    def compare_portfolios(self, quantum_portfolio: Portfolio, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare quantum portfolio against all baseline portfolios
        
        Args:
            quantum_portfolio: The quantum-optimized portfolio
            returns_data: Historical returns data
            
        Returns:
            Comprehensive comparison results
        """
        logger.info("Starting comprehensive portfolio comparison")
        
        comparison_results = {
            'quantum_portfolio': {},
            'baselines': {},
            'relative_performance': {},
            'summary': {}
        }
        
        try:
            # Calculate quantum portfolio metrics
            quantum_metrics = self.calculate_performance_metrics(quantum_portfolio, returns_data)
            comparison_results['quantum_portfolio'] = {
                'portfolio': quantum_portfolio,
                'metrics': quantum_metrics
            }
            
            # Create and evaluate baseline portfolios
            baselines = {}
            
            # Equal-weight baseline
            try:
                eq_weight_portfolio = self.create_equal_weight_portfolio(quantum_portfolio.assets, returns_data)
                eq_weight_metrics = self.calculate_performance_metrics(eq_weight_portfolio, returns_data)
                baselines['equal_weight'] = {
                    'portfolio': eq_weight_portfolio,
                    'metrics': eq_weight_metrics
                }
                logger.info("Equal-weight baseline created successfully")
            except Exception as e:
                logger.error(f"Failed to create equal-weight baseline: {e}")
                baselines['equal_weight'] = None
            
            # Market-cap weighted baseline
            try:
                market_cap_portfolio = self.create_market_cap_weighted_portfolio(quantum_portfolio.assets, returns_data)
                market_cap_metrics = self.calculate_performance_metrics(market_cap_portfolio, returns_data)
                baselines['market_cap_weighted'] = {
                    'portfolio': market_cap_portfolio,
                    'metrics': market_cap_metrics
                }
                logger.info("Market-cap weighted baseline created successfully")
            except Exception as e:
                logger.error(f"Failed to create market-cap weighted baseline: {e}")
                baselines['market_cap_weighted'] = None
            
            # Markowitz baseline
            try:
                markowitz_portfolio = self.create_markowitz_portfolio(quantum_portfolio.assets, returns_data)
                markowitz_metrics = self.calculate_performance_metrics(markowitz_portfolio, returns_data)
                baselines['markowitz'] = {
                    'portfolio': markowitz_portfolio,
                    'metrics': markowitz_metrics
                }
                logger.info("Markowitz baseline created successfully")
            except Exception as e:
                logger.error(f"Failed to create Markowitz baseline: {e}")
                baselines['markowitz'] = None
            
            comparison_results['baselines'] = baselines
            
            # Calculate relative performance
            relative_performance = {}
            
            for baseline_name, baseline_data in baselines.items():
                if baseline_data is not None:
                    baseline_metrics = baseline_data['metrics']
                    relative_perf = {}
                    
                    for metric_name in quantum_metrics:
                        if metric_name in baseline_metrics:
                            quantum_val = quantum_metrics[metric_name]
                            baseline_val = baseline_metrics[metric_name]
                            
                            if baseline_val != 0:
                                relative_perf[f'{metric_name}_improvement'] = (quantum_val - baseline_val) / abs(baseline_val)
                            else:
                                relative_perf[f'{metric_name}_improvement'] = 0
                            
                            relative_perf[f'{metric_name}_difference'] = quantum_val - baseline_val
                    
                    relative_performance[baseline_name] = relative_perf
            
            comparison_results['relative_performance'] = relative_performance
            
            # Generate summary
            summary = {}
            
            # Best performing baseline
            best_baseline = None
            best_sharpe = -float('inf')
            
            for baseline_name, baseline_data in baselines.items():
                if baseline_data is not None:
                    baseline_sharpe = baseline_data['metrics'].get('sharpe_ratio', -float('inf'))
                    if baseline_sharpe > best_sharpe:
                        best_sharpe = baseline_sharpe
                        best_baseline = baseline_name
            
            if best_baseline:
                summary['best_baseline'] = best_baseline
                summary['best_baseline_sharpe'] = best_sharpe
                
                # Quantum vs best baseline
                quantum_sharpe = quantum_metrics.get('sharpe_ratio', 0)
                if best_sharpe != 0:
                    sharpe_improvement = (quantum_sharpe - best_sharpe) / abs(best_sharpe)
                    summary['sharpe_improvement_vs_best'] = sharpe_improvement
                    summary['sharpe_improvement_percentage'] = sharpe_improvement * 100
                
                # Check if we meet the 44% improvement target
                summary['meets_target_improvement'] = sharpe_improvement >= 0.44 if 'sharpe_improvement_vs_best' in summary else False
            
            # Count successful baselines
            successful_baselines = sum(1 for baseline in baselines.values() if baseline is not None)
            summary['successful_baselines'] = successful_baselines
            summary['total_baselines'] = len(baselines)
            
            comparison_results['summary'] = summary
            
            # Log summary results
            logger.info("=== Portfolio Comparison Summary ===")
            logger.info(f"Quantum Portfolio Sharpe Ratio: {quantum_metrics.get('sharpe_ratio', 0):.4f}")
            
            for baseline_name, baseline_data in baselines.items():
                if baseline_data is not None:
                    baseline_sharpe = baseline_data['metrics'].get('sharpe_ratio', 0)
                    logger.info(f"{baseline_name.replace('_', ' ').title()} Sharpe Ratio: {baseline_sharpe:.4f}")
            
            if 'sharpe_improvement_percentage' in summary:
                logger.info(f"Sharpe Ratio Improvement vs Best Baseline: {summary['sharpe_improvement_percentage']:.1f}%")
                logger.info(f"Meets 44% Target: {summary['meets_target_improvement']}")
            
            logger.info("Portfolio comparison completed successfully")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Portfolio comparison failed: {e}")
            comparison_results['error'] = str(e)
            return comparison_results
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable comparison report
        
        Args:
            comparison_results: Results from compare_portfolios
            
        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("=" * 70)
            report.append("PORTFOLIO PERFORMANCE COMPARISON REPORT")
            report.append("=" * 70)
            report.append("")
            
            # Check for errors
            if 'error' in comparison_results:
                report.append(f"ERROR: {comparison_results['error']}")
                return "\n".join(report)
            
            # Quantum portfolio information
            quantum_data = comparison_results['quantum_portfolio']
            quantum_portfolio = quantum_data['portfolio']
            quantum_metrics = quantum_data['metrics']
            
            report.append("QUANTUM PORTFOLIO:")
            report.append(f"  Assets: {', '.join(quantum_portfolio.assets)}")
            report.append(f"  Weights: {dict(zip(quantum_portfolio.assets, [f'{w:.3f}' for w in quantum_portfolio.weights]))}")
            report.append(f"  Expected Return: {quantum_portfolio.expected_return:.4f}")
            report.append(f"  Volatility: {quantum_portfolio.volatility:.4f}")
            report.append(f"  Sharpe Ratio: {quantum_portfolio.sharpe_ratio:.4f}")
            report.append("")
            
            # Baseline portfolios
            report.append("BASELINE PORTFOLIOS:")
            baselines = comparison_results['baselines']
            
            for baseline_name, baseline_data in baselines.items():
                if baseline_data is not None:
                    baseline_portfolio = baseline_data['portfolio']
                    baseline_metrics = baseline_data['metrics']
                    
                    report.append(f"  {baseline_name.replace('_', ' ').upper()}:")
                    report.append(f"    Sharpe Ratio: {baseline_portfolio.sharpe_ratio:.4f}")
                    report.append(f"    Expected Return: {baseline_portfolio.expected_return:.4f}")
                    report.append(f"    Volatility: {baseline_portfolio.volatility:.4f}")
                    report.append(f"    Max Drawdown: {baseline_metrics.get('max_drawdown', 0):.4f}")
                else:
                    report.append(f"  {baseline_name.replace('_', ' ').upper()}: FAILED TO CREATE")
                report.append("")
            
            # Performance comparison
            report.append("PERFORMANCE COMPARISON:")
            relative_performance = comparison_results['relative_performance']
            
            for baseline_name, relative_perf in relative_performance.items():
                report.append(f"  QUANTUM vs {baseline_name.replace('_', ' ').upper()}:")
                
                # Key metrics
                sharpe_improvement = relative_perf.get('sharpe_ratio_improvement', 0)
                return_improvement = relative_perf.get('annualized_return_improvement', 0)
                vol_improvement = relative_perf.get('annualized_volatility_improvement', 0)
                
                report.append(f"    Sharpe Ratio Improvement: {sharpe_improvement:.1%}")
                report.append(f"    Return Improvement: {return_improvement:.1%}")
                report.append(f"    Volatility Change: {vol_improvement:.1%}")
                
                # Risk metrics
                if 'max_drawdown_improvement' in relative_perf:
                    dd_improvement = relative_perf['max_drawdown_improvement']
                    report.append(f"    Max Drawdown Improvement: {dd_improvement:.1%}")
                
                report.append("")
            
            # Summary
            summary = comparison_results['summary']
            report.append("SUMMARY:")
            
            if 'best_baseline' in summary:
                best_baseline = summary['best_baseline'].replace('_', ' ').title()
                report.append(f"  Best Baseline: {best_baseline}")
                report.append(f"  Best Baseline Sharpe: {summary['best_baseline_sharpe']:.4f}")
                
                if 'sharpe_improvement_percentage' in summary:
                    improvement = summary['sharpe_improvement_percentage']
                    report.append(f"  Quantum Improvement: {improvement:.1f}%")
                    
                    target_met = summary.get('meets_target_improvement', False)
                    report.append(f"  Meets 44% Target: {'YES' if target_met else 'NO'}")
            
            successful = summary.get('successful_baselines', 0)
            total = summary.get('total_baselines', 0)
            report.append(f"  Successful Baselines: {successful}/{total}")
            
            report.append("")
            report.append("=" * 70)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            return f"Error generating comparison report: {e}"