"""
Risk analysis modules including risk metrics calculation and Monte Carlo stress testing.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from config import MonteCarloConfig
from exceptions import QuantumMLOptimizationError

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """Calculator for various portfolio risk metrics"""
    
    def __init__(self, risk_free_rate: float = 0.07):
        """
        Initialize risk metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 7% for Indian bonds)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = risk_free_rate / 252  # Convert to daily
        logger.info(f"RiskMetricsCalculator initialized with risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of portfolio returns
            confidence: Confidence level (default: 95%)
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate percentile for VaR
        alpha = 1 - confidence
        var_value = -np.percentile(returns, alpha * 100)
        
        logger.debug(f"VaR at {confidence:.1%} confidence: {var_value:.4f}")
        return var_value
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Args:
            returns: Array of portfolio returns
            confidence: Confidence level (default: 95%)
            
        Returns:
            CVaR value (positive number representing expected loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR threshold
        alpha = 1 - confidence
        var_threshold = -np.percentile(returns, alpha * 100)
        
        # Calculate CVaR as mean of losses beyond VaR
        tail_losses = returns[returns <= -var_threshold]
        if len(tail_losses) > 0:
            cvar_value = -np.mean(tail_losses)
        else:
            cvar_value = var_threshold
        
        logger.debug(f"CVaR at {confidence:.1%} confidence: {cvar_value:.4f}")
        return cvar_value
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate Maximum Drawdown
        
        Args:
            returns: Array of portfolio returns
            
        Returns:
            Maximum drawdown (positive number representing largest peak-to-trough decline)
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown at each point
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown is the most negative drawdown
        max_drawdown = -np.min(drawdown)
        
        logger.debug(f"Maximum Drawdown: {max_drawdown:.4f}")
        return max_drawdown
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = None) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Array of portfolio returns
            risk_free_rate: Risk-free rate (if None, uses instance default)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        if risk_free_rate is None:
            risk_free_rate = self.daily_risk_free_rate
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        
        # Calculate Sharpe ratio
        if np.std(excess_returns) > 1e-10:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
            # Annualize
            sharpe_ratio = sharpe_ratio * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        logger.debug(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = None) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total volatility)
        
        Args:
            returns: Array of portfolio returns
            risk_free_rate: Risk-free rate (if None, uses instance default)
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        if risk_free_rate is None:
            risk_free_rate = self.daily_risk_free_rate
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        
        # Calculate downside deviation (only negative excess returns)
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns)
        else:
            downside_deviation = 1e-10  # Avoid division by zero
        
        # Calculate Sortino ratio
        sortino_ratio = np.mean(excess_returns) / downside_deviation
        # Annualize
        sortino_ratio = sortino_ratio * np.sqrt(252)
        
        logger.debug(f"Sortino Ratio: {sortino_ratio:.4f}")
        return sortino_ratio
    
    def calculate_all_metrics(self, returns: np.ndarray, confidence_levels: List[float] = None) -> Dict[str, float]:
        """
        Calculate all risk metrics for a portfolio
        
        Args:
            returns: Array of portfolio returns
            confidence_levels: List of confidence levels for VaR/CVaR calculation
            
        Returns:
            Dictionary containing all calculated risk metrics
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = np.mean(returns) if len(returns) > 0 else 0.0
        metrics['volatility'] = np.std(returns) if len(returns) > 0 else 0.0
        metrics['skewness'] = self._calculate_skewness(returns)
        metrics['kurtosis'] = self._calculate_kurtosis(returns)
        
        # Risk metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
        
        # VaR and CVaR at different confidence levels
        for confidence in confidence_levels:
            conf_str = f"{int(confidence*100)}"
            metrics[f'var_{conf_str}'] = self.calculate_var(returns, confidence)
            metrics[f'cvar_{conf_str}'] = self.calculate_cvar(returns, confidence)
        
        # Annualized metrics
        if len(returns) > 0:
            metrics['annualized_return'] = (1 + metrics['mean_return']) ** 252 - 1
            metrics['annualized_volatility'] = metrics['volatility'] * np.sqrt(252)
        else:
            metrics['annualized_return'] = 0.0
            metrics['annualized_volatility'] = 0.0
        
        logger.info(f"Calculated {len(metrics)} risk metrics")
        return metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 1e-10:
            skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        else:
            skewness = 0.0
        
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns"""
        if len(returns) < 4:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 1e-10:
            kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        else:
            kurtosis = 0.0
        
        return kurtosis


class MonteCarloStressTester:
    """Monte Carlo simulation engine for portfolio stress testing"""
    
    def __init__(self, config: MonteCarloConfig):
        """
        Initialize Monte Carlo stress tester
        
        Args:
            config: Monte Carlo configuration parameters
        """
        self.config = config
        self.risk_calculator = RiskMetricsCalculator()
        self.random_state = np.random.RandomState(42)  # For reproducible results
        logger.info(f"MonteCarloStressTester initialized with {config.n_simulations} simulations")
    
    def generate_market_scenarios(self, returns_data: pd.DataFrame, scenario_length: int = None) -> Dict[str, np.ndarray]:
        """
        Generate different market scenarios (normal, bear, crisis)
        
        Args:
            returns_data: Historical returns data
            scenario_length: Length of each scenario in days (default from config)
            
        Returns:
            Dictionary containing scenarios for each market regime
        """
        if scenario_length is None:
            scenario_length = self.config.scenario_length
        
        logger.info(f"Generating market scenarios of length {scenario_length}")
        
        # Calculate historical statistics
        historical_mean = returns_data.mean()
        historical_std = returns_data.std()
        historical_corr = returns_data.corr()
        
        scenarios = {}
        
        # Normal market regime (70% probability)
        scenarios['normal'] = self._generate_regime_scenarios(
            historical_mean, historical_std, historical_corr,
            scenario_length, regime_type='normal'
        )
        
        # Bear market regime (20% probability)
        scenarios['bear'] = self._generate_regime_scenarios(
            historical_mean, historical_std, historical_corr,
            scenario_length, regime_type='bear'
        )
        
        # Crisis market regime (10% probability)
        scenarios['crisis'] = self._generate_regime_scenarios(
            historical_mean, historical_std, historical_corr,
            scenario_length, regime_type='crisis'
        )
        
        logger.info(f"Generated scenarios for {len(scenarios)} market regimes")
        return scenarios
    
    def _generate_regime_scenarios(self, mean_returns: pd.Series, std_returns: pd.Series, 
                                 correlation_matrix: pd.DataFrame, scenario_length: int, 
                                 regime_type: str) -> np.ndarray:
        """
        Generate scenarios for a specific market regime
        
        Args:
            mean_returns: Historical mean returns
            std_returns: Historical standard deviations
            correlation_matrix: Historical correlation matrix
            scenario_length: Length of scenarios
            regime_type: Type of market regime ('normal', 'bear', 'crisis')
            
        Returns:
            Array of simulated returns [n_simulations, scenario_length, n_assets]
        """
        n_assets = len(mean_returns)
        n_sims = int(self.config.n_simulations * self.config.market_regimes.get(regime_type, 0.1))
        
        # Adjust parameters based on regime type
        if regime_type == 'normal':
            # Normal market: use historical parameters
            regime_mean = mean_returns.values
            regime_std = std_returns.values
            regime_corr = correlation_matrix.values
            
        elif regime_type == 'bear':
            # Bear market: lower returns, higher volatility, higher correlations
            regime_mean = mean_returns.values * 0.5  # 50% lower returns
            regime_std = std_returns.values * 1.5    # 50% higher volatility
            regime_corr = correlation_matrix.values * 1.2  # Higher correlations
            regime_corr = np.clip(regime_corr, -0.99, 0.99)  # Keep correlations valid
            
        elif regime_type == 'crisis':
            # Crisis market: negative returns, very high volatility, very high correlations
            regime_mean = mean_returns.values * -1.0  # Negative returns
            regime_std = std_returns.values * 2.5     # 150% higher volatility
            regime_corr = correlation_matrix.values * 1.5  # Much higher correlations
            regime_corr = np.clip(regime_corr, -0.99, 0.99)  # Keep correlations valid
        
        else:
            raise ValueError(f"Unknown regime type: {regime_type}")
        
        # Ensure correlation matrix is positive semi-definite
        regime_corr = self._make_correlation_matrix_valid(regime_corr)
        
        # Generate multivariate normal scenarios
        scenarios = np.zeros((n_sims, scenario_length, n_assets))
        
        for sim in range(n_sims):
            # Generate correlated random returns
            random_returns = self.random_state.multivariate_normal(
                regime_mean, 
                np.outer(regime_std, regime_std) * regime_corr,
                size=scenario_length
            )
            scenarios[sim] = random_returns
        
        logger.debug(f"Generated {n_sims} scenarios for {regime_type} regime")
        return scenarios
    
    def _make_correlation_matrix_valid(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Ensure correlation matrix is positive semi-definite
        
        Args:
            corr_matrix: Input correlation matrix
            
        Returns:
            Valid correlation matrix
        """
        try:
            # Check if matrix is already valid
            eigenvals = np.linalg.eigvals(corr_matrix)
            if np.all(eigenvals >= -1e-8):  # Allow small numerical errors
                return corr_matrix
            
            # Fix matrix using eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
            
            # Set negative eigenvalues to small positive value
            eigenvals = np.maximum(eigenvals, 1e-8)
            
            # Reconstruct matrix
            fixed_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Ensure diagonal is 1
            np.fill_diagonal(fixed_corr, 1.0)
            
            # Ensure symmetry
            fixed_corr = (fixed_corr + fixed_corr.T) / 2
            
            return fixed_corr
            
        except Exception as e:
            logger.warning(f"Error fixing correlation matrix: {e}")
            # Fallback to identity matrix
            return np.eye(corr_matrix.shape[0])
    
    def simulate_portfolio_performance(self, assets: List[str], weights: np.ndarray, 
                                     returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate portfolio performance across different market scenarios
        
        Args:
            assets: List of asset symbols
            weights: Portfolio weights
            returns_data: Historical returns data
            
        Returns:
            Dictionary containing simulation results and risk metrics
        """
        logger.info(f"Starting Monte Carlo simulation for portfolio with {len(assets)} assets")
        
        # Filter returns data for selected assets
        asset_returns = returns_data[assets]
        
        # Generate market scenarios
        scenarios = self.generate_market_scenarios(asset_returns)
        
        # Simulate portfolio returns for each scenario
        simulation_results = {}
        
        for regime, regime_scenarios in scenarios.items():
            logger.info(f"Simulating {regime} market regime...")
            
            # Calculate portfolio returns for each simulation
            portfolio_returns = []
            
            for sim_idx in range(regime_scenarios.shape[0]):
                # Get scenario returns for this simulation
                scenario_returns = regime_scenarios[sim_idx]  # [scenario_length, n_assets]
                
                # Calculate portfolio returns
                portfolio_sim_returns = np.dot(scenario_returns, weights)
                portfolio_returns.append(portfolio_sim_returns)
            
            # Convert to numpy array
            portfolio_returns = np.array(portfolio_returns)  # [n_sims, scenario_length]
            
            # Calculate risk metrics for this regime
            regime_metrics = self._calculate_regime_metrics(portfolio_returns, regime)
            simulation_results[regime] = regime_metrics
        
        # Calculate overall metrics (weighted by regime probabilities)
        overall_metrics = self._calculate_overall_metrics(simulation_results)
        
        # Compile final results
        stress_test_results = {
            'regime_results': simulation_results,
            'overall_metrics': overall_metrics,
            'simulation_parameters': {
                'n_simulations': self.config.n_simulations,
                'scenario_length': self.config.scenario_length,
                'market_regimes': self.config.market_regimes,
                'confidence_levels': self.config.confidence_levels
            },
            'portfolio_info': {
                'assets': assets,
                'weights': weights.tolist(),
                'n_assets': len(assets)
            }
        }
        
        logger.info("Monte Carlo stress testing completed")
        return stress_test_results
    
    def _calculate_regime_metrics(self, portfolio_returns: np.ndarray, regime: str) -> Dict[str, Any]:
        """
        Calculate risk metrics for a specific market regime
        
        Args:
            portfolio_returns: Portfolio returns array [n_sims, scenario_length]
            regime: Market regime name
            
        Returns:
            Dictionary containing regime-specific metrics
        """
        # Flatten all returns for overall statistics
        all_returns = portfolio_returns.flatten()
        
        # Calculate basic statistics
        regime_metrics = {
            'regime': regime,
            'n_simulations': portfolio_returns.shape[0],
            'scenario_length': portfolio_returns.shape[1],
            'total_returns': len(all_returns)
        }
        
        # Calculate risk metrics using RiskMetricsCalculator
        risk_metrics = self.risk_calculator.calculate_all_metrics(
            all_returns, self.config.confidence_levels
        )
        regime_metrics.update(risk_metrics)
        
        # Calculate scenario-level statistics
        scenario_final_values = np.cumprod(1 + portfolio_returns, axis=1)[:, -1]  # Final portfolio values
        scenario_total_returns = scenario_final_values - 1  # Total returns for each scenario
        
        regime_metrics.update({
            'scenario_final_values': {
                'mean': np.mean(scenario_final_values),
                'std': np.std(scenario_final_values),
                'min': np.min(scenario_final_values),
                'max': np.max(scenario_final_values),
                'percentiles': {
                    '5': np.percentile(scenario_final_values, 5),
                    '25': np.percentile(scenario_final_values, 25),
                    '50': np.percentile(scenario_final_values, 50),
                    '75': np.percentile(scenario_final_values, 75),
                    '95': np.percentile(scenario_final_values, 95)
                }
            },
            'scenario_total_returns': {
                'mean': np.mean(scenario_total_returns),
                'std': np.std(scenario_total_returns),
                'min': np.min(scenario_total_returns),
                'max': np.max(scenario_total_returns),
                'positive_scenarios_pct': np.mean(scenario_total_returns > 0) * 100
            }
        })
        
        # Calculate probability of loss
        loss_probability = np.mean(all_returns < 0) * 100
        regime_metrics['loss_probability'] = loss_probability
        
        # Calculate tail risk metrics
        tail_returns = all_returns[all_returns < np.percentile(all_returns, 5)]
        if len(tail_returns) > 0:
            regime_metrics['tail_risk'] = {
                'mean_tail_loss': np.mean(tail_returns),
                'worst_day_loss': np.min(all_returns),
                'tail_volatility': np.std(tail_returns)
            }
        else:
            regime_metrics['tail_risk'] = {
                'mean_tail_loss': 0.0,
                'worst_day_loss': 0.0,
                'tail_volatility': 0.0
            }
        
        logger.debug(f"Calculated metrics for {regime} regime: Sharpe={risk_metrics['sharpe_ratio']:.3f}")
        return regime_metrics
    
    def _calculate_overall_metrics(self, regime_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate overall portfolio metrics weighted by regime probabilities
        
        Args:
            regime_results: Results from each market regime
            
        Returns:
            Dictionary containing overall weighted metrics
        """
        overall_metrics = {}
        
        # Weight metrics by regime probabilities
        weighted_metrics = {}
        
        for regime, results in regime_results.items():
            regime_prob = self.config.market_regimes.get(regime, 0.0)
            
            # Weight key metrics
            for metric in ['mean_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 
                          'var_95', 'cvar_95', 'loss_probability']:
                if metric in results:
                    if metric not in weighted_metrics:
                        weighted_metrics[metric] = 0.0
                    weighted_metrics[metric] += results[metric] * regime_prob
        
        overall_metrics.update(weighted_metrics)
        
        # Calculate additional overall metrics
        overall_metrics['regime_probabilities'] = self.config.market_regimes.copy()
        
        # Calculate worst-case scenario metrics
        worst_case_metrics = {}
        for metric in ['max_drawdown', 'var_95', 'cvar_95']:
            worst_values = [results.get(metric, 0) for results in regime_results.values()]
            worst_case_metrics[f'worst_case_{metric}'] = max(worst_values) if worst_values else 0.0
        
        overall_metrics.update(worst_case_metrics)
        
        # Calculate best-case scenario metrics
        best_case_metrics = {}
        for metric in ['sharpe_ratio', 'mean_return']:
            best_values = [results.get(metric, 0) for results in regime_results.values()]
            best_case_metrics[f'best_case_{metric}'] = max(best_values) if best_values else 0.0
        
        overall_metrics.update(best_case_metrics)
        
        logger.info(f"Calculated overall weighted metrics: Sharpe={overall_metrics.get('sharpe_ratio', 0):.3f}")
        return overall_metrics
    
    def stress_test_portfolio(self, assets: List[str], weights: np.ndarray, 
                            returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete stress testing pipeline for a portfolio
        
        Args:
            assets: List of asset symbols
            weights: Portfolio weights
            returns_data: Historical returns data
            
        Returns:
            Comprehensive stress test results
        """
        logger.info("Starting comprehensive portfolio stress testing")
        
        try:
            # Validate inputs
            if len(assets) != len(weights):
                raise ValueError("Number of assets must match number of weights")
            
            if not np.isclose(np.sum(weights), 1.0, rtol=1e-3):
                logger.warning(f"Weights sum to {np.sum(weights):.4f}, normalizing to 1.0")
                weights = weights / np.sum(weights)
            
            # Run Monte Carlo simulation
            simulation_results = self.simulate_portfolio_performance(assets, weights, returns_data)
            
            # Add stress test metadata
            simulation_results['stress_test_metadata'] = {
                'test_date': datetime.now().isoformat(),
                'assets_tested': assets,
                'weights_tested': weights.tolist(),
                'data_period': f"{returns_data.index[0]} to {returns_data.index[-1]}",
                'total_simulations': self.config.n_simulations,
                'scenario_length_days': self.config.scenario_length
            }
            
            # Generate summary report
            summary = self._generate_stress_test_summary(simulation_results)
            simulation_results['summary'] = summary
            
            logger.info("Portfolio stress testing completed successfully")
            return simulation_results
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            raise QuantumMLOptimizationError(f"Monte Carlo stress testing failed: {e}")
    
    def _generate_stress_test_summary(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate human-readable summary of stress test results
        
        Args:
            results: Stress test results
            
        Returns:
            Dictionary containing summary strings
        """
        overall = results['overall_metrics']
        
        summary = {
            'overall_performance': f"Expected Sharpe Ratio: {overall.get('sharpe_ratio', 0):.3f}",
            'risk_assessment': f"95% VaR: {overall.get('var_95', 0):.3f}, Max Drawdown: {overall.get('max_drawdown', 0):.3f}",
            'tail_risk': f"95% CVaR: {overall.get('cvar_95', 0):.3f}",
            'loss_probability': f"Daily Loss Probability: {overall.get('loss_probability', 0):.1f}%"
        }
        
        # Add regime-specific insights
        regime_insights = []
        for regime, regime_data in results['regime_results'].items():
            sharpe = regime_data.get('sharpe_ratio', 0)
            regime_insights.append(f"{regime.title()}: Sharpe {sharpe:.3f}")
        
        summary['regime_performance'] = ", ".join(regime_insights)
        
        return summary