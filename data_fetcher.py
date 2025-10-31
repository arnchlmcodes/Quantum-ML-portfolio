"""
Data fetching and preprocessing for NSE stocks.
"""

import time
import pandas as pd
import yfinance as yf
from typing import List, Tuple
from datetime import datetime

from config import SystemConfig
from data_models import MarketData


class DataFetcher:
    """Handles data fetching and preprocessing for NSE stocks"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.retry_attempts = 3
        self.retry_delay = 1.0  # seconds
    
    def fetch_nse_data(self, symbols: List[str], period: str = None) -> pd.DataFrame:
        """
        Fetch NSE stock data using yfinance with retry mechanism
        
        Args:
            symbols: List of NSE stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
            period: Data period (default from config)
            
        Returns:
            DataFrame with stock price data
        """
        if period is None:
            period = self.config.data_period
        
        # Use yfinance to download data
        data = yf.download(
            symbols,
            period=period,
            interval=self.config.data_interval,
            group_by='ticker',
            auto_adjust=True,
            prepost=True,
            threads=True,
            progress=False
        )
        
        # Handle single symbol case (yfinance returns different structure)
        if len(symbols) == 1:
            data.columns = pd.MultiIndex.from_product([symbols, data.columns])
        
        return data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Validate data quality and return quality score
        
        Args:
            data: Raw price data DataFrame
            
        Returns:
            Tuple of (is_valid, quality_score)
        """
        if data.empty:
            return False, 0.0
        
        total_points = 0
        valid_points = 0
        
        # Check each symbol's data quality
        symbols = data.columns.get_level_values(0).unique()
        
        for symbol in symbols:
            symbol_data = data[symbol]
            close_prices = symbol_data['Close'].dropna()
            
            if len(close_prices) == 0:
                continue
            
            total_points += len(symbol_data)
            valid_points += len(close_prices)
        
        quality_score = valid_points / total_points if total_points > 0 else 0.0
        is_valid = quality_score >= 0.8  # Require at least 80% valid data
        
        return is_valid, quality_score
    
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data using forward fill and interpolation
        
        Args:
            data: Raw price data with potential missing values
            
        Returns:
            DataFrame with missing data handled
        """
        processed_data = data.copy()
        symbols = data.columns.get_level_values(0).unique()
        
        for symbol in symbols:
            symbol_data = processed_data[symbol]
            
            # Forward fill missing values first
            symbol_data = symbol_data.ffill()
            
            # For remaining NaN values, use interpolation
            symbol_data = symbol_data.interpolate(method='linear')
            
            # If still NaN at the beginning, backward fill
            symbol_data = symbol_data.bfill()
            
            processed_data[symbol] = symbol_data
        
        return processed_data
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            prices: Price data DataFrame
            
        Returns:
            DataFrame with calculated returns
        """
        returns_data = {}
        symbols = prices.columns.get_level_values(0).unique()
        
        for symbol in symbols:
            close_prices = prices[symbol]['Close']
            
            # Calculate daily returns
            daily_returns = close_prices.pct_change().dropna()
            
            # Store returns
            returns_data[symbol] = daily_returns
        
        returns_df = pd.DataFrame(returns_data)
        
        return returns_df
    
    def preprocess_data(self, symbols: List[str]) -> MarketData:
        """
        Complete data preprocessing pipeline
        
        Args:
            symbols: List of stock symbols to process
            
        Returns:
            MarketData object with processed data
        """
        # Fetch raw data
        raw_data = self.fetch_nse_data(symbols)
        
        # Validate data quality
        is_valid, quality_score = self.validate_data_quality(raw_data)
        
        # Handle missing data
        clean_data = self.handle_missing_data(raw_data)
        
        # Calculate returns
        returns = self.calculate_returns(clean_data)
        
        # Create MarketData object
        market_data = MarketData(
            symbols=symbols,
            prices=clean_data,
            returns=returns,
            data_quality_score=quality_score,
            last_updated=datetime.now()
        )
        
        return market_data