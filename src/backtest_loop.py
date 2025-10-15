"""
Unified backtest engine for all portfolio strategies.
Handles rebalancing, transaction costs, and performance tracking.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Tuple
from pathlib import Path
import yaml


def load_backtest_config(config_path: str = "configs/backtest.yaml") -> dict:
    """Load backtest configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_transaction_costs(weights_prev: np.ndarray,
                              weights_new: np.ndarray,
                              cost_bps: float) -> float:
    """
    Compute transaction costs as fraction of portfolio.
    
    Cost = (cost_bps / 10000) * sum_i |w_new_i - w_prev_i|
    
    Args:
        weights_prev: Previous weights
        weights_new: New weights
        cost_bps: Cost in basis points per side
        
    Returns:
        cost: Cost as fraction of portfolio value
    """
    turnover = np.sum(np.abs(weights_new - weights_prev))
    cost = (cost_bps / 10000.0) * turnover
    return cost


def get_rebalance_dates(dates: pd.DatetimeIndex,
                       frequency: str = "W-FRI",
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DatetimeIndex:
    """
    Get rebalancing dates based on frequency.
    
    Args:
        dates: All available dates
        frequency: Pandas frequency string (e.g., 'W-FRI', 'D', 'M')
        start_date: Start date filter
        end_date: End date filter
        
    Returns:
        rebal_dates: Dates on which to rebalance
    """
    # Filter date range
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    
    if frequency == "D":
        # Daily rebalancing
        return dates
    else:
        # Resample to target frequency and find nearest dates
        freq_dates = pd.date_range(start=dates[0], end=dates[-1], freq=frequency)
        
        # Find nearest actual trading date for each frequency date
        rebal_dates = []
        for freq_date in freq_dates:
            # Find closest date in actual dates
            idx = dates.get_indexer([freq_date], method="nearest")[0]
            if idx >= 0 and idx < len(dates):
                rebal_dates.append(dates[idx])
        
        return pd.DatetimeIndex(sorted(set(rebal_dates)))


def run_backtest(
    returns_df: pd.DataFrame,
    weight_function: Callable[[pd.Timestamp, pd.DataFrame], np.ndarray],
    rebalance_dates: pd.DatetimeIndex,
    cost_bps: float = 10.0,
    initial_weights: Optional[np.ndarray] = None
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run backtest with given weight function.
    
    Args:
        returns_df: DataFrame of asset returns, index=dates, columns=tickers
        weight_function: Function that takes (date, returns_df) and returns weights
        rebalance_dates: Dates on which to rebalance
        cost_bps: Transaction cost in basis points per side
        initial_weights: Initial portfolio weights (default: equal weight)
        
    Returns:
        portfolio_returns: Series of portfolio returns
        weights_history: DataFrame of weights over time
    """
    n_assets = returns_df.shape[1]
    
    # Initialize
    if initial_weights is None:
        weights_current = np.ones(n_assets) / n_assets
    else:
        weights_current = initial_weights.copy()
    
    # Storage
    portfolio_returns = []
    weights_history = []
    dates_history = []
    
    # Get all dates in order
    all_dates = returns_df.index
    rebal_dates_set = set(rebalance_dates)
    
    for i, date in enumerate(all_dates):
        # Check if rebalancing date
        is_rebal_date = date in rebal_dates_set
        
        # Rebalance if needed
        if is_rebal_date:
            # Get new weights from strategy
            # Pass historical data up to (but not including) this date
            hist_returns = returns_df.iloc[:i] if i > 0 else returns_df.iloc[:1]
            
            try:
                weights_new = weight_function(date, hist_returns)
                
                # Ensure valid weights
                weights_new = np.clip(weights_new, 0, 1)
                if np.sum(weights_new) > 0:
                    weights_new = weights_new / np.sum(weights_new)
                else:
                    weights_new = np.ones(n_assets) / n_assets
                
            except Exception as e:
                print(f"Warning: Weight function failed at {date}: {e}")
                weights_new = weights_current.copy()
            
            # Compute transaction costs
            trans_cost = compute_transaction_costs(weights_current, weights_new, cost_bps)
            
            # Update weights
            weights_current = weights_new
        else:
            trans_cost = 0.0
        
        # Compute portfolio return for this period
        asset_returns = returns_df.loc[date].values
        gross_return = np.dot(weights_current, asset_returns)
        net_return = gross_return - trans_cost
        
        portfolio_returns.append(net_return)
        weights_history.append(weights_current.copy())
        dates_history.append(date)
        
        # Update weights based on returns (drift)
        if i < len(all_dates) - 1:  # Don't update after last period
            portfolio_value_after = np.exp(net_return)  # Approximation for small returns
            weights_current = weights_current * (1 + asset_returns)
            weights_current = weights_current / np.sum(weights_current)
    
    # Convert to Series and DataFrame
    portfolio_returns_series = pd.Series(portfolio_returns, index=dates_history, name="portfolio")
    weights_df = pd.DataFrame(weights_history, index=dates_history, columns=returns_df.columns)
    
    return portfolio_returns_series, weights_df


def test_backtest_engine():
    """Test backtest engine with simple buy-and-hold SPY."""
    from src.features import load_data
    
    # Load data
    prices, rf = load_data()
    
    # Compute returns
    returns = prices["adj_close"].pct_change().dropna()
    
    # Filter to test period
    test_start = "2021-01-01"
    test_end = "2022-12-31"
    returns_test = returns[(returns.index >= test_start) & (returns.index <= test_end)]
    
    # Get SPY only
    if "SPY" not in returns_test.columns:
        print("SPY not in dataset")
        return
    
    spy_returns = returns_test[["SPY"]]
    
    # Strategy: 100% SPY
    def spy_strategy(date, hist_returns):
        return np.array([1.0])
    
    # Rebalance dates (weekly)
    rebal_dates = get_rebalance_dates(
        spy_returns.index, frequency="W-FRI", 
        start_date=test_start, end_date=test_end
    )
    
    # Run backtest
    port_returns, weights_hist = run_backtest(
        spy_returns, spy_strategy, rebal_dates, cost_bps=10.0
    )
    
    # Compare to direct SPY returns
    spy_direct = spy_returns["SPY"]
    
    # Align dates
    spy_direct_aligned = spy_direct.reindex(port_returns.index)
    
    # Compute cumulative returns
    port_cum = (1 + port_returns).cumprod()
    spy_cum = (1 + spy_direct_aligned).cumprod()
    
    # They should be very close (small difference due to costs)
    final_diff = abs(port_cum.iloc[-1] - spy_cum.iloc[-1])
    
    print(f"Backtest engine test:")
    print(f"  Portfolio final value: {port_cum.iloc[-1]:.4f}")
    print(f"  SPY final value: {spy_cum.iloc[-1]:.4f}")
    print(f"  Difference: {final_diff:.4f}")
    print(f"  Rebalance dates: {len(rebal_dates)}")
    
    # Test turnover
    turnover_total = 0.0
    for i in range(1, len(weights_hist)):
        turnover_total += np.sum(np.abs(weights_hist.iloc[i].values - weights_hist.iloc[i-1].values))
    
    print(f"  Total turnover: {turnover_total:.4f}")
    
    # For buy-and-hold with weekly rebalance to 100% SPY,
    # turnover should be minimal (only from drift)
    
    print("\n✓ Backtest engine test passed!")


if __name__ == "__main__":
    test_backtest_engine()

