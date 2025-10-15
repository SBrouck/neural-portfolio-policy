"""
Generate oracle labels for supervised distillation.
Creates momentum-based target weights for each rebalancing date.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from src.features import load_data
from src.constraints import project_to_caps


def compute_momentum_weights(returns_window: pd.DataFrame,
                            lookback_days: int = 252,
                            max_weight: float = 0.20) -> np.ndarray:
    """
    Compute momentum-based weights using past returns only.
    
    Args:
        returns_window: Historical returns up to (but not including) current date
        lookback_days: Momentum lookback period
        max_weight: Maximum weight per asset
        
    Returns:
        weights: Long-only portfolio weights with caps
    """
    if len(returns_window) < lookback_days:
        # Not enough history - equal weight
        n_assets = returns_window.shape[1]
        return np.ones(n_assets) / n_assets
    
    # Calculate momentum: cumulative return over lookback period
    mom_window = returns_window.iloc[-lookback_days:]
    cum_returns = (1 + mom_window).prod(axis=0) - 1
    
    # Rank assets by momentum
    ranks = cum_returns.rank(pct=True)
    
    # Linear weighting based on ranks
    # Top momentum assets get higher weight
    raw_weights = ranks.values
    
    # Normalize to sum to 1
    weights = raw_weights / raw_weights.sum()
    
    # Apply caps and renormalize
    weights = project_to_caps(weights, max_weight=max_weight)
    
    return weights


def generate_oracle_labels(start_date: str = "2007-01-01",
                          end_date: str = "2020-12-31",
                          rebalance_freq: str = "W-FRI",
                          lookback_days: int = 252,
                          max_weight: float = 0.20) -> pd.DataFrame:
    """
    Generate oracle momentum weights for all rebalancing dates.
    
    Args:
        start_date: Start date for label generation
        end_date: End date for label generation (typically train+val)
        rebalance_freq: Rebalancing frequency
        lookback_days: Momentum lookback
        max_weight: Maximum weight per asset
        
    Returns:
        oracle_weights: DataFrame with dates as index, assets as columns
    """
    print("Generating Momentum Oracle Labels...")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    
    # Filter to date range (including history for momentum calculation)
    returns_full = returns.loc[:"2020-12-31"]  # Up to end of val
    
    # Get rebalancing dates
    returns_filt = returns[(returns.index >= start_date) & (returns.index <= end_date)]
    rebal_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    
    # Find actual trading dates closest to rebal dates
    actual_rebal_dates = []
    for date in rebal_dates:
        idx = returns_filt.index.get_indexer([date], method='nearest')[0]
        if idx >= 0:
            actual_rebal_dates.append(returns_filt.index[idx])
    
    actual_rebal_dates = pd.DatetimeIndex(sorted(set(actual_rebal_dates)))
    
    print(f"Date range: {start_date} to {end_date}")
    print(f"Rebalancing frequency: {rebalance_freq}")
    print(f"Number of rebalancing dates: {len(actual_rebal_dates)}")
    
    # Generate weights for each rebalancing date
    oracle_weights = []
    
    for i, date in enumerate(actual_rebal_dates):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(actual_rebal_dates)}...")
        
        # Get all history up to (but not including) this date
        date_idx = returns_full.index.get_loc(date)
        if date_idx < lookback_days + 20:  # Need enough history
            continue
        
        returns_window = returns_full.iloc[:date_idx]
        
        # Compute momentum weights
        weights = compute_momentum_weights(
            returns_window,
            lookback_days=lookback_days,
            max_weight=max_weight
        )
        
        # Store
        oracle_weights.append({
            'date': date,
            **{ticker: w for ticker, w in zip(returns.columns, weights)}
        })
    
    # Convert to DataFrame
    oracle_df = pd.DataFrame(oracle_weights)
    oracle_df = oracle_df.set_index('date')
    
    print(f"\n✓ Generated {len(oracle_df)} oracle weight vectors")
    print(f"  Assets: {oracle_df.shape[1]}")
    print(f"  Date range: {oracle_df.index.min().date()} to {oracle_df.index.max().date()}")
    
    # Sanity checks
    print("\nSanity checks:")
    print(f"  Weights sum (mean): {oracle_df.sum(axis=1).mean():.6f}")
    print(f"  Max weight (overall): {oracle_df.values.max():.4f}")
    print(f"  Min weight (overall): {oracle_df.values.min():.4f}")
    
    return oracle_df


def align_oracle_to_sequences(oracle_weights: pd.DataFrame,
                              sequence_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Align oracle weights to sequence prediction dates.
    
    Args:
        oracle_weights: Oracle weights DataFrame (rebal_date x assets)
        sequence_dates: Dates for which we need oracle labels
        
    Returns:
        aligned_weights: Array of shape (n_sequences, n_assets)
    """
    aligned = []
    
    for date in sequence_dates:
        # Find closest rebalancing date <= this date
        valid_dates = oracle_weights.index[oracle_weights.index <= date]
        
        if len(valid_dates) == 0:
            # No oracle available - skip or use equal weight
            n_assets = oracle_weights.shape[1]
            aligned.append(np.ones(n_assets) / n_assets)
        else:
            closest_date = valid_dates[-1]  # Most recent rebalancing
            weights = oracle_weights.loc[closest_date].values
            aligned.append(weights)
    
    return np.array(aligned)


def main():
    """Generate and save oracle labels."""
    # Generate oracle weights for train + val period
    oracle_weights = generate_oracle_labels(
        start_date="2007-01-01",
        end_date="2020-12-31",  # Train + Val
        rebalance_freq="W-FRI",
        lookback_days=252,
        max_weight=0.20
    )
    
    # Save
    Path("data/oracle").mkdir(parents=True, exist_ok=True)
    oracle_weights.to_parquet("data/oracle/momentum_weights.parquet")
    
    print("\n" + "=" * 60)
    print("✓ Oracle labels saved to data/oracle/momentum_weights.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()

