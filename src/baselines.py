"""
Baseline portfolio strategies for comparison.
Equal weight, risk parity, minimum variance, and momentum.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import minimize


class BaselineStrategy:
    """Base class for baseline strategies."""
    
    def __init__(self, n_assets: int, lookback_days: int = 252):
        self.n_assets = n_assets
        self.lookback_days = lookback_days
    
    def compute_weights(self, date: pd.Timestamp, hist_returns: pd.DataFrame) -> np.ndarray:
        """
        Compute portfolio weights.
        
        Args:
            date: Current date
            hist_returns: Historical returns up to (but not including) date
            
        Returns:
            weights: Array of length n_assets summing to 1
        """
        raise NotImplementedError


class EqualWeight(BaselineStrategy):
    """Equal weight (1/N) strategy."""
    
    def compute_weights(self, date: pd.Timestamp, hist_returns: pd.DataFrame) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets


class RiskParity(BaselineStrategy):
    """
    Risk parity: weights inversely proportional to volatility.
    w_i = (1/vol_i) / sum_j (1/vol_j)
    """
    
    def compute_weights(self, date: pd.Timestamp, hist_returns: pd.DataFrame) -> np.ndarray:
        if len(hist_returns) < 20:
            # Not enough data, use equal weight
            return np.ones(self.n_assets) / self.n_assets
        
        # Use recent returns for vol estimation
        recent = hist_returns.iloc[-self.lookback_days:]
        
        # Compute volatility
        vols = recent.std().values
        
        # Avoid division by zero
        vols = np.maximum(vols, 1e-8)
        
        # Inverse vol weights
        inv_vol = 1.0 / vols
        weights = inv_vol / np.sum(inv_vol)
        
        return weights


class MinimumVariance(BaselineStrategy):
    """
    Minimum variance portfolio.
    Solve: min w^T Sigma w s.t. sum(w) = 1, w >= 0
    """
    
    def compute_weights(self, date: pd.Timestamp, hist_returns: pd.DataFrame) -> np.ndarray:
        if len(hist_returns) < 60:
            # Not enough data, use equal weight
            return np.ones(self.n_assets) / self.n_assets
        
        # Use recent returns for covariance
        recent = hist_returns.iloc[-self.lookback_days:]
        
        # Covariance matrix
        cov = recent.cov().values
        
        # Add regularization
        cov = cov + np.eye(self.n_assets) * 1e-5
        
        # Solve quadratic program
        def objective(w):
            return w @ cov @ w
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds: 0 <= w_i <= 1
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Initial guess: equal weight
        w0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9}
        )
        
        if result.success:
            weights = result.x
            # Normalize
            weights = np.clip(weights, 0, 1)
            weights = weights / np.sum(weights)
            return weights
        else:
            # Fallback to equal weight
            return np.ones(self.n_assets) / self.n_assets


class MomentumStrategy(BaselineStrategy):
    """
    Simple momentum strategy.
    Rank assets by past 12-month returns and overweight top performers.
    """
    
    def __init__(self, n_assets: int, lookback_days: int = 252, top_k: Optional[int] = None):
        super().__init__(n_assets, lookback_days)
        self.top_k = top_k if top_k is not None else max(n_assets // 2, 1)
    
    def compute_weights(self, date: pd.Timestamp, hist_returns: pd.DataFrame) -> np.ndarray:
        if len(hist_returns) < self.lookback_days:
            # Not enough data, use equal weight
            return np.ones(self.n_assets) / self.n_assets
        
        # Use lookback period
        recent = hist_returns.iloc[-self.lookback_days:]
        
        # Compute cumulative returns
        cum_returns = (1 + recent).prod() - 1
        
        # Rank
        ranks = cum_returns.rank(ascending=False).values.astype(int)
        
        # Allocate to top_k
        weights = np.zeros(self.n_assets)
        for i, rank in enumerate(ranks):
            if rank <= self.top_k:
                weights[i] = 1.0
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights


class BuyAndHoldSPY(BaselineStrategy):
    """
    Buy and hold SPY (first asset assumed to be SPY).
    """
    
    def compute_weights(self, date: pd.Timestamp, hist_returns: pd.DataFrame) -> np.ndarray:
        weights = np.zeros(self.n_assets)
        weights[0] = 1.0  # Assume first asset is SPY
        return weights


def get_baseline_strategies(n_assets: int, lookback_days: int = 252) -> dict:
    """
    Get all baseline strategies.
    
    Args:
        n_assets: Number of assets
        lookback_days: Lookback window for estimation
        
    Returns:
        strategies: Dict of {name: strategy_object}
    """
    strategies = {
        "Equal_Weight": EqualWeight(n_assets, lookback_days),
        "Risk_Parity": RiskParity(n_assets, lookback_days),
        "Min_Variance": MinimumVariance(n_assets, lookback_days),
        "Momentum": MomentumStrategy(n_assets, lookback_days),
        "SPY_Only": BuyAndHoldSPY(n_assets, lookback_days),
    }
    
    return strategies


def test_baselines():
    """Test baseline strategies."""
    from src.features import load_data
    
    # Load data
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    
    # Filter to test period
    test_start = "2020-01-01"
    test_end = "2021-12-31"
    returns_test = returns[(returns.index >= test_start) & (returns.index <= test_end)]
    
    n_assets = returns_test.shape[1]
    
    # Get strategies
    strategies = get_baseline_strategies(n_assets, lookback_days=252)
    
    # Test each
    test_date = returns_test.index[300]  # Some date with enough history
    hist = returns_test.iloc[:300]
    
    print("Testing baseline strategies:")
    for name, strategy in strategies.items():
        weights = strategy.compute_weights(test_date, hist)
        print(f"\n{name}:")
        print(f"  Weights sum: {np.sum(weights):.4f}")
        print(f"  Max weight: {np.max(weights):.4f}")
        print(f"  Min weight: {np.min(weights):.4f}")
        print(f"  Non-zero assets: {np.sum(weights > 0.001)}")
    
    print("\n✓ Baseline strategies test passed!")


if __name__ == "__main__":
    test_baselines()

