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
    Inverse-volatility weights (naive risk parity, ignores correlations).
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
    Long-only minimum variance on the sample covariance (+1e-5 I ridge).
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
    Cross-sectional momentum: equal weight on the top_k assets by trailing
    12-month return (top_k = n_assets // 2 = 7 here).
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


class SingleAsset(BaselineStrategy):
    """100% in one named asset, rebalanced like the others (e.g. SPY)."""

    def __init__(self, n_assets: int, index: int):
        super().__init__(n_assets)
        self.index = index

    def compute_weights(self, date: pd.Timestamp, hist_returns: pd.DataFrame) -> np.ndarray:
        weights = np.zeros(self.n_assets)
        weights[self.index] = 1.0
        return weights


def get_baseline_strategies(tickers: list, lookback_days: int = 252) -> dict:
    """All baselines. Estimation windows use the last `lookback_days` returns before each rebalance."""
    n_assets = len(tickers)
    return {
        "Equal weight": EqualWeight(n_assets, lookback_days),
        "Inverse volatility": RiskParity(n_assets, lookback_days),
        "Minimum variance": MinimumVariance(n_assets, lookback_days),
        "Momentum (top 7, 12m)": MomentumStrategy(n_assets, lookback_days),
        "SPY": SingleAsset(n_assets, list(tickers).index("SPY")),
    }
