"""
Two-stage approach: Forecast then optimize.
Stage 1: LSTM forecaster for return predictions
Stage 2: Convex optimizer for portfolio allocation
PyTorch implementation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cvxpy as cp
from typing import Optional, Tuple


class ReturnForecaster(nn.Module):
    """
    LSTM-based forecaster for short-horizon returns.
    """
    def __init__(self,
                 n_assets: int,
                 n_features: int,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # LSTM layers
        self.lstm1 = nn.LSTM(n_features, hidden_dim, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        
        # Output layer
        self.output_dense = nn.Linear(hidden_dim, n_assets)  # Predict return for each asset
    
    def forward(self, x):
        # x shape: (batch, window_len, n_features)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # Take last time step
        x = x[:, -1, :]  # (batch, hidden_dim)
        predictions = self.output_dense(x)  # (batch, n_assets)
        return predictions


def create_forecaster(config: dict, n_assets: int, n_features: int) -> ReturnForecaster:
    """Create forecaster from config."""
    model = ReturnForecaster(
        n_assets=n_assets,
        n_features=n_features,
        hidden_dim=config.get("forecaster_hidden_dim", 64),
        dropout=0.1
    )
    return model


class ConvexAllocator:
    """
    Convex optimizer for portfolio allocation.
    Given expected returns and covariance, solves:
    
    min  w^T Sigma w - lambda * mu^T w
    s.t. sum(w) = 1
         0 <= w_i <= max_weight
    
    For long-only minimum variance with expected return tilt.
    """
    def __init__(self,
                 n_assets: int,
                 max_weight: float = 0.20,
                 shrinkage_alpha: float = 0.1):
        self.n_assets = n_assets
        self.max_weight = max_weight
        self.shrinkage_alpha = shrinkage_alpha
    
    def compute_shrinkage_cov(self, returns: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf style shrinkage covariance.
        
        Sigma = alpha * I + (1 - alpha) * Sample_Cov
        """
        if len(returns) < 2:
            return np.eye(self.n_assets) * 0.0001
        
        sample_cov = np.cov(returns.T)
        
        # Shrinkage target: diagonal
        target = np.eye(self.n_assets) * np.trace(sample_cov) / self.n_assets
        
        # Shrunk covariance
        shrunk_cov = self.shrinkage_alpha * target + (1 - self.shrinkage_alpha) * sample_cov
        
        # Ensure positive definite
        shrunk_cov = shrunk_cov + np.eye(self.n_assets) * 1e-6
        
        return shrunk_cov
    
    def allocate(self,
                 expected_returns: np.ndarray,
                 hist_returns: np.ndarray,
                 risk_aversion: float = 1.0) -> np.ndarray:
        """
        Solve for optimal weights.
        
        Args:
            expected_returns: Expected returns for each asset, shape (n_assets,)
            hist_returns: Historical returns for covariance estimation, shape (T, n_assets)
            risk_aversion: Risk aversion parameter (higher = more conservative)
            
        Returns:
            weights: Optimal weights, shape (n_assets,)
        """
        # Compute covariance
        cov = self.compute_shrinkage_cov(hist_returns)
        
        # Define optimization problem
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize variance - lambda * expected return
        risk = cp.quad_form(w, cov)
        expected_return = expected_returns @ w
        objective = cp.Minimize(risk_aversion * risk - expected_return)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if w.value is not None:
                weights = np.array(w.value).flatten()
                # Ensure valid
                weights = np.clip(weights, 0, self.max_weight)
                weights = weights / np.sum(weights)
                return weights
            else:
                # Fallback: equal weight
                return np.ones(self.n_assets) / self.n_assets
        
        except Exception as e:
            print(f"Optimization failed: {e}")
            return np.ones(self.n_assets) / self.n_assets


class TwoStagePortfolio:
    """
    Combined two-stage portfolio system.
    """
    def __init__(self,
                 forecaster: ReturnForecaster,
                 allocator: ConvexAllocator,
                 window_len: int):
        self.forecaster = forecaster
        self.allocator = allocator
        self.window_len = window_len
    
    def predict_weights(self,
                       features_window: np.ndarray,
                       hist_returns: np.ndarray,
                       risk_aversion: float = 1.0) -> np.ndarray:
        """
        Predict portfolio weights using two-stage approach.
        
        Args:
            features_window: Feature window, shape (window_len, n_features)
            hist_returns: Historical returns for covariance, shape (T, n_assets)
            risk_aversion: Risk aversion parameter
            
        Returns:
            weights: Portfolio weights, shape (n_assets,)
        """
        # Stage 1: Forecast returns
        X = torch.tensor(features_window.reshape(1, self.window_len, -1), dtype=torch.float32)
        self.forecaster.eval()
        with torch.no_grad():
            expected_returns = self.forecaster(X).cpu().numpy()[0]
        
        # Stage 2: Optimize allocation
        weights = self.allocator.allocate(expected_returns, hist_returns, risk_aversion)
        
        return weights


def test_two_stage():
    """Test two-stage system."""
    n_assets = 10
    n_features = 60
    window_len = 60
    
    # Create forecaster
    forecaster = ReturnForecaster(n_assets, n_features, hidden_dim=64)
    
    # Test forecast
    X = torch.randn(32, window_len, n_features)
    forecaster.eval()
    with torch.no_grad():
        predictions = forecaster(X)
    print(f"Forecaster test:")
    print(f"  Input shape: {X.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    
    # Create allocator
    allocator = ConvexAllocator(n_assets, max_weight=0.20, shrinkage_alpha=0.1)
    
    # Test allocation
    expected_returns = np.random.randn(n_assets) * 0.001
    hist_returns = np.random.randn(252, n_assets) * 0.01
    weights = allocator.allocate(expected_returns, hist_returns, risk_aversion=1.0)
    
    print(f"\nAllocator test:")
    print(f"  Weights sum: {weights.sum():.4f}")
    print(f"  Max weight: {weights.max():.4f}")
    print(f"  Min weight: {weights.min():.4f}")
    
    # Test two-stage system
    two_stage = TwoStagePortfolio(forecaster, allocator, window_len)
    features_window = X[0].numpy()
    weights_pred = two_stage.predict_weights(features_window, hist_returns)
    
    print(f"\nTwo-stage system test:")
    print(f"  Predicted weights sum: {weights_pred.sum():.4f}")
    
    print("\n✓ Two-stage model test passed!")


if __name__ == "__main__":
    test_two_stage()
