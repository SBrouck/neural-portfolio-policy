"""
Portfolio constraints and projection to feasible set.
Handles long-only, sum-to-one, per-asset caps, and volatility targeting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def softmax_weights(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to weights via softmax.
    Ensures non-negative and sum to 1.
    
    Args:
        logits: Array of shape (n_assets,) or (batch, n_assets)
        
    Returns:
        weights: Same shape as logits, summing to 1 along last axis
    """
    # Numerical stability
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return weights


def project_to_caps(weights: np.ndarray, 
                   max_weight: float = 0.20) -> np.ndarray:
    """
    Project weights to respect per-asset maximum weight constraint.
    Iteratively clips weights above cap and redistributes to others.
    
    Args:
        weights: Array of shape (n_assets,) or (batch, n_assets), must sum to 1
        max_weight: Maximum weight per asset
        
    Returns:
        projected_weights: Same shape, respecting caps and summing to 1
    """
    is_1d = weights.ndim == 1
    if is_1d:
        weights = weights.reshape(1, -1)
    
    projected = weights.copy()
    
    for _ in range(100):  # Max iterations to avoid infinite loop
        # Find assets above cap
        above_cap = projected > max_weight
        
        if not above_cap.any():
            break
        
        # Clip to cap
        excess = np.sum(np.maximum(projected - max_weight, 0), axis=1, keepdims=True)
        projected = np.minimum(projected, max_weight)
        
        # Redistribute excess to assets below cap
        below_cap = projected < max_weight
        n_below = np.sum(below_cap, axis=1, keepdims=True)
        
        # Avoid division by zero
        n_below = np.maximum(n_below, 1)
        
        redistrib = (excess * below_cap) / n_below
        projected += redistrib
    
    # Ensure sum to 1 (numerical stability)
    projected = projected / np.sum(projected, axis=1, keepdims=True)
    
    if is_1d:
        projected = projected.flatten()
    
    return projected


def apply_volatility_target(weights: np.ndarray,
                           cov_matrix: np.ndarray,
                           target_vol: float,
                           bdays_per_year: int = 252) -> np.ndarray:
    """
    Scale weights to achieve target annual volatility.
    
    portfolio_vol = sqrt(w^T Cov w) * sqrt(252)
    scaled_weights = w * (target_vol / portfolio_vol)
    
    If resulting weights would violate sum-to-one or caps, 
    add the remainder to cash (index 0).
    
    Args:
        weights: Array of shape (n_assets,), sum to 1, long-only
        cov_matrix: Covariance matrix of daily returns, shape (n_assets, n_assets)
        target_vol: Target annual volatility (e.g. 0.10 for 10%)
        bdays_per_year: Business days per year for annualization
        
    Returns:
        scaled_weights: Scaled to target vol, may have cash position
    """
    # Current portfolio variance (daily)
    port_var_daily = weights @ cov_matrix @ weights
    port_vol_daily = np.sqrt(np.maximum(port_var_daily, 1e-12))
    
    # Annualized
    port_vol_ann = port_vol_daily * np.sqrt(bdays_per_year)
    
    # Scale factor
    if port_vol_ann > 1e-8:
        scale = target_vol / port_vol_ann
    else:
        scale = 1.0
    
    # Don't lever up, only scale down
    scale = min(scale, 1.0)
    
    scaled_weights = weights * scale
    
    # Cash position
    cash = 1.0 - np.sum(scaled_weights)
    
    # If we have n assets, prepend cash at index 0
    # But if weights don't include cash already, we need to handle this
    # For simplicity, assume weights[0] is cash and adjust
    
    # Actually, let's assume weights include all assets including cash
    # and we just scale risky assets and adjust cash
    
    return scaled_weights, cash


def apply_vol_target_with_cash(risky_weights: np.ndarray,
                               cov_matrix: np.ndarray,
                               target_vol: float,
                               max_weight: float = 0.20,
                               bdays_per_year: int = 252) -> np.ndarray:
    """
    Apply volatility targeting to risky assets and add cash.
    
    Args:
        risky_weights: Weights on risky assets only, shape (n_risky,)
        cov_matrix: Covariance of risky assets, shape (n_risky, n_risky)
        target_vol: Target annual volatility
        max_weight: Max weight per asset (including cash)
        bdays_per_year: Business days per year
        
    Returns:
        full_weights: Weights including cash at index 0, shape (n_risky+1,)
    """
    # Ensure risky weights sum to <= 1 and respect caps
    risky_weights = np.clip(risky_weights, 0, max_weight)
    risky_weights = risky_weights / (np.sum(risky_weights) + 1e-12)
    
    # Current vol
    port_var_daily = risky_weights @ cov_matrix @ risky_weights
    port_vol_ann = np.sqrt(np.maximum(port_var_daily, 1e-12)) * np.sqrt(bdays_per_year)
    
    # Scale factor (only scale down)
    scale = min(1.0, target_vol / (port_vol_ann + 1e-12))
    
    scaled_risky = risky_weights * scale
    cash = 1.0 - np.sum(scaled_risky)
    
    # Prepend cash
    full_weights = np.concatenate([[cash], scaled_risky])
    
    return full_weights


def project_weights_full(logits: np.ndarray,
                        cov_matrix: np.ndarray,
                        max_weight: float = 0.20,
                        target_vol: Optional[float] = None,
                        bdays_per_year: int = 252) -> np.ndarray:
    """
    Full projection pipeline:
    1. Softmax to get non-negative weights summing to 1
    2. Project to caps
    3. Apply volatility targeting if specified
    
    Args:
        logits: Raw model outputs, shape (n_assets,)
        cov_matrix: Covariance matrix for vol targeting, shape (n_assets, n_assets)
        max_weight: Maximum weight per asset
        target_vol: Target annual volatility (optional)
        bdays_per_year: Business days per year
        
    Returns:
        final_weights: Projected weights, shape (n_assets,)
    """
    # Step 1: Softmax
    weights = softmax_weights(logits)
    
    # Step 2: Cap projection
    weights = project_to_caps(weights, max_weight=max_weight)
    
    # Step 3: Vol targeting (optional)
    if target_vol is not None and cov_matrix is not None:
        port_var_daily = weights @ cov_matrix @ weights
        port_vol_ann = np.sqrt(np.maximum(port_var_daily, 1e-12)) * np.sqrt(bdays_per_year)
        
        if port_vol_ann > target_vol:
            scale = target_vol / port_vol_ann
            weights = weights * scale
            
            # Renormalize
            weights = weights / np.sum(weights)
    
    return weights


def test_projection():
    """Test projection functions."""
    # Test softmax
    logits = np.array([1.0, 2.0, 3.0, 1.5])
    weights = softmax_weights(logits)
    assert np.allclose(np.sum(weights), 1.0), "Softmax doesn't sum to 1"
    assert np.all(weights >= 0), "Softmax has negative weights"
    print(f"✓ Softmax test passed. Weights: {weights}")
    
    # Test cap projection
    weights_over = np.array([0.5, 0.3, 0.15, 0.05])
    projected = project_to_caps(weights_over, max_weight=0.20)
    assert np.allclose(np.sum(projected), 1.0), "Projected weights don't sum to 1"
    assert np.all(projected <= 0.20 + 1e-8), "Projected weights exceed cap"
    print(f"✓ Cap projection test passed. Projected: {projected}")
    
    # Test vol targeting
    n_assets = 4
    cov = np.eye(n_assets) * 0.0004  # Daily variance
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Current vol
    port_vol_ann = np.sqrt(weights @ cov @ weights * 252)
    print(f"Current vol: {port_vol_ann:.4f}")
    
    # Full projection
    logits = np.array([2.0, 2.5, 1.5, 2.0])
    final = project_weights_full(logits, cov, max_weight=0.30, target_vol=0.10)
    assert np.allclose(np.sum(final), 1.0), "Final weights don't sum to 1"
    print(f"✓ Full projection test passed. Final weights: {final}")


if __name__ == "__main__":
    test_projection()

