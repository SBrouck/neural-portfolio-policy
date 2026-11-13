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
