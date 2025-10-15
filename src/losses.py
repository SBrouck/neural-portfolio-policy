"""
Custom loss functions for portfolio optimization.
Includes Sharpe loss, turnover penalty, and CVaR via pinball loss.
PyTorch implementation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


def sharpe_loss(portfolio_returns: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Negative Sharpe ratio as loss.
    
    Loss = -mean(R) / sqrt(var(R) + epsilon)
    
    Args:
        portfolio_returns: Tensor of shape (batch_size,) with portfolio returns
        epsilon: Small constant for numerical stability
        
    Returns:
        loss: Scalar tensor (negative Sharpe)
    """
    mean_return = portfolio_returns.mean()
    variance = portfolio_returns.var()
    
    # Negative Sharpe (we minimize)
    sharpe = mean_return / torch.sqrt(variance + epsilon)
    loss = -sharpe
    
    return loss


def turnover_penalty(weights_t: torch.Tensor, 
                    weights_t_minus_1: torch.Tensor) -> torch.Tensor:
    """
    L1 turnover penalty.
    
    Turnover = sum_i |w_{i,t} - w_{i,t-1}|
    
    Args:
        weights_t: Current weights, shape (batch_size, n_assets)
        weights_t_minus_1: Previous weights, shape (batch_size, n_assets)
        
    Returns:
        turnover: Mean turnover across batch
    """
    delta_w = weights_t - weights_t_minus_1
    turnover = torch.sum(torch.abs(delta_w), dim=-1)
    return turnover.mean()


def cvar_pinball_loss(returns: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """
    CVaR via pinball (quantile) loss.
    
    We want to penalize tail risk, so we compute the alpha-quantile
    of NEGATIVE returns (losses) using pinball loss.
    
    Pinball loss at quantile alpha:
    L(y, q) = max(alpha * (y - q), (alpha - 1) * (y - q))
    
    For CVaR, we use alpha=0.95 to focus on the worst 5% of outcomes.
    
    Args:
        returns: Portfolio returns, shape (batch_size,)
        alpha: Quantile level (e.g., 0.95 for 95th percentile)
        
    Returns:
        cvar_loss: Scalar CVaR estimate
    """
    # Convert returns to losses (negative returns)
    losses = -returns
    
    # Estimate the alpha-quantile
    quantile = torch_quantile(losses, alpha)
    
    # Pinball loss
    errors = losses - quantile
    pinball = torch.where(
        errors >= 0,
        alpha * errors,
        (alpha - 1) * errors
    )
    
    return pinball.mean()


def torch_quantile(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute empirical quantile using PyTorch operations.
    
    Args:
        values: Tensor of shape (batch_size,)
        alpha: Quantile level in [0, 1]
        
    Returns:
        quantile: Scalar tensor
    """
    sorted_values, _ = torch.sort(values)
    n = sorted_values.shape[0]
    index = alpha * (n - 1)
    
    # Linear interpolation between floor and ceil
    index_floor = int(torch.floor(torch.tensor(index)).item())
    index_ceil = int(torch.ceil(torch.tensor(index)).item())
    
    # Clamp indices
    index_floor = max(0, min(index_floor, n - 1))
    index_ceil = max(0, min(index_ceil, n - 1))
    
    val_floor = sorted_values[index_floor]
    val_ceil = sorted_values[index_ceil]
    
    # Interpolation weight
    weight = index - torch.floor(torch.tensor(index))
    quantile = val_floor + weight * (val_ceil - val_floor)
    
    return quantile


def combined_portfolio_loss(portfolio_returns: torch.Tensor,
                           weights_t: torch.Tensor,
                           weights_t_minus_1: torch.Tensor,
                           lambda_turn: float = 2.0,
                           lambda_cvar: float = 0.5,
                           alpha_cvar: float = 0.95,
                           epsilon: float = 1e-6) -> torch.Tensor:
    """
    Combined loss for portfolio optimization.
    
    Loss = -Sharpe + lambda_turn * Turnover + lambda_cvar * CVaR
    
    Args:
        portfolio_returns: Portfolio returns, shape (batch_size,)
        weights_t: Current weights, shape (batch_size, n_assets)
        weights_t_minus_1: Previous weights, shape (batch_size, n_assets)
        lambda_turn: Weight for turnover penalty
        lambda_cvar: Weight for CVaR penalty
        alpha_cvar: Quantile level for CVaR
        epsilon: Numerical stability constant
        
    Returns:
        total_loss: Scalar loss
    """
    # Sharpe component
    loss_sharpe = sharpe_loss(portfolio_returns, epsilon=epsilon)
    
    # Turnover component
    loss_turn = turnover_penalty(weights_t, weights_t_minus_1)
    
    # CVaR component
    loss_cvar = cvar_pinball_loss(portfolio_returns, alpha=alpha_cvar)
    
    # Combine
    total_loss = loss_sharpe + lambda_turn * loss_turn + lambda_cvar * loss_cvar
    
    return total_loss


def compute_portfolio_returns(weights: torch.Tensor,
                              asset_returns: torch.Tensor,
                              transaction_cost_bps: float = 10.0,
                              weights_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute portfolio returns including transaction costs.
    
    R_p = w^T r - c * |delta_w|
    
    Args:
        weights: Portfolio weights, shape (batch_size, n_assets)
        asset_returns: Asset returns, shape (batch_size, n_assets)
        transaction_cost_bps: Transaction cost in basis points per side
        weights_prev: Previous weights for cost calculation
        
    Returns:
        portfolio_returns: Shape (batch_size,)
    """
    # Gross returns
    gross_returns = torch.sum(weights * asset_returns, dim=-1)
    
    # Transaction costs
    if weights_prev is not None:
        cost_rate = transaction_cost_bps / 10000.0  # Convert bps to decimal
        turnover = torch.sum(torch.abs(weights - weights_prev), dim=-1)
        transaction_costs = cost_rate * turnover
        net_returns = gross_returns - transaction_costs
    else:
        net_returns = gross_returns
    
    return net_returns


class PortfolioLoss(nn.Module):
    """
    PyTorch loss module for portfolio optimization.
    """
    def __init__(self,
                 lambda_turn: float = 2.0,
                 lambda_cvar: float = 0.5,
                 alpha_cvar: float = 0.95,
                 epsilon: float = 1e-6):
        super().__init__()
        self.lambda_turn = lambda_turn
        self.lambda_cvar = lambda_cvar
        self.alpha_cvar = alpha_cvar
        self.epsilon = epsilon
    
    def forward(self, weights_pred, asset_returns, weights_prev):
        """
        Compute loss.
        
        Args:
            weights_pred: Predicted weights, shape (batch, n_assets)
            asset_returns: Asset returns, shape (batch, n_assets)
            weights_prev: Previous weights, shape (batch, n_assets)
            
        Returns:
            loss: Scalar
        """
        # Compute portfolio returns
        portfolio_returns = compute_portfolio_returns(
            weights_pred, asset_returns, transaction_cost_bps=10.0, weights_prev=weights_prev
        )
        
        # Combined loss
        loss = combined_portfolio_loss(
            portfolio_returns,
            weights_pred,
            weights_prev,
            lambda_turn=self.lambda_turn,
            lambda_cvar=self.lambda_cvar,
            alpha_cvar=self.alpha_cvar,
            epsilon=self.epsilon
        )
        
        return loss


def test_losses():
    """Test loss functions."""
    import numpy as np
    
    # Create synthetic data
    batch_size = 128
    n_assets = 10
    
    returns = torch.tensor(np.random.randn(batch_size) * 0.01 + 0.0003, dtype=torch.float32)
    weights_t = torch.tensor(np.random.dirichlet(np.ones(n_assets), batch_size), dtype=torch.float32)
    weights_t_minus_1 = torch.tensor(np.random.dirichlet(np.ones(n_assets), batch_size), dtype=torch.float32)
    
    # Test Sharpe loss
    loss_sharpe = sharpe_loss(returns)
    print(f"Sharpe loss: {loss_sharpe.item():.6f}")
    
    # Test turnover
    loss_turn = turnover_penalty(weights_t, weights_t_minus_1)
    print(f"Turnover: {loss_turn.item():.6f}")
    
    # Test CVaR
    loss_cvar = cvar_pinball_loss(returns, alpha=0.95)
    print(f"CVaR loss: {loss_cvar.item():.6f}")
    
    # Test combined
    total_loss = combined_portfolio_loss(returns, weights_t, weights_t_minus_1)
    print(f"Total loss: {total_loss.item():.6f}")
    
    print("\n✓ Loss function tests passed!")


if __name__ == "__main__":
    test_losses()
