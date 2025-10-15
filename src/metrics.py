"""
Performance metrics for portfolio evaluation.
Sharpe, Sortino, drawdown, turnover, hit ratio, etc.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_sharpe_ratio(returns: pd.Series,
                        rf_rate: float = 0.0,
                        periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio.
    
    Args:
        returns: Series of period returns (daily, weekly, etc.)
        rf_rate: Risk-free rate in same frequency as returns
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        sharpe: Annualized Sharpe ratio
    """
    excess = returns - rf_rate
    if len(excess) < 2:
        return 0.0
    
    mean_excess = excess.mean()
    std_excess = excess.std()
    
    if std_excess < 1e-12:
        return 0.0
    
    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    return sharpe


def compute_sortino_ratio(returns: pd.Series,
                         rf_rate: float = 0.0,
                         periods_per_year: int = 252) -> float:
    """
    Annualized Sortino ratio (using downside deviation).
    
    Args:
        returns: Series of period returns
        rf_rate: Risk-free rate
        periods_per_year: Periods per year
        
    Returns:
        sortino: Annualized Sortino ratio
    """
    excess = returns - rf_rate
    if len(excess) < 2:
        return 0.0
    
    mean_excess = excess.mean()
    
    # Downside deviation
    downside = excess[excess < 0]
    if len(downside) < 1:
        downside_std = excess.std()  # Fallback
    else:
        downside_std = downside.std()
    
    if downside_std < 1e-12:
        return 0.0
    
    sortino = mean_excess / downside_std * np.sqrt(periods_per_year)
    return sortino


def compute_max_drawdown(returns: pd.Series) -> Dict[str, float]:
    """
    Compute maximum drawdown and drawdown duration.
    
    Args:
        returns: Series of period returns
        
    Returns:
        dict with max_dd (decimal) and max_dd_days
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    max_dd = drawdown.min()
    
    # Drawdown duration
    in_dd = drawdown < -0.001  # Consider > 0.1% as drawdown
    if in_dd.any():
        # Find longest consecutive stretch
        dd_lengths = []
        current_length = 0
        for is_dd in in_dd:
            if is_dd:
                current_length += 1
            else:
                if current_length > 0:
                    dd_lengths.append(current_length)
                current_length = 0
        if current_length > 0:
            dd_lengths.append(current_length)
        
        max_dd_days = max(dd_lengths) if dd_lengths else 0
    else:
        max_dd_days = 0
    
    return {"max_dd": max_dd, "max_dd_days": max_dd_days}


def compute_turnover(weights_history: pd.DataFrame) -> float:
    """
    Compute average turnover.
    Turnover_t = sum_i |w_{i,t} - w_{i,t-1}|
    
    Args:
        weights_history: DataFrame with dates as index and assets as columns
        
    Returns:
        avg_turnover: Mean turnover per rebalance
    """
    if len(weights_history) < 2:
        return 0.0
    
    weight_changes = weights_history.diff().abs()
    turnover_per_period = weight_changes.sum(axis=1)
    
    # Exclude first period (NaN)
    avg_turnover = turnover_per_period.iloc[1:].mean()
    
    return avg_turnover


def compute_hit_ratio(returns: pd.Series) -> float:
    """
    Fraction of periods with positive returns.
    
    Args:
        returns: Series of period returns
        
    Returns:
        hit_ratio: Fraction in [0, 1]
    """
    if len(returns) == 0:
        return 0.0
    
    hit_ratio = (returns > 0).sum() / len(returns)
    return hit_ratio


def compute_calmar_ratio(returns: pd.Series,
                        periods_per_year: int = 252) -> float:
    """
    Calmar ratio = annualized return / abs(max drawdown).
    
    Args:
        returns: Series of period returns
        periods_per_year: Periods per year
        
    Returns:
        calmar: Calmar ratio
    """
    ann_return = returns.mean() * periods_per_year
    max_dd = compute_max_drawdown(returns)["max_dd"]
    
    if abs(max_dd) < 1e-12:
        return 0.0
    
    calmar = ann_return / abs(max_dd)
    return calmar


def compute_all_metrics(returns: pd.Series,
                       weights_history: Optional[pd.DataFrame] = None,
                       rf_rate: float = 0.0,
                       periods_per_year: int = 252) -> Dict[str, float]:
    """
    Compute all performance metrics.
    
    Args:
        returns: Series of portfolio returns
        weights_history: DataFrame of weights over time (optional, for turnover)
        rf_rate: Risk-free rate in same frequency as returns
        periods_per_year: Periods per year
        
    Returns:
        metrics: Dictionary of all metrics
    """
    if len(returns) < 2:
        return {
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "max_dd_days": 0,
            "calmar": 0.0,
            "hit_ratio": 0.0,
            "avg_turnover": 0.0,
        }
    
    # Basic stats
    total_return = (1 + returns).prod() - 1
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    
    # Ratios
    sharpe = compute_sharpe_ratio(returns, rf_rate, periods_per_year)
    sortino = compute_sortino_ratio(returns, rf_rate, periods_per_year)
    calmar = compute_calmar_ratio(returns, periods_per_year)
    
    # Drawdown
    dd_stats = compute_max_drawdown(returns)
    
    # Hit ratio
    hit_ratio = compute_hit_ratio(returns)
    
    # Turnover
    avg_turnover = 0.0
    if weights_history is not None:
        avg_turnover = compute_turnover(weights_history)
    
    metrics = {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": dd_stats["max_dd"],
        "max_dd_days": int(dd_stats["max_dd_days"]),
        "calmar": calmar,
        "hit_ratio": hit_ratio,
        "avg_turnover": avg_turnover,
    }
    
    return metrics


def create_metrics_table(returns_dict: Dict[str, pd.Series],
                        weights_dict: Optional[Dict[str, pd.DataFrame]] = None,
                        rf_rate: float = 0.0,
                        periods_per_year: int = 252) -> pd.DataFrame:
    """
    Create comparison table of metrics across multiple strategies.
    
    Args:
        returns_dict: Dict of {strategy_name: returns_series}
        weights_dict: Dict of {strategy_name: weights_df} (optional)
        rf_rate: Risk-free rate
        periods_per_year: Periods per year
        
    Returns:
        metrics_df: DataFrame with strategies as rows and metrics as columns
    """
    all_metrics = {}
    
    for strategy_name, returns in returns_dict.items():
        weights = weights_dict.get(strategy_name) if weights_dict else None
        metrics = compute_all_metrics(returns, weights, rf_rate, periods_per_year)
        all_metrics[strategy_name] = metrics
    
    metrics_df = pd.DataFrame(all_metrics).T
    
    # Sort columns for readability
    col_order = [
        "total_return", "ann_return", "ann_vol", "sharpe", "sortino",
        "max_dd", "max_dd_days", "calmar", "hit_ratio", "avg_turnover"
    ]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]
    
    return metrics_df


def test_metrics():
    """Test metrics computation."""
    # Create synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003, 
                       index=pd.date_range("2020-01-01", periods=252, freq="D"))
    
    # Compute metrics
    metrics = compute_all_metrics(returns, rf_rate=0.0, periods_per_year=252)
    
    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test turnover
    dates = pd.date_range("2020-01-01", periods=10, freq="W")
    weights = pd.DataFrame({
        "A": np.linspace(0.5, 0.3, 10),
        "B": np.linspace(0.3, 0.4, 10),
        "C": np.linspace(0.2, 0.3, 10),
    }, index=dates)
    
    turnover = compute_turnover(weights)
    print(f"\nTest Turnover: {turnover:.4f}")
    
    print("\n✓ Metrics tests passed!")


if __name__ == "__main__":
    test_metrics()

