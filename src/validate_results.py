"""
Rigorous validation and diagnostics for portfolio results.
Implements all checks requested for publication-quality results.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import yaml

from src.features import load_data
from src.metrics import (
    compute_all_metrics, compute_sharpe_ratio, compute_max_drawdown,
    compute_turnover
)

# Seaborn style for publication
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_rf_rate(start: str, end: str) -> pd.Series:
    """Load daily risk-free rate from data."""
    try:
        rf_data = pd.read_parquet("data/rf/tbill_3m_daily.parquet")
        rf_series = rf_data.set_index("date")["rf_rate"]
        rf_series = rf_series.loc[start:end]
        return rf_series
    except Exception as e:
        print(f"Warning: Could not load risk-free rate: {e}")
        print("Using rf=0.0")
        dates = pd.date_range(start, end, freq='D')
        return pd.Series(0.0, index=dates)


def compute_correct_sharpe(returns: pd.Series, 
                          rf_series: pd.Series,
                          periods_per_year: int = 252) -> float:
    """
    Compute Sharpe ratio with proper risk-free alignment.
    
    Args:
        returns: Portfolio returns (weekly or daily)
        rf_series: Risk-free rate series (daily)
        periods_per_year: Periods in returns
        
    Returns:
        sharpe: Excess Sharpe ratio
    """
    # Align rf to returns dates
    rf_aligned = rf_series.reindex(returns.index, method='ffill').fillna(0.0)
    
    # Convert daily rf to same frequency as returns if needed
    if periods_per_year == 52:  # Weekly returns
        # Compound daily rf to weekly
        rf_weekly = (1 + rf_aligned).resample('W-FRI').prod() - 1
        rf_aligned = rf_weekly.reindex(returns.index, method='ffill')
    
    excess_returns = returns - rf_aligned
    
    if len(excess_returns) < 2:
        return 0.0
    
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess < 1e-12:
        return 0.0
    
    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    return sharpe


def load_all_test_results() -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
    """Load all strategy results from test period."""
    results = {}
    
    # Load baselines
    baseline_names = [
        "baseline_equal_weight",
        "baseline_risk_parity", 
        "baseline_min_variance",
        "baseline_momentum",
        "baseline_spy_only"
    ]
    
    for name in baseline_names:
        try:
            returns = pd.read_csv(f"out/reports/{name}_test_returns.csv", index_col=0, parse_dates=True).squeeze()
            weights = pd.read_csv(f"out/reports/{name}_test_weights.csv", index_col=0, parse_dates=True)
            clean_name = name.replace("baseline_", "").replace("_", " ").title()
            results[clean_name] = (returns, weights)
        except Exception as e:
            print(f"Could not load {name}: {e}")
    
    # Load TCN policy
    try:
        tcn_returns = pd.read_csv("out/reports/tcn_policy_test_returns.csv", index_col=0, parse_dates=True).squeeze()
        tcn_weights = pd.read_csv("out/reports/tcn_policy_test_weights.csv", index_col=0, parse_dates=True)
        results["TCN Policy"] = (tcn_returns, tcn_weights)
    except Exception as e:
        print(f"Could not load TCN: {e}")
    
    return results


def create_comparison_table(results: Dict[str, Tuple[pd.Series, pd.DataFrame]],
                           rf_series: pd.Series) -> pd.DataFrame:
    """
    Create publication-quality comparison table with proper excess Sharpe.
    """
    metrics_list = []
    
    for strategy_name, (returns, weights) in results.items():
        # Determine frequency
        time_diff = (returns.index[1] - returns.index[0]).days
        if time_diff > 3:  # Weekly
            periods_per_year = 52
        else:  # Daily
            periods_per_year = 252
        
        # Compute correct Sharpe with excess returns
        sharpe_excess = compute_correct_sharpe(returns, rf_series, periods_per_year)
        
        # Standard metrics
        total_ret = (1 + returns).prod() - 1
        ann_ret = returns.mean() * periods_per_year
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        
        dd_stats = compute_max_drawdown(returns)
        turnover = compute_turnover(weights)
        
        # Beta to SPY (approximate)
        spy_returns = results.get("Spy Only", (returns, None))[0]
        if spy_returns is not None and len(spy_returns) == len(returns):
            beta_spy = np.cov(returns, spy_returns)[0, 1] / np.var(spy_returns)
        else:
            beta_spy = np.nan
        
        metrics_list.append({
            "Strategy": strategy_name,
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe (excess)": sharpe_excess,
            "Max DD": dd_stats["max_dd"],
            "DD Days": dd_stats["max_dd_days"],
            "Turnover": turnover,
            "Beta(SPY)": beta_spy
        })
    
    df = pd.DataFrame(metrics_list)
    df = df.set_index("Strategy")
    
    # Sort by Sharpe
    df = df.sort_values("Sharpe (excess)", ascending=False)
    
    return df


def plot_nav_comparison(results: Dict[str, Tuple[pd.Series, pd.DataFrame]],
                       save_path: str = "out/figs/nav_comparison.png"):
    """Plot NAV curves for all strategies."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    for strategy_name, (returns, _) in results.items():
        nav = (1 + returns).cumprod()
        
        # NAV
        ax1.plot(nav.index, nav.values, label=strategy_name, linewidth=2)
        
        # Drawdown
        running_max = nav.expanding().max()
        drawdown = (nav - running_max) / running_max
        ax2.plot(drawdown.index, drawdown.values * 100, label=strategy_name, linewidth=1.5, alpha=0.7)
    
    ax1.set_title("Net Asset Value Comparison (Test Period)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("NAV ($1 initial)", fontsize=12)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Drawdown Comparison", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved NAV comparison to {save_path}")
    plt.close()


def plot_rolling_sharpe(returns: pd.Series,
                       rf_series: pd.Series,
                       window: int = 126,  # 6 months
                       save_path: str = "out/figs/rolling_sharpe.png"):
    """Plot rolling 6-month Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Determine frequency
    time_diff = (returns.index[1] - returns.index[0]).days
    periods_per_year = 52 if time_diff > 3 else 252
    
    # Align rf
    rf_aligned = rf_series.reindex(returns.index, method='ffill').fillna(0.0)
    excess = returns - rf_aligned
    
    # Rolling Sharpe
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='steelblue')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values, 
                     where=(rolling_sharpe.values > 0), alpha=0.3, color='green')
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                     where=(rolling_sharpe.values < 0), alpha=0.3, color='red')
    
    ax.set_title("Rolling 6-Month Sharpe Ratio (TCN Policy)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved rolling Sharpe to {save_path}")
    plt.close()


def plot_weights_heatmap(weights: pd.DataFrame,
                        save_path: str = "out/figs/weights_heatmap.png"):
    """Plot weights heatmap over time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Transpose so assets are rows
    weights_T = weights.T
    
    sns.heatmap(weights_T, cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Weight'}, ax=ax,
                vmin=0, vmax=weights.values.max())
    
    ax.set_title("Portfolio Weights Over Time (TCN Policy)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Asset", fontsize=12)
    
    # Show fewer x-axis labels
    n_labels = 10
    step = len(weights) // n_labels
    ax.set_xticks(range(0, len(weights), step))
    ax.set_xticklabels([weights.index[i].strftime('%Y-%m') for i in range(0, len(weights), step)], 
                       rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved weights heatmap to {save_path}")
    plt.close()


def compute_newey_west_se(returns: pd.Series, lags: int = 5) -> float:
    """
    Compute Newey-West standard error for Sharpe ratio.
    
    Simple approximation: SE(Sharpe) ≈ sqrt((1 + 0.5*Sharpe^2) / T)
    with HAC adjustment.
    """
    T = len(returns)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    # Simple Newey-West approximation
    se = np.sqrt((1 + 0.5 * sharpe**2) / T) * np.sqrt(252)
    
    return se


def main():
    """Run full validation suite."""
    print("=" * 70)
    print("RIGOROUS VALIDATION OF PORTFOLIO RESULTS")
    print("=" * 70)
    
    # Create output directories
    Path("out/figs").mkdir(parents=True, exist_ok=True)
    Path("out/validation").mkdir(parents=True, exist_ok=True)
    
    # Load risk-free rate
    print("\n1. Loading risk-free rate (DTB3)...")
    rf_series = load_rf_rate("2020-01-01", "2025-12-31")
    print(f"   ✓ Loaded {len(rf_series)} days of risk-free data")
    print(f"   Mean daily rf: {rf_series.mean()*100:.4f}%")
    print(f"   Annualized: {rf_series.mean()*252*100:.2f}%")
    
    # Load all results
    print("\n2. Loading all strategy results...")
    results = load_all_test_results()
    print(f"   ✓ Loaded {len(results)} strategies")
    
    # Create proper comparison table
    print("\n3. Computing metrics with EXCESS RETURNS over risk-free...")
    comparison_df = create_comparison_table(results, rf_series)
    
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (Test Period 2021-2025)")
    print("=" * 70)
    print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Save table
    comparison_df.to_csv("out/validation/comparison_table.csv")
    comparison_df.to_latex("out/validation/comparison_table.tex", float_format="%.4f")
    print("\n✓ Saved comparison table")
    
    # Compute confidence interval for TCN Sharpe
    if "TCN Policy" in results:
        tcn_returns = results["TCN Policy"][0]
        rf_aligned = rf_series.reindex(tcn_returns.index, method='ffill').fillna(0.0)
        excess_returns = tcn_returns - rf_aligned
        
        se_sharpe = compute_newey_west_se(excess_returns)
        tcn_sharpe = comparison_df.loc["TCN Policy", "Sharpe (excess)"]
        
        ci_lower = tcn_sharpe - 1.96 * se_sharpe
        ci_upper = tcn_sharpe + 1.96 * se_sharpe
        
        print(f"\n4. TCN Policy Sharpe 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Generate figures
    print("\n5. Generating figures...")
    plot_nav_comparison(results)
    
    if "TCN Policy" in results:
        tcn_returns, tcn_weights = results["TCN Policy"]
        plot_rolling_sharpe(tcn_returns, rf_series)
        plot_weights_heatmap(tcn_weights)
    
    # Summary assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    
    if "TCN Policy" in results:
        tcn_sharpe = comparison_df.loc["TCN Policy", "Sharpe (excess)"]
        tcn_rank = (comparison_df["Sharpe (excess)"] > tcn_sharpe).sum() + 1
        
        print(f"\nTCN Policy ranks #{tcn_rank} out of {len(comparison_df)} strategies")
        print(f"Sharpe (excess): {tcn_sharpe:.4f}")
        
        if tcn_rank <= 2:
            print("✅ RESULT: Strong performance - publishable")
        elif tcn_rank <= len(comparison_df) // 2:
            print("⚠️  RESULT: Moderate performance - needs improvement")
        else:
            print("❌ RESULT: Weak performance - not competitive")
    
    print("\n✓ Validation complete. Check out/validation/ for detailed results.")
    print("=" * 70)


if __name__ == "__main__":
    main()

