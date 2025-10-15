"""
Create final report with figures and comparison tables.
Combines results from baselines and both model tracks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from typing import Dict

from src.backtest_loop import load_backtest_config
from src.metrics import create_metrics_table, compute_all_metrics

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def load_all_results() -> Dict[str, Dict]:
    """
    Load all saved results from out/reports/.
    
    Returns dict of {strategy_name: {returns: Series, weights: DataFrame, metrics: dict}}
    """
    reports_dir = Path("out/reports")
    
    results = {}
    
    # Load baselines
    baseline_names = ["Equal_Weight", "Risk_Parity", "Min_Variance", "Momentum", "SPY_Only"]
    
    for name in baseline_names:
        safe_name = name.lower().replace(" ", "_")
        returns_path = reports_dir / f"baseline_{safe_name}_test_returns.csv"
        weights_path = reports_dir / f"baseline_{safe_name}_test_weights.csv"
        
        if returns_path.exists() and weights_path.exists():
            returns = pd.read_csv(returns_path, index_col=0, parse_dates=True).squeeze()
            weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
            metrics = compute_all_metrics(returns, weights, periods_per_year=252)
            
            results[name] = {
                "returns": returns,
                "weights": weights,
                "metrics": metrics
            }
    
    # Load Track A (TCN Policy)
    tcn_returns_path = reports_dir / "tcn_policy_test_returns.csv"
    tcn_weights_path = reports_dir / "tcn_policy_test_weights.csv"
    
    if tcn_returns_path.exists() and tcn_weights_path.exists():
        returns = pd.read_csv(tcn_returns_path, index_col=0, parse_dates=True).squeeze()
        weights = pd.read_csv(tcn_weights_path, index_col=0, parse_dates=True)
        metrics = compute_all_metrics(returns, weights, periods_per_year=252)
        
        results["TCN_Policy"] = {
            "returns": returns,
            "weights": weights,
            "metrics": metrics
        }
    
    # Load Track B (Two-Stage)
    ts_returns_path = reports_dir / "two_stage_test_returns.csv"
    ts_weights_path = reports_dir / "two_stage_test_weights.csv"
    
    if ts_returns_path.exists() and ts_weights_path.exists():
        returns = pd.read_csv(ts_returns_path, index_col=0, parse_dates=True).squeeze()
        weights = pd.read_csv(ts_weights_path, index_col=0, parse_dates=True)
        metrics = compute_all_metrics(returns, weights, periods_per_year=252)
        
        results["Two_Stage"] = {
            "returns": returns,
            "weights": weights,
            "metrics": metrics
        }
    
    return results


def plot_nav_curves(results: Dict, save_path: str = "out/figs/nav_test.png"):
    """Plot NAV curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, data in results.items():
        returns = data["returns"]
        nav = (1 + returns).cumprod()
        ax.plot(nav.index, nav.values, label=name, linewidth=2)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (Starting at 1.0)", fontsize=12)
    ax.set_title("Net Asset Value - Test Period", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ NAV plot saved to {save_path}")
    plt.close()


def plot_drawdowns(results: Dict, save_path: str = "out/figs/drawdown_test.png"):
    """Plot drawdown curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, data in results.items():
        returns = data["returns"]
        nav = (1 + returns).cumprod()
        running_max = nav.expanding().max()
        drawdown = (nav - running_max) / running_max * 100  # Percentage
        
        ax.plot(drawdown.index, drawdown.values, label=name, linewidth=2)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title("Portfolio Drawdowns - Test Period", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Drawdown plot saved to {save_path}")
    plt.close()


def plot_weights_heatmap(results: Dict, strategy_name: str = "TCN_Policy", 
                         save_path: str = "out/figs/weights_heatmap.png"):
    """Plot weights heatmap over time for a specific strategy."""
    if strategy_name not in results:
        print(f"  Warning: {strategy_name} not found in results")
        return
    
    weights = results[strategy_name]["weights"]
    
    # Sample every N days for readability
    sample_freq = max(len(weights) // 50, 1)
    weights_sampled = weights.iloc[::sample_freq]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Transpose for heatmap (assets as rows, time as columns)
    sns.heatmap(
        weights_sampled.T * 100,  # Convert to percentage
        cmap="YlOrRd",
        cbar_kws={"label": "Weight (%)"},
        ax=ax,
        vmin=0,
        vmax=20
    )
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Asset", fontsize=12)
    ax.set_title(f"Portfolio Weights Over Time - {strategy_name}", fontsize=14, fontweight="bold")
    
    # Format x-axis dates
    xtick_labels = [weights_sampled.index[i].strftime("%Y-%m") 
                    for i in range(0, len(weights_sampled), max(len(weights_sampled)//10, 1))]
    xtick_positions = list(range(0, len(weights_sampled), max(len(weights_sampled)//10, 1)))
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Weights heatmap saved to {save_path}")
    plt.close()


def plot_metrics_comparison(results: Dict, save_path: str = "out/figs/metrics_comparison.png"):
    """Plot bar charts comparing key metrics."""
    # Extract key metrics
    metrics_list = []
    names = []
    
    for name, data in results.items():
        metrics_list.append(data["metrics"])
        names.append(name)
    
    metrics_df = pd.DataFrame(metrics_list, index=names)
    
    # Select key metrics to plot
    key_metrics = ["sharpe", "sortino", "max_dd", "ann_return", "ann_vol"]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        if metric in metrics_df.columns:
            ax = axes[i]
            values = metrics_df[metric].values
            
            # Create bar plot
            bars = ax.bar(range(len(names)), values, alpha=0.7)
            
            # Color code: green for positive metrics, red for drawdown
            if metric == "max_dd":
                colors = ['red' if v < 0 else 'green' for v in values]
            else:
                colors = ['green' if v > 0 else 'red' for v in values]
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
            ax.grid(True, alpha=0.3, axis='y')
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.suptitle("Performance Metrics Comparison - Test Period", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Metrics comparison saved to {save_path}")
    plt.close()


def create_final_table(results: Dict, save_path: str = "out/reports/final_comparison.csv"):
    """Create final comparison table with all strategies."""
    metrics_list = []
    names = []
    
    for name, data in results.items():
        metrics_list.append(data["metrics"])
        names.append(name)
    
    metrics_df = pd.DataFrame(metrics_list, index=names)
    
    # Sort by Sharpe ratio
    metrics_df = metrics_df.sort_values("sharpe", ascending=False)
    
    # Round for readability
    metrics_df_rounded = metrics_df.round(4)
    
    # Save
    metrics_df_rounded.to_csv(save_path)
    print(f"\n  ✓ Final comparison table saved to {save_path}")
    
    return metrics_df_rounded


def main():
    """Generate all reports and figures."""
    print("=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)
    
    # Create output directories
    Path("out/figs").mkdir(parents=True, exist_ok=True)
    Path("out/reports").mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("\nLoading results...")
    results = load_all_results()
    
    if not results:
        print("  ✗ No results found. Please run baselines and models first.")
        return
    
    print(f"  Loaded {len(results)} strategies:")
    for name in results.keys():
        print(f"    - {name}")
    
    # Create final comparison table
    print("\nCreating comparison table...")
    final_table = create_final_table(results, "out/reports/final_comparison.csv")
    
    print("\nFinal Comparison (Test Period):")
    print("=" * 70)
    print(final_table)
    print("=" * 70)
    
    # Generate plots
    print("\nGenerating figures...")
    
    plot_nav_curves(results, "out/figs/nav_test.png")
    plot_drawdowns(results, "out/figs/drawdown_test.png")
    plot_metrics_comparison(results, "out/figs/metrics_comparison.png")
    
    # Create weights heatmaps for model strategies
    if "TCN_Policy" in results:
        plot_weights_heatmap(results, "TCN_Policy", "out/figs/weights_heatmap_tcn.png")
    
    if "Two_Stage" in results:
        plot_weights_heatmap(results, "Two_Stage", "out/figs/weights_heatmap_two_stage.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Reports:")
    print("    - out/reports/final_comparison.csv")
    print("  Figures:")
    print("    - out/figs/nav_test.png")
    print("    - out/figs/drawdown_test.png")
    print("    - out/figs/metrics_comparison.png")
    print("    - out/figs/weights_heatmap_*.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

