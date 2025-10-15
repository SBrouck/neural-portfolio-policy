"""
Generate publication-quality figures for portfolio analysis.
All figures in English, ready for thesis/paper.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_strategy_results(name_pattern: str):
    """Load returns and weights for a strategy."""
    try:
        returns = pd.read_csv(f"out/reports/{name_pattern}_test_returns.csv", 
                            index_col=0, parse_dates=True).squeeze()
        weights = pd.read_csv(f"out/reports/{name_pattern}_test_weights.csv",
                            index_col=0, parse_dates=True)
        return returns, weights
    except:
        return None, None


def plot_nav_comparison(save_path="out/figs/nav_comparison.png"):
    """Plot NAV curves for all strategies."""
    strategies = {
        "TCN Enhanced": "tcn_policy_enhanced",
        "Momentum": "baseline_momentum",
        "Equal Weight": "baseline_equal_weight",
        "SPY Only": "baseline_spy_only"
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (name, pattern) in enumerate(strategies.items()):
        returns, _ = load_strategy_results(pattern)
        if returns is not None:
            nav = (1 + returns).cumprod()
            ax.plot(nav.index, nav.values, label=name, linewidth=2.5, 
                   color=colors[i], alpha=0.9)
    
    ax.set_title("Net Asset Value Comparison (Test Period 2021-2025)", 
                fontsize=14, fontweight='bold')
    ax.set_ylabel("NAV ($1 initial)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_drawdown_comparison(save_path="out/figs/drawdown_comparison.png"):
    """Plot drawdown curves."""
    strategies = {
        "TCN Enhanced": "tcn_policy_enhanced",
        "Momentum": "baseline_momentum",
        "Equal Weight": "baseline_equal_weight",
        "SPY Only": "baseline_spy_only"
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (name, pattern) in enumerate(strategies.items()):
        returns, _ = load_strategy_results(pattern)
        if returns is not None:
            nav = (1 + returns).cumprod()
            running_max = nav.expanding().max()
            drawdown = (nav - running_max) / running_max * 100
            
            ax.plot(drawdown.index, drawdown.values, label=name, 
                   linewidth=2, color=colors[i], alpha=0.8)
    
    ax.set_title("Drawdown Comparison (Test Period)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_rolling_sharpe(returns, window=126, save_path="out/figs/rolling_sharpe_tcn.png"):
    """Plot rolling 6-month Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Rolling Sharpe (6 months = ~126 trading days for weekly data)
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(52)  # Weekly data
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2.5, color='#2E86AB')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    
    # Fill positive/negative
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                     where=(rolling_sharpe.values > 0), alpha=0.3, color='green', label='Positive')
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                     where=(rolling_sharpe.values < 0), alpha=0.3, color='red', label='Negative')
    
    ax.set_title("Rolling 6-Month Sharpe Ratio (TCN Enhanced)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_weights_heatmap(weights, save_path="out/figs/weights_heatmap_tcn.png"):
    """Plot portfolio weights heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Subsample for readability (every 4th week)
    weights_sub = weights.iloc[::4]
    
    # Transpose for heatmap (assets as rows)
    weights_T = weights_sub.T
    
    im = ax.imshow(weights_T.values, aspect='auto', cmap='RdYlGn', 
                   vmin=0, vmax=weights.values.max())
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', fontsize=12)
    
    # Labels
    ax.set_title("Portfolio Weights Over Time (TCN Enhanced)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Asset", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # Y-axis: asset names
    ax.set_yticks(range(len(weights_T)))
    ax.set_yticklabels(weights_T.index, fontsize=9)
    
    # X-axis: dates (show every 10th)
    n_labels = 10
    step = len(weights_sub) // n_labels
    if step == 0:
        step = 1
    ax.set_xticks(range(0, len(weights_sub), step))
    ax.set_xticklabels([weights_sub.index[i].strftime('%Y-%m') 
                       for i in range(0, len(weights_sub), step)], 
                       rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_turnover_analysis(weights, save_path="out/figs/turnover_analysis.png"):
    """Plot turnover over time and cumulative cost."""
    # Compute turnover
    turnover = weights.diff().abs().sum(axis=1)
    turnover = turnover.iloc[1:]  # Skip first NaN
    
    # Cumulative cost (10 bps per side)
    cumulative_cost = (turnover * 0.001).cumsum() * 100  # in percent
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Turnover per rebalance
    ax1.bar(turnover.index, turnover.values * 100, width=7, color='steelblue', alpha=0.7)
    ax1.set_title("Weekly Turnover (TCN Enhanced)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Turnover (%)", fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=turnover.mean()*100, color='red', linestyle='--', 
               label=f'Mean: {turnover.mean()*100:.2f}%', linewidth=2)
    ax1.legend()
    
    # Cumulative cost
    ax2.plot(cumulative_cost.index, cumulative_cost.values, 
            linewidth=2.5, color='#C73E1D')
    ax2.set_title("Cumulative Transaction Costs (10 bps per side)", 
                 fontsize=13, fontweight='bold')
    ax2.set_ylabel("Cumulative Cost (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(cumulative_cost.index, 0, cumulative_cost.values, alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_performance_bars(save_path="out/figs/performance_bars.png"):
    """Bar chart comparing key metrics."""
    # Load comparison table
    df = pd.read_csv("out/validation/comparison_excess_sharpe.csv")
    
    # Filter top strategies
    df_top = df.head(4)
    
    # Extract metrics - handle both string and float types
    strategies = df_top["Strategy"].values
    
    # Convert to float, removing % if present
    def to_float(col):
        if df_top[col].dtype == 'object':
            return df_top[col].str.replace("%", "").astype(float).values
        else:
            return df_top[col].values
    
    sharpe = to_float("Sharpe (EXCESS)")
    returns = to_float("Ann Return")
    turnover = to_float("Turnover")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    
    # Sharpe
    ax1.bar(strategies, sharpe, color=colors, alpha=0.8)
    ax1.set_title("Excess Sharpe Ratio", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Sharpe", fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Returns
    ax2.bar(strategies, returns, color=colors, alpha=0.8)
    ax2.set_title("Annualized Return", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Return (%)", fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Turnover
    ax3.bar(strategies, turnover, color=colors, alpha=0.8)
    ax3.set_title("Average Turnover", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Turnover (%)", fontsize=11)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Generate all figures."""
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    
    # Create output directory
    Path("out/figs").mkdir(parents=True, exist_ok=True)
    
    # Load TCN Enhanced results
    print("\n1. Loading TCN Enhanced results...")
    tcn_returns, tcn_weights = load_strategy_results("tcn_policy_enhanced")
    
    if tcn_returns is None:
        print("❌ Could not load TCN Enhanced results")
        return
    
    print(f"   ✓ Returns: {len(tcn_returns)} periods")
    print(f"   ✓ Weights: {tcn_weights.shape}")
    
    # Generate figures
    print("\n2. Generating figures...")
    
    try:
        plot_nav_comparison()
    except Exception as e:
        print(f"   ⚠️ NAV comparison failed: {e}")
    
    try:
        plot_drawdown_comparison()
    except Exception as e:
        print(f"   ⚠️ Drawdown comparison failed: {e}")
    
    try:
        plot_rolling_sharpe(tcn_returns)
    except Exception as e:
        print(f"   ⚠️ Rolling Sharpe failed: {e}")
    
    try:
        plot_weights_heatmap(tcn_weights)
    except Exception as e:
        print(f"   ⚠️ Weights heatmap failed: {e}")
    
    try:
        plot_turnover_analysis(tcn_weights)
    except Exception as e:
        print(f"   ⚠️ Turnover analysis failed: {e}")
    
    try:
        plot_performance_bars()
    except Exception as e:
        print(f"   ⚠️ Performance bars failed: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Figure generation complete")
    print(f"✓ Check out/figs/ for all visualizations")
    print("=" * 60)


if __name__ == "__main__":
    main()

