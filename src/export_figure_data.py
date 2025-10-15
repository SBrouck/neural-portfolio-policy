"""
Export numerical data for figures (matplotlib-free).
Generates CSV files that can be visualized in Excel, R, or Python later.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def export_nav_data():
    """Export NAV time series for all strategies."""
    strategies = {
        "TCN_Enhanced": "tcn_policy_enhanced",
        "Momentum": "baseline_momentum",
        "Equal_Weight": "baseline_equal_weight",
        "SPY_Only": "baseline_spy_only"
    }
    
    nav_df = pd.DataFrame()
    
    for name, pattern in strategies.items():
        try:
            returns = pd.read_csv(f"out/reports/{pattern}_test_returns.csv",
                                index_col=0, parse_dates=True).squeeze()
            nav = (1 + returns).cumprod()
            nav_df[name] = nav
        except:
            continue
    
    nav_df.to_csv("out/figs/nav_timeseries.csv")
    print(f"✓ Exported NAV data: out/figs/nav_timeseries.csv")


def export_drawdown_data():
    """Export drawdown time series."""
    strategies = {
        "TCN_Enhanced": "tcn_policy_enhanced",
        "Momentum": "baseline_momentum",
        "Equal_Weight": "baseline_equal_weight"
    }
    
    dd_df = pd.DataFrame()
    
    for name, pattern in strategies.items():
        try:
            returns = pd.read_csv(f"out/reports/{pattern}_test_returns.csv",
                                index_col=0, parse_dates=True).squeeze()
            nav = (1 + returns).cumprod()
            running_max = nav.expanding().max()
            drawdown = (nav - running_max) / running_max * 100
            dd_df[name] = drawdown
        except:
            continue
    
    dd_df.to_csv("out/figs/drawdown_timeseries.csv")
    print(f"✓ Exported Drawdown data: out/figs/drawdown_timeseries.csv")


def export_rolling_sharpe():
    """Export rolling 6-month Sharpe for TCN."""
    try:
        returns = pd.read_csv("out/reports/tcn_policy_enhanced_test_returns.csv",
                            index_col=0, parse_dates=True).squeeze()
        
        window = 126  # ~6 months for weekly data
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(52)
        
        rolling_sharpe.to_csv("out/figs/rolling_sharpe_tcn.csv")
        print(f"✓ Exported Rolling Sharpe: out/figs/rolling_sharpe_tcn.csv")
    except Exception as e:
        print(f"   ⚠️ Rolling Sharpe export failed: {e}")


def export_weights_matrix():
    """Export weights matrix for heatmap."""
    try:
        weights = pd.read_csv("out/reports/tcn_policy_enhanced_test_weights.csv",
                            index_col=0, parse_dates=True)
        
        weights.to_csv("out/figs/weights_matrix_tcn.csv")
        print(f"✓ Exported Weights matrix: out/figs/weights_matrix_tcn.csv ({weights.shape})")
    except Exception as e:
        print(f"   ⚠️ Weights export failed: {e}")


def export_turnover_data():
    """Export turnover time series."""
    try:
        weights = pd.read_csv("out/reports/tcn_policy_enhanced_test_weights.csv",
                            index_col=0, parse_dates=True)
        
        turnover = weights.diff().abs().sum(axis=1)
        cumulative_cost = (turnover * 0.001).cumsum() * 100  # 10bps, in percent
        
        turnover_df = pd.DataFrame({
            'turnover_pct': turnover * 100,
            'cumulative_cost_pct': cumulative_cost
        })
        
        turnover_df.to_csv("out/figs/turnover_analysis.csv")
        print(f"✓ Exported Turnover data: out/figs/turnover_analysis.csv")
        print(f"   Mean turnover: {turnover.mean()*100:.2f}%")
        print(f"   Total cost: {cumulative_cost.iloc[-1]:.2f}%")
    except Exception as e:
        print(f"   ⚠️ Turnover export failed: {e}")


def export_summary_stats():
    """Export summary statistics table."""
    df = pd.read_csv("out/validation/comparison_excess_sharpe.csv")
    
    # Save as CSV
    df.to_csv("out/figs/performance_table.csv", index=False)
    
    # Save as simple text table
    with open("out/figs/performance_table.txt", "w") as f:
        f.write("Performance Comparison Table\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
    
    print(f"✓ Exported Performance table: out/figs/performance_table.csv")


def main():
    """Generate all figure data exports."""
    print("="*60)
    print("EXPORTING FIGURE DATA (CSV format)")
    print("="*60)
    
    Path("out/figs").mkdir(parents=True, exist_ok=True)
    
    print("\n1. NAV time series...")
    export_nav_data()
    
    print("\n2. Drawdown time series...")
    export_drawdown_data()
    
    print("\n3. Rolling Sharpe...")
    export_rolling_sharpe()
    
    print("\n4. Weights matrix...")
    export_weights_matrix()
    
    print("\n5. Turnover analysis...")
    export_turnover_data()
    
    print("\n6. Summary statistics...")
    export_summary_stats()
    
    print("\n" + "="*60)
    print("✓ All data exported to out/figs/")
    print("  Use Excel, R, or Python to visualize")
    print("="*60)


if __name__ == "__main__":
    main()

