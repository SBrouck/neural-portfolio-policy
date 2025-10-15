"""
Quick numerical validation without plotting dependencies.
Computes proper excess Sharpe ratios over risk-free rate.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def load_rf_rate(start: str, end: str) -> pd.Series:
    """Load daily risk-free rate."""
    try:
        rf_data = pd.read_parquet("data/rf/tbill_3m_daily.parquet")
        rf_series = rf_data.set_index("date")["rf_daily"]  # Column is rf_daily not rf_rate
        rf_series = rf_series.loc[start:end]
        return rf_series
    except Exception as e:
        print(f"Warning: Could not load rf: {e}")
        print(f"Using rf=0.0")
        return pd.Series(0.0, index=pd.date_range(start, end, freq='D'))


def compute_excess_sharpe(returns: pd.Series, rf_series: pd.Series, periods_per_year: int = 252) -> float:
    """Compute Sharpe with proper risk-free alignment."""
    rf_aligned = rf_series.reindex(returns.index, method='ffill').fillna(0.0)
    
    # Convert to same frequency if needed
    if periods_per_year == 52:  # Weekly
        rf_weekly = (1 + rf_aligned).resample('W-FRI').prod() - 1
        rf_aligned = rf_weekly.reindex(returns.index, method='ffill')
    
    excess = returns - rf_aligned
    if len(excess) < 2:
        return 0.0
    
    sharpe = (excess.mean() / excess.std()) * np.sqrt(periods_per_year)
    return sharpe


def compute_max_dd(returns: pd.Series) -> tuple:
    """Compute max drawdown."""
    cum = (1 + returns).cumprod()
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max
    return dd.min(), len(dd)


def compute_turnover(weights: pd.DataFrame) -> float:
    """Average turnover per rebalance."""
    if len(weights) < 2:
        return 0.0
    changes = weights.diff().abs().sum(axis=1)
    return changes.iloc[1:].mean()


def load_strategy(name_pattern: str) -> tuple:
    """Load returns and weights for a strategy."""
    try:
        ret = pd.read_csv(f"out/reports/{name_pattern}_test_returns.csv", 
                         index_col=0, parse_dates=True).squeeze()
        wgt = pd.read_csv(f"out/reports/{name_pattern}_test_weights.csv",
                         index_col=0, parse_dates=True)
        return ret, wgt
    except:
        return None, None


def main():
    print("="*80)
    print("VALIDATION: PROPER EXCESS SHARPE RATIOS (over DTB3)")
    print("="*80)
    
    # Load risk-free
    print("\n1. Loading 3-month T-Bill rate...")
    rf = load_rf_rate("2020-01-01", "2026-01-01")
    rf_ann = rf.mean() * 252 * 100
    print(f"   Mean annualized rf: {rf_ann:.2f}%")
    
    # Define strategies
    strategies = {
        "TCN Enhanced": "tcn_policy_enhanced",
        "TCN Simple FT": "tcn_simple",
        "Equal Weight": "baseline_equal_weight",
        "Risk Parity": "baseline_risk_parity",
        "Min Variance": "baseline_min_variance",
        "Momentum": "baseline_momentum",
        "SPY Only": "baseline_spy_only"
    }
    
    results = []
    
    print("\n2. Computing metrics for all strategies...")
    for name, pattern in strategies.items():
        ret, wgt = load_strategy(pattern)
        if ret is None:
            continue
        
        # Determine frequency
        freq_days = (ret.index[1] - ret.index[0]).days
        periods = 52 if freq_days > 3 else 252
        
        # Metrics
        total_ret = (1 + ret).prod() - 1
        ann_ret = ret.mean() * periods
        ann_vol = ret.std() * np.sqrt(periods)
        
        # CORRECTED: Excess Sharpe
        sharpe_excess = compute_excess_sharpe(ret, rf, periods)
        
        # Naive Sharpe (for comparison)
        sharpe_naive = (ret.mean() / ret.std()) * np.sqrt(periods)
        
        max_dd, dd_len = compute_max_dd(ret)
        turnover = compute_turnover(wgt)
        
        # Beta to SPY
        spy_ret, _ = load_strategy("baseline_spy_only")
        if spy_ret is not None and len(spy_ret) == len(ret):
            beta = np.cov(ret, spy_ret)[0,1] / np.var(spy_ret)
        else:
            beta = np.nan
        
        results.append({
            "Strategy": name,
            "Ann Return": f"{ann_ret*100:.2f}%",
            "Ann Vol": f"{ann_vol*100:.2f}%",
            "Sharpe (naive)": f"{sharpe_naive:.4f}",
            "Sharpe (EXCESS)": f"{sharpe_excess:.4f}",
            "Max DD": f"{max_dd*100:.2f}%",
            "Turnover": f"{turnover*100:.2f}%",
            "Beta(SPY)": f"{beta:.3f}" if not np.isnan(beta) else "N/A"
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by excess Sharpe
    df["_sort"] = df["Sharpe (EXCESS)"].str.replace("%", "").astype(float)
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    
    print("\n" + "="*80)
    print("TEST PERIOD RESULTS (2021-2025)")
    print("="*80)
    print()
    print(df.to_string(index=False))
    print()
    
    # Save
    Path("out/validation").mkdir(parents=True, exist_ok=True)
    df.to_csv("out/validation/comparison_excess_sharpe.csv", index=False)
    
    # Assessment
    print("="*80)
    print("ASSESSMENT")
    print("="*80)
    
    tcn_row = df[df["Strategy"] == "TCN Policy"]
    if not tcn_row.empty:
        rank = (df.index == tcn_row.index[0]).argmax() + 1
        tcn_sharpe = float(tcn_row["Sharpe (EXCESS)"].values[0])
        
        print(f"\nTCN Policy: Rank #{rank} out of {len(df)}")
        print(f"Excess Sharpe: {tcn_sharpe:.4f}")
        
        if rank == 1:
            print("✅ WINNER: Best excess Sharpe - publishable result")
        elif rank <= 3:
            print("✅ COMPETITIVE: Top-3 performance - good result")
        elif rank <= len(df)//2:
            print("⚠️  MODERATE: Mid-tier performance - needs improvement")
        else:
            print("❌ WEAK: Below-median - model needs rework")
        
        # Compare to Equal Weight
        ew_row = df[df["Strategy"] == "Equal Weight"]
        if not ew_row.empty:
            ew_sharpe = float(ew_row["Sharpe (EXCESS)"].values[0])
            diff = tcn_sharpe - ew_sharpe
            print(f"\nvsEqual Weight: {diff:+.4f} ({diff/ew_sharpe*100:+.1f}%)")
            
            if abs(diff) < 0.05:
                print("⚠️  Difference is marginal - not economically significant")
    
    print("\n✓ Validation complete")
    print(f"✓ Results saved to out/validation/comparison_excess_sharpe.csv")
    print("="*80)


if __name__ == "__main__":
    main()

