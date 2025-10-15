"""
Master runner script: Execute all baselines and model tracks.
Produces comparison tables and figures.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict

from src.features import load_data, build_feature_matrix
from src.baselines import get_baseline_strategies
from src.backtest_loop import run_backtest, get_rebalance_dates, load_backtest_config
from src.metrics import create_metrics_table


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def run_baselines(
    returns_df: pd.DataFrame,
    config_bt: dict,
    split_name: str = "test"
) -> Dict[str, tuple]:
    """
    Run all baseline strategies.
    
    Returns dict of {strategy_name: (returns, weights)}
    """
    if split_name == "val":
        start, end = config_bt["val_start"], config_bt["val_end"]
    else:
        start, end = config_bt["test_start"], config_bt["test_end"]
    
    returns_split = returns_df[(returns_df.index >= start) & (returns_df.index <= end)]
    
    # Get baseline strategies
    n_assets = returns_split.shape[1]
    strategies = get_baseline_strategies(n_assets, lookback_days=252)
    
    # Get rebalance dates
    rebal_dates = get_rebalance_dates(
        returns_split.index,
        frequency=config_bt["rebalance"],
        start_date=start,
        end_date=end
    )
    
    # Run each baseline
    results = {}
    
    print(f"\nRunning baselines on {split_name} set...")
    print("=" * 60)
    
    for name, strategy in strategies.items():
        print(f"  Running {name}...")
        
        def weight_func(date, hist_returns):
            return strategy.compute_weights(date, hist_returns)
        
        port_returns, weights_hist = run_backtest(
            returns_split,
            weight_func,
            rebal_dates,
            cost_bps=config_bt["cost_bps_per_side"]
        )
        
        results[name] = (port_returns, weights_hist)
    
    print("=" * 60)
    
    return results


def main():
    """Main execution pipeline."""
    # Load config
    bt_cfg = load_backtest_config("configs/backtest.yaml")
    set_seeds(bt_cfg["seed"])
    
    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION: Full Pipeline Execution")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    
    print(f"  Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"  Assets: {returns.shape[1]}")
    print(f"  Trading days: {len(returns)}")
    
    # Run baselines on validation
    print("\n" + "=" * 70)
    print("VALIDATION SET RESULTS")
    print("=" * 70)
    
    val_results = run_baselines(returns, bt_cfg, split_name="val")
    
    # Compute metrics
    val_returns_dict = {name: rets for name, (rets, _) in val_results.items()}
    val_weights_dict = {name: weights for name, (_, weights) in val_results.items()}
    
    val_metrics_table = create_metrics_table(
        val_returns_dict, val_weights_dict, rf_rate=0.0, periods_per_year=252
    )
    
    print("\nValidation Metrics:")
    print(val_metrics_table.round(4))
    
    # Run baselines on test
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    
    test_results = run_baselines(returns, bt_cfg, split_name="test")
    
    # Compute metrics
    test_returns_dict = {name: rets for name, (rets, _) in test_results.items()}
    test_weights_dict = {name: weights for name, (_, weights) in test_results.items()}
    
    test_metrics_table = create_metrics_table(
        test_returns_dict, test_weights_dict, rf_rate=0.0, periods_per_year=252
    )
    
    print("\nTest Metrics:")
    print(test_metrics_table.round(4))
    
    # Save baseline results
    print("\nSaving baseline results...")
    Path("out/reports").mkdir(parents=True, exist_ok=True)
    
    val_metrics_table.to_csv("out/reports/baselines_val_metrics.csv")
    test_metrics_table.to_csv("out/reports/baselines_test_metrics.csv")
    
    # Save individual baseline returns and weights
    for name, (rets, weights) in test_results.items():
        safe_name = name.lower().replace(" ", "_")
        rets.to_csv(f"out/reports/baseline_{safe_name}_test_returns.csv")
        weights.to_csv(f"out/reports/baseline_{safe_name}_test_weights.csv")
    
    print("  ✓ Baseline results saved to out/reports/")
    
    # Instructions for model training
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nBaselines complete. To train models, run:")
    print("  1. Track A (TCN Policy):     python3 -m src.train_policy")
    print("  2. Track B (Two-Stage):      python3 -m src.train_two_stage")
    print("  3. Generate final report:    python3 -m src.create_report")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

