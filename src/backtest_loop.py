"""
Backtest engine shared by the learned policy and the baselines.

Timing convention. The panel holds daily close-to-close simple returns, indexed by
the date of the closing price. On a rebalance date t the strategy receives the
returns strictly before t (information available at the close of t-1), sets new
weights, pays the transaction cost, and earns the return of date t. Between
rebalances, weights drift with realised returns.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def load_backtest_config(config_path: str = "configs/backtest.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_rebalance_dates(dates: pd.DatetimeIndex,
                        frequency: str = "W-FRI",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DatetimeIndex:
    """Trading dates closest to each calendar point of `frequency` (e.g. every Friday)."""
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if frequency == "D":
        return dates
    freq_dates = pd.date_range(start=dates[0], end=dates[-1], freq=frequency)
    idx = dates.get_indexer(freq_dates, method="nearest")
    return pd.DatetimeIndex(sorted(set(dates[idx])))


def run_backtest(
    returns_df: pd.DataFrame,
    weight_function: Callable[[pd.Timestamp, pd.DataFrame], np.ndarray],
    rebalance_dates: pd.DatetimeIndex,
    cost_bps: float = 10.0,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Run a long-only backtest over [eval_start, eval_end].

    `returns_df` may start before `eval_start`: earlier rows are only passed to
    `weight_function` as history (warm-up), never traded.

    Returns:
        portfolio_returns: daily net returns
        weights: weights held over each date (after rebalancing, before drift)
        turnover: sum_i |w_new - w_drifted| on each date (0 between rebalances)
    """
    n_assets = returns_df.shape[1]
    start = pd.Timestamp(eval_start) if eval_start else returns_df.index[0]
    end = pd.Timestamp(eval_end) if eval_end else returns_df.index[-1]
    rebal = set(rebalance_dates)

    w = np.ones(n_assets) / n_assets
    out_ret, out_w, out_turn, out_dates = [], [], [], []
    first = True
    for i, date in enumerate(returns_df.index):
        if date < start or date > end:
            continue
        turnover = 0.0
        if first or date in rebal:
            hist = returns_df.iloc[:i]  # strictly before `date`
            w_new = np.clip(np.asarray(weight_function(date, hist), dtype=float), 0, None)
            w_new = w_new / w_new.sum() if w_new.sum() > 0 else np.ones(n_assets) / n_assets
            # The initial allocation from cash is not charged
            turnover = 0.0 if first else float(np.abs(w_new - w).sum())
            w = w_new
            first = False
        r = returns_df.iloc[i].values
        out_ret.append(float(w @ r) - cost_bps / 1e4 * turnover)
        out_w.append(w.copy())
        out_turn.append(turnover)
        out_dates.append(date)
        w = w * (1 + r)
        w = w / w.sum()

    idx = pd.DatetimeIndex(out_dates)
    return (pd.Series(out_ret, index=idx, name="portfolio"),
            pd.DataFrame(out_w, index=idx, columns=returns_df.columns),
            pd.Series(out_turn, index=idx, name="turnover"))
