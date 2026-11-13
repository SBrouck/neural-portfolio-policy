"""Performance metrics on daily net returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 252


def excess(returns: pd.Series, rf_daily: pd.Series) -> pd.Series:
    """Daily excess returns over the 3-month T-bill (DTB3 / 252, last value carried forward)."""
    return returns - rf_daily.reindex(returns.index).ffill().fillna(0.0)


def sharpe(ex: pd.Series) -> float:
    return float(ex.mean() / ex.std() * np.sqrt(DAYS))


def max_drawdown(returns: pd.Series) -> float:
    nav = (1 + returns).cumprod()
    return float((nav / nav.cummax() - 1).min())


def summary(returns: pd.Series, turnover: pd.Series, rf_daily: pd.Series,
            bench: pd.Series | None = None) -> dict:
    """
    Annualised metrics. Turnover is sum_i |w_new - w_drifted| per rebalance, so
    10 bps per side costs 10 bps x turnover; cost drag is reported per year.
    """
    ex = excess(returns, rf_daily)
    years = len(returns) / DAYS
    n_rebal = int((turnover > 0).sum()) or 1
    out = {
        "cagr": float((1 + returns).prod() ** (1 / years) - 1),
        "vol": float(returns.std() * np.sqrt(DAYS)),
        "sharpe_excess": sharpe(ex),
        "max_dd": max_drawdown(returns),
        "turnover_per_rebalance": float(turnover.sum() / n_rebal),
        "turnover_per_year": float(turnover.sum() / years),
    }
    if bench is not None:
        b = bench.reindex(returns.index)
        out["beta_spy"] = float(np.cov(returns, b)[0, 1] / np.var(b, ddof=1))
    return out


def block_bootstrap_sharpe_diff(ex_a: pd.Series, ex_b: pd.Series, n_boot: int = 2000,
                                block: int = 21, seed: int = 0) -> tuple[float, float, float]:
    """
    Sharpe(a) - Sharpe(b) on paired daily excess returns, with a moving-block
    bootstrap 95% interval (blocks of `block` days keep short-range dependence).
    """
    a, b = ex_a.values, ex_b.values
    n = len(a)
    rng = np.random.default_rng(seed)
    point = sharpe(ex_a) - sharpe(ex_b)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        sa, sb = a[idx], b[idx]
        diffs[k] = (sa.mean() / sa.std(ddof=1) - sb.mean() / sb.std(ddof=1)) * np.sqrt(DAYS)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)
