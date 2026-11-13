"""
Run the baselines with the same engine and summarise all strategies.

    python -m src.evaluate

Baselines see the full return history before each rebalance (no cold start at the
beginning of a split). Writes results/metrics_{val,test}.csv and
results/sharpe_diff_test.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest_loop import get_rebalance_dates, load_backtest_config, run_backtest
from src.baselines import get_baseline_strategies
from src.features import load_data
from src.metrics import block_bootstrap_sharpe_diff, excess, summary

OUT = Path("results")


def run_baselines(returns: pd.DataFrame, cfg: dict, split: str) -> dict:
    start, end = cfg[f"{split}_start"], cfg[f"{split}_end"]
    rebal = get_rebalance_dates(returns.index, cfg["rebalance"], start, end)
    res = {}
    for name, strat in get_baseline_strategies(list(returns.columns)).items():
        r, _, t = run_backtest(returns, strat.compute_weights, rebal,
                               cost_bps=cfg["cost_bps_per_side"], eval_start=start, eval_end=end)
        res[name] = (r, t)
    return res


def load_tcn_runs(split: str) -> dict:
    runs = {}
    for p in sorted((OUT / "runs").glob(f"tcn_seed*_{split}.csv")):
        seed = int(p.stem.split("_")[1].removeprefix("seed"))
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        runs[seed] = (df["ret"], df["turnover"])
    return runs


def main() -> None:
    cfg = load_backtest_config()
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()

    for split in ["val", "test"]:
        base = run_baselines(returns, cfg, split)
        spy = base["SPY"][0]
        rows = []
        tcn = load_tcn_runs(split)
        seed_rows = [summary(r, t, rf, spy) for r, t in tcn.values()]
        if seed_rows:
            df = pd.DataFrame(seed_rows)
            row = {"strategy": "TCN policy", "n_seeds": len(df)}
            for c in df.columns:
                row[c] = df[c].mean()
                row[c + "_sd"] = df[c].std(ddof=1) if len(df) > 1 else np.nan
            rows.append(row)
        for name, (r, t) in base.items():
            rows.append({"strategy": name, "n_seeds": 1, **summary(r, t, rf, spy)})
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"metrics_{split}.csv", index=False, float_format="%.6f")
        print(f"\n{split} {cfg[f'{split}_start']} .. {cfg[f'{split}_end']}")
        print(table[["strategy", "cagr", "vol", "sharpe_excess", "max_dd",
                     "turnover_per_year", "beta_spy"]].round(4).to_string(index=False))

        if split == "test" and tcn:
            diffs = []
            for seed, (r, _) in tcn.items():
                ex_t = excess(r, rf)
                w = pd.read_csv(OUT / "runs" / f"tcn_seed{seed}_test_weights.csv",
                                index_col=0, parse_dates=True)
                l1_to_ew = float(np.abs(w.values - 1 / w.shape[1]).sum(axis=1).mean())
                for other in ["Equal weight", "Momentum (top 7, 12m)"]:
                    d, lo, hi = block_bootstrap_sharpe_diff(ex_t, excess(base[other][0], rf), seed=seed)
                    diffs.append({"seed": seed, "vs": other, "sharpe_diff": d, "ci_lo": lo,
                                  "ci_hi": hi, "mean_l1_to_equal_weight": l1_to_ew})
            dd = pd.DataFrame(diffs)
            dd.to_csv(OUT / "sharpe_diff_test.csv", index=False, float_format="%.6f")
            print("\nTCN minus baseline, excess Sharpe, 95% block-bootstrap CI")
            print(dd.round(3).to_string(index=False))

        if split == "test":
            nav = pd.DataFrame({k: v[0] for k, v in base.items()})
            if tcn:
                nav["TCN policy (seed mean)"] = pd.concat([r for r, _ in tcn.values()], axis=1).mean(axis=1)
            nav.to_csv(OUT / "daily_returns_test.csv", float_format="%.8f")


if __name__ == "__main__":
    main()
