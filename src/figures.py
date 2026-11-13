"""
Paper-style figures from the CSV files in results/.

    python -m src.figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES, FIG = Path("results"), Path("figures")

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
    "legend.frameon": False, "svg.fonttype": "none",
})
STYLE = {
    "TCN policy (seed mean)": dict(color="#b2182b", lw=1.3, zorder=5),
    "Equal weight": dict(color="#222222", lw=0.9, ls=(0, (4, 2))),
    "Momentum (top 7, 12m)": dict(color="#2166ac"),
    "SPY": dict(color="#8c8c8c"),
    "Inverse volatility": dict(color="#5aae61"),
    "Minimum variance": dict(color="#9970ab"),
}


def fig_nav() -> None:
    r = pd.read_csv(RES / "daily_returns_test.csv", index_col=0, parse_dates=True)
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})
    for name, st in STYLE.items():
        a.plot(nav.index, nav[name], label=name.replace(" (seed mean)", ""), **st)
        b.plot(dd.index, dd[name], **st)
    a.set_ylabel("Growth of 1 (net of costs)")
    b.set_ylabel("Drawdown")
    a.legend(ncol=3, loc="upper left")
    a.set_title("(a) Test period, weekly rebalancing, 10 bps per side", loc="left")
    b.set_title("(b) Drawdown", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_test_nav.svg")
    plt.close(fig)


def fig_policy() -> None:
    log = pd.read_csv(RES / "training_log.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1, 1.4]})
    for seed, g in log.groupby("seed"):
        a.plot(g["epoch"], g["val_sharpe_daily"], marker="o", ms=2, lw=0.8, label=f"seed {seed}")
        best = g.loc[g["val_sharpe_daily"].idxmax()]
        a.plot(best["epoch"], best["val_sharpe_daily"], "k*", ms=5, zorder=5)
    a.set_xlabel("Epoch")
    a.set_ylabel("Validation Sharpe (selection metric)")
    a.set_title("(a) Early stopping; * = selected epoch", loc="left")
    a.legend(fontsize=7, ncol=2, loc="lower right")

    devs = []
    for p in sorted((RES / "runs").glob("tcn_seed*_test_weights.csv")):
        w = pd.read_csv(p, index_col=0, parse_dates=True)
        devs.append(w - 1 / w.shape[1])
    dev = pd.concat(devs) * 100
    order = dev.columns
    b.boxplot([dev[c].values for c in order], tick_labels=order, whis=(1, 99), showfliers=False,
              widths=0.6, medianprops=dict(color="#b2182b"), boxprops=dict(lw=0.6),
              whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    b.axhline(0, color="k", lw=0.5)
    b.set_ylabel("Weight minus 1/15 (pp)")
    b.set_title("(b) TCN weights on test rebalances, 5 seeds", loc="left")
    b.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_training_and_weights.svg")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_nav()
    fig_policy()


if __name__ == "__main__":
    main()
