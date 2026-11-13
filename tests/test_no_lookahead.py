"""Causality checks for the feature pipeline and the backtest engine."""
import numpy as np
import pandas as pd

from src.backtest_loop import get_rebalance_dates, run_backtest
from src.features import build_feature_matrix

FEATURES = ["ret_1d", "ret_5d", "ret_20d", "ewma_vol", "mom_12m_z", "mom_12m_rank",
            "skew_60d", "drawdown_local", "dispersion", "vol_regime"]


def _panel(seed=0, n=400, tickers=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(tickers))), axis=0)) * 100
    cols = pd.MultiIndex.from_product([["adj_close", "volume"], tickers])
    panel = pd.DataFrame(np.hstack([close, np.ones_like(close)]), index=idx, columns=cols)
    rf = pd.Series(1e-4, index=idx)
    return panel, rf


def test_features_do_not_use_future_prices():
    panel, rf = _panel()
    cut = 300
    shocked = panel.copy()
    shocked.iloc[cut:, :3] *= 1.5  # change every price after the cut
    f0, _ = build_feature_matrix(panel, rf, FEATURES)
    f1, _ = build_feature_matrix(shocked, rf, FEATURES)
    pd.testing.assert_frame_equal(f0.iloc[:cut], f1.iloc[:cut])
    assert not np.allclose(f0.iloc[cut:].fillna(0), f1.iloc[cut:].fillna(0))


def test_target_is_next_day_return():
    panel, rf = _panel()
    _, targets = build_feature_matrix(panel, rf, ["ret_1d"])
    close = panel["adj_close"]["AAA"]
    expected = np.log(close.shift(-1) / close) - 1e-4
    np.testing.assert_allclose(targets[("target", "AAA")].iloc[:-1], expected.iloc[:-1])


def test_backtest_weights_only_see_past_returns():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=120)
    rets = pd.DataFrame(rng.normal(0, 0.01, (120, 3)), index=idx, columns=list("abc"))
    rebal = get_rebalance_dates(idx, "W-FRI")
    seen = {}

    def fn(date, hist):
        assert hist.index.max() < date if len(hist) else True
        seen[date] = hist.index.max() if len(hist) else None
        w = np.maximum(hist.tail(20).mean().values, 0) + 1e-6 if len(hist) else np.ones(3)
        return w

    r0, w0, _ = run_backtest(rets, fn, rebal, cost_bps=10, eval_start=idx[40])
    shocked = rets.copy()
    shocked.iloc[80:] += 0.05
    r1, w1, _ = run_backtest(shocked, fn, rebal, cost_bps=10, eval_start=idx[40])
    pd.testing.assert_frame_equal(w0.loc[: idx[79]], w1.loc[: idx[79]])
    pd.testing.assert_series_equal(r0.loc[: idx[79]], r1.loc[: idx[79]])


def test_costs_are_charged_on_turnover():
    idx = pd.bdate_range("2020-01-06", periods=10)
    rets = pd.DataFrame(0.0, index=idx, columns=list("ab"))
    flip = iter([[1, 0], [0, 1], [1, 0]])
    rebal = pd.DatetimeIndex([idx[0], idx[5]])
    r, _, t = run_backtest(rets, lambda d, h: np.array(next(flip), float), rebal, cost_bps=10)
    assert t.iloc[0] == 0 and t.iloc[5] == 2.0
    assert np.isclose(r.iloc[5], -10 / 1e4 * 2.0)
