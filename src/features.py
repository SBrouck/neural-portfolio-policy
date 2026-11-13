"""
Feature engineering for portfolio optimization.
All features computed with past data only - no look-ahead bias.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


def load_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices panel and risk-free rates."""
    prices = pd.read_parquet(Path(data_dir) / "processed" / "prices_panel.parquet")
    rf = pd.read_parquet(Path(data_dir) / "rf" / "tbill_3m_daily.parquet")
    rf = rf.set_index("date")["rf_daily"]
    return prices, rf


def compute_returns(prices: pd.DataFrame, horizons: List[int] = [1, 5, 20]) -> Dict[str, pd.DataFrame]:
    """
    Compute log returns at multiple horizons.
    Returns are computed using past prices only.
    
    For horizon h: ret_t = log(price_t / price_{t-h})
    This is the return realized OVER the period [t-h, t].
    
    Returns dict of {f"ret_{h}d": DataFrame with returns for each asset}
    """
    close = prices["adj_close"]
    returns_dict = {}
    
    for h in horizons:
        log_ret = np.log(close / close.shift(h))
        returns_dict[f"ret_{h}d"] = log_ret
    
    return returns_dict


def compute_ewma_vol_dict(returns_1d: pd.DataFrame, 
                          lambda_decay: float = 0.94,
                          min_periods: int = 20) -> pd.DataFrame:
    """
    Compute EWMA volatility per asset.
    Uses only past returns at each time t.
    
    Args:
        returns_1d: DataFrame of 1-day returns, columns are tickers
        
    Returns:
        DataFrame of EWMA volatilities with same structure
    """
    # Squared returns for volatility
    ret_sq = returns_1d ** 2
    
    # EWMA of squared returns, then sqrt
    # Use adjust=False to ensure we only use past data
    ewma_var = ret_sq.ewm(alpha=1-lambda_decay, min_periods=min_periods, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    
    return ewma_vol


def compute_momentum_12m(prices: pd.DataFrame, 
                         lookback_days: int = 252) -> pd.DataFrame:
    """
    12-month momentum as z-score using expanding window.
    mom_t = (price_t / price_{t-252} - 1)
    z_t = (mom_t - expanding_mean_t) / expanding_std_t
    """
    close = prices["adj_close"]
    
    # Raw momentum
    mom_raw = (close / close.shift(lookback_days)) - 1.0
    
    # Z-score with expanding window (only past data)
    mom_z = pd.DataFrame(index=close.index, columns=close.columns)
    
    for col in close.columns:
        series = mom_raw[col]
        # expanding mean and std use only past data
        expanding_mean = series.expanding(min_periods=60).mean()
        expanding_std = series.expanding(min_periods=60).std()
        mom_z[col] = (series - expanding_mean) / (expanding_std + 1e-8)
    
    mom_z.columns = [f"mom_12m_z_{col}" for col in mom_z.columns]
    
    return mom_z


def compute_momentum_rank(prices: pd.DataFrame,
                         lookback_days: int = 252) -> pd.DataFrame:
    """
    Cross-sectional rank of 12-month momentum.
    Rank is computed within each date across all assets.
    Returns percentile rank [0, 1].
    """
    close = prices["adj_close"]
    mom_raw = (close / close.shift(lookback_days)) - 1.0
    
    # Rank across columns (assets) for each row (date)
    mom_rank = mom_raw.rank(axis=1, pct=True)
    mom_rank.columns = [f"mom_12m_rank_{col}" for col in mom_rank.columns]
    
    return mom_rank


def compute_skewness_60d(returns_1d: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling skewness as asymmetry proxy.
    Uses only past 60 days at each time t.
    """
    skew = returns_1d.rolling(window=window, min_periods=30).skew()
    return skew


def compute_drawdown_local(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Rolling drawdown over past window.
    DD_t = (price_t - rolling_max_t) / rolling_max_t
    """
    close = prices["adj_close"]
    rolling_max = close.rolling(window=window, min_periods=60).max()
    drawdown = (close - rolling_max) / (rolling_max + 1e-8)
    return drawdown


def compute_cross_sectional_dispersion(returns_1d: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Cross-sectional dispersion of returns.
    Std of returns across assets at each time t, smoothed with rolling window.
    """
    # Std across columns (assets) for each row (date)
    cross_std = returns_1d.std(axis=1)
    
    # Smooth with rolling window
    dispersion = cross_std.rolling(window=window, min_periods=10).mean()
    
    return dispersion


def compute_vol_regime(returns_1d: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.Series:
    """
    Volatility regime indicator.
    ratio = short_term_vol / long_term_vol
    > 1 means elevated volatility regime
    """
    # Market-wide volatility (average across assets)
    avg_returns = returns_1d.mean(axis=1)
    
    short_vol = avg_returns.rolling(window=short_window, min_periods=10).std()
    long_vol = avg_returns.rolling(window=long_window, min_periods=30).std()
    
    vol_ratio = short_vol / (long_vol + 1e-8)
    
    return vol_ratio


def build_feature_matrix(prices: pd.DataFrame,
                        rf: pd.Series,
                        feature_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build complete feature matrix with all requested features.
    
    Args:
        prices: Price panel with MultiIndex columns (field, ticker)
        rf: Risk-free rate series
        feature_list: List of feature names to compute
        
    Returns:
        features: Wide DataFrame with MultiIndex columns (feature, ticker)
        targets: Forward excess returns for each asset
    """
    tickers = prices["adj_close"].columns.tolist()
    
    # Compute base features
    returns_dict = compute_returns(prices, horizons=[1, 5, 20])
    
    # Build feature dict
    all_features = {}
    
    if "ret_1d" in feature_list:
        for ticker in tickers:
            all_features[("ret_1d", ticker)] = returns_dict["ret_1d"][ticker]
    
    if "ret_5d" in feature_list:
        for ticker in tickers:
            all_features[("ret_5d", ticker)] = returns_dict["ret_5d"][ticker]
    
    if "ret_20d" in feature_list:
        for ticker in tickers:
            all_features[("ret_20d", ticker)] = returns_dict["ret_20d"][ticker]
    
    if "ewma_vol" in feature_list:
        ewma = compute_ewma_vol_dict(returns_dict["ret_1d"])
        for ticker in tickers:
            all_features[("ewma_vol", ticker)] = ewma[ticker]
    
    if "mom_12m_z" in feature_list:
        mom_z = compute_momentum_12m(prices, lookback_days=252)
        for ticker in tickers:
            all_features[("mom_12m_z", ticker)] = mom_z[f"mom_12m_z_{ticker}"]
    
    if "mom_12m_rank" in feature_list:
        mom_rank = compute_momentum_rank(prices, lookback_days=252)
        for ticker in tickers:
            all_features[("mom_12m_rank", ticker)] = mom_rank[f"mom_12m_rank_{ticker}"]
    
    if "skew_60d" in feature_list:
        skew = compute_skewness_60d(returns_dict["ret_1d"], window=60)
        for ticker in tickers:
            all_features[("skew_60d", ticker)] = skew[ticker]
    
    if "drawdown_local" in feature_list:
        dd = compute_drawdown_local(prices, window=252)
        for ticker in tickers:
            all_features[("drawdown_local", ticker)] = dd[ticker]
    
    # Cross-sectional features (shared across all assets)
    if "dispersion" in feature_list:
        dispersion = compute_cross_sectional_dispersion(returns_dict["ret_1d"], window=20)
        for ticker in tickers:
            all_features[("dispersion", ticker)] = dispersion
    
    if "vol_regime" in feature_list:
        vol_regime = compute_vol_regime(returns_dict["ret_1d"], short_window=20, long_window=60)
        for ticker in tickers:
            all_features[("vol_regime", ticker)] = vol_regime
    
    # Assemble into MultiIndex DataFrame
    features = pd.DataFrame(all_features)
    features.columns = pd.MultiIndex.from_tuples(features.columns, names=["feature", "ticker"])
    
    # Compute targets: forward 1-period excess return
    # target_t = r_{t+1} - rf_{t+1}
    ret_1d = returns_dict["ret_1d"]
    
    # Align rf with returns
    rf_aligned = rf.reindex(ret_1d.index)
    
    # Forward shift returns to get next period return
    targets = ret_1d.shift(-1)  # This is r_{t+1} when indexed at t
    
    # Subtract risk-free rate
    for ticker in tickers:
        targets[ticker] = targets[ticker] - rf_aligned.shift(-1)
    
    targets.columns = pd.MultiIndex.from_product([["target"], targets.columns], 
                                                  names=["feature", "ticker"])
    
    return features, targets


def create_sequences(features: pd.DataFrame,
                    targets: pd.DataFrame,
                    window_len: int,
                    start_date: str,
                    end_date: str) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create sequences for training.
    
    Args:
        features: Feature matrix with MultiIndex (feature, ticker)
        targets: Target matrix with MultiIndex (feature, ticker)
        window_len: Number of time steps in each sequence
        start_date: Start of period
        end_date: End of period
        
    Returns:
        X: Array of shape (n_samples, window_len, n_features)
        y: Array of shape (n_samples, n_assets)
        dates: DatetimeIndex of shape (n_samples,) - the prediction date
    """
    # Filter to date range
    mask = (features.index >= start_date) & (features.index <= end_date)
    features_filt = features[mask]
    targets_filt = targets[mask]
    
    # Forward fill (past values only) for at most 5 rows, then drop incomplete rows
    features_filt = features_filt.ffill(limit=5)
    
    # Drop rows that still have NaN after forward fill
    valid_idx = features_filt.dropna().index
    valid_idx = valid_idx.intersection(targets_filt.dropna().index)
    
    features_filt = features_filt.loc[valid_idx]
    targets_filt = targets_filt.loc[valid_idx]
    
    n_samples = len(features_filt) - window_len + 1
    if n_samples <= 0:
        return np.array([]), np.array([]), pd.DatetimeIndex([])
    
    # Get number of features per asset
    n_features_per_asset = len(features_filt.columns.get_level_values(0).unique())
    n_assets = len(features_filt.columns.get_level_values(1).unique())
    
    X = np.zeros((n_samples, window_len, n_features_per_asset * n_assets))
    y = np.zeros((n_samples, n_assets))
    dates = []
    
    for i in range(n_samples):
        # Features from [i:i+window_len]
        window = features_filt.iloc[i:i+window_len]
        X[i] = window.values
        
        # Target at i+window_len-1 (last timestep of window)
        y[i] = targets_filt.iloc[i+window_len-1].values
        dates.append(features_filt.index[i+window_len-1])
    
    return X, y, pd.DatetimeIndex(dates)
