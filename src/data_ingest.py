from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from pandas_datareader import data as pdr

# ----------------------------
# Helpers
# ----------------------------
def _ensure_dirs():
    Path("data/raw/yahoo").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/rf").mkdir(parents=True, exist_ok=True)

def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _nyse_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    cal = mcal.get_calendar("XNYS")
    sched = cal.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index.tz_localize(None))

# ----------------------------
# Universe and RF
# ----------------------------
def load_universe(path: str | Path = "meta/universe.yaml") -> List[str]:
    cfg = load_yaml(path)
    return list(cfg["universe"])

def fetch_ticker_to_parquet(ticker: str, start: str, end: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.parquet"
    # incremental update if cache exists
    df_old = None
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        last = pd.to_datetime(df_old["date"]).max()
        start_fetch = (last + pd.Timedelta(days=1)).date().isoformat()
    else:
        start_fetch = start
    if (not out_path.exists()) or (start_fetch <= end):
        df_new = yf.download(ticker, start=start_fetch, end=end,
                             auto_adjust=True, progress=False)
        if not df_new.empty:
            df_new = df_new.reset_index()
            # When auto_adjust=True, column is 'Close' not 'Adj Close'
            if "Close" in df_new.columns and "Adj Close" not in df_new.columns:
                df_new = df_new[["Date", "Close", "Volume"]]
                df_new.columns = ["date", "adj_close", "volume"]
            else:
                df_new = df_new[["Date", "Adj Close", "Volume"]]
                df_new.columns = ["date", "adj_close", "volume"]
            df_new["date"] = pd.to_datetime(df_new["date"]).dt.tz_localize(None)
            if df_old is not None and not df_old.empty:
                df = pd.concat([df_old, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=["date"]).sort_values("date")
            else:
                df = df_new.sort_values("date")
            df.to_parquet(out_path, index=False)
    return out_path

def _safe_positive(df: pd.DataFrame) -> pd.DataFrame:
    # Drop non positive adj_close, clip volume
    df = df[df["adj_close"] > 0].copy()
    df["volume"] = df["volume"].clip(lower=0)
    return df

def build_prices_panel(tickers: List[str], start: str, end: str,
                       raw_dir: Path = Path("data/raw/yahoo")) -> pd.DataFrame:
    dfs = []
    for tk in tickers:
        path = raw_dir / f"{tk}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache for {tk}. Run fetch first.")
        d = pd.read_parquet(path)
        d = d[(pd.to_datetime(d["date"]) >= pd.to_datetime(start)) &
              (pd.to_datetime(d["date"]) <= pd.to_datetime(end))]
        d = _safe_positive(d)
        d["ticker"] = tk
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    # NYSE calendar intersection
    nyse_idx = _nyse_days(pd.to_datetime(start), pd.to_datetime(end))
    common = set(nyse_idx)
    for tk in tickers:
        di = df.loc[df["ticker"] == tk, "date"]
        common &= set(pd.to_datetime(di))
    common = pd.DatetimeIndex(sorted(common))
    df = df[df["date"].isin(common)]
    # Pivot to MultiIndex columns
    panel = df.pivot(index="date", columns="ticker", values=["adj_close", "volume"]).sort_index()
    # Hard invariants
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    assert panel.isna().sum().sum() == 0, "No NA allowed after intersection"
    assert (panel["adj_close"] > 0).all().all()
    # Persist
    panel.to_parquet("data/processed/prices_panel.parquet")
    # Mask for availability
    mask = panel["adj_close"].notna()
    mask.to_parquet("data/processed/masks.parquet")
    return panel

def fetch_risk_free(start: str, end: str, bday_per_year: int = 252) -> pd.DataFrame:
    # Try FRED DTB3 first
    try:
        rf = pdr.DataReader("DTB3", "fred", start=start, end=end).reset_index()
        rf.columns = ["date", "tbill_3m_annual_pct"]
        rf["rf_daily"] = (rf["tbill_3m_annual_pct"] / 100.0) / bday_per_year
    except Exception:
        # Fallback to Yahoo ^IRX (13-week T-bill yield in percent)
        df = yf.download("^IRX", start=start, end=end, progress=False)
        if df.empty:
            raise RuntimeError("Risk-free fetch failed for both FRED and ^IRX")
        df = df.reset_index()
        # Handle column name variations
        close_col = "Close" if "Close" in df.columns else "Adj Close"
        rf = df[["Date", close_col]]
        rf.columns = ["date", "tbill_3m_annual_pct"]
        rf["rf_daily"] = (rf["tbill_3m_annual_pct"] / 100.0) / bday_per_year
    rf["date"] = pd.to_datetime(rf["date"]).dt.tz_localize(None)
    rf = rf.sort_values("date")
    rf.to_parquet("data/rf/tbill_3m_daily.parquet", index=False)
    return rf

# ----------------------------
# Orchestration
# ----------------------------
def run_data_pipeline(universe_yaml="meta/universe.yaml",
                      data_yaml="configs/data.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()
    uv = load_universe(universe_yaml)
    cfg = load_yaml(data_yaml)
    start = cfg["start_date"]; end = cfg["end_date"]
    # fetch and cache
    for tk in uv:
        fetch_ticker_to_parquet(tk, start=start, end=end, out_dir=Path("data/raw/yahoo"))
    # build panel and rf
    panel = build_prices_panel(uv, start=start, end=end, raw_dir=Path("data/raw/yahoo"))
    rf = fetch_risk_free(start=start, end=end, bday_per_year=cfg.get("business_days_per_year", 252))
    # sanity log
    msg = (
        f"Built panel with {panel.shape[0]} dates and {len(uv)} tickers\n"
        f"Panel path: data/processed/prices_panel.parquet\n"
        f"RF path: data/rf/tbill_3m_daily.parquet\n"
        f"Date range: {panel.index.min().date()} to {panel.index.max().date()}"
    )
    print(msg)
    return panel, rf

if __name__ == "__main__":
    run_data_pipeline()

