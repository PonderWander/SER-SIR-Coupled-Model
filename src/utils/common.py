"""
utils/common.py
───────────────
Shared utilities: config loading, logging setup, time transforms,
rolling statistics, reproducibility helpers.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console and optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML config file. Returns nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_sector_graph_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load sector graph YAML definition."""
    return load_config(path)


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Time Series Transforms ───────────────────────────────────────────────────

def resample_daily_to_weekly(
    df: pd.DataFrame,
    agg: str = "mean",
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Resample daily DataFrame to weekly (Sunday end-of-period)."""
    df = _ensure_dt_index(df, date_col)
    return df.resample("W").agg(agg)


def resample_daily_to_monthly(
    df: pd.DataFrame,
    agg: str = "mean",
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Resample daily DataFrame to monthly (month-end)."""
    df = _ensure_dt_index(df, date_col)
    return df.resample("ME").agg(agg)


def align_panel_by_timestamp(
    panels: Dict[str, pd.DataFrame],
    freq: str = "ME",
    method: str = "ffill",
    limit: int = 3,
) -> pd.DataFrame:
    """
    Align multiple sector-level DataFrames to a common DatetimeIndex.

    Parameters
    ----------
    panels : dict of {label: DataFrame} with DatetimeIndex
    freq   : target frequency ('ME', 'W', 'D')
    method : fill method for missing periods
    limit  : max consecutive fill periods

    Returns
    -------
    Wide panel DataFrame with MultiIndex columns (sector, variable)
    """
    # Build union index
    all_dates: set = set()
    for df in panels.values():
        all_dates.update(df.index.tolist())
    idx = pd.date_range(
        start=min(all_dates), end=max(all_dates), freq=freq
    )
    aligned = {}
    for label, df in panels.items():
        df2 = df.reindex(idx)
        df2 = df2.fillna(method=method, limit=limit)
        aligned[label] = df2
    return pd.concat(aligned, axis=1)


def apply_sector_specific_lags(
    df: pd.DataFrame,
    lag_map: Dict[str, int],
) -> pd.DataFrame:
    """
    Shift columns forward by sector-specific lag (in periods).

    Parameters
    ----------
    df      : DataFrame with column names matching lag_map keys
    lag_map : {column: n_periods_lag}
    """
    out = df.copy()
    for col, lag in lag_map.items():
        if col in out.columns and lag > 0:
            out[col] = out[col].shift(lag)
    return out


def rolling_zscore(
    series: pd.Series,
    window: int = 12,
    min_periods: int = 3,
) -> pd.Series:
    """Rolling z-score normalisation."""
    mu = series.rolling(window, min_periods=min_periods).mean()
    sigma = series.rolling(window, min_periods=min_periods).std()
    return (series - mu) / sigma.replace(0, np.nan)


def rolling_volatility(
    series: pd.Series,
    window: int = 3,
    min_periods: int = 2,
) -> pd.Series:
    """Rolling standard deviation as volatility proxy."""
    return series.rolling(window, min_periods=min_periods).std()


def percent_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change over `periods`."""
    return series.pct_change(periods=periods)


def log_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Log difference over `periods`."""
    return np.log(series).diff(periods)


def minmax_normalize(series: pd.Series, floor: float = 0.0, cap: float = 1.0) -> pd.Series:
    """Min-max normalize to [0,1]; clip to [floor, cap]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    norm = (series - mn) / (mx - mn)
    return norm.clip(floor, cap)


def _ensure_dt_index(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Ensure DataFrame has DatetimeIndex, optionally from a column."""
    df = df.copy()
    if date_col and date_col in df.columns:
        df.index = pd.to_datetime(df[date_col])
        df = df.drop(columns=[date_col])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


# ─── Panel Utilities ──────────────────────────────────────────────────────────

def audit_alignment(df: pd.DataFrame, name: str = "panel") -> Dict[str, Any]:
    """Log alignment audit for a panel DataFrame."""
    log = get_logger("alignment_audit")
    info: Dict[str, Any] = {
        "name": name,
        "shape": df.shape,
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "freq": pd.infer_freq(df.index),
        "missing_pct": df.isnull().mean().to_dict(),
    }
    log.info(
        f"[{name}] shape={info['shape']}, "
        f"range={info['start']}→{info['end']}, "
        f"freq={info['freq']}, "
        f"missing={round(df.isnull().mean().mean()*100, 1)}%"
    )
    return info


def safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division, filling div-by-zero with `fill`."""
    return num.div(den.replace(0, np.nan)).fillna(fill)


# ─── Output Helpers ───────────────────────────────────────────────────────────

def save_outputs(
    df: pd.DataFrame,
    name: str,
    output_dir: Union[str, Path],
    parquet: bool = True,
    csv: bool = True,
) -> List[Path]:
    """Save DataFrame to parquet and/or CSV. Returns list of written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if parquet:
        p = output_dir / f"{name}.parquet"
        df.to_parquet(p)
        paths.append(p)
    if csv:
        p = output_dir / f"{name}.csv"
        df.to_csv(p)
        paths.append(p)
    return paths
