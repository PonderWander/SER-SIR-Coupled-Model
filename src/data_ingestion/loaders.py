"""
data_ingestion/loaders.py
──────────────────────────
Pluggable data adapters for:
  - FRED API (CPI, PPI, efficiency proxies)
  - Our World In Data COVID series (cached CSV)
  - Synthetic fallback generator (when APIs unavailable)

All adapters return clean pandas DataFrames with DatetimeIndex.
Raw files are cached in data/raw/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

# ─── FRED Adapter ─────────────────────────────────────────────────────────────

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(
    series_id: str,
    api_key: str,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
    cache_dir: Optional[Path] = None,
) -> pd.Series:
    """
    Fetch a single FRED series. Returns pd.Series with DatetimeIndex.
    Caches raw JSON to cache_dir if provided.
    """
    cache_dir = Path(cache_dir) if cache_dir else None
    cache_key = f"fred_{series_id}_{start_date}_{end_date}.json"
    cache_path = cache_dir / cache_key if cache_dir else None

    # Try cache first
    if cache_path and cache_path.exists():
        logger.info(f"[FRED] Loading cached {series_id}")
        with open(cache_path) as f:
            data = json.load(f)
    else:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": "m",
        }
        logger.info(f"[FRED] Fetching {series_id}...")
        resp = requests.get(FRED_BASE, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"FRED API error for {series_id}: {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
        time.sleep(0.3)  # rate-limit courtesy

    obs = data.get("observations", [])
    records = [(r["date"], r["value"]) for r in obs if r["value"] != "."]
    if not records:
        logger.warning(f"[FRED] No observations for {series_id}")
        return pd.Series(dtype=float, name=series_id)

    dates, values = zip(*records)
    s = pd.Series(
        pd.to_numeric(values, errors="coerce"),
        index=pd.to_datetime(dates),
        name=series_id,
    )
    return s.sort_index()


def fetch_fred_panel(
    series_map: Dict[str, str],
    api_key: str,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch multiple FRED series. Returns wide DataFrame, columns = sector names.
    series_map: {sector_name: fred_series_id}
    """
    frames = {}
    for name, sid in series_map.items():
        try:
            s = fetch_fred_series(sid, api_key, start_date, end_date, cache_dir)
            frames[name] = s
        except Exception as e:
            logger.warning(f"[FRED] Failed {name} ({sid}): {e}. Will use synthetic.")
            frames[name] = None
    df = pd.DataFrame({k: v for k, v in frames.items() if v is not None})
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ─── OWID COVID Adapter ───────────────────────────────────────────────────────

OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)

OWID_COLS = [
    "date",
    "new_cases_smoothed",
    "new_deaths_smoothed",
    "hosp_patients",
    "people_vaccinated",
    "stringency_index",
    "new_cases_smoothed_per_million",
]


def fetch_owid_covid(
    country: str = "USA",
    cache_dir: Optional[Path] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
) -> pd.DataFrame:
    """
    Fetch OWID COVID data for a given ISO country code.
    Returns DataFrame with DatetimeIndex.
    """
    cache_dir = Path(cache_dir) if cache_dir else None
    cache_path = cache_dir / f"owid_covid_{country}.parquet" if cache_dir else None

    if cache_path and cache_path.exists():
        logger.info(f"[OWID] Loading cached COVID data for {country}")
        df = pd.read_parquet(cache_path)
    else:
        logger.info(f"[OWID] Downloading COVID data for {country}...")
        try:
            raw = pd.read_csv(OWID_URL, low_memory=False)
            df = raw[raw["iso_code"] == country].copy()
            keep = [c for c in OWID_COLS if c in df.columns]
            df = df[keep].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"[OWID] Download failed: {e}. Using synthetic.")
            df = _synthetic_covid(start_date, end_date)

    mask = (df.index >= start_date) & (df.index <= end_date)
    return df.loc[mask]


# ─── Synthetic Data Generator ─────────────────────────────────────────────────
# Used as fallback when API keys are missing or network unavailable.

def generate_synthetic_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic but plausible COVID-era data:
    - covid_df: epidemic series
    - cpi_df:   sector CPI indices
    - efficiency_df: supply-chain proxies

    Returns (covid_df, cpi_df, efficiency_df)
    """
    rng = np.random.default_rng(seed)
    idx_daily = pd.date_range(start_date, end_date, freq="D")
    idx_monthly = pd.date_range(start_date, end_date, freq="ME")
    T_d = len(idx_daily)
    T_m = len(idx_monthly)

    # ── Epidemic ────────────────────────────────────────────────────
    covid_df = _synthetic_covid(start_date, end_date, rng)

    # ── CPI sectors ─────────────────────────────────────────────────
    # Base 100 in Jan 2020; apply realistic shock profiles
    sectors = [
        "food_at_home",
        "food_away_from_home",
        "energy",
        "gasoline",
        "electricity",
        "shelter",
        "transportation_services",
        "household_goods",
        "medical_services",
        "apparel",
        "ppi_goods",
        "ppi_food",
        "ppi_energy",
    ]

    # Shock parameters per sector: (peak_month, peak_magnitude, recovery_months)
    shock_params = {
        "food_at_home":           (14,  8.5, 18),
        "food_away_from_home":    (12,  6.0, 24),
        "energy":                 (18, 35.0, 12),
        "gasoline":               (16, 50.0, 10),
        "electricity":            (20, 12.0, 14),
        "shelter":                (24, 10.0, 30),
        "transportation_services":(10, -15.0, 20),
        "household_goods":        (15, 12.0, 18),
        "medical_services":       (8,   5.0, 12),
        "apparel":                (8, -10.0, 16),
        "ppi_goods":              (18, 18.0, 14),
        "ppi_food":               (16, 22.0, 14),
        "ppi_energy":             (16, 60.0, 10),
    }

    cpi_data = {}
    t = np.arange(T_m)
    for sec, (peak_m, peak_mag, rec_m) in shock_params.items():
        trend = 2.0 * t / 12  # 2% annual base drift
        shock = peak_mag * np.exp(-0.5 * ((t - peak_m) / (rec_m / 2.5)) ** 2)
        noise = rng.normal(0, abs(peak_mag) * 0.05, T_m)
        idx_series = 100 + trend + shock + noise
        cpi_data[sec] = np.maximum(idx_series, 50)

    cpi_df = pd.DataFrame(cpi_data, index=idx_monthly)

    # ── Efficiency proxies ───────────────────────────────────────────
    inv_sales = 1.35 + 0.25 * np.exp(-0.5 * ((t - 5) / 4) ** 2) + rng.normal(0, 0.03, T_m)
    util = 77 - 10 * np.exp(-0.5 * ((t - 4) / 3) ** 2) + 8 * (t / T_m) + rng.normal(0, 1.2, T_m)
    unemp = 3.5 + 11 * np.exp(-0.5 * ((t - 4) / 2) ** 2) - 9 * (t / T_m).clip(0, 1) + rng.normal(0, 0.2, T_m)

    efficiency_df = pd.DataFrame(
        {
            "inventory_sales_ratio": inv_sales.clip(1.1, 2.0),
            "industrial_utilization": util.clip(60, 90),
            "unemployment_rate": unemp.clip(3.0, 15.0),
            "initial_claims": (unemp * 1e5 + rng.normal(0, 5000, T_m)).clip(100000, 1e7),
            "retail_inventories": (1000 + 50 * rng.standard_normal(T_m)).clip(800, 1200),
            "mfg_inventories": (700 + 40 * rng.standard_normal(T_m)).clip(550, 900),
        },
        index=idx_monthly,
    )

    logger.info("[Synthetic] Generated synthetic dataset.")
    return covid_df, cpi_df, efficiency_df


def _synthetic_covid(
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Generate synthetic COVID epidemic daily series with multiple waves."""
    if rng is None:
        rng = np.random.default_rng(42)
    idx = pd.date_range(start_date, end_date, freq="D")
    T = len(idx)
    t = np.arange(T)

    # Three epidemic waves
    waves = [
        (90,  20, 30000),    # Wave 1: spring 2020
        (270, 25, 80000),    # Wave 2: winter 2020-21
        (450, 30, 150000),   # Wave 3: delta/omicron
    ]
    cases = np.zeros(T)
    for center, width, peak in waves:
        cases += peak * np.exp(-((t - center) ** 2) / (2 * width ** 2))
    cases += rng.poisson(cases.clip(1) * 0.1)

    hosp = np.convolve(cases * 0.05, np.ones(7) / 7, mode="same")
    deaths = np.convolve(cases * 0.012, np.ones(14) / 14, mode="same")

    return pd.DataFrame(
        {
            "new_cases_smoothed": cases.clip(0),
            "new_deaths_smoothed": deaths.clip(0),
            "hosp_patients": hosp.clip(0),
            "stringency_index": np.clip(70 * (cases / cases.max()) + rng.normal(0, 3, T), 0, 100),
        },
        index=idx,
    )
