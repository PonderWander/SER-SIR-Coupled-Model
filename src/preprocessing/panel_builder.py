"""
preprocessing/panel_builder.py
────────────────────────────────
Builds the aligned sector observables panel from raw data sources.

Responsibilities:
  - Resample daily → monthly
  - Compute price pressure, volatility, efficiency proxies
  - Assemble composite panel with metadata
  - Produce audit logs and missing-data reports
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.common import (
    audit_alignment,
    log_change,
    minmax_normalize,
    percent_change,
    resample_daily_to_monthly,
    rolling_volatility,
    rolling_zscore,
    save_outputs,
)

logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


# ─── Main Panel Builder ───────────────────────────────────────────────────────

class SectorPanelBuilder:
    """
    Constructs the aligned sector observables panel.

    Parameters
    ----------
    cpi_df         : Monthly CPI index DataFrame (columns = sectors)
    efficiency_df  : Monthly efficiency/supply proxies
    covid_daily    : Daily COVID DataFrame
    config         : Dict from base_config.yaml
    sector_meta    : Dict of sector metadata from sector_graph.yaml
    """

    def __init__(
        self,
        cpi_df: pd.DataFrame,
        efficiency_df: pd.DataFrame,
        covid_daily: pd.DataFrame,
        config: Dict,
        sector_meta: Dict,
    ):
        self.cpi = cpi_df.copy()
        self.eff = efficiency_df.copy()
        self.covid_daily = covid_daily.copy()
        self.cfg = config
        self.sector_meta = sector_meta
        self.vol_window: int = config.get("ser", {}).get("volatility_window", 3)

    # ──────────────────────────────────────────────────────────────────────────

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Build the full sector observables panel.

        Returns
        -------
        panel      : MultiIndex DataFrame (date x (sector, variable))
        covid_monthly : Monthly COVID pressure series
        audit      : Dict of alignment audit info
        """
        logger.info("Building sector observables panel...")

        # 1. COVID monthly aggregation
        covid_monthly = self._aggregate_covid()

        # 2. Price pressure + volatility per sector
        price_panels = self._compute_price_transforms()

        # 3. Efficiency proxies
        eff_panel = self._compute_efficiency_proxies()

        # 4. Assemble per-sector frames
        sector_frames = {}
        sectors = [c for c in self.cpi.columns if c in self.sector_meta or c in self.cpi.columns]

        for sec in self.cpi.columns:
            if sec not in price_panels:
                continue
            meta = self.sector_meta.get(sec, {})
            covid_lag = meta.get("covid_lag_months", 1)
            ep = covid_monthly["epidemic_pressure"].shift(covid_lag)

            frame = pd.DataFrame(index=price_panels[sec].index)
            frame["price_index"]    = self.cpi[sec]
            frame["price_change"]   = price_panels[sec]["pct_change"]
            frame["log_change"]     = price_panels[sec]["log_change"]
            frame["price_zscore"]   = price_panels[sec]["zscore"]
            frame["price_pressure"] = price_panels[sec]["pressure"]
            frame["volatility"]     = price_panels[sec]["volatility"]
            frame["epidemic_pressure"] = ep.reindex(frame.index)

            # Efficiency / supply proxy: use best available
            frame["efficiency_loss"] = eff_panel.get(
                "efficiency_loss_composite", pd.Series(np.nan, index=frame.index)
            ).reindex(frame.index)
            frame["inventory_slack"] = eff_panel.get(
                "inventory_slack", pd.Series(np.nan, index=frame.index)
            ).reindex(frame.index)
            frame["labor_disruption"] = eff_panel.get(
                "labor_disruption", pd.Series(np.nan, index=frame.index)
            ).reindex(frame.index)

            # Shortage proxy: blend price pressure and efficiency loss
            frame["shortage_proxy"] = (
                0.6 * frame["price_pressure"].fillna(0)
                + 0.4 * frame["efficiency_loss"].fillna(0)
            )

            sector_frames[sec] = frame

        # 5. Concat into MultiIndex panel
        panel = pd.concat(sector_frames, axis=1)
        audit = audit_alignment(panel, "sector_observables_panel")

        logger.info(f"Panel built: {panel.shape}")
        return panel, covid_monthly, audit

    # ──────────────────────────────────────────────────────────────────────────

    def _aggregate_covid(self) -> pd.DataFrame:
        """Aggregate daily COVID to monthly; compute epidemic pressure [0,1]."""
        df = self.covid_daily.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        monthly = df.resample("ME").mean()
        cases_col = "new_cases_smoothed" if "new_cases_smoothed" in monthly.columns else monthly.columns[0]

        # Normalize to [0,1] epidemic pressure
        raw = monthly[cases_col].fillna(0)
        monthly["epidemic_pressure"] = minmax_normalize(raw)

        if "stringency_index" in monthly.columns:
            monthly["stringency_norm"] = monthly["stringency_index"] / 100.0

        logger.info(
            f"COVID monthly: {monthly.index.min()} → {monthly.index.max()}, "
            f"peak pressure={monthly['epidemic_pressure'].max():.3f}"
        )
        return monthly

    def _compute_price_transforms(self) -> Dict[str, pd.DataFrame]:
        """
        For each CPI sector column, compute:
        - pct_change (MoM)
        - log_change
        - rolling z-score
        - rolling volatility
        - pressure = normalized deviation from trend
        """
        result = {}
        for col in self.cpi.columns:
            s = self.cpi[col].dropna()
            if len(s) < 4:
                continue
            pct = percent_change(s, periods=1)
            logd = log_change(s, periods=1)
            zscore = rolling_zscore(pct, window=12)
            vol = rolling_volatility(pct, window=self.vol_window)

            # Pressure: absolute YoY% change, normalized over full period
            yoy = percent_change(s, periods=12)
            pressure = minmax_normalize(yoy.fillna(0).abs())
            # Signed version for direction-aware analysis
            pressure_signed = minmax_normalize(yoy.fillna(0))

            out = pd.DataFrame(
                {
                    "pct_change":       pct,
                    "log_change":       logd,
                    "zscore":           zscore,
                    "volatility":       vol.pipe(minmax_normalize),
                    "pressure":         pressure,
                    "pressure_signed":  pressure_signed,
                },
                index=s.index,
            )
            result[col] = out
        return result

    def _compute_efficiency_proxies(self) -> Dict[str, pd.Series]:
        """
        Build efficiency / supply-capacity proxies from raw efficiency panel.
        Returns dict of named pd.Series aligned to efficiency_df index.
        """
        e = self.eff.copy()
        idx = e.index
        proxies: Dict[str, pd.Series] = {}

        # ── Industrial utilization ──────────────────────────────────
        if "industrial_utilization" in e:
            # Low utilization → high efficiency loss
            util_norm = minmax_normalize(e["industrial_utilization"])
            proxies["capacity_utilization_norm"] = util_norm
            proxies["efficiency_loss"] = (1.0 - util_norm).clip(0, 1)

        # ── Inventory/sales ratio ────────────────────────────────────
        # High ratio → slack (buffering); sharp drop → shortage pressure
        if "inventory_sales_ratio" in e:
            inv_sal = e["inventory_sales_ratio"]
            inv_norm = minmax_normalize(inv_sal)
            proxies["inventory_slack"] = inv_norm
            # Shortage proxy: falling inventory_sales
            proxies["inventory_pressure"] = (1 - inv_norm).clip(0, 1)

        # ── Labor market ────────────────────────────────────────────
        if "unemployment_rate" in e:
            unemp_norm = minmax_normalize(e["unemployment_rate"])
            proxies["labor_disruption"] = unemp_norm

        # ── Composite efficiency loss ───────────────────────────────
        components = [
            proxies.get("efficiency_loss", pd.Series(0, index=idx)),
            proxies.get("inventory_pressure", pd.Series(0, index=idx)),
            proxies.get("labor_disruption", pd.Series(0, index=idx)),
        ]
        composite = sum(c.reindex(idx).fillna(0) for c in components) / len(components)
        proxies["efficiency_loss_composite"] = composite.clip(0, 1)

        return proxies
