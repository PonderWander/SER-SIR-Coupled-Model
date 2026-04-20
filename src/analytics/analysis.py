"""
analytics/analysis.py
──────────────────────
Post-simulation analysis and figure-ready data exports.

Produces:
  - Peak stress windows
  - Cross-sector propagation tables
  - Lagged epidemic-to-sector sensitivity
  - Regime transition matrices
  - Retention vs spillover summaries
  - Sector decomposition exports
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("analytics")
logger.setLevel(logging.INFO)


def _flatten_if_multi(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Flatten MultiIndex columns to 'sector_variable' strings if needed."""
    if df is None:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        flat = df.copy()
        flat.columns = ["_".join(str(c) for c in col).strip("_") for col in flat.columns]
        return flat
    return df


class SimulationAnalytics:
    """
    Post-simulation analysis layer.

    Accepts either MultiIndex or flat-column DataFrames (handles both).

    Parameters
    ----------
    results : Dict from SimulationRunner.results
    """

    def __init__(self, results: Dict):
        self.res = results
        # Flatten MultiIndex columns if needed so string operations work
        self._ser    = _flatten_if_multi(results.get("sector_ser_panel"))
        self._prop   = _flatten_if_multi(results.get("propagation_panel"))
        self._regime = _flatten_if_multi(results.get("regime_panel"))
        self._covid  = results.get("covid_monthly")

    # ──────────────────────────────────────────────────────────────────────────

    def peak_stress_windows(self, n: int = 5) -> pd.DataFrame:
        """Return top-N stress windows per sector."""
        ser = self._ser
        if ser is None:
            return pd.DataFrame()

        rows = []
        for col in ser.columns:
            if not col.endswith("_stress"):
                continue
            sec = col.replace("_stress", "")
            s = ser[col].dropna().sort_values(ascending=False)
            for rank, (dt, val) in enumerate(s.head(n).items(), 1):
                rows.append({
                    "sector": sec,
                    "rank": rank,
                    "date": dt,
                    "stress_value": round(val, 4),
                })
        return pd.DataFrame(rows)

    def epidemic_sector_sensitivity(
        self, lag_range: range = range(0, 7)
    ) -> pd.DataFrame:
        """
        Lagged correlation between epidemic pressure and sector stress.
        Returns table: sector x lag → correlation.
        """
        ser = self._ser
        covid = self._covid
        if ser is None or covid is None:
            return pd.DataFrame()

        ep = covid.get("epidemic_pressure", covid.iloc[:, 0])
        rows = []

        for col in ser.columns:
            if not col.endswith("_stress"):
                continue
            sec = col.replace("_stress", "")
            stress = ser[col].dropna()

            # Align
            common = ep.index.intersection(stress.index)
            ep_a = ep.reindex(common)
            st_a = stress.reindex(common)

            lag_corrs = {}
            for lag in lag_range:
                ep_shifted = ep_a.shift(lag)
                combined = pd.DataFrame({"ep": ep_shifted, "st": st_a}).dropna()
                if len(combined) >= 5:
                    r, p = stats.pearsonr(combined["ep"], combined["st"])
                    lag_corrs[f"lag_{lag}"] = round(r, 3)
                else:
                    lag_corrs[f"lag_{lag}"] = np.nan

            row = {"sector": sec, **lag_corrs}
            # Best lag
            non_nan = {k: v for k, v in lag_corrs.items() if not np.isnan(v)}
            if non_nan:
                row["best_lag"] = max(non_nan, key=lambda k: abs(non_nan[k]))
                row["best_corr"] = non_nan[row["best_lag"]]
            rows.append(row)

        return pd.DataFrame(rows).set_index("sector")

    def cross_sector_propagation_table(self) -> pd.DataFrame:
        """
        For each sector pair (i→j), compute mean effective transmission.
        Returns wide matrix.
        """
        sg = self.res.get("sector_graph")
        if sg is None:
            return pd.DataFrame()

        sectors = sg.sectors
        A = sg.adjacency_matrix()

        prop = self._prop
        if prop is None:
            return A

        # Enrich with mean propagation received by each sector
        received = {}
        for sec in sectors:
            col = f"{sec}_propagation_raw"
            if col in prop.columns:
                received[sec] = round(float(prop[col].mean()), 4)
            else:
                received[sec] = 0.0

        enriched = A.copy()
        enriched.loc["mean_received"] = pd.Series(received)
        return enriched

    def retention_spillover_summary(self) -> pd.DataFrame:
        """Retention vs spillover share per sector, full period."""
        ser = self._ser
        if ser is None:
            return pd.DataFrame()

        rows = []
        for col in ser.columns:
            if col.endswith("_retention"):
                sec = col.replace("_retention", "")
                sp_col = f"{sec}_spillover_share"
                rows.append({
                    "sector": sec,
                    "retention_mean":       round(float(ser[col].mean()), 4),
                    "retention_peak":       round(float(ser[col].max()),  4),
                    "spillover_mean":       round(float(ser[sp_col].mean()), 4) if sp_col in ser.columns else np.nan,
                    "spillover_peak":       round(float(ser[sp_col].max()),  4) if sp_col in ser.columns else np.nan,
                })
        return pd.DataFrame(rows).set_index("sector")

    def aggregate_system_metrics(self) -> pd.DataFrame:
        """Time series of system-level aggregate metrics."""
        ser = self._ser
        if ser is None:
            return pd.DataFrame()

        # Columns ending in key variables
        def mean_across_sectors(var_suffix: str) -> pd.Series:
            cols = [c for c in ser.columns if c.endswith(var_suffix)]
            if not cols:
                return pd.Series(dtype=float)
            return ser[cols].mean(axis=1)

        df = pd.DataFrame(index=ser.index)
        df["mean_stress"]        = mean_across_sectors("_stress")
        df["mean_leaked_stress"] = mean_across_sectors("_leaked_stress")
        df["mean_elasticity"]    = mean_across_sectors("_elasticity")
        df["mean_reaction"]      = mean_across_sectors("_reaction")
        df["mean_retention"]     = mean_across_sectors("_retention")

        if self._covid is not None:
            covid_flat = _flatten_if_multi(self._covid)
            ep = None
            if isinstance(covid_flat, pd.DataFrame):
                if "epidemic_pressure" in covid_flat.columns:
                    ep = covid_flat["epidemic_pressure"]
                elif len(covid_flat.columns) > 0:
                    ep = covid_flat.iloc[:, 0]
            elif isinstance(covid_flat, pd.Series):
                ep = covid_flat
            if ep is not None:
                df["epidemic_pressure"] = ep.reindex(df.index)

        if self._prop is not None:
            conn_col = "_system_effective_connectivity"
            if conn_col in self._prop.columns:
                df["effective_connectivity"] = self._prop[conn_col].reindex(df.index)

        if self.res.get("system_regime") is not None:
            df["system_regime"] = self.res["system_regime"].reindex(df.index)

        return df

    def export_figure_ready(self, output_dir: Path) -> Dict[str, Path]:
        """
        Write figure-ready CSV exports:
          - line chart data
          - heatmap data (sector x time)
          - regime map
          - retention vs spillover

        Returns dict of {name: path}.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = {}

        # System metrics (for line charts)
        sys_metrics = self.aggregate_system_metrics()
        if not sys_metrics.empty:
            p = output_dir / "figure_system_metrics.csv"
            sys_metrics.to_csv(p)
            written["system_metrics"] = p

        # Sector stress heatmap: rows=dates, cols=sectors
        ser = self._ser
        if ser is not None:
            stress_cols = [c for c in ser.columns if c.endswith("_stress") and not c.endswith("_leaked_stress") and not c.endswith("_absorbed_stress")]
            if stress_cols:
                heatmap = ser[stress_cols].copy()
                heatmap.columns = [c.replace("_stress","") for c in heatmap.columns]
                p = output_dir / "figure_stress_heatmap.csv"
                heatmap.to_csv(p)
                written["stress_heatmap"] = p

            # Leaked stress heatmap
            lk_cols = [c for c in ser.columns if c.endswith("_leaked_stress")]
            if lk_cols:
                lk = ser[lk_cols].copy()
                lk.columns = [c.replace("_leaked_stress","") for c in lk.columns]
                p = output_dir / "figure_leaked_stress_heatmap.csv"
                lk.to_csv(p)
                written["leaked_stress_heatmap"] = p

        # Regime map: sector x time
        regime = self._regime
        if regime is not None:
            regime_cols = [c for c in regime.columns if c.endswith("_regime")]
            if regime_cols:
                rmap = regime[regime_cols].copy()
                rmap.columns = [c.replace("_regime","") for c in rmap.columns]
                p = output_dir / "figure_regime_map.csv"
                rmap.to_csv(p)
                written["regime_map"] = p

        # Retention vs spillover
        ret_sp = self.retention_spillover_summary()
        if not ret_sp.empty:
            p = output_dir / "figure_retention_spillover.csv"
            ret_sp.to_csv(p)
            written["retention_spillover"] = p

        # Epidemic sensitivity
        ep_sens = self.epidemic_sector_sensitivity()
        if not ep_sens.empty:
            p = output_dir / "figure_epidemic_sensitivity.csv"
            ep_sens.to_csv(p)
            written["epidemic_sensitivity"] = p

        logger.info(f"[Analytics] Exported {len(written)} figure-ready files.")
        return written
