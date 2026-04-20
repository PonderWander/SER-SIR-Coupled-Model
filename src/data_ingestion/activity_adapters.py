"""
data_ingestion/activity_adapters.py
─────────────────────────────────────
Adapters for:
  1. BEA 2017 Benchmark IO Use Table  →  W^0_{ij} sector flow matrix
  2. Opportunity Insights Economic Tracker  →  A_{i,t} activity panel
  3. Synthetic activity fallback (stress-derived proxy)

DATA INGESTION — NO API KEYS REQUIRED:

  BEA IO:
    Download: https://www.bea.gov/industry/input-output-accounts-data
    File: AllTablesSUP.zip → extract → "Use_SUT_Framework_2007_2012_DET.xlsx"
    Or summary table: https://apps.bea.gov/iTable/iTable.cfm?reqid=150
    Save to: data/raw/bea/use_table_2017.csv  (or .xlsx)
    No registration or API key needed.

  Opportunity Insights:
    Download: git clone https://github.com/OpportunityInsights/EconomicTracker
    Or direct CSV: data/Affinity - National - Weekly.csv
    Save to: data/raw/oi/affinity_national_weekly.csv
    No API key needed. Public repository, free to use with attribution.

  If neither file is present, the adapter returns a stress-derived proxy
  that is clearly flagged in the blob visualization as "proxied" (not observed).

SECTOR MAPPING TABLE:
  Simulation sector  ←→  BEA IO code(s)  ←→  OI Affinity category
  (see SECTOR_MAPPING dict below — this is the explicit concordance)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("activity_adapters")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


# ─── Explicit Sector Concordance ─────────────────────────────────────────────
#
# Three-way mapping: simulation sector ↔ BEA IO code(s) ↔ OI Affinity category
#
# Observation status:
#   "oi_direct"  : OI Affinity spend data maps cleanly
#   "oi_partial" : OI category partially covers sector (split or overlap)
#   "fred_proxy" : No OI coverage; use FRED CPI/PPI as activity proxy
#   "stress_proxy": Fallback to 1 - stress_{i,t} when nothing else available

SECTOR_MAPPING = {
    "food_at_home": {
        "bea_codes":      ["311FT", "4A00"],          # food mfg + retail trade
        "bea_label":      "Food/beverage products + Retail trade",
        "oi_category":    "Grocery",
        "oi_col":         "spend_grocery",             # column in OI CSV
        "obs_status":     "oi_direct",
        "notes":          "OI Grocery maps well to food-at-home CPI basket",
    },
    "food_away_from_home": {
        "bea_codes":      ["722", "7211"],
        "bea_label":      "Food services + Hotels",
        "oi_category":    "Restaurants and Hotels",
        "oi_col":         "spend_restbar",
        "obs_status":     "oi_direct",
        "notes":          "OI Restaurants/Hotels is the primary signal",
    },
    "energy": {
        "bea_codes":      ["211", "213", "324"],
        "bea_label":      "Oil/gas extraction + Petroleum products",
        "oi_category":    None,
        "oi_col":         None,
        "obs_status":     "fred_proxy",
        "notes":          "No OI Affinity category for upstream energy; use FRED CPIENGSL",
    },
    "gasoline": {
        "bea_codes":      ["324", "447"],
        "bea_label":      "Petroleum products + Gas stations",
        "oi_category":    "Transportation",
        "oi_col":         "spend_transport",
        "obs_status":     "oi_partial",
        "notes":          "OI Transportation includes gasoline retail but is broader",
    },
    "electricity": {
        "bea_codes":      ["2200A"],
        "bea_label":      "Utilities",
        "oi_category":    None,
        "oi_col":         None,
        "obs_status":     "fred_proxy",
        "notes":          "No OI category; use FRED electricity price index",
    },
    "shelter": {
        "bea_codes":      ["531"],
        "bea_label":      "Real estate",
        "oi_category":    None,
        "oi_col":         None,
        "obs_status":     "fred_proxy",
        "notes":          "Housing excluded from card spend; use FRED shelter CPI",
    },
    "transportation_services": {
        "bea_codes":      ["484", "481", "482", "487OS"],
        "bea_label":      "Truck/air/rail/other transport",
        "oi_category":    "Transportation",
        "oi_col":         "spend_transport",
        "obs_status":     "oi_partial",
        "notes":          "OI Transportation covers consumer side; freight not included",
    },
    "household_goods": {
        "bea_codes":      ["315AL", "321", "337", "4A00"],
        "bea_label":      "Apparel/wood/furniture + Retail",
        "oi_category":    "Apparel and General Merchandise",
        "oi_col":         "spend_apmerchandise",
        "obs_status":     "oi_direct",
        "notes":          "OI Apparel/General Merch captures household commodities well",
    },
    "medical_services": {
        "bea_codes":      ["621", "622HO"],
        "bea_label":      "Ambulatory care + Hospitals",
        "oi_category":    "Health Care",
        "oi_col":         "spend_hcs",
        "obs_status":     "oi_direct",
        "notes":          "OI Health Care is a clean match; excludes insurance",
    },
    "apparel": {
        "bea_codes":      ["315AL"],
        "bea_label":      "Apparel and leather",
        "oi_category":    "Apparel and General Merchandise",
        "oi_col":         "spend_apmerchandise",
        "obs_status":     "oi_partial",
        "notes":          "Shares OI category with household_goods; weight split 40/60",
    },
    "ppi_goods": {
        "bea_codes":      ["3259", "3369", "339"],
        "bea_label":      "Misc manufacturing",
        "oi_category":    None,
        "oi_col":         None,
        "obs_status":     "fred_proxy",
        "notes":          "Upstream PPI; no consumer-facing OI signal; use FRED WPUFD49207",
    },
    "ppi_food": {
        "bea_codes":      ["311FT"],
        "bea_label":      "Food and beverage products (upstream)",
        "oi_category":    "Grocery",
        "oi_col":         "spend_grocery",
        "obs_status":     "oi_partial",
        "notes":          "Farm/upstream PPI; OI Grocery is downstream signal only",
    },
    "ppi_energy": {
        "bea_codes":      ["211", "324"],
        "bea_label":      "Petroleum/coal upstream",
        "oi_category":    None,
        "oi_col":         None,
        "obs_status":     "fred_proxy",
        "notes":          "Upstream energy PPI; no OI coverage; use FRED WPU0543",
    },
}

# OI Affinity column names as published in the national weekly CSV
OI_COLUMN_MAP = {
    "spend_all":          "All sectors (total card spend)",
    "spend_apmerchandise":"Apparel and General Merchandise",
    "spend_entertainment":"Entertainment and Recreation",
    "spend_grocery":      "Grocery",
    "spend_hcs":          "Health Care",
    "spend_restbar":      "Restaurants and Hotels",
    "spend_transport":    "Transportation",
}


# ─── BEA IO Adapter ───────────────────────────────────────────────────────────

class BEAIOAdapter:
    """
    Loads the BEA 2017 Benchmark Use Table and aggregates it to the
    simulation sector taxonomy.

    The BEA table gives dollar flows: how much sector i purchases from sector j.
    We use this as the static structural prior W^0_{ij}, normalised to [0,1]
    relative to the maximum flow in the table.

    If the file is not present, returns uniform weights (equal to current
    sector_graph.yaml weights) with a warning.

    Download (no registration):
      https://www.bea.gov/industry/input-output-accounts-data
      File: "2017 Benchmark I-O Use Table" → save as data/raw/bea/use_table_2017.csv
    """

    def __init__(self, io_path: Optional[str] = None):
        self.io_path = Path(io_path) if io_path else None
        self._raw: Optional[pd.DataFrame] = None

    def load(self) -> Optional[pd.DataFrame]:
        """Load raw BEA IO table. Returns None if file not present."""
        if self.io_path is None or not self.io_path.exists():
            logger.warning(
                "[BEA IO] File not found. Using structural weights from sector_graph.yaml.\n"
                "  To enable BEA IO anchoring:\n"
                "  1. Download from https://www.bea.gov/industry/input-output-accounts-data\n"
                "  2. Save to data/raw/bea/use_table_2017.csv\n"
                "  3. Set blob.bea_io_path in base_config.yaml"
            )
            return None
        try:
            if str(self.io_path).endswith(".csv"):
                df = pd.read_csv(self.io_path, index_col=0)
            else:
                df = pd.read_excel(self.io_path, index_col=0, header=0)
            self._raw = df
            logger.info(f"[BEA IO] Loaded: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"[BEA IO] Load failed: {e}")
            return None

    def build_prior_matrix(
        self,
        sectors: list,
        sector_mapping: Dict = None,
    ) -> pd.DataFrame:
        """
        Aggregate BEA IO flows to simulation taxonomy.
        Returns a (sectors x sectors) DataFrame of normalised W^0_{ij} values.

        If BEA data unavailable, returns None (caller falls back to graph.yaml weights).
        """
        if self._raw is None:
            return None

        mapping = sector_mapping or SECTOR_MAPPING
        sim_to_bea = {s: mapping[s]["bea_codes"] for s in sectors if s in mapping}

        W0 = pd.DataFrame(0.0, index=sectors, columns=sectors)

        for src in sectors:
            for tgt in sectors:
                if src == tgt:
                    continue
                bea_src = sim_to_bea.get(src, [])
                bea_tgt = sim_to_bea.get(tgt, [])
                total = 0.0
                count = 0
                for bs in bea_src:
                    for bt in bea_tgt:
                        # Look for bt row (purchaser) buying from bs column (supplier)
                        if bs in self._raw.columns and bt in self._raw.index:
                            val = self._raw.loc[bt, bs]
                            if pd.notna(val) and val > 0:
                                total += float(val)
                                count += 1
                if count > 0:
                    W0.loc[src, tgt] = total / count

        # Normalise to [0,1] relative to max flow
        max_val = W0.values.max()
        if max_val > 0:
            W0 = W0 / max_val

        logger.info(f"[BEA IO] Prior matrix built: {W0.shape}, max={W0.values.max():.3f}")
        return W0


# ─── Opportunity Insights Activity Adapter ───────────────────────────────────

class OIActivityAdapter:
    """
    Loads Opportunity Insights Economic Tracker Affinity spend data and
    constructs a monthly activity panel A_{i,t} for simulation sectors.

    Data is indexed relative to Jan 2020 baseline (OI's standard indexing).
    We transform to a [0,1] activity level: 1.0 = at or above baseline.

    Download (no API key, free to use with attribution):
      git clone https://github.com/OpportunityInsights/EconomicTracker
      Or: download data/Affinity - National - Weekly.csv directly
      Save to: data/raw/oi/affinity_national_weekly.csv

    Attribution: Chetty, Friedman, Stepner et al. (2020) NBER WP 27431.
    """

    def __init__(self, oi_path: Optional[str] = None):
        self.oi_path = Path(oi_path) if oi_path else None
        self._raw: Optional[pd.DataFrame] = None

    def load(self) -> Optional[pd.DataFrame]:
        """Load OI Affinity CSV. Returns None if not present."""
        if self.oi_path is None or not self.oi_path.exists():
            logger.warning(
                "[OI Activity] File not found. Using stress-derived activity proxy.\n"
                "  To enable OI spend data:\n"
                "  1. git clone https://github.com/OpportunityInsights/EconomicTracker\n"
                "  2. Copy data/Affinity - National - Weekly.csv\n"
                "  3. Save to data/raw/oi/affinity_national_weekly.csv\n"
                "  4. Set blob.oi_spend_path in base_config.yaml\n"
                "  Attribution: Chetty, Friedman et al. (2020) NBER WP 27431"
            )
            return None
        try:
            df = pd.read_csv(self.oi_path)
            # OI date columns are year + day_endofweek
            if "year" in df.columns and "month" in df.columns and "day" in df.columns:
                df["date"] = pd.to_datetime(df[["year","month","day"]])
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            self._raw = df
            logger.info(f"[OI Activity] Loaded: {df.shape}, "
                        f"{df.index.min()} → {df.index.max()}")
            return df
        except Exception as e:
            logger.error(f"[OI Activity] Load failed: {e}")
            return None

    def build_activity_panel(
        self,
        sectors: list,
        start_date: str,
        end_date: str,
        sector_mapping: Dict = None,
    ) -> pd.DataFrame:
        """
        Build monthly A_{i,t} panel for simulation sectors.
        Values are normalised spend index: 0 = complete collapse, 1 = at baseline.

        Sectors with no OI coverage are marked NaN and should be proxied.
        """
        if self._raw is None:
            return pd.DataFrame()

        mapping = sector_mapping or SECTOR_MAPPING
        idx_monthly = pd.date_range(start_date, end_date, freq="ME")
        result = pd.DataFrame(index=idx_monthly, columns=sectors, dtype=float)

        for sec in sectors:
            m = mapping.get(sec, {})
            oi_col = m.get("oi_col")
            obs_status = m.get("obs_status", "stress_proxy")

            if oi_col and oi_col in self._raw.columns and "oi" in obs_status:
                # OI spend index: values around 0 in 2020 = percent change from Jan 2020
                # Transform: activity = 1 + (spend_index / 100), clipped to [0, 1.5]
                series = self._raw[oi_col].dropna()
                # Resample weekly → monthly mean
                monthly = series.resample("ME").mean()
                # Convert from percent-change-from-baseline to activity level
                activity = (1.0 + monthly / 100.0).clip(0.0, 1.5)
                # Normalize to [0,1]
                act_norm = ((activity - activity.min()) /
                            (activity.max() - activity.min() + 1e-8)).clip(0, 1)
                result[sec] = act_norm.reindex(idx_monthly)
                logger.info(f"[OI Activity] {sec}: obs_status={obs_status}, "
                            f"coverage={act_norm.notna().mean()*100:.0f}%")
            else:
                # Leave as NaN — caller will fill with proxy
                result[sec] = np.nan

        return result

    def observation_status(self, sectors: list) -> pd.DataFrame:
        """Return observation status table for all sectors."""
        mapping = SECTOR_MAPPING
        rows = []
        for sec in sectors:
            m = mapping.get(sec, {})
            rows.append({
                "sector":      sec,
                "obs_status":  m.get("obs_status", "stress_proxy"),
                "oi_category": m.get("oi_category", "—"),
                "oi_col":      m.get("oi_col", "—"),
                "bea_codes":   ", ".join(m.get("bea_codes", [])),
                "notes":       m.get("notes", ""),
            })
        return pd.DataFrame(rows).set_index("sector")


# ─── Activity Panel Builder ───────────────────────────────────────────────────

def build_activity_panel(
    sectors: list,
    ser_panel: pd.DataFrame,
    bea_io_path: Optional[str] = None,
    oi_path: Optional[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the full activity panel combining OI (where available) and
    stress-derived proxy (where not).

    Returns
    -------
    activity_panel : pd.DataFrame — A_{i,t} for all sectors, monthly
    obs_status_df  : pd.DataFrame — observation status per sector
    W0_matrix      : pd.DataFrame — BEA IO prior matrix (or None)
    """
    # Load OI data
    oi_adapter = OIActivityAdapter(oi_path)
    oi_adapter.load()
    oi_panel = oi_adapter.build_activity_panel(
        sectors, start_date, end_date
    ) if oi_adapter._raw is not None else pd.DataFrame()

    obs_status = oi_adapter.observation_status(sectors)

    # Load BEA IO
    bea_adapter = BEAIOAdapter(bea_io_path)
    bea_adapter.load()
    W0_matrix = bea_adapter.build_prior_matrix(sectors) if bea_adapter._raw is not None else None

    # Build final activity panel
    idx = pd.date_range(start_date, end_date, freq="ME")
    activity = pd.DataFrame(index=idx, columns=sectors, dtype=float)

    for sec in sectors:
        # Priority: OI observed > stress proxy
        if not oi_panel.empty and sec in oi_panel.columns and oi_panel[sec].notna().any():
            activity[sec] = oi_panel[sec].reindex(idx)
        # Stress-derived fallback: A_{i,t} = 1 - stress_{i,t}
        if activity[sec].isna().any():
            if sec in ser_panel.columns.get_level_values(0):
                stress_col = ser_panel[sec]["stress"].reindex(idx)
                proxy = (1.0 - stress_col).clip(0, 1)
            else:
                proxy = pd.Series(0.7, index=idx)  # neutral default
            activity[sec] = activity[sec].fillna(proxy)

    logger.info(
        f"[Activity Panel] Built: {activity.shape}  "
        f"OI-observed: {obs_status['obs_status'].str.startswith('oi').sum()}/{len(sectors)}  "
        f"BEA IO: {'available' if W0_matrix is not None else 'not loaded'}"
    )
    return activity, obs_status, W0_matrix
