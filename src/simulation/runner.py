"""
simulation/runner.py
─────────────────────
Main simulation orchestrator.

Stages:
  1. Data ingestion
  2. Panel building (preprocessing)
  3. Epidemic engine
  4. SER engine
  5. Graph propagation
  6. Regime classification
  7. Output writing

Usage:
  runner = SimulationRunner(config_path="configs/base_config.yaml")
  runner.run()
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Load .env from project root so FRED_API_KEY is available regardless of
# how the runner is invoked (CLI, notebook, or direct import)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

from src.data_ingestion.loaders import (
    fetch_fred_panel,
    fetch_owid_covid,
    generate_synthetic_dataset,
)
from src.epidemic.sir_model import ObservedPressureAdapter
from src.graph.propagation import PropagationEngine, SectorGraph
from src.preprocessing.panel_builder import SectorPanelBuilder
from src.ser.ser_engine import SEREngine
from src.simulation.regime import RegimeClassifier
from src.utils.common import (
    get_logger,
    load_config,
    load_sector_graph_config,
    save_outputs,
    set_seed,
)

logger = get_logger("simulation_runner")


class SimulationRunner:
    """
    Orchestrates the full epidemic-market simulation pipeline.

    Parameters
    ----------
    config_path      : Path to base_config.yaml
    graph_config_path: Path to sector_graph.yaml
    output_dir       : Override output directory
    """

    def __init__(
        self,
        config_path: str = "configs/base_config.yaml",
        graph_config_path: str = "configs/sector_graph.yaml",
        output_dir: Optional[str] = None,
        refresh_cache: bool = False,
    ):
        self.config_path = config_path
        self.graph_config_path = graph_config_path
        self.cfg = load_config(config_path)
        self.graph_cfg = load_sector_graph_config(graph_config_path)
        self.refresh_cache = refresh_cache

        # Apply seed
        set_seed(self.cfg.get("run", {}).get("seed", 42))

        # Output
        self.output_dir = Path(output_dir or self.cfg.get("outputs", {}).get("dir", "outputs"))
        self.parquet_dir = self.output_dir / "parquet"
        self.csv_dir = self.output_dir / "csv"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # Results store
        self.results: Dict = {}
        self._start_time: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute the full simulation pipeline. Returns dict of output frames."""
        self._start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"Starting simulation: {self.cfg['run']['name']}")
        logger.info("=" * 60)

        # ── Stage 1: Data ─────────────────────────────────────────────
        covid_df, cpi_df, eff_df = self._stage_data()

        # ── Stage 2: Epidemic engine ──────────────────────────────────
        sir_df, epidemic_monthly = self._stage_epidemic(covid_df)
        self.results["sir_timeseries"] = sir_df

        # ── Stage 3: Sector panel ─────────────────────────────────────
        panel, covid_monthly, audit = self._stage_panel(cpi_df, eff_df, covid_df)
        self.results["sector_observables_panel"] = panel
        self.results["covid_monthly"] = covid_monthly

        # ── Stage 4: SER engine ───────────────────────────────────────
        ser_panel = self._stage_ser(panel)
        self.results["sector_ser_panel"] = ser_panel

        # ── Stage 5: Graph propagation ────────────────────────────────
        prop_panel = self._stage_propagation(ser_panel)
        self.results["propagation_panel"] = prop_panel

        # ── Stage 6: Regime classification ───────────────────────────
        regime_panel, system_regime = self._stage_regime(ser_panel, prop_panel)
        self.results["regime_panel"] = regime_panel
        self.results["system_regime"] = system_regime

        # ── Stage 7: Write outputs ────────────────────────────────────
        self._stage_outputs()

        elapsed = time.time() - self._start_time
        logger.info(f"Simulation complete in {elapsed:.1f}s")
        logger.info(f"Outputs written to {self.output_dir}")
        return self.results

    # ──────────────────────────────────────────────────────────────────────────

    def _purge_cache(self, cache_dir: Path) -> None:
        """Delete all cached raw data files so they are re-fetched on next run."""
        patterns = ["fred_*.json", "owid_covid_*.parquet"]
        removed = []
        for pattern in patterns:
            for f in cache_dir.glob(pattern):
                f.unlink()
                removed.append(f.name)
        if removed:
            logger.info(f"[Cache] Purged {len(removed)} file(s): {', '.join(removed)}")
        else:
            logger.info("[Cache] No cached files found to purge.")

    def _stage_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stage 1: Data ingestion."""
        logger.info("[Stage 1] Data ingestion...")
        dcfg = self.cfg.get("data", {})
        run_cfg = self.cfg.get("run", {})
        cache_dir = Path(dcfg.get("cache_dir", "data/raw"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.refresh_cache:
            logger.info("[Stage 1] --refresh-cache requested: purging cached files...")
            self._purge_cache(cache_dir)

        fred_key = dcfg.get("fred_api_key") or os.environ.get("FRED_API_KEY")
        start = run_cfg.get("start_date", "2020-01-01")
        end   = run_cfg.get("end_date",   "2022-12-31")

        use_synthetic = not bool(fred_key)
        if use_synthetic:
            logger.warning("[Stage 1] No FRED API key found. Using synthetic data.")

        # COVID
        try:
            covid_df = fetch_owid_covid(
                country="USA",
                cache_dir=cache_dir,
                start_date=start,
                end_date=end,
            )
            if covid_df.empty or len(covid_df) < 10:
                raise ValueError("Empty COVID data")
        except Exception as e:
            logger.warning(f"[Stage 1] COVID fetch failed ({e}). Using synthetic.")
            covid_df, _, _ = generate_synthetic_dataset(start, end)

        # CPI / Market data
        if not use_synthetic:
            cpi_series = dcfg.get("sources", {}).get("cpi", {}).get("series", {})
            eff_series = dcfg.get("sources", {}).get("efficiency_proxies", {}).get("series", {})
            try:
                cpi_df = fetch_fred_panel(cpi_series, fred_key, start, end, cache_dir)
                eff_df = fetch_fred_panel(eff_series, fred_key, start, end, cache_dir)
                # Backfill to pre-2020 for context
                cpi_df_context = fetch_fred_panel(
                    cpi_series, fred_key,
                    "2018-01-01", end, cache_dir
                )
                cpi_df = cpi_df_context  # use full range
            except Exception as e:
                logger.warning(f"[Stage 1] FRED fetch failed ({e}). Falling back to synthetic.")
                _, cpi_df, eff_df = generate_synthetic_dataset(start, end)
        else:
            _, cpi_df, eff_df = generate_synthetic_dataset(start, end)

        logger.info(
            f"[Stage 1] COVID: {len(covid_df)} days | "
            f"CPI: {cpi_df.shape} | Efficiency: {eff_df.shape}"
        )
        return covid_df, cpi_df, eff_df

    def _stage_epidemic(
        self, covid_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Stage 2: Epidemic engine."""
        logger.info("[Stage 2] Epidemic engine...")
        epi_cfg = self.cfg.get("epidemic", {})
        lag_days = epi_cfg.get("pressure_lag_days", 14)

        adapter = ObservedPressureAdapter(
            covid_df=covid_df,
            pressure_col="new_cases_smoothed",
            smoothing_sigma=7.0,
            normalize=epi_cfg.get("normalize_pressure", True),
            lag_days=lag_days,
        )
        sir_df = adapter.build_sir_output()
        monthly_pressure = adapter.get_monthly_pressure()

        logger.info(
            f"[Stage 2] SIR output: {sir_df.shape}, "
            f"monthly pressure: {monthly_pressure.shape}"
        )
        return sir_df, monthly_pressure

    def _stage_panel(
        self,
        cpi_df: pd.DataFrame,
        eff_df: pd.DataFrame,
        covid_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Stage 3: Sector panel construction."""
        logger.info("[Stage 3] Building sector observables panel...")
        sector_meta = {
            s["name"] if isinstance(s, dict) else s: (
                self.graph_cfg.get("sector_metadata", {}).get(
                    s["name"] if isinstance(s, dict) else s, {}
                )
            )
            for s in self.graph_cfg.get("sectors", [])
        }
        # sectors is a list of strings
        sector_meta = self.graph_cfg.get("sector_metadata", {})

        builder = SectorPanelBuilder(
            cpi_df=cpi_df,
            efficiency_df=eff_df,
            covid_daily=covid_df,
            config=self.cfg,
            sector_meta=sector_meta,
        )
        panel, covid_monthly, audit = builder.build()

        # Trim to run date range
        run_cfg = self.cfg.get("run", {})
        start = run_cfg.get("start_date", "2020-01-01")
        end   = run_cfg.get("end_date",   "2022-12-31")
        panel = panel.loc[start:end]
        covid_monthly = covid_monthly.loc[start:end]

        logger.info(f"[Stage 3] Panel: {panel.shape}")
        return panel, covid_monthly, audit

    def _stage_ser(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Stage 4: SER engine."""
        logger.info("[Stage 4] Computing SER states...")
        ser_engine = SEREngine(
            panel=panel,
            config=self.cfg,
            sector_meta=self.graph_cfg.get("sector_metadata", {}),
        )
        ser_panel = ser_engine.run(rng_seed=self.cfg["run"].get("seed", 42))

        # Attach summary for Stage 7
        self.results["sector_summary"] = ser_engine.sector_summary()

        logger.info(f"[Stage 4] SER panel: {ser_panel.shape}")
        return ser_panel

    def _stage_propagation(self, ser_panel: pd.DataFrame) -> pd.DataFrame:
        """Stage 5: Cross-sector propagation.

        Pass the raw sector observables panel as activity_panel so that
        connectivity modes using A_{i,t} (stress_elasticity_activity,
        delta_deformation) use the actual efficiency/inventory proxies rather
        than falling back to the 1-S stress proxy.
        """
        logger.info("[Stage 5] Graph propagation...")
        sg = SectorGraph(self.graph_cfg)
        # Retrieve raw panel built in Stage 3 (stored by _stage_panel)
        activity_panel = self.results.get("sector_observables_panel")
        engine = PropagationEngine(
            sector_graph=sg,
            ser_panel=ser_panel,
            config=self.cfg,
            activity_panel=activity_panel,
        )
        prop_panel = engine.run()

        # Attach diagnostics
        self.results["emitter_absorber_table"] = engine.emitter_absorber_table()
        self.results["sector_graph"] = sg

        logger.info(f"[Stage 5] Propagation panel: {prop_panel.shape}")
        return prop_panel

    def _stage_regime(
        self, ser_panel: pd.DataFrame, prop_panel: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Stage 6: Regime classification."""
        logger.info("[Stage 6] Regime classification...")
        clf = RegimeClassifier(
            ser_panel=ser_panel,
            prop_panel=prop_panel,
            config=self.cfg,
        )
        regime_panel, system_regime = clf.run()

        # Attach analytical tables
        self.results["transition_matrix"]      = clf.transition_matrix()
        self.results["regime_duration_table"]  = clf.regime_duration_table()
        self.results["threshold_breach_summary"] = clf.threshold_breach_summary()

        return regime_panel, system_regime

    def _stage_outputs(self) -> None:
        """Stage 7: Write all outputs to parquet and CSV."""
        logger.info("[Stage 7] Writing outputs...")
        do_parquet = self.cfg.get("outputs", {}).get("parquet", True)
        do_csv     = self.cfg.get("outputs", {}).get("csv", True)

        # Core DataFrames to save
        frames = {
            "sir_timeseries":          self.results.get("sir_timeseries"),
            "covid_monthly":           self.results.get("covid_monthly"),
            "sector_ser_panel":        self._flatten_multiindex(self.results.get("sector_ser_panel")),
            "propagation_panel":       self._flatten_multiindex(self.results.get("propagation_panel")),
            "regime_panel":            self._flatten_multiindex(self.results.get("regime_panel")),
            "system_regime":           self.results.get("system_regime").to_frame() if self.results.get("system_regime") is not None else None,
            "sector_summary":          self.results.get("sector_summary"),
            "emitter_absorber_table":  self.results.get("emitter_absorber_table"),
            "transition_matrix":       self.results.get("transition_matrix"),
            "regime_duration_table":   self.results.get("regime_duration_table"),
            "threshold_breach_summary":self.results.get("threshold_breach_summary"),
        }

        for name, df in frames.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            try:
                save_outputs(df, name, self.parquet_dir, parquet=do_parquet, csv=False)
                save_outputs(df, name, self.csv_dir,     parquet=False,      csv=do_csv)
                logger.info(f"  Saved: {name} {df.shape}")
            except Exception as e:
                logger.warning(f"  Failed to save {name}: {e}")

    @staticmethod
    def _flatten_multiindex(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Flatten MultiIndex columns to 'sector_variable' strings."""
        if df is None:
            return None
        flat = df.copy()
        if isinstance(flat.columns, pd.MultiIndex):
            flat.columns = ["_".join(str(c) for c in col).strip("_") for col in flat.columns]
        return flat
