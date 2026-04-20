"""
simulation/regime.py
─────────────────────
Regime classification engine.

Classifies each (sector, time) pair into one of:
  dispersal, accumulation, isolation, recovery, fragmented, amplification

Also classifies system-level regimes from aggregate metrics.

Outputs:
  - regime_panel : MultiIndex (date x (sector, regime_variable))
  - system_regime: pd.Series with system regime per date
  - transition_matrix: sector-level regime transition counts
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("regime")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

# Regime labels
REGIMES = [
    "dispersal",
    "accumulation",
    "isolation",
    "recovery",
    "fragmented",
    "amplification",
    "unknown",
]
REGIME_CODES = {r: i for i, r in enumerate(REGIMES)}


class RegimeClassifier:
    """
    Rule-based regime classifier for sectors and system.

    Classification logic (in priority order):
    1. amplification: E <= E_crit AND leaked_stress > threshold
    2. fragmented   : effective_connectivity < isolation_conn_thresh
    3. isolation    : sector incoming propagation near zero but stress high
    4. recovery     : leaked_stress falling AND elasticity rising
    5. accumulation : leaked_stress above threshold AND not falling
    6. dispersal    : low leaked_stress, propagation active, elasticity sufficient
    7. unknown      : catch-all

    Parameters
    ----------
    ser_panel  : SER output panel
    prop_panel : Propagation output panel
    config     : regime section of base config
    """

    def __init__(
        self,
        ser_panel: pd.DataFrame,
        prop_panel: pd.DataFrame,
        config: Dict,
    ):
        self.ser = ser_panel
        self.prop = prop_panel
        self.cfg = config.get("regime", config)
        self.sectors = ser_panel.columns.get_level_values(0).unique().tolist()
        self._sector_results: Optional[pd.DataFrame] = None
        self._system_results: Optional[pd.Series] = None

    def run(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Classify regimes for all sectors and system.

        Returns
        -------
        regime_panel  : MultiIndex DataFrame
        system_regime : pd.Series
        """
        cfg = self.cfg
        E_crit = cfg.get("elasticity", {}).get("E_crit", 0.25)

        # Thresholds (configurable)
        disp = cfg.get("dispersal", {})
        accum = cfg.get("accumulation", {})
        iso = cfg.get("isolation", {})

        L_thresh_hi   = accum.get("leaked_stress_min", 0.20)
        E_thresh_min  = disp.get("elasticity_min", 0.35)
        L_thresh_lo   = disp.get("leaked_stress_max", 0.20)
        conn_thresh   = iso.get("effective_connectivity_max", 0.20)

        sector_dfs = {}
        for sec in self.sectors:
            if sec not in self.ser.columns.get_level_values(0):
                continue
            df = self._classify_sector(sec, E_crit, L_thresh_hi, E_thresh_min, L_thresh_lo, conn_thresh)
            sector_dfs[sec] = df
            logger.info(
                f"[Regime] {sec}: modal_regime={df['regime'].mode().iloc[0]}, "
                f"n_amplification={int((df['regime']=='amplification').sum())}"
            )

        regime_panel = pd.concat(sector_dfs, axis=1)
        system_regime = self._classify_system(regime_panel)

        self._sector_results = regime_panel
        self._system_results = system_regime
        logger.info(f"[Regime] Classification complete: {regime_panel.shape}")
        return regime_panel, system_regime

    def _classify_sector(
        self, sec: str, E_crit, L_hi, E_min, L_lo, conn_thresh
    ) -> pd.DataFrame:
        """Classify sector regimes over time."""
        s  = self.ser[sec]["stress"]
        L  = self.ser[sec]["leaked_stress"]
        E  = self.ser[sec]["elasticity"]
        d  = self.ser[sec]["damping"]
        sp = self.ser[sec]["spillover_share"]
        ret = self.ser[sec]["retention"]

        # Propagation received (if available)
        if sec in self.prop.columns.get_level_values(0):
            P_in = self.prop[sec].get("propagation_raw", pd.Series(0, index=s.index))
        else:
            P_in = pd.Series(0, index=s.index)

        # System connectivity
        conn_col = ("_system", "effective_connectivity")
        if conn_col in self.prop.columns:
            eff_conn = self.prop[conn_col]
        else:
            eff_conn = pd.Series(1.0, index=s.index)

        # Trend indicators (rolling slope over 3 periods)
        L_trend  = L.diff(2)
        E_trend  = E.diff(2)

        T = len(s)
        regime_arr   = ["unknown"] * T
        code_arr     = np.zeros(T, dtype=int)
        breach_arr   = (E.values <= E_crit).astype(int)
        hfail_arr    = (L.values >= 0.6).astype(int)   # headroom failure
        afail_arr    = (E.values <= 0.15).astype(int)  # acceptance failure

        for t in range(T):
            lv  = float(L.iloc[t])
            ev  = float(E.iloc[t])
            sv  = float(s.iloc[t])
            pv  = float(P_in.iloc[t])
            cv  = float(eff_conn.iloc[t]) if t < len(eff_conn) else 1.0
            lt  = float(L_trend.iloc[t]) if not np.isnan(L_trend.iloc[t]) else 0
            et  = float(E_trend.iloc[t]) if not np.isnan(E_trend.iloc[t]) else 0

            # Priority classification
            if ev <= E_crit and lv >= L_hi:
                r = "amplification"
            elif cv <= conn_thresh and sv > 0.2:
                r = "fragmented"
            elif pv < 0.01 and sv > 0.3:
                r = "isolation"
            elif lv <= L_lo and lt <= 0 and et >= 0 and ev >= E_min:
                r = "dispersal"
            elif lt < -0.01 and et > 0.01:
                r = "recovery"
            elif lv >= L_hi or (lt > 0 and ret.iloc[t] > 0.3):
                r = "accumulation"
            else:
                r = "dispersal"  # default to dispersal if mild

            regime_arr[t] = r
            code_arr[t]   = REGIME_CODES[r]

        return pd.DataFrame(
            {
                "regime":           regime_arr,
                "regime_code":      code_arr,
                "ecrit_breach":     breach_arr,
                "headroom_failure": hfail_arr,
                "accept_failure":   afail_arr,
            },
            index=s.index,
        )

    def _classify_system(self, regime_panel: pd.DataFrame) -> pd.Series:
        """
        Classify the system-level regime from aggregate sector states.

        System regime logic:
        - fragmented: >50% sectors in fragmented or isolation
        - amplification: >40% sectors in amplification
        - accumulation: >50% sectors in accumulation
        - recovery: >50% sectors in recovery or dispersal
        - dispersal: default
        """
        dates = regime_panel.index
        system_regimes = []

        for t in range(len(dates)):
            counts = {"dispersal":0,"accumulation":0,"isolation":0,
                      "recovery":0,"fragmented":0,"amplification":0,"unknown":0}
            total = 0
            for sec in self.sectors:
                if sec not in regime_panel.columns.get_level_values(0):
                    continue
                r = regime_panel[sec]["regime"].iloc[t]
                counts[r] = counts.get(r, 0) + 1
                total += 1
            if total == 0:
                system_regimes.append("unknown")
                continue

            fracs = {k: v/total for k, v in counts.items()}
            if fracs.get("fragmented",0) + fracs.get("isolation",0) > 0.50:
                sr = "fragmented"
            elif fracs.get("amplification",0) > 0.40:
                sr = "amplification"
            elif fracs.get("accumulation",0) > 0.45:
                sr = "accumulation"
            elif fracs.get("recovery",0) + fracs.get("dispersal",0) > 0.55:
                sr = "recovery"
            else:
                sr = "dispersal"
            system_regimes.append(sr)

        return pd.Series(system_regimes, index=dates, name="system_regime")

    def transition_matrix(self) -> pd.DataFrame:
        """Compute regime transition count matrix per sector (averaged)."""
        if self._sector_results is None:
            raise RuntimeError("Run .run() first")
        combined_trans = pd.DataFrame(0, index=REGIMES[:-1], columns=REGIMES[:-1])
        for sec in self.sectors:
            if sec not in self._sector_results.columns.get_level_values(0):
                continue
            regimes = self._sector_results[sec]["regime"]
            for t in range(1, len(regimes)):
                r_from = regimes.iloc[t-1]
                r_to   = regimes.iloc[t]
                if r_from in combined_trans.index and r_to in combined_trans.columns:
                    combined_trans.loc[r_from, r_to] += 1
        return combined_trans

    def regime_duration_table(self) -> pd.DataFrame:
        """Mean duration (in periods) per regime per sector."""
        if self._sector_results is None:
            raise RuntimeError("Run .run() first")
        rows = []
        for sec in self.sectors:
            if sec not in self._sector_results.columns.get_level_values(0):
                continue
            regimes = self._sector_results[sec]["regime"]
            for r in REGIMES[:-1]:
                mask = regimes == r
                # Find run lengths
                runs = _run_lengths(mask.values)
                rows.append({
                    "sector": sec,
                    "regime": r,
                    "total_periods": int(mask.sum()),
                    "pct_time": float(mask.mean() * 100),
                    "mean_run_length": float(np.mean(runs)) if runs else 0,
                    "max_run_length": int(max(runs)) if runs else 0,
                })
        return pd.DataFrame(rows)

    def threshold_breach_summary(self) -> pd.DataFrame:
        """Count E_crit and headroom breaches per sector."""
        if self._sector_results is None:
            raise RuntimeError("Run .run() first")
        rows = []
        for sec in self.sectors:
            if sec not in self._sector_results.columns.get_level_values(0):
                continue
            df = self._sector_results[sec]
            rows.append({
                "sector": sec,
                "ecrit_breaches": int(df["ecrit_breach"].sum()),
                "ecrit_pct": float(df["ecrit_breach"].mean() * 100),
                "headroom_failures": int(df["headroom_failure"].sum()),
                "accept_failures":   int(df["accept_failure"].sum()),
            })
        return pd.DataFrame(rows).set_index("sector")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_lengths(bool_arr: np.ndarray) -> List[int]:
    """Return list of run lengths where bool_arr is True."""
    runs = []
    count = 0
    for v in bool_arr:
        if v:
            count += 1
        else:
            if count > 0:
                runs.append(count)
                count = 0
    if count > 0:
        runs.append(count)
    return runs

from typing import List  # noqa: E402 (for _run_lengths annotation)
