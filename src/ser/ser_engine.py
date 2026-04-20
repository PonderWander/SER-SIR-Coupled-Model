"""
ser/ser_engine.py
──────────────────
Stress–Elasticity–Reaction (SER) state construction engine.

For each sector i and time t, computes:
  S_{i,t}          : composite stress
  S_absorbed_{i,t} : absorbed portion
  L_{i,t}          : leaked stress
  E_{i,t}          : elasticity
  R_{i,t}          : reaction
  Phi_{i,t}        : discrepancy field (price - supply signal)
  d_{i,t}          : damping coefficient
  retention_{i,t}  : fraction of stress retained (absorbed / total)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.common import minmax_normalize, rolling_zscore

logger = logging.getLogger("ser_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


# ─── SER Engine ───────────────────────────────────────────────────────────────

class SEREngine:
    """
    Sector SER state construction engine.

    Operates over the full sector panel, producing per-sector SER states
    at each time period.

    Parameters
    ----------
    panel    : MultiIndex DataFrame (date x (sector, variable))
               as produced by SectorPanelBuilder
    config   : Nested dict from base_config.yaml ['ser'] section
    sector_meta : Metadata dict from sector_graph.yaml
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        config: Dict,
        sector_meta: Optional[Dict] = None,
    ):
        self.panel = panel
        self.cfg = config.get("ser", config)
        self.sector_meta = sector_meta or {}
        self._validate_config()
        self.sectors = panel.columns.get_level_values(0).unique().tolist()
        self._results: Optional[pd.DataFrame] = None

    def _validate_config(self) -> None:
        """Set defaults for any missing config keys."""
        c = self.cfg
        c.setdefault("default_weights", {})
        dw = c["default_weights"]
        dw.setdefault("w_price", 0.35)
        dw.setdefault("w_volatility", 0.20)
        dw.setdefault("w_efficiency", 0.20)
        dw.setdefault("w_covid", 0.15)
        dw.setdefault("w_shortage", 0.10)

        el = c.setdefault("elasticity", {})
        el.setdefault("alpha",       0.25)   # buffering gain (was 0.15)
        el.setdefault("beta_L",      0.60)   # leaked-stress drain (replaces beta)
        el.setdefault("beta_S",      0.30)   # direct stress-level drain (new)
        el.setdefault("chi",         0.18)   # clearance gain (was 0.10)
        el.setdefault("delta",       0.05)   # natural decay rate per period (new)
        el.setdefault("gamma_dL",    0.80)   # sensitivity to rising leaked stress (new)
        el.setdefault("S_drain_thresh", 0.40) # stress level above which beta_S activates (new)
        el.setdefault("E_init",      0.35)   # sectors start with modest buffering (was 0.5)
        el.setdefault("E_crit",      0.25)
        el.setdefault("adaptive",    False)  # enable adaptive elasticity variant
        el.setdefault("adapt_window", 6)     # periods of high stress before recovery slows
        el.setdefault("adapt_penalty", 0.4)  # max recovery speed reduction under prolonged stress

        rx = c.setdefault("reaction", {})
        rx.setdefault("a", 0.4)
        rx.setdefault("b", 0.3)
        rx.setdefault("gamma_R", 0.5)
        rx.setdefault("noise_std", 0.02)

        ab = c.setdefault("absorption", {})
        ab.setdefault("k_E", 1.5)
        ab.setdefault("k_theta", 0.5)

    # ──────────────────────────────────────────────────────────────────────────

    def run(self, rng_seed: int = 42) -> pd.DataFrame:
        """
        Compute full SER panel for all sectors.

        Returns
        -------
        MultiIndex DataFrame: (date x (sector, ser_variable))
        """
        rng = np.random.default_rng(rng_seed)
        sector_results = {}

        for sec in self.sectors:
            sec_data = self.panel[sec] if sec in self.panel.columns.get_level_values(0) else None
            if sec_data is None or sec_data.empty:
                logger.warning(f"[SER] No panel data for sector '{sec}'. Skipping.")
                continue
            ser_df = self._compute_sector_ser(sec, sec_data, rng)
            sector_results[sec] = ser_df
            logger.info(
                f"[SER] {sec}: stress_peak={ser_df['stress'].max():.3f}, "
                f"elasticity_min={ser_df['elasticity'].min():.3f}"
            )

        result = pd.concat(sector_results, axis=1)
        self._results = result
        logger.info(f"[SER] Full panel computed: {result.shape}")
        return result

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_sector_ser(
        self, sector: str, data: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Compute SER states for a single sector.

        Implements the full SER specification:
        S, S_absorbed, L, E, d, R, Phi, retention, spillover_share
        """
        cfg = self.cfg
        dw = cfg["default_weights"]
        el = cfg["elasticity"]
        rx = cfg["reaction"]
        ab = cfg["absorption"]
        meta = self.sector_meta.get(sector, {})

        T = len(data)
        idx = data.index

        # ── 1. Composite Stress ──────────────────────────────────────────
        # S_{i,t} = weighted normalized composite
        w_p = dw["w_price"]
        w_v = dw["w_volatility"]
        w_e = dw["w_efficiency"]
        w_c = dw["w_covid"]
        w_s = dw["w_shortage"]

        price_p   = _safe_col(data, "price_pressure",   T)
        vol       = _safe_col(data, "volatility",        T)
        eff_loss  = _safe_col(data, "efficiency_loss",   T)
        ep        = _safe_col(data, "epidemic_pressure", T)
        shortage  = _safe_col(data, "shortage_proxy",    T)

        stress = (
            w_p * price_p
            + w_v * vol
            + w_e * eff_loss
            + w_c * ep
            + w_s * shortage
        )
        # Ensure [0,1]
        stress = np.clip(stress, 0, 1)

        # ── 2. Elasticity (state variable, evolves over time) ────────────
        #
        # REPAIRED EQUATION (four mechanisms):
        #
        # E_{t+1} = (1-δ)·E_t                              [natural decay]
        #         + α·B_t·(1-E_t) + χ·C_t·(1-E_t)          [diminishing-returns gains]
        #         - β_L·(L_t + γ_dL·ΔL_t⁺)                 [leak + rising-leak drain]
        #         - β_S·max(0, S_t - S_thresh)              [direct high-stress drain]
        #
        # Mechanism 1 — Natural decay (δ): E erodes passively each period.
        #   Prevents indefinite accumulation without active buffering input.
        #
        # Mechanism 2 — Diminishing returns (1-E_t headroom factor): gains
        #   from buffering and clearance weaken as E approaches 1. Prevents
        #   saturation at the ceiling.
        #
        # Mechanism 3 — Rate-sensitive leak drain (γ_dL·ΔL⁺): penalises not
        #   just the level of leaked stress but its *rise*, so rapidly worsening
        #   sectors lose elasticity faster than slowly-building ones.
        #
        # Mechanism 4 — Stress-level drain (β_S): when total stress exceeds a
        #   threshold, elasticity is directly eroded. Captures the reality that
        #   sustained high overall stress impairs buffering capacity independent
        #   of what fraction is leaking.
        #
        # Adaptive variant: if el["adaptive"]=True, recovery speed (α, χ) is
        #   further reduced after prolonged high-stress episodes, capturing
        #   structural impairment of buffering capacity.

        alpha      = el["alpha"]
        beta_L     = el["beta_L"]
        beta_S     = el["beta_S"]
        chi        = el["chi"]
        delta      = el["delta"]
        gamma_dL   = el["gamma_dL"]
        S_thresh   = el["S_drain_thresh"]
        adaptive   = el["adaptive"]
        adapt_win  = int(el["adapt_window"])
        adapt_pen  = el["adapt_penalty"]

        E = np.zeros(T)
        E[0] = el["E_init"]

        # Buffering inputs: inventory slack + inverse labor disruption
        inv_slack = _safe_col(data, "inventory_slack", T)
        labor_d   = _safe_col(data, "labor_disruption", T)
        B = np.clip(inv_slack - 0.3, 0, 1)   # slack above floor
        C = np.clip(1.0 - labor_d,  0, 1)    # labor normalization

        # ── 3–4. Absorbed stress, Leaked stress ──────────────────────────
        k_E     = ab["k_E"]
        k_theta = ab["k_theta"]

        S_absorbed = np.zeros(T)
        L          = np.zeros(T)
        d          = np.zeros(T)  # damping coefficient
        R          = np.zeros(T)  # reaction
        R[0]       = 0.0

        noise_std = rx["noise_std"]
        noise     = rng.normal(0, noise_std, T)

        for t in range(T):
            # Damping: d(E,theta) = 1 / (1 + k_E*E + k_theta*theta)
            # theta approximated as persistence of stress
            theta_t = stress[t-1] if t > 0 else 0.0
            d[t] = 1.0 / (1.0 + k_E * E[t] + k_theta * theta_t)

            # Absorbed: fraction of stress the sector can buffer
            # proportional to elasticity via sigmoid-like scaling
            absorb_capacity = E[t] / (E[t] + d[t] * stress[t] + 1e-8)
            S_absorbed[t] = absorb_capacity * stress[t]
            S_absorbed[t] = np.clip(S_absorbed[t], 0, stress[t])

            # Leaked stress
            L[t] = max(0.0, stress[t] - S_absorbed[t])

            # Reaction: R_{i,t} = a*L - b*E + gamma_R*R_{t-1} + noise
            R_prev = R[t-1] if t > 0 else 0.0
            R[t] = (
                rx["a"] * L[t]
                - rx["b"] * E[t]
                + rx["gamma_R"] * R_prev
                + noise[t]
            )
            R[t] = np.clip(R[t], -1, 2)

            # ── Elasticity update for next period ────────────────────────
            if t < T - 1:
                L_t   = L[t]
                S_t   = stress[t]
                dL_t  = max(0.0, L[t] - (L[t-1] if t > 0 else 0.0))  # rising-leak only

                # Adaptive recovery modifier: if sector has been under sustained
                # high stress, α and χ are penalised proportionally.
                if adaptive and t >= adapt_win:
                    recent_high = np.mean(stress[t - adapt_win: t] > S_thresh)
                    adapt_factor = 1.0 - adapt_pen * recent_high
                else:
                    adapt_factor = 1.0

                headroom = 1.0 - E[t]                          # diminishing returns
                gain  = adapt_factor * (alpha * B[t] * headroom + chi * C[t] * headroom)
                decay = delta * E[t]                            # natural depreciation
                drain = (beta_L * (L_t + gamma_dL * dL_t)      # leak + rising-leak
                         + beta_S * max(0.0, S_t - S_thresh))  # high-stress drain

                E[t+1] = np.clip(E[t] + gain - decay - drain, 0.01, 1.30)

        # ── 5. Discrepancy field (Phi) ───────────────────────────────────
        # Phi = normalized_price_signal - normalized_supply_signal
        price_sig  = minmax_normalize(pd.Series(price_p, index=idx))
        supply_sig = minmax_normalize(pd.Series(1.0 - eff_loss, index=idx))
        phi = (price_sig - supply_sig).values

        # ── 6. Regime-diagnostic quantities ──────────────────────────────
        retention = np.where(stress > 0, S_absorbed / (stress + 1e-8), 0.0)
        spillover_share = np.where(stress > 0, L / (stress + 1e-8), 0.0)
        above_crit = (E <= el["E_crit"]).astype(float)

        result = pd.DataFrame(
            {
                "stress":           stress,
                "absorbed_stress":  S_absorbed,
                "leaked_stress":    L,
                "elasticity":       E,
                "reaction":         R,
                "damping":          d,
                "phi":              phi,
                "retention":        retention,
                "spillover_share":  spillover_share,
                "below_ecrit":      above_crit,
                # decomposition terms for dashboard
                "stress_price":     w_p * price_p,
                "stress_volatility":w_v * vol,
                "stress_efficiency":w_e * eff_loss,
                "stress_covid":     w_c * ep,
                "stress_shortage":  w_s * shortage,
                "buffering_B":      B,
                "clearance_C":      C,
            },
            index=idx,
        )
        return result

    def get_results(self) -> Optional[pd.DataFrame]:
        return self._results

    def sector_summary(self) -> pd.DataFrame:
        """Summary statistics across sectors for the full simulation period."""
        if self._results is None:
            raise RuntimeError("Run .run() first.")
        rows = []
        for sec in self.sectors:
            if sec not in self._results.columns.get_level_values(0):
                continue
            df = self._results[sec]
            rows.append({
                "sector": sec,
                "stress_mean":        df["stress"].mean(),
                "stress_peak":        df["stress"].max(),
                "stress_peak_date":   str(df["stress"].idxmax().date()),
                "leaked_stress_mean": df["leaked_stress"].mean(),
                "leaked_stress_peak": df["leaked_stress"].max(),
                "elasticity_mean":    df["elasticity"].mean(),
                "elasticity_min":     df["elasticity"].min(),
                "elasticity_min_date":str(df["elasticity"].idxmin().date()),
                "reaction_peak":      df["reaction"].max(),
                "below_ecrit_pct":    df["below_ecrit"].mean() * 100,
                "retention_mean":     df["retention"].mean(),
                "spillover_mean":     df["spillover_share"].mean(),
            })
        return pd.DataFrame(rows).set_index("sector")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_col(data: pd.DataFrame, col: str, T: int) -> np.ndarray:
    """Extract column from DataFrame, fill NaN with 0, return ndarray of length T."""
    if col in data.columns:
        arr = data[col].fillna(0).values
    else:
        arr = np.zeros(T)
    return np.clip(arr, 0, 1)
