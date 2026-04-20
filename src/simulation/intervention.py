"""
simulation/intervention.py
───────────────────────────
Exogenous elasticity inflow intervention layer.

Adds to the SER elasticity update equation:
  E_{i,t+1} += ω_i * H_t

Where:
  H_t = A_t * G_t
  A_t = logistic adoption ramp (sector-common)
  G_t = normalized monthly gain (default constant)
  ω_i = sector-specific diffusion weight

This module does NOT modify:
  - stress S_{i,t} construction
  - leaked stress L_{i,t} directly (it changes via elasticity)
  - regime classification logic
  - graph structure

Usage:
  engine = InterventionEngine(cfg, intervention_cfg)
  for scenario in ["baseline","low","medium","high"]:
      result = engine.run_scenario(scenario, inputs)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("intervention")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


# ─── H_t Construction ─────────────────────────────────────────────────────────

def build_H_t(
    T: int,
    t_start: int,
    A_max: float,
    k: float,
    t0: float,
    G_base: float,
    seasonal_amp: float = 0.0,
    season_offset: int = 0,
) -> np.ndarray:
    """
    Construct H_t = A_t * G_t for t in [0, T).

    A_t = A_max / (1 + exp(-k * (t - t_start - t0)))   for t >= t_start
        = 0                                              for t < t_start

    G_t = G_base * (1 + seasonal_amp * sin(2π(t - season_offset)/12))

    Parameters
    ----------
    T           : number of time periods
    t_start     : first period intervention is active (0-indexed months)
    A_max       : maximum adoption level [0,1]
    k           : logistic growth rate
    t0          : logistic inflection offset from t_start (months)
    G_base      : constant monthly gain per unit adoption
    seasonal_amp: optional seasonal amplitude (0 = constant)
    season_offset: seasonal phase shift (months)

    Returns
    -------
    H : np.ndarray of shape (T,), values in [0, A_max * G_base]
    """
    t_arr = np.arange(T, dtype=float)

    # Adoption ramp A_t
    A = np.zeros(T)
    active = t_arr >= t_start
    t_shifted = t_arr - t_start - t0
    A[active] = A_max / (1.0 + np.exp(-k * t_shifted[active]))

    # Gain G_t (with optional seasonality)
    G = G_base * (1.0 + seasonal_amp * np.sin(2 * np.pi * (t_arr - season_offset) / 12))
    G = np.maximum(G, 0.0)

    H = A * G
    return H


# ─── Intervention Engine ──────────────────────────────────────────────────────

class InterventionEngine:
    """
    Runs multiple intervention scenarios over fixed sector inputs.

    Each scenario re-executes the elasticity loop with the same
    stress/B/C inputs but with ω_i * H_t added at each step.

    Parameters
    ----------
    ser_cfg         : SER section of base_config (elasticity params)
    intervention_cfg: Full intervention config dict
    """

    def __init__(self, ser_cfg: Dict, intervention_cfg: Dict):
        self.el  = ser_cfg.get("elasticity", ser_cfg)
        self.icfg = intervention_cfg
        self.scenarios = intervention_cfg["scenarios"]
        self.weights: Dict[str, float] = intervention_cfg["sector_weights"]
        self.secondary = intervention_cfg.get("secondary_clearance", False)
        self.clearance_boost = intervention_cfg.get("clearance_boost_fraction", 0.20)

    # ──────────────────────────────────────────────────────────────────────────

    def run_scenario(
        self,
        scenario_name: str,
        inputs: Dict[str, Dict[str, np.ndarray]],
        T: int = 36,
        rng_seed: int = 42,
    ) -> Dict:
        """
        Run one intervention scenario across all sectors.

        Returns dict with:
          - 'H': H_t array
          - 'A': A_t array
          - per-sector DataFrames keyed by sector name
          - 'scenario': scenario name
          - 'label': human-readable label
        """
        scfg  = self.scenarios[scenario_name]
        label = scfg["label"]
        rng   = np.random.default_rng(rng_seed)
        sources = scfg.get("sources", ["generic"])

        # Build composite H_t by summing across all listed source channels.
        # Each source uses its own A_t / G_t parameters and sector weight map.
        H_total   = np.zeros(T)
        H_by_src  = {}
        A_primary = np.zeros(T)

        for src_name in sources:
            if not sources or src_name == "generic":
                # Generic: use scenario-level params directly
                sp = scfg
            else:
                sp = self.icfg.get("sources", {}).get(src_name, scfg)

            H_src = build_H_t(
                T=T,
                t_start=sp.get("t_start", scfg.get("t_start", 0)),
                A_max=sp.get("A_max",  scfg.get("A_max",  0.0)),
                k=sp.get("k",          scfg.get("k",      0.3)),
                t0=sp.get("t0",        scfg.get("t0",     10)),
                G_base=sp.get("G_base", scfg.get("G_base", 0.0)),
                seasonal_amp=sp.get("seasonal_amp", 0.0),
                season_offset=sp.get("season_offset", 0),
            )
            H_by_src[src_name] = H_src
            H_total += H_src

            if src_name in ("generic", sources[0]):
                t0_s = sp.get("t0", scfg.get("t0", 10))
                ts_s = sp.get("t_start", scfg.get("t_start", 0))
                am_s = sp.get("A_max",  scfg.get("A_max",  0.0))
                k_s  = sp.get("k",      scfg.get("k",      0.3))
                A_primary = np.array([
                    am_s / (1.0 + np.exp(-k_s * (t - ts_s - t0_s)))
                    if t >= ts_s else 0.0
                    for t in range(T)
                ])

        logger.info(
            f"[Intervention] scenario={scenario_name}  sources={sources}  "
            f"H_total_max={H_total.max():.4f}  H_total_mean={H_total.mean():.4f}"
        )

        # Resolve per-sector omega as sum of source-specific weighted contributions
        # H_effective_i = sum_src(omega_i^src * H_src)
        sector_results = {}
        for sec, inp in inputs.items():
            # Compute sector-specific effective inflow: sum over sources
            H_eff = np.zeros(T)
            for src_name in sources:
                weight_map = self._get_weight_map(src_name)
                omega_src  = weight_map.get(sec, 0.0)
                H_eff += omega_src * H_by_src.get(src_name, np.zeros(T))

            df = self._compute_sector(sec, inp, H_eff, 1.0, T, rng)
            sector_results[sec] = df

        return {
            "scenario":      scenario_name,
            "label":         label,
            "H":             H_total,
            "A":             A_primary,
            "H_by_source":   H_by_src,
            "sectors":       sector_results,
        }

    def _get_weight_map(self, source_name: str) -> Dict[str, float]:
        """Return the correct weight map for a given source name."""
        if source_name == "generic":
            return self.icfg.get("sector_weights", {})
        elif source_name == "food":
            return self.icfg.get("food_weights", self.icfg.get("sector_weights", {}))
        elif source_name == "filter":
            return self.icfg.get("filter_weights", self.icfg.get("sector_weights", {}))
        else:
            return self.icfg.get("sector_weights", {})

    def run_all_scenarios(
        self,
        inputs: Dict[str, Dict[str, np.ndarray]],
        T: int = 36,
    ) -> Dict[str, Dict]:
        """Run all scenarios; return dict keyed by scenario name."""
        results = {}
        for name in self.scenarios:
            results[name] = self.run_scenario(name, inputs, T)
        return results

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_sector(
        self,
        sector: str,
        inp: Dict[str, np.ndarray],
        H: np.ndarray,
        omega: float,
        T: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Elasticity loop for one sector under the intervention inflow.

        Implements:
          E_{t+1} = (1-δ)·E_t
                  + α·B_t·(1-E_t) + χ·C_t·(1-E_t)
                  - β_L·(L_t + γ_dL·ΔL_t⁺)
                  + ω_i·H_t                           ← intervention term
        """
        el = self.el
        alpha    = el["alpha"]
        beta_L   = el["beta_L"]
        beta_S   = el.get("beta_S", 0.0)
        chi      = el["chi"]
        delta    = el["delta"]
        gamma_dL = el["gamma_dL"]
        S_thresh = el.get("S_drain_thresh", 0.40)
        E_init   = el.get("E_init", 0.35)

        stress = inp["stress"]
        B      = inp["B"]
        C_base = inp["C"]

        noise = rng.normal(0, el.get("noise_std", 0.015), T)

        E = np.zeros(T)
        L = np.zeros(T)
        R = np.zeros(T)
        d = np.zeros(T)
        E[0] = E_init

        for t in range(T):
            # Clearance — optionally boosted by secondary channel
            C_t = C_base[t]
            if self.secondary:
                C_t = np.clip(C_t + self.clearance_boost * H[t], 0, 1)

            # Damping
            theta_t = stress[t-1] if t > 0 else 0.0
            k_E, k_theta = 1.5, 0.5
            d[t] = 1.0 / (1.0 + k_E * E[t] + k_theta * theta_t)

            # Absorption and leaked stress
            absorb = E[t] / (E[t] + d[t] * stress[t] + 1e-8) * stress[t]
            absorb = np.clip(absorb, 0, stress[t])
            L[t]   = max(0.0, stress[t] - absorb)

            # Reaction
            R_prev = R[t-1] if t > 0 else 0.0
            R[t] = np.clip(
                0.4 * L[t] - 0.3 * E[t] + 0.5 * R_prev + noise[t],
                -1, 2
            )

            # Elasticity update
            if t < T - 1:
                dL_t  = max(0.0, L[t] - (L[t-1] if t > 0 else 0.0))
                headroom = 1.0 - E[t]
                gain  = alpha * B[t] * headroom + chi * C_t * headroom
                decay = delta * E[t]
                drain = (beta_L * (L[t] + gamma_dL * dL_t)
                         + beta_S * max(0.0, stress[t] - S_thresh))
                inflow = omega * H[t]           # ← intervention term

                E[t+1] = np.clip(E[t] + gain - decay - drain + inflow, 0.01, 1.30)

        idx = inp.get("index", pd.RangeIndex(T))
        return pd.DataFrame({
            "elasticity":     E,
            "leaked_stress":  L,
            "stress":         stress,
            "reaction":       R,
            "damping":        d,
            "H_t":            H,
            "H_contribution": H,   # already sector-weighted (omega=1.0 passthrough)
            "below_ecrit":    (E <= 0.25).astype(float),
        }, index=idx)


# ─── Metrics extraction ───────────────────────────────────────────────────────

REGIME_ORDER = ["dispersal","accumulation","isolation","recovery","fragmented","amplification"]


def classify_regime_simple(E, L, S, E_prev=None, L_prev=None):
    L_trend = (L - L_prev) if L_prev is not None else 0.0
    E_trend = (E - E_prev) if E_prev is not None else 0.0
    if E <= 0.25 and L >= 0.20:
        return "amplification"
    if L <= 0.20 and L_trend <= 0 and E_trend >= 0 and E >= 0.35:
        return "dispersal"
    if L_trend < -0.01 and E_trend > 0.01:
        return "recovery"
    if L >= 0.20 or L_trend > 0:
        return "accumulation"
    if S > 0.3 and L < 0.05:
        return "isolation"
    return "dispersal"


def extract_metrics(result: Dict, scenario_name: str) -> Dict:
    """Compute all required metrics from a scenario run result."""
    sectors = result["sectors"]
    T = len(result["H"])

    all_E, all_L, all_S, all_R = [], [], [], []
    sector_metrics = {}
    all_regimes = []

    for sec, df in sectors.items():
        E = df["elasticity"].values
        L = df["leaked_stress"].values
        S = df["stress"].values
        R = df["reaction"].values

        all_E.append(E); all_L.append(L)
        all_S.append(S); all_R.append(R)

        # Sector lag
        t_sp = int(np.argmax(S))
        t_em = int(np.argmin(E))
        t_rp = int(np.argmax(R))

        # Per-sector regime series
        regs = []
        for t in range(T):
            regs.append(classify_regime_simple(
                E[t], L[t], S[t],
                E[t-1] if t>0 else None,
                L[t-1] if t>0 else None,
            ))
        all_regimes.extend(regs)

        sector_metrics[sec] = {
            "E_mean":              float(E.mean()),
            "E_max":               float(E.max()),
            "E_min":               float(E.min()),
            "pct_below_ecrit":     float((E <= 0.25).mean() * 100),
            "L_mean":              float(L.mean()),
            "L_peak":              float(L.max()),
            "t_stress_peak":       t_sp,
            "t_elast_min":         t_em,
            "t_react_peak":        t_rp,
            "lag_E":               t_em - t_sp,
            "H_contribution_mean": float(df["H_contribution"].mean()),
        }

    E_all = np.concatenate(all_E)
    L_all = np.concatenate(all_L)
    S_all = np.concatenate(all_S)
    R_all = np.concatenate(all_R)

    # Below-E_crit run lengths
    below_mask = E_all <= 0.25
    runs, count = [], 0
    for v in below_mask:
        if v: count += 1
        else:
            if count: runs.append(count); count = 0
    if count: runs.append(count)
    mean_run = float(np.mean(runs)) if runs else 0.0

    # Regime distribution
    total_reg = len(all_regimes)
    rdist = {r: all_regimes.count(r) / total_reg * 100 for r in REGIME_ORDER}
    n_active = sum(1 for v in rdist.values() if v > 1.0)

    # Mean lag across sectors
    mean_lag = float(np.mean([m["lag_E"] for m in sector_metrics.values()]))

    return {
        "scenario":           scenario_name,
        "label":              result["label"],
        # Elasticity
        "E_mean":             float(E_all.mean()),
        "E_std":              float(E_all.std()),
        "E_max":              float(E_all.max()),
        "E_min":              float(E_all.min()),
        "pct_below_ecrit":    float(below_mask.mean() * 100),
        "mean_run_below":     mean_run,
        # Lag
        "mean_lag":           mean_lag,
        # System
        "S_mean":             float(S_all.mean()),
        "L_mean":             float(L_all.mean()),
        "R_mean":             float(R_all.mean()),
        # Regimes
        "pct_dispersal":      rdist.get("dispersal", 0),
        "pct_accumulation":   rdist.get("accumulation", 0),
        "pct_amplification":  rdist.get("amplification", 0),
        "pct_recovery":       rdist.get("recovery", 0),
        "pct_isolation":      rdist.get("isolation", 0),
        "pct_fragmented":     rdist.get("fragmented", 0),
        "n_active_regimes":   n_active,
        "H_max":              float(result["H"].max()),
        "H_mean":             float(result["H"].mean()),
        # Per-sector
        "sector_metrics":     sector_metrics,
    }
