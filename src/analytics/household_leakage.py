"""
analytics/household_leakage.py
────────────────────────────────
Leakage decomposition and household-channel breakpoint diagnostics.

DECOMPOSITION:
  L_total[i,t] = L_market[i,t] + L_household[i,t]

  L_market   : leaked stress exiting into observable exchange / transaction /
               institutional channels (= SER engine's leaked_stress)

  L_household: leaked stress exiting into household / local / temporal / human
               channels outside the standard market measurement frame —
               inferred as residual between transmission-consistent expected
               activity and observed activity. This is a propagated exit
               channel, not suppression.

IMPORTANT EPISTEMIC STATUS:
  L_household is an inferred residual. It is not directly observed.
  The estimate depends on the quality of A_obs (currently efficiency/inventory
  proxies; improves materially with real OI spend data).

A^expected construction:
  A_exp[i,t] = clip( (1-S[i,t]) * (0.6 + 0.25*E_norm[i,t] + 0.15*W_in_norm[i,t]), 0, 1 )
  Captures what activity level the model's capacity + connectivity structure implies.

A^observed construction (independent of stress construction):
  A_obs[i,t] = 0.6*(1-efficiency_loss[i,t]) + 0.4*inventory_slack[i,t]
  Uses FRED TCU/ISRATIO proxies — different weighting from stress composite.

L_household[i,t] = clip(A_exp[i,t] - A_obs[i,t], 0, 1)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro, jarque_bera, anderson, skew, kurtosis,
    norm as scipy_norm
)

logger = logging.getLogger("household_leakage")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


# ─── Leakage Construction ────────────────────────────────────────────────────

def build_leakage_decomposition(
    ser_panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    W_hist: pd.DataFrame,
    avail_sectors: List[str],
) -> pd.DataFrame:
    """
    Build the full leakage decomposition panel.

    Returns long-format DataFrame with columns:
      date, sector, L_market, L_household, L_household_alt, L_total,
      A_obs, A_exp, S, E, regime
    """
    T      = len(ser_panel)
    dates  = ser_panel.index

    # Incoming mean effective weight per sector
    W_in_mean: Dict[str, np.ndarray] = {}
    for s in avail_sectors:
        inc = [col for col in W_hist.columns if isinstance(col, tuple) and col[1] == s]
        W_in_mean[s] = W_hist[inc].mean(axis=1).values if inc else np.full(T, 0.3)

    rows = []
    for s in avail_sectors:
        S   = ser_panel[(s, "stress")].values
        E   = ser_panel[(s, "elasticity")].values
        Lm  = ser_panel[(s, "leaked_stress")].values   # L_market

        A_obs = _build_A_obs(s, raw_panel, T)
        A_exp = _build_A_exp(S, E, W_in_mean[s])

        Lh     = np.clip(A_exp - A_obs, 0.0, 1.0)   # L_household (primary)
        Lh_alt = _build_L_household_robust(A_obs, S, E)  # robust z-score form
        L_tot  = Lm + Lh

        for t in range(T):
            rows.append({
                "date":            dates[t],
                "sector":          s,
                "L_market":        float(Lm[t]),
                "L_household":     float(Lh[t]),
                "L_household_alt": float(Lh_alt[t]),
                "L_total":         float(L_tot[t]),
                "A_obs":           float(A_obs[t]),
                "A_exp":           float(A_exp[t]),
                "S":               float(S[t]),
                "E":               float(E[t]),
            })

    df = pd.DataFrame(rows)
    logger.info(f"[Leakage] Decomposition built: {df.shape}")
    return df


def _build_A_obs(sector: str, raw_panel: pd.DataFrame, T: int) -> np.ndarray:
    """Independent activity signal from efficiency + inventory proxies."""
    if sector not in raw_panel.columns.get_level_values(0):
        return np.full(T, 0.55)
    sp = raw_panel[sector]
    eff = sp["efficiency_loss"].fillna(0).values if "efficiency_loss" in sp.columns else np.zeros(T)
    inv = sp["inventory_slack"].fillna(0).values if "inventory_slack" in sp.columns else np.full(T, 0.5)
    return np.clip(0.6 * (1 - eff) + 0.4 * inv, 0, 1)


def _build_A_exp(S: np.ndarray, E: np.ndarray, W_in: np.ndarray) -> np.ndarray:
    """Transmission-consistent expected activity."""
    E_norm = (E - E.min()) / (E.max() - E.min() + 1e-8)
    W_norm = (W_in - W_in.min()) / (W_in.max() - W_in.min() + 1e-8)
    return np.clip((1 - S) * (0.6 + 0.25 * E_norm + 0.15 * W_norm), 0, 1)


def _build_L_household_robust(A_obs: np.ndarray, S: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Robust z-score residual form of L_household."""
    z_A   = (A_obs - A_obs.mean()) / (A_obs.std() + 1e-8)
    cap   = E * (1 - S)
    z_cap = (cap - cap.mean()) / (cap.std() + 1e-8)
    return np.clip(z_A - z_cap, -1, 1)


# ─── Distributional Analysis ─────────────────────────────────────────────────

class HouseholdLeakageDiagnostics:
    """
    Full distributional breakpoint analysis of L_household.

    Tests whether L_household behaves as an ordinary bounded stochastic
    variable or as a statistical breakpoint / exit-point process.
    """

    NEAR_ZERO_THRESH = 0.005   # values below this treated as "near zero"
    ACTIVATION_THRESH_PCTILE = 75  # percentile above which "activated"

    def __init__(self, decomp_df: pd.DataFrame):
        self.df = decomp_df
        self.sectors = sorted(decomp_df["sector"].unique())
        self._results: Optional[Dict] = None

    def run_all(self) -> Dict:
        results = {}
        for sec in self.sectors:
            sec_data = self.df[self.df["sector"] == sec].sort_values("date")
            Lh = sec_data["L_household"].values
            Lm = sec_data["L_market"].values
            S  = sec_data["S"].values
            E  = sec_data["E"].values
            results[sec] = self._analyze_sector(sec, Lh, Lm, S, E)

        # Pooled analysis
        Lh_pool = self.df["L_household"].values
        Lm_pool = self.df["L_market"].values
        S_pool  = self.df["S"].values
        E_pool  = self.df["E"].values
        results["_pooled"] = self._analyze_sector("_pooled", Lh_pool, Lm_pool, S_pool, E_pool)

        self._results = results
        return results

    def _analyze_sector(
        self, sector: str,
        Lh: np.ndarray, Lm: np.ndarray,
        S:  np.ndarray, E:  np.ndarray,
    ) -> Dict:
        T = len(Lh)
        out = {"sector": sector, "T": T}

        # ── A: Distribution shape ────────────────────────────────────────
        out["mean"]     = float(np.mean(Lh))
        out["std"]      = float(np.std(Lh))
        out["skew"]     = float(skew(Lh))
        out["kurtosis"] = float(kurtosis(Lh))   # excess kurtosis
        out["min"]      = float(Lh.min())
        out["max"]      = float(Lh.max())
        out["nonzero_pct"] = float(np.mean(Lh > self.NEAR_ZERO_THRESH) * 100)

        # Normality tests (only meaningful for n >= 8)
        if T >= 8:
            sw_stat, sw_p = shapiro(Lh)
            jb_stat, jb_p = jarque_bera(Lh)
            out["shapiro_stat"] = float(sw_stat)
            out["shapiro_p"]    = float(sw_p)
            out["jarque_bera_stat"] = float(jb_stat)
            out["jarque_bera_p"]    = float(jb_p)
            out["normality_rejected_sw"]  = bool(sw_p < 0.05)
            out["normality_rejected_jb"]  = bool(jb_p < 0.05)
            out["normality_rejected_both"]= bool(sw_p < 0.05 and jb_p < 0.05)

            # Anderson-Darling
            ad = anderson(Lh, dist="norm")
            # 5% significance level = index 2
            out["anderson_stat"]     = float(ad.statistic)
            out["anderson_cv_5pct"]  = float(ad.critical_values[2])
            out["anderson_rejected_5pct"] = bool(ad.statistic > ad.critical_values[2])
        else:
            for k in ["shapiro_stat","shapiro_p","jarque_bera_stat","jarque_bera_p",
                      "normality_rejected_sw","normality_rejected_jb","normality_rejected_both",
                      "anderson_stat","anderson_cv_5pct","anderson_rejected_5pct"]:
                out[k] = None

        # ── B: Rolling distribution diagnostics ─────────────────────────
        win = max(6, T // 5)
        roll_stats = self._rolling_stats(Lh, window=win)
        out["rolling"] = roll_stats
        out["rolling_window"] = win
        # Convergence assessment: does rolling skew/kurtosis stabilize?
        if len(roll_stats["skew"]) > 3:
            skew_range = max(roll_stats["skew"]) - min(roll_stats["skew"])
            kurt_range = max(roll_stats["kurtosis"]) - min(roll_stats["kurtosis"])
            out["rolling_skew_range"]  = float(skew_range)
            out["rolling_kurt_range"]  = float(kurt_range)
            out["distributional_stable"] = bool(skew_range < 1.0 and kurt_range < 3.0)
        else:
            out["rolling_skew_range"] = None
            out["rolling_kurt_range"] = None
            out["distributional_stable"] = None

        # ── C: Mixture / breakpoint behavior ─────────────────────────────
        out["mixture"]    = self._mixture_test(Lh)
        out["changepoint"]= self._changepoint_detection(Lh)

        # Zero-inflation characterization
        near_zero_mass = float(np.mean(Lh <= self.NEAR_ZERO_THRESH))
        activ_thresh   = float(np.percentile(Lh[Lh > self.NEAR_ZERO_THRESH], 75)
                               if Lh[Lh > self.NEAR_ZERO_THRESH].size > 0 else 0)
        activated_mass = float(np.mean(Lh > activ_thresh))
        out["near_zero_mass_pct"] = near_zero_mass * 100
        out["activated_mass_pct"] = activated_mass * 100
        out["activation_threshold_value"] = activ_thresh
        out["zero_inflated"] = bool(near_zero_mass > 0.40)   # >40% near zero = zero-inflated

        # ── D: Exit-point characterization ──────────────────────────────
        out["exit_point"] = self._exit_point_analysis(Lh, Lm, S, T)

        # ── E: Threshold analysis ────────────────────────────────────────
        out["threshold"] = self._threshold_analysis(Lh, S, E, Lm)

        return out

    # ── Rolling stats ─────────────────────────────────────────────────────

    def _rolling_stats(self, x: np.ndarray, window: int) -> Dict:
        T   = len(x)
        out = {"mean": [], "std": [], "skew": [], "kurtosis": [], "sw_p": []}
        for i in range(window, T + 1):
            seg = x[i - window: i]
            out["mean"].append(float(np.mean(seg)))
            out["std"].append(float(np.std(seg)))
            out["skew"].append(float(skew(seg)))
            out["kurtosis"].append(float(kurtosis(seg)))
            if len(seg) >= 8:
                _, p = shapiro(seg)
                out["sw_p"].append(float(p))
            else:
                out["sw_p"].append(None)
        return out

    # ── Two-component Gaussian mixture ───────────────────────────────────

    def _mixture_test(self, x: np.ndarray, n_init: int = 20) -> Dict:
        """
        Fit single Gaussian vs two-component Gaussian mixture.
        Compare log-likelihoods; report BIC improvement.
        """
        x_pos = x[x > self.NEAR_ZERO_THRESH]
        if len(x_pos) < 8:
            return {"bic_improvement": None, "mixture_preferred": False,
                    "component_means": None, "component_stds": None}

        # Single Gaussian log-likelihood
        mu1, sd1 = scipy_norm.fit(x_pos)
        ll_single = np.sum(scipy_norm.logpdf(x_pos, mu1, sd1))
        bic_single = -2 * ll_single + 2 * np.log(len(x_pos))

        # Two-component mixture (EM-style grid search)
        best_ll = -np.inf
        best_params = None
        rng = np.random.default_rng(42)
        for _ in range(n_init):
            split = rng.uniform(0.2, 0.8)
            # Two Gaussians fitted on random partition
            mask  = x_pos < np.quantile(x_pos, split)
            c1    = x_pos[mask]
            c2    = x_pos[~mask]
            if len(c1) < 3 or len(c2) < 3:
                continue
            mu_1, sd_1 = np.mean(c1), max(np.std(c1), 1e-6)
            mu_2, sd_2 = np.mean(c2), max(np.std(c2), 1e-6)
            pi_1 = len(c1) / len(x_pos)
            pi_2 = 1 - pi_1
            ll = np.sum(np.log(
                pi_1 * scipy_norm.pdf(x_pos, mu_1, sd_1) +
                pi_2 * scipy_norm.pdf(x_pos, mu_2, sd_2) + 1e-12
            ))
            if ll > best_ll:
                best_ll = ll
                best_params = (mu_1, sd_1, mu_2, sd_2, pi_1)

        if best_params is None:
            return {"bic_improvement": None, "mixture_preferred": False,
                    "component_means": None, "component_stds": None}

        bic_mixture = -2 * best_ll + 5 * np.log(len(x_pos))  # 5 free params
        bic_improvement = bic_single - bic_mixture
        mu_1, sd_1, mu_2, sd_2, pi_1 = best_params
        return {
            "bic_improvement": float(bic_improvement),
            "mixture_preferred": bool(bic_improvement > 0),
            "component_means": [float(mu_1), float(mu_2)],
            "component_stds":  [float(sd_1), float(sd_2)],
            "component_weights": [float(pi_1), float(1 - pi_1)],
            "bic_single":   float(bic_single),
            "bic_mixture":  float(bic_mixture),
        }

    # ── Change-point detection ───────────────────────────────────────────

    def _changepoint_detection(self, x: np.ndarray) -> Dict:
        """
        CUSUM-based change-point detection.
        Returns detected break indices and mean-shift magnitude.
        """
        T = len(x)
        if T < 8:
            return {"n_breakpoints": 0, "break_indices": [], "mean_shift": []}

        mu = np.mean(x); sd = np.std(x) + 1e-8
        cusum = np.cumsum((x - mu) / sd)
        cusum_range = cusum.max() - cusum.min()

        # Detect breakpoints where CUSUM reverses direction significantly
        breaks = []
        for t in range(2, T - 2):
            left_mean  = np.mean(x[:t])
            right_mean = np.mean(x[t:])
            shift      = abs(right_mean - left_mean)
            if shift > 0.5 * sd:
                # Local extremum in CUSUM
                if (cusum[t] > cusum[t-1] and cusum[t] > cusum[t+1]) or \
                   (cusum[t] < cusum[t-1] and cusum[t] < cusum[t+1]):
                    breaks.append({
                        "index":      int(t),
                        "mean_before": float(left_mean),
                        "mean_after":  float(right_mean),
                        "shift":       float(shift),
                    })

        # Merge nearby breakpoints (within 2 periods)
        merged = []
        for b in sorted(breaks, key=lambda x: x["shift"], reverse=True)[:4]:
            if not merged or abs(b["index"] - merged[-1]["index"]) > 2:
                merged.append(b)

        return {
            "n_breakpoints": len(merged),
            "break_indices": [b["index"] for b in merged],
            "mean_shift":    [b["shift"] for b in merged],
            "cusum_range":   float(cusum_range),
            "details":       merged,
        }

    # ── Exit-point analysis ──────────────────────────────────────────────

    def _exit_point_analysis(
        self, Lh: np.ndarray, Lm: np.ndarray,
        S: np.ndarray, T: int,
    ) -> Dict:
        """
        Characterize whether Lh behaves as an exit channel:
          - clustering near zero with episodic excursions
          - abrupt activation after stress thresholds
          - persistence after activation
          - lagged transition into L_market
        """
        thresh = self.NEAR_ZERO_THRESH

        # Episode detection: contiguous runs above threshold
        in_episode = Lh > thresh
        episodes   = []
        ep_start   = None
        for t in range(T):
            if in_episode[t] and ep_start is None:
                ep_start = t
            elif not in_episode[t] and ep_start is not None:
                episodes.append({"start": ep_start, "end": t-1,
                                  "length": t - ep_start,
                                  "peak": float(Lh[ep_start:t].max())})
                ep_start = None
        if ep_start is not None:
            episodes.append({"start": ep_start, "end": T-1,
                              "length": T - ep_start,
                              "peak": float(Lh[ep_start:].max())})

        ep_lengths = [e["length"] for e in episodes]

        # Lagged transition: does high Lh precede rising Lm?
        lag_corrs = {}
        for lag in [0, 1, 2, 3]:
            if lag == 0:
                r = np.corrcoef(Lh, Lm)[0, 1]
            else:
                r = np.corrcoef(Lh[:-lag], Lm[lag:])[0, 1]
            lag_corrs[lag] = float(r)

        # Abruptness: compare median rise speed of Lh episodes vs Lm episodes
        Lh_rises = np.diff(np.clip(Lh, 0, 1))
        abrupt_activations = int(np.sum(
            (Lh[1:] > thresh) & (Lh[:-1] <= thresh)
        ))

        return {
            "n_episodes":         len(episodes),
            "ep_lengths_mean":    float(np.mean(ep_lengths)) if ep_lengths else 0,
            "ep_lengths_max":     int(max(ep_lengths)) if ep_lengths else 0,
            "ep_lengths_median":  float(np.median(ep_lengths)) if ep_lengths else 0,
            "abrupt_activations": abrupt_activations,
            "reentry_to_zero":    int(np.sum((Lh[1:] <= thresh) & (Lh[:-1] > thresh))),
            "lag_corr_Lh_to_Lm": lag_corrs,
            "peak_lag": max(lag_corrs, key=lambda k: lag_corrs[k]),
            "episodes": episodes[:6],  # store first 6 for reference
        }

    # ── Threshold analysis ───────────────────────────────────────────────

    def _threshold_analysis(
        self,
        Lh: np.ndarray, S: np.ndarray,
        E: np.ndarray, Lm: np.ndarray,
    ) -> Dict:
        """
        Estimate stress/leakage threshold above which Lh activates
        disproportionately. Test against S, E, Lm, C.
        """
        results = {}
        predictors = {"S": S, "E": E, "L_market": Lm}

        for name, pred in predictors.items():
            # Scan percentile thresholds; find inflection in Lh rate
            pcts     = np.arange(20, 85, 5)
            threshs  = np.percentile(pred, pcts)
            above_rates = []
            for thr in threshs:
                mask = pred >= thr
                if mask.sum() < 3:
                    above_rates.append(np.nan)
                    continue
                above_rates.append(float(np.mean(Lh[mask] > self.NEAR_ZERO_THRESH)))
            below_rates = []
            for thr in threshs:
                mask = pred < thr
                if mask.sum() < 3:
                    below_rates.append(np.nan)
                    continue
                below_rates.append(float(np.mean(Lh[mask] > self.NEAR_ZERO_THRESH)))

            above_rates = np.array(above_rates)
            below_rates = np.array(below_rates)
            valid        = ~np.isnan(above_rates) & ~np.isnan(below_rates)

            if valid.sum() > 2:
                contrasts = above_rates[valid] - below_rates[valid]
                best_idx  = int(np.argmax(contrasts))
                best_pct  = float(pcts[valid][best_idx])
                best_thr  = float(threshs[valid][best_idx])
                best_contrast = float(contrasts[best_idx])
            else:
                best_pct = best_thr = best_contrast = None

            results[name] = {
                "candidate_threshold":  best_thr,
                "threshold_percentile": best_pct,
                "activation_contrast":  best_contrast,  # above-rate - below-rate
                "thresholds_tested":    threshs[valid].tolist() if valid.sum() else [],
                "above_rates":          above_rates[valid].tolist() if valid.sum() else [],
                "below_rates":          below_rates[valid].tolist() if valid.sum() else [],
            }

        return results

    # ── Summary table ────────────────────────────────────────────────────

    def summary_table(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Run run_all() first.")
        rows = []
        for sec, r in self._results.items():
            row = {
                "sector":              sec,
                "mean":                r["mean"],
                "std":                 r["std"],
                "skew":                r["skew"],
                "excess_kurtosis":     r["kurtosis"],
                "nonzero_pct":         r["nonzero_pct"],
                "shapiro_p":           r.get("shapiro_p"),
                "jarque_bera_p":       r.get("jarque_bera_p"),
                "normality_rejected":  r.get("normality_rejected_both"),
                "anderson_rejected":   r.get("anderson_rejected_5pct"),
                "near_zero_mass_pct":  r.get("near_zero_mass_pct"),
                "zero_inflated":       r.get("zero_inflated"),
                "mixture_preferred":   r.get("mixture", {}).get("mixture_preferred"),
                "bic_improvement":     r.get("mixture", {}).get("bic_improvement"),
                "n_changepoints":      r.get("changepoint", {}).get("n_breakpoints"),
                "cusum_range":         r.get("changepoint", {}).get("cusum_range"),
                "n_episodes":          r.get("exit_point", {}).get("n_episodes"),
                "ep_length_mean":      r.get("exit_point", {}).get("ep_lengths_mean"),
                "abrupt_activations":  r.get("exit_point", {}).get("abrupt_activations"),
                "reentry_to_zero":     r.get("exit_point", {}).get("reentry_to_zero"),
                "distributional_stable": r.get("distributional_stable"),
                "rolling_skew_range":  r.get("rolling_skew_range"),
                "s_threshold":         r.get("threshold", {}).get("S", {}).get("candidate_threshold"),
                "s_contrast":          r.get("threshold", {}).get("S", {}).get("activation_contrast"),
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("sector")
