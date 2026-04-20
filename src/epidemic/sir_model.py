"""
epidemic/sir_model.py
──────────────────────
Standard time-varying SIR epidemic engine.

Provides:
  - SIRModel class: ODE + discrete-step solvers
  - ObservedPressureAdapter: wraps observed case data as epidemic pressure
  - epidemic_pressure_from_sir(): extract normalized I_t pressure series
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger("epidemic")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


# ─── SIR Model ────────────────────────────────────────────────────────────────

@dataclass
class SIRParams:
    """Parameters for time-varying SIR model."""
    population: float = 331_000_000
    beta_init: float = 0.25          # initial transmission rate
    gamma_init: float = 0.07         # initial recovery rate
    beta_min: float = 0.05
    beta_max: float = 0.80
    smoothing_window: int = 14       # days for Gaussian smoothing
    I0_fraction: float = 1e-5        # initial infected fraction


class SIRModel:
    """
    Time-varying SIR model with multiple solver backends.

    The epidemic layer acts as a temporal forcing process. The key output
    is a normalized epidemic pressure time series P_t ∈ [0,1] representing
    the burden imposed on market sectors.

    Parameters
    ----------
    params    : SIRParams instance
    beta_path : Optional external beta_t time series (pd.Series with DatetimeIndex)
    gamma_path: Optional external gamma_t time series
    """

    def __init__(
        self,
        params: Optional[SIRParams] = None,
        beta_path: Optional[pd.Series] = None,
        gamma_path: Optional[pd.Series] = None,
    ):
        self.params = params or SIRParams()
        self._beta_path = beta_path
        self._gamma_path = gamma_path
        self._result: Optional[pd.DataFrame] = None

    def run_discrete(
        self,
        dates: pd.DatetimeIndex,
        beta_series: Optional[pd.Series] = None,
        gamma_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Discrete-step SIR.

        Parameters
        ----------
        dates        : DatetimeIndex for the simulation
        beta_series  : Time-varying beta_t aligned to dates (optional)
        gamma_series : Time-varying gamma_t aligned to dates (optional)

        Returns
        -------
        DataFrame with columns [S, I, R, beta_t, gamma_t, pressure]
        """
        N = self.params.population
        T = len(dates)
        p = self.params

        # Build beta and gamma arrays
        if beta_series is not None:
            beta = beta_series.reindex(dates).ffill().bfill().values
        elif self._beta_path is not None:
            beta = self._beta_path.reindex(dates).ffill().bfill().values
        else:
            beta = np.full(T, p.beta_init)

        if gamma_series is not None:
            gamma = gamma_series.reindex(dates).ffill().bfill().values
        elif self._gamma_path is not None:
            gamma = self._gamma_path.reindex(dates).ffill().bfill().values
        else:
            gamma = np.full(T, p.gamma_init)

        beta = np.clip(beta, p.beta_min, p.beta_max)

        # Initial conditions
        S = np.zeros(T)
        I = np.zeros(T)
        R = np.zeros(T)
        S[0] = N * (1 - p.I0_fraction)
        I[0] = N * p.I0_fraction
        R[0] = 0.0

        for t in range(1, T):
            new_infected = beta[t-1] * S[t-1] * I[t-1] / N
            new_recovered = gamma[t-1] * I[t-1]
            S[t] = S[t-1] - new_infected
            I[t] = I[t-1] + new_infected - new_recovered
            R[t] = R[t-1] + new_recovered
            # Clamp
            S[t] = max(S[t], 0)
            I[t] = max(I[t], 0)
            R[t] = max(R[t], 0)

        df = pd.DataFrame(
            {"S": S, "I": I, "R": R, "beta_t": beta, "gamma_t": gamma},
            index=dates,
        )
        df["pressure"] = _normalize_series(df["I"])
        df["R0_t"] = df["beta_t"] / df["gamma_t"]
        self._result = df
        logger.info(
            f"[SIR-Discrete] Run complete: peak I={I.max():.0f}, "
            f"final R={R[-1]/N:.3f}"
        )
        return df

    def run_ode(
        self,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        ODE-based SIR via scipy.integrate.solve_ivp (constant beta/gamma).
        For time-varying ODE, use run_discrete with small dt.
        """
        N = self.params.population
        b = beta or self.params.beta_init
        g = gamma or self.params.gamma_init
        I0 = N * self.params.I0_fraction
        S0 = N - I0

        def rhs(t, y):
            S_, I_, R_ = y
            dS = -b * S_ * I_ / N
            dI = b * S_ * I_ / N - g * I_
            dR = g * I_
            return [dS, dI, dR]

        sol = solve_ivp(rhs, t_span, [S0, I0, 0.0], t_eval=t_eval, method="RK45")
        df = pd.DataFrame(
            {"S": sol.y[0], "I": sol.y[1], "R": sol.y[2]},
            index=t_eval,
        )
        df["beta_t"] = b
        df["gamma_t"] = g
        df["pressure"] = _normalize_series(df["I"])
        df["R0_t"] = b / g
        self._result = df
        return df

    def get_pressure(self) -> Optional[pd.Series]:
        """Return the epidemic pressure series if a run has been executed."""
        if self._result is None:
            return None
        return self._result["pressure"]


# ─── Observed-Pressure Adapter ────────────────────────────────────────────────

class ObservedPressureAdapter:
    """
    Wraps observed COVID case/hospitalization data as a normalized
    epidemic pressure series, bypassing the SIR engine.

    This is the primary mode when mode='observed' in config.
    """

    def __init__(
        self,
        covid_df: pd.DataFrame,
        pressure_col: str = "new_cases_smoothed",
        smoothing_sigma: float = 7.0,
        normalize: bool = True,
        lag_days: int = 14,
    ):
        self.covid_df = covid_df
        self.pressure_col = pressure_col
        self.smoothing_sigma = smoothing_sigma
        self.normalize = normalize
        self.lag_days = lag_days

    def get_daily_pressure(self) -> pd.Series:
        """Return smoothed, normalized daily epidemic pressure."""
        col = self.pressure_col
        if col not in self.covid_df.columns:
            col = self.covid_df.columns[0]
            logger.warning(f"Column '{self.pressure_col}' not found. Using '{col}'.")

        raw = self.covid_df[col].fillna(0).values
        smoothed = gaussian_filter1d(raw.astype(float), sigma=self.smoothing_sigma)
        s = pd.Series(smoothed, index=self.covid_df.index, name="epidemic_pressure_daily")
        s = s.shift(self.lag_days)  # apply market lag
        if self.normalize:
            s = _normalize_series(s)
        return s

    def get_monthly_pressure(self) -> pd.Series:
        """Resample daily pressure to monthly mean."""
        daily = self.get_daily_pressure()
        return daily.resample("ME").mean().rename("epidemic_pressure")

    def build_sir_output(self) -> pd.DataFrame:
        """
        Build a pseudo-SIR DataFrame from observed data.
        S,I,R are approximate reconstructions for visualization.
        """
        df = self.covid_df.copy()
        N = 331_000_000
        pressure = self.get_daily_pressure()

        # Approximate I from cases
        I_approx = df.get(
            "new_cases_smoothed",
            pd.Series(0, index=df.index)
        ).cumsum().diff(7).clip(0) * 7
        I_approx = I_approx.fillna(0)

        R_approx = I_approx.cumsum() * 0.14  # rough
        S_approx = N - I_approx - R_approx
        S_approx = S_approx.clip(0, N)

        out = pd.DataFrame(
            {
                "S": S_approx,
                "I": I_approx,
                "R": R_approx.clip(0, N),
                "pressure": pressure,
                "new_cases_smoothed": df.get("new_cases_smoothed", pd.Series(0, index=df.index)),
                "new_deaths_smoothed": df.get("new_deaths_smoothed", pd.Series(0, index=df.index)),
                "hosp_patients":       df.get("hosp_patients", pd.Series(0, index=df.index)),
                "stringency_index":    df.get("stringency_index", pd.Series(50.0, index=df.index)),
            },
            index=df.index,
        )
        logger.info(
            f"[ObservedPressure] Built SIR-like output: "
            f"{out.index.min()} → {out.index.max()}"
        )
        return out


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0,1]."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return ((s - mn) / (mx - mn)).clip(0, 1)


def estimate_beta_from_cases(
    cases: pd.Series,
    N: float = 331_000_000,
    gamma: float = 0.07,
    smoothing: int = 7,
) -> pd.Series:
    """
    Heuristic daily beta estimate from observed case growth.
    beta_t ≈ (growth_rate + gamma) / I_fraction
    This is an approximate method; proper MLE would use SEIR fitting.
    """
    I_approx = cases.rolling(smoothing, min_periods=1).mean().fillna(1).clip(1)
    S_approx = (N - I_approx.cumsum()).clip(N * 0.01, N)
    growth = I_approx.pct_change().fillna(0).clip(-0.5, 0.5)
    beta = (growth + gamma) * N / S_approx
    return beta.clip(0.01, 1.0)
