"""
analytics/blob_analytics.py
────────────────────────────
Sector blob construction and delta analytics.

Concepts:
  Blob size        = A_{i,t}  (realized activity level, OI or proxy)
  Blob composition = IO-weighted internal structure from BEA prior
  Intra-blob delta = period-over-period change in internal slice shares
  Inter-blob delta = divergence/convergence between sector blobs over time

The blobs do not represent a claim of directly observing inter-sector
transaction flows. They represent inferred deformation of transmission
structure from joint activity and IO-anchored composition.

Sectors are flagged by observation status:
  "oi_direct"  — OI Affinity spend (solid border in visualization)
  "oi_partial" — OI partial coverage (dashed border)
  "fred_proxy" — FRED CPI/PPI proxy (dotted border)
  "stress_proxy" — 1 - stress fallback (light border, lowest confidence)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("blob_analytics")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


class BlobAnalytics:
    """
    Constructs sector blobs and computes intra/inter deltas over time.

    Parameters
    ----------
    activity_panel : pd.DataFrame — A_{i,t} monthly, columns=sectors
    W0_matrix      : pd.DataFrame — BEA IO prior (sectors x sectors) or None
    obs_status_df  : pd.DataFrame — observation status per sector
    graph_edges    : list of (i,j) with structural weights
    """

    def __init__(
        self,
        activity_panel: pd.DataFrame,
        W0_matrix: Optional[pd.DataFrame],
        obs_status_df: pd.DataFrame,
        graph_edges: List[Tuple[str, str, float]],
    ):
        self.activity  = activity_panel
        self.W0        = W0_matrix
        self.obs_status= obs_status_df
        self.edges     = graph_edges
        self.sectors   = list(activity_panel.columns)
        self.dates     = activity_panel.index
        self.T         = len(self.dates)

        self._blob_sizes: Optional[pd.DataFrame]       = None
        self._blob_composition: Optional[Dict]         = None
        self._intra_deltas: Optional[pd.DataFrame]     = None
        self._inter_deltas: Optional[pd.DataFrame]     = None
        self._delta_flags: Optional[pd.DataFrame]      = None

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Compute all blob metrics. Returns dict of DataFrames."""
        self._build_blob_sizes()
        self._build_blob_composition()
        self._build_intra_deltas()
        self._build_inter_deltas()
        self._build_delta_flags()
        return {
            "blob_sizes":       self._blob_sizes,
            "intra_deltas":     self._intra_deltas,
            "inter_deltas":     self._inter_deltas,
            "delta_flags":      self._delta_flags,
            "obs_status":       self.obs_status,
        }

    def _build_blob_sizes(self):
        """Blob size = A_{i,t}. Simple pass-through with NaN fill."""
        self._blob_sizes = self.activity.copy().fillna(0.5)
        logger.info(f"[Blob] Sizes computed: {self._blob_sizes.shape}")

    def _build_blob_composition(self):
        """
        Internal blob composition: for each sector i at time t, what fraction
        of its IO structure comes from each upstream source?

        If BEA IO is available:
          composition_{i,k} = W0_{k,i} / sum_k W0_{k,i}   (column-normalised)
          This gives the structural share of each input source in sector i's basket.

        If BEA IO not available:
          composition_{i,k} = 1/n (uniform across graph predecessors)
        """
        comp = {}
        for sec in self.sectors:
            if self.W0 is not None and sec in self.W0.columns:
                col = self.W0[sec]  # inputs TO sec from each source
                total = col.sum()
                if total > 0:
                    shares = col / total
                else:
                    shares = pd.Series(1.0/len(self.sectors), index=self.sectors)
            else:
                # Uniform over structural predecessors
                preds = [i for (i,j,_) in self.edges if j == sec]
                if preds:
                    shares = pd.Series(1.0/len(preds), index=preds)
                    shares = shares.reindex(self.sectors).fillna(0)
                else:
                    shares = pd.Series(1.0/len(self.sectors), index=self.sectors)
            comp[sec] = shares
        self._blob_composition = comp
        logger.info(f"[Blob] Composition built for {len(comp)} sectors")

    def _build_intra_deltas(self):
        """
        Intra-blob delta: how does the effective internal composition change
        each period due to differential activity changes?

        effective_slice_{i,k,t} = composition_{i,k} * A_{k,t}
        slice_share_{i,k,t} = effective_slice_{i,k,t} / sum_k effective_slice_{i,k,t}
        intra_delta_{i,k,t} = slice_share_{i,k,t} - slice_share_{i,k,t-1}
        """
        if self._blob_composition is None:
            return

        rows = []
        for sec in self.sectors:
            comp = self._blob_composition[sec]
            for t_idx, date in enumerate(self.dates):
                # Effective slice = structural share * source activity
                eff_slices = {}
                for src in self.sectors:
                    c_share = float(comp.get(src, 0.0))
                    a_src   = float(self._blob_sizes.loc[date, src]) if src in self._blob_sizes.columns else 0.5
                    eff_slices[src] = c_share * a_src

                total = sum(eff_slices.values()) + 1e-8
                share = {src: v/total for src,v in eff_slices.items()}

                if t_idx > 0:
                    prev_date = self.dates[t_idx-1]
                    # recompute previous shares
                    prev_eff = {}
                    for src in self.sectors:
                        c_share = float(comp.get(src, 0.0))
                        a_src   = float(self._blob_sizes.loc[prev_date, src]) if src in self._blob_sizes.columns else 0.5
                        prev_eff[src] = c_share * a_src
                    prev_total = sum(prev_eff.values()) + 1e-8
                    prev_share = {src: v/prev_total for src,v in prev_eff.items()}

                    for src in self.sectors:
                        delta = share[src] - prev_share[src]
                        if abs(delta) > 0.001:  # only store meaningful deltas
                            rows.append({
                                "date": date, "sector": sec, "source": src,
                                "slice_share": share[src],
                                "prev_share":  prev_share[src],
                                "intra_delta": delta,
                                "abs_delta":   abs(delta),
                            })

        self._intra_deltas = pd.DataFrame(rows)
        logger.info(f"[Blob] Intra-deltas: {len(self._intra_deltas)} records")

    def _build_inter_deltas(self):
        """
        Inter-blob delta: divergence/convergence between sector blobs.

        distance_{ij,t} = |A_{i,t} - A_{j,t}| weighted by W0_{ij} (IO coupling)
        inter_delta_{ij,t} = distance_{ij,t} - distance_{ij,t-1}

        Positive inter_delta = diverging (weakening coupling expression)
        Negative inter_delta = converging (strengthening coupling expression)
        """
        rows = []
        for t_idx in range(1, self.T):
            date      = self.dates[t_idx]
            prev_date = self.dates[t_idx-1]

            for (i, j, w0) in self.edges:
                if i not in self._blob_sizes.columns or j not in self._blob_sizes.columns:
                    continue
                A_i  = float(self._blob_sizes.loc[date, i])
                A_j  = float(self._blob_sizes.loc[date, j])
                pA_i = float(self._blob_sizes.loc[prev_date, i])
                pA_j = float(self._blob_sizes.loc[prev_date, j])

                dist      = abs(A_i - A_j) * w0
                prev_dist = abs(pA_i - pA_j) * w0
                delta     = dist - prev_dist

                rows.append({
                    "date":        date,
                    "source":      i,
                    "target":      j,
                    "W0":          w0,
                    "A_source":    A_i,
                    "A_target":    A_j,
                    "distance":    dist,
                    "inter_delta": delta,
                    "diverging":   delta > 0,
                    "converging":  delta < 0,
                })

        self._inter_deltas = pd.DataFrame(rows)
        logger.info(f"[Blob] Inter-deltas: {len(self._inter_deltas)} records")

    def _build_delta_flags(self):
        """
        Flag significant intra and inter events each period.
        Returns a ranked list per time period for the dashboard panel.
        """
        rows = []

        # Top intra-delta events
        if self._intra_deltas is not None and len(self._intra_deltas):
            threshold = self._intra_deltas["abs_delta"].quantile(0.80)
            top_intra = self._intra_deltas[self._intra_deltas["abs_delta"] >= threshold].copy()
            top_intra["event_type"] = top_intra["intra_delta"].apply(
                lambda d: "intra_gain" if d > 0 else "intra_loss"
            )
            top_intra["description"] = top_intra.apply(
                lambda r: f"{r['sector']} ← {r['source']}: "
                          f"slice {'grew' if r['intra_delta']>0 else 'shrank'} "
                          f"{r['intra_delta']:+.3f}",
                axis=1
            )
            rows.append(top_intra[["date","sector","event_type","abs_delta","description"]])

        # Top inter-delta events
        if self._inter_deltas is not None and len(self._inter_deltas):
            threshold = self._inter_deltas["inter_delta"].abs().quantile(0.80)
            top_inter = self._inter_deltas[
                self._inter_deltas["inter_delta"].abs() >= threshold
            ].copy()
            top_inter["sector"]     = top_inter["source"]
            top_inter["abs_delta"]  = top_inter["inter_delta"].abs()
            top_inter["event_type"] = top_inter["diverging"].apply(
                lambda d: "inter_diverging" if d else "inter_converging"
            )
            top_inter["description"] = top_inter.apply(
                lambda r: f"{r['source']} ↔ {r['target']}: "
                          f"{'diverging' if r['diverging'] else 'converging'} "
                          f"(Δ{r['inter_delta']:+.4f}, W0={r['W0']:.2f})",
                axis=1
            )
            rows.append(top_inter[["date","sector","event_type","abs_delta","description"]])

        if rows:
            self._delta_flags = pd.concat(rows, ignore_index=True).sort_values(
                ["date","abs_delta"], ascending=[True, False]
            )
        else:
            self._delta_flags = pd.DataFrame(
                columns=["date","sector","event_type","abs_delta","description"]
            )

        logger.info(f"[Blob] Delta flags: {len(self._delta_flags)} events")

    # ── Accessors ─────────────────────────────────────────────────────────────

    def top_flags_for_date(self, date: pd.Timestamp, n: int = 10) -> pd.DataFrame:
        """Return top-N flagged events for a specific date."""
        if self._delta_flags is None or self._delta_flags.empty:
            return pd.DataFrame()
        mask = self._delta_flags["date"] == date
        return self._delta_flags[mask].head(n)

    def blob_state_at(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Return blob state (size, top composition slices, obs_status)
        for all sectors at a given date.
        """
        rows = []
        for sec in self.sectors:
            size = float(self._blob_sizes.loc[date, sec]) if date in self.dates else 0.5
            comp = self._blob_composition.get(sec, {})
            # Top 3 composition sources
            top_srcs = sorted(comp.items(), key=lambda x: x[1], reverse=True)[:3]
            obs = self.obs_status.loc[sec, "obs_status"] if sec in self.obs_status.index else "stress_proxy"
            rows.append({
                "sector":      sec,
                "size":        size,
                "obs_status":  obs,
                "top_src_1":   top_srcs[0][0] if len(top_srcs) > 0 else "",
                "top_src_2":   top_srcs[1][0] if len(top_srcs) > 1 else "",
                "top_src_3":   top_srcs[2][0] if len(top_srcs) > 2 else "",
                "top_share_1": top_srcs[0][1] if len(top_srcs) > 0 else 0,
            })
        return pd.DataFrame(rows).set_index("sector")

    def composition_matrix_at(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Effective composition matrix at date:
        comp_{i,k} * A_{k,t} / sum_k (normalized).
        """
        result = pd.DataFrame(index=self.sectors, columns=self.sectors, dtype=float)
        for sec in self.sectors:
            comp = self._blob_composition.get(sec, {})
            for src in self.sectors:
                c  = float(comp.get(src, 0.0))
                a  = float(self._blob_sizes.loc[date, src]) if (
                    date in self.dates and src in self._blob_sizes.columns) else 0.5
                result.loc[sec, src] = c * a
            row_sum = result.loc[sec].sum() + 1e-8
            result.loc[sec] = result.loc[sec] / row_sum
        return result.fillna(0)
