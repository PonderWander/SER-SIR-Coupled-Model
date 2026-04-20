"""
graph/propagation.py
─────────────────────
REFACTORED: explicit edge-level connectivity state C_{ij,t}.

Architecture:
  W^0_{ij}  : structural prior (sector_graph.yaml, fixed)
  C_{ij,t}  : dynamic connectivity state (evolves each period)
  W_{ij,t}  : effective edge weight = W^0_{ij} * C_{ij,t}

Three C_{ij,t} realizations (graph.connectivity.mode):

  persistence_only
    C_{t} = clip((1-rho)*C_{t-1} + rho*G)
    G = f(elasticity, leaked stress) — capacity signal

  stress_elasticity_activity  [default]
    G = sqrt(E_i*E_j) * sqrt(A_i*A_j) * (1 - 0.5*|Phi_i-Phi_j|) - mean_leak
    Connectivity tracks joint capacity and discrepancy alignment

  delta_deformation
    divergence = |dA_i - dA_j| / (|dA_i|+|dA_j|+eps)
    G = exp(-kappa*divergence) * (1 - mean_leak)
    Connectivity deforms when activity trajectories diverge
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger("graph_propagation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


class SectorGraph:
    def __init__(self, graph_config: Dict):
        self.config  = graph_config
        self.sectors: List[str] = graph_config.get("sectors", [])
        self.G: nx.DiGraph = self._build_structural_graph()

    def _build_structural_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(self.sectors)
        for edge in self.config.get("edges", []):
            src = edge["source"]; tgt = edge["target"]
            w   = float(edge.get("weight", 0.3))
            G.add_edge(src, tgt, weight=w, structural_weight=w,
                       note=edge.get("note", ""))
        logger.info(f"[SectorGraph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def adjacency_matrix(self) -> pd.DataFrame:
        return nx.to_pandas_adjacency(self.G, weight="weight", nodelist=self.sectors)

    def structural_matrix(self) -> pd.DataFrame:
        return nx.to_pandas_adjacency(self.G, weight="structural_weight", nodelist=self.sectors)

    def get_successors(self, node: str): return [(v, self.G[node][v]["weight"]) for v in self.G.successors(node)]
    def get_predecessors(self, node: str): return [(u, self.G[u][node]["weight"]) for u in self.G.predecessors(node)]


class ConnectivityState:
    """Manages C_{ij,t} for every edge."""
    MODES = ("persistence_only", "stress_elasticity_activity", "delta_deformation")

    def __init__(self, edges, W0, conn_cfg):
        self.edges   = edges
        self.W0      = W0
        self.mode    = conn_cfg.get("mode", "stress_elasticity_activity")
        self.rho     = float(conn_cfg.get("rho_c", 0.25))
        self.C_init  = float(conn_cfg.get("C_init", 1.0))
        self.C_floor = float(conn_cfg.get("C_floor", 0.05))
        self.C_ceil  = float(conn_cfg.get("C_ceil", 1.20))
        self.kappa   = float(conn_cfg.get("kappa", 2.0))
        self.warn_thr= float(conn_cfg.get("flatness_warn_threshold", 0.02))
        assert self.mode in self.MODES, f"Unknown mode '{self.mode}'"
        self.C       = {e: self.C_init for e in edges}
        self.history = {e: [] for e in edges}

    def step(self, t, sector_state):
        for (i, j) in self.edges:
            G_ij  = self._compute_G(t, sector_state.get(i,{}), sector_state.get(j,{}))
            C_new = float(np.clip((1-self.rho)*self.C[(i,j)] + self.rho*G_ij,
                                  self.C_floor, self.C_ceil))
            self.C[(i,j)] = C_new
            self.history[(i,j)].append(C_new)
        return {e: self.W0[e]*self.C[e] for e in self.edges}

    def _compute_G(self, t, si, sj):
        L_i  = si.get("leaked_stress", 0.0); L_j  = sj.get("leaked_stress", 0.0)
        E_i  = max(si.get("elasticity",0.5), 0.01); E_j = max(sj.get("elasticity",0.5), 0.01)
        Phi_i= si.get("phi",0.0);  Phi_j= sj.get("phi",0.0)
        A_i  = si.get("activity", 1.0-si.get("stress",0.0))
        A_j  = sj.get("activity", 1.0-sj.get("stress",0.0))
        dA_i = si.get("dA",0.0);  dA_j  = sj.get("dA",0.0)

        if self.mode == "persistence_only":
            G = float(np.clip((E_i+E_j)/2*(1-(L_i+L_j)/2) - 0.1*(L_i+L_j)/2, 0.05, 1.0))

        elif self.mode == "stress_elasticity_activity":
            phi_div   = float(np.clip(abs(Phi_i-Phi_j)/2, 0, 1))
            e_joint   = float((max(E_i,0)*max(E_j,0))**0.5)
            a_joint   = float(max((max(A_i,0)*max(A_j,0))**0.5, 0))
            leak_avg  = (L_i+L_j)/2
            G = float(np.clip(e_joint*a_joint*(1-0.5*phi_div)-leak_avg, 0.05, 1.0))

        elif self.mode == "delta_deformation":
            total  = abs(dA_i)+abs(dA_j)+1e-6
            div    = float(np.clip(abs(dA_i-dA_j)/total, 0, 1))
            deform = float(np.exp(-self.kappa*div))
            G = float(np.clip(deform*(1-(L_i+L_j)/2), 0.05, 1.0))
        else:
            G = 0.5
        return G

    def get_history_df(self, dates):
        T    = len(dates)
        data = {}
        for e, hist in self.history.items():
            if hist:
                data[e] = (hist[:T] + [hist[-1]]*max(0,T-len(hist)))
        return pd.DataFrame(data, index=dates)

    def effective_W_df(self, dates):
        C_df = self.get_history_df(dates)
        W_df = C_df.copy()
        for e in W_df.columns:
            W_df[e] = W_df[e]*self.W0.get(e,1.0)
        return W_df

    def diagnostic_flatness(self):
        results = {}; flat_count = 0
        for e, hist in self.history.items():
            if not hist: continue
            arr  = np.array(hist)
            std  = float(arr.std())
            flat = std < self.warn_thr
            if flat: flat_count += 1
            results[e] = {"mean":float(arr.mean()),"std":std,
                          "min":float(arr.min()),"max":float(arr.max()),"flat":flat}
        n = len(results)
        if flat_count > 0:
            logger.warning(f"[Connectivity] FLATNESS: {flat_count}/{n} edges have std(C)<{self.warn_thr}")
        else:
            logger.info(f"[Connectivity] All {n} edges evolving (std>={self.warn_thr})")
        results["_system"] = {"n_edges":n,"n_flat":flat_count,
                              "pct_flat":flat_count/max(n,1)*100,
                              "is_evolving":flat_count==0}
        return results


class PropagationEngine:
    """Stress propagation using W_{ij,t} = W^0_{ij} * C_{ij,t}."""

    def __init__(self, sector_graph, ser_panel, config, activity_panel=None):
        self.graph          = sector_graph
        self.ser            = ser_panel
        self.cfg            = config.get("graph", config)
        self.sectors        = sector_graph.sectors
        self.activity_panel = activity_panel
        self._results       = None
        self._conn_state    = None
        self._conn_history  = None
        self._W_history     = None

    def run(self) -> pd.DataFrame:
        mode        = self.cfg.get("propagation_mode","blended")
        lambda_base = float(self.cfg.get("lambda_base",0.3))
        ba          = float(self.cfg.get("blended_weights",{}).get("a_p",0.5))
        bb          = float(self.cfg.get("blended_weights",{}).get("b_p",0.5))
        conn_cfg    = self.cfg.get("connectivity",{})

        dates = self.ser.index; T = len(dates)
        avail = [s for s in self.sectors if s in self.ser.columns.get_level_values(0)]

        stress     = {s: self.ser[s]["stress"].values         for s in avail}
        leaked     = {s: self.ser[s]["leaked_stress"].values  for s in avail}
        elasticity = {s: self.ser[s]["elasticity"].values     for s in avail}
        damping    = {s: self.ser[s]["damping"].values        for s in avail}
        phi        = {s: self.ser[s]["phi"].values            for s in avail
                      if "phi" in self.ser[s].columns}

        activity = {}
        for s in avail:
            if self.activity_panel is not None:
                # activity_panel may be a flat DataFrame (sector columns) or
                # a MultiIndex DataFrame ((sector, variable) columns).
                top_level = self.activity_panel.columns.get_level_values(0)
                if s in top_level:
                    sec_df = self.activity_panel[s]
                    # Prefer efficiency_loss + inventory_slack composite (independent of stress)
                    if isinstance(sec_df, pd.DataFrame) and "efficiency_loss" in sec_df.columns:
                        eff = sec_df["efficiency_loss"].reindex(dates).ffill().bfill().fillna(0).values
                        inv = sec_df.get("inventory_slack",
                                         pd.Series(0.5, index=dates)).reindex(dates).ffill().bfill().fillna(0.5).values
                        act = np.clip(0.6*(1-eff) + 0.4*inv, 0.0, 1.0)
                    elif isinstance(sec_df, pd.Series):
                        act = sec_df.reindex(dates).ffill().bfill().values
                        act = np.clip(act, 0.0, 1.0)
                    else:
                        act = np.clip(1.0 - stress[s], 0.0, 1.0)
                    activity[s] = act
                else:
                    activity[s] = np.clip(1.0 - stress[s], 0.0, 1.0)
            else:
                activity[s] = np.clip(1.0 - stress[s], 0.0, 1.0)

        edges = [(i,j) for (i,j) in self.graph.G.edges() if i in avail and j in avail]
        W0    = {(i,j): float(self.graph.G[i][j]["structural_weight"]) for (i,j) in edges}

        conn_state       = ConnectivityState(edges, W0, conn_cfg)
        self._conn_state = conn_state

        P_raw    = {s: np.zeros(T) for s in avail}
        P_damped = {s: np.zeros(T) for s in avail}
        S_next   = {s: np.zeros(T) for s in avail}
        A_struct = self.graph.adjacency_matrix()

        for t in range(T):
            sector_state = {}
            for s in avail:
                sector_state[s] = {
                    "stress":        float(stress[s][t]),
                    "leaked_stress": float(leaked[s][t]),
                    "elasticity":    float(elasticity[s][t]),
                    "phi":           float(phi[s][t]) if s in phi else 0.0,
                    "activity":      float(activity[s][t]),
                    "dA":            float(activity[s][t]-activity[s][t-1]) if t>0 else 0.0,
                }

            W_eff = conn_state.step(t, sector_state)

            for j in avail:
                P_sum = 0.0
                for i in list(self.graph.G.predecessors(j)):
                    if i not in avail: continue
                    w_ij = W_eff.get((i,j),
                        float(A_struct.loc[i,j]) if i in A_struct.index and j in A_struct.columns else 0.0)
                    if mode=="stress_gradient":
                        P_sum += w_ij*max(0.0, stress[i][t]-stress[j][t])
                    elif mode=="leaked_stress":
                        P_sum += w_ij*max(0.0, leaked[i][t]-leaked[j][t])
                    elif mode=="blended":
                        mi = stress[i][t]-stress[i][t-1] if t>0 else 0.0
                        mj = stress[j][t]-stress[j][t-1] if t>0 else 0.0
                        P_sum += w_ij*(ba*max(0.0,mi-mj) + bb*max(0.0,leaked[i][t]-leaked[j][t]))
                P_raw[j][t]    = lambda_base*P_sum
                P_damped[j][t] = P_raw[j][t]*damping[j][t]
                S_next[j][t]   = stress[j][t]+P_damped[j][t]

        self._conn_history = conn_state.get_history_df(dates)
        self._W_history    = conn_state.effective_W_df(dates)

        C_df = self._conn_history
        eff_conn = C_df.mean(axis=1).values if not C_df.empty else np.zeros(T)
        total_em = np.array([sum(leaked[s][t] for s in avail) for t in range(T)])
        total_ab = np.array([sum(P_raw[s][t]  for s in avail) for t in range(T)])

        flat_diag   = conn_state.diagnostic_flatness()
        is_evolving = flat_diag.get("_system",{}).get("is_evolving", False)
        n_flat      = flat_diag.get("_system",{}).get("n_flat", 0)
        logger.info(
            f"[Connectivity] mode={conn_state.mode}  "
            f"C_mean={eff_conn.mean():.3f}  C_std={eff_conn.std():.4f}  "
            f"evolving={'YES' if is_evolving else 'NO ('+str(n_flat)+' flat edges)'}"
        )

        sector_dfs = {}
        for s in avail:
            sector_dfs[s] = pd.DataFrame({
                "propagation_raw":    P_raw[s],
                "propagation_damped": P_damped[s],
                "stress_forecast":    S_next[s],
                "propagation_share":  np.where(
                    stress[s]+P_raw[s]>0, P_raw[s]/(stress[s]+P_raw[s]+1e-8), 0),
                "activity":           activity[s],
            }, index=dates)

        result = pd.concat(sector_dfs, axis=1)
        result[("_system","effective_connectivity")] = eff_conn
        result[("_system","total_emission")]         = total_em
        result[("_system","total_absorption")]       = total_ab
        result[("_system","net_flow")]               = total_em - total_ab

        self._results = result
        logger.info(f"[Propagation] Complete: {result.shape}")
        return result

    def get_results(self): return self._results
    def get_connectivity_history(self): return self._conn_history
    def get_effective_W_history(self): return self._W_history

    def connectivity_diagnostic_df(self) -> pd.DataFrame:
        if self._conn_state is None: return pd.DataFrame()
        diag = self._conn_state.diagnostic_flatness()
        rows = []
        for e, stats in diag.items():
            if e == "_system": continue
            i,j = e
            rows.append({"source":i,"target":j,
                         "W0":self._conn_state.W0.get(e,0.0),
                         "C_mean":stats["mean"],"C_std":stats["std"],
                         "C_min":stats["min"],"C_max":stats["max"],"flat":stats["flat"]})
        return pd.DataFrame(rows).sort_values("C_std",ascending=False)

    def top_n_connectivity_histories(self, n=5, dates=None) -> pd.DataFrame:
        if self._conn_history is None or self._conn_history.empty: return pd.DataFrame()
        top = self._conn_history.std().nlargest(n).index
        return self._conn_history[top]

    def emitter_absorber_table(self) -> pd.DataFrame:
        if self._results is None: raise RuntimeError("Run .run() first")
        rows = []; avail=[s for s in self.sectors if s in self.ser.columns.get_level_values(0)]
        for s in avail:
            rows.append({"sector":s,
                "mean_propagation_emitted":  self.ser[s]["leaked_stress"].mean(),
                "mean_propagation_received": self._results[s]["propagation_raw"].mean()
                    if s in self._results.columns.get_level_values(0) else 0,
                "peak_emission": self.ser[s]["leaked_stress"].max(),
                "peak_received": self._results[s]["propagation_raw"].max()
                    if s in self._results.columns.get_level_values(0) else 0})
        df = pd.DataFrame(rows).set_index("sector")
        df["net_emission"] = df["mean_propagation_emitted"]-df["mean_propagation_received"]
        return df.sort_values("net_emission",ascending=False)

    def effective_graph_snapshot(self, date: pd.Timestamp) -> nx.DiGraph:
        G_eff = nx.DiGraph(); G_eff.add_nodes_from(self.sectors)
        if self._W_history is None: return G_eff
        t_idx = min(self.ser.index.get_indexer([date],method="nearest")[0],
                    len(self._W_history)-1)
        for e in self._W_history.columns:
            i,j  = e
            w_eff = float(self._W_history.iloc[t_idx][e])
            w0    = self._conn_state.W0.get(e,0.0) if self._conn_state else 0.0
            G_eff.add_edge(i,j,weight=w_eff,structural_weight=w0,
                           C=w_eff/(w0+1e-8))
        return G_eff
