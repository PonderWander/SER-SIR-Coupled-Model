# Architecture — epi_market_sim

## System Overview

`epi_market_sim` is a coupled epidemic–market simulation pipeline. An SIR epidemic model produces time-varying pressure on a network of economic sectors. Each sector runs a Stress–Elasticity–Reaction (SER) engine that computes how much stress it absorbs versus leaks. Leaked stress propagates across a sector graph whose edges have dynamic connectivity weights. Leaked stress is decomposed into two exit channels: `L_market` (observable structural failure) and `L_household` (residual absorbed through household/local/temporal channels).

---

## Data Flow

```
Raw data sources
  FRED (CPI, TCU, ISRATIO)         ──┐
  OWID / JHU (COVID)                ──┼──► data_ingestion/loaders.py
  BEA IO table (optional)           ──┘
  OI EconomicTracker (optional)              │
                                             ▼
                                  preprocessing/panel_builder.py
                                  sector_observables_panel
                                             │
                         ┌───────────────────┤
                         ▼                   ▼
                 epidemic/sir_model.py    ser/ser_engine.py
                 S,I,R,pressure_t         S,E,L_market per sector
                                             │
                                             ▼
                                  graph/propagation.py
                                  C_{ij,t}, W_{ij,t}
                                             │
                         ┌───────────────────┤
                         ▼                   ▼
            simulation/regime.py    analytics/household_leakage.py
            regime labels           L_household per sector
                         │
                         ▼
            analytics/analysis.py   scripts/elasticity_sweep.py
            summary stats           200-run LHS sweep
                         │
                         ▼
            src/dashboard/app.py  (11-tab Streamlit)
            scripts/export_paper_figures.py  (Figures 2-18)
```

---

## Key Equations

### SER Engine

```
S_{i,t} = Σ w_k · feature_k                              [composite stress]
d_{i,t} = 1 / (1 + k_E·E + k_θ·θ_{t-1})                [damping]
L_market_{i,t} = max(0, S - E/(E + d·S) · S)            [hard-channel leakage]

E_{i,t+1} = (1-δ)·E_t
           + α·B_t·(1-E_t) + χ·C_t·(1-E_t)             [diminishing-returns gains]
           - β_L·(L_t + γ_dL·ΔL_t⁺)                    [leak drain]
           - β_S·max(0, S_t - S_thresh)                  [stress drain]
           clipped [0.01, 1.30]
```

Calibrated defaults: `δ=0.05`, `β_L=0.60`, `β_S=0.30`, `α=0.25`, `χ=0.18`, `γ_dL=0.80`

### Connectivity

```
W⁰_{ij}   — structural prior (sector_graph.yaml)
C_{ij,t}  — dynamic state ∈ [0,1]
W_{ij,t}  = W⁰_{ij} × C_{ij,t}

stress_elasticity_activity mode (default):
  G = √(E_i·E_j) · √(A_i·A_j) · (1 - 0.5·|Φ_i-Φ_j|/2) - L_avg
  C_{t+1} = clip(ρ_c·C_t + (1-ρ_c)·G, 0.05, 0.95)

A_{i,t} = 0.6·(1-efficiency_loss) + 0.4·inventory_slack  [FRED proxy]
```

### Leakage Decomposition

```
L_total[i,t] = L_market[i,t] + L_household[i,t]

A_expected = clip((1-S)·(0.6 + 0.25·E_norm + 0.15·W_in_norm), 0, 1)
A_observed = 0.6·(1-efficiency_loss) + 0.4·inventory_slack
L_household = clip(A_expected - A_observed, 0, 1)
```

### Regime Boundary (fitted)

```
P(household_dominant) = σ(z)
z = −47.02·S + 18.50·E + 10.25·w_covid + 10.12·χ + 9.06·δ − 1.35·α − 0.70·ρ − 1.66

Tree decision rule:
  E ≤ 0.48                           → market dominant
  E > 0.48, S ≤ 0.27                 → household dominant
  E > 0.48, S > 0.31                 → market dominant
  E > 0.48, 0.27 < S ≤ 0.31, E>0.61 → household dominant

Critical point: S* ≈ 0.265, E* ≈ 0.521
```

---

## Module Reference

| Module | File | Purpose |
|---|---|---|
| Data ingestion | `src/data_ingestion/loaders.py` | Load/synthesize CPI, efficiency, COVID data |
| Activity adapters | `src/data_ingestion/activity_adapters.py` | BEA IO + OI concordance |
| Panel builder | `src/preprocessing/panel_builder.py` | sector_observables_panel |
| SIR model | `src/epidemic/sir_model.py` | Epidemic forcing |
| SER engine | `src/ser/ser_engine.py` | Per-sector SER dynamics |
| Propagation | `src/graph/propagation.py` | Dynamic connectivity + propagation |
| Runner | `src/simulation/runner.py` | 7-stage pipeline orchestrator |
| Regime | `src/simulation/regime.py` | Regime classification |
| Intervention | `src/simulation/intervention.py` | Multi-scenario H_t engine |
| Analysis | `src/analytics/analysis.py` | Summary stats, tables |
| Blobs | `src/analytics/blob_analytics.py` | Sector blob dynamics |
| Leakage | `src/analytics/household_leakage.py` | L_market/L_household decomposition |
| Dashboard | `src/dashboard/app.py` | 11-tab Streamlit app |
| Paper figures | `scripts/export_paper_figures.py` | Figures 2–18 export |

---

## Output Files

| Path | Content |
|---|---|
| `outputs/parquet/sector_ser_panel.parquet` | Full SER state panel |
| `outputs/parquet/propagation_panel.parquet` | Propagation results |
| `outputs/parquet/leakage_decomposition.parquet` | L_market + L_household |
| `outputs/parquet/leakage_sweep.parquet` | 200-run LHS sweep |
| `outputs/parquet/connectivity_C_history.parquet` | C_{ij,t} all edges |
| `outputs/parquet/regime_probabilities.parquet` | Run-level regime probs |
| `outputs/parquet/regime_classifier.txt` | Fitted equations + rules |
| `outputs/paper_figures/*.html` | Figures 2–18 |

---

## Known Constraints

- All current runs use **synthetic data** — FRED/OI blocked in sandbox
- `L_household` quality improves materially with real OI spend data
- BEA IO required for real W⁰ edge weights (`data/raw/bea/`)
- PNG export requires `kaleido`: `pip install kaleido`
