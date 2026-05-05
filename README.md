# epi_market_sim

**Epidemic–Market Simulation Pipeline**

A coupled simulation framework that models how epidemic forcing propagates through sectoral economic networks, decomposes stress leakage into market and household exit channels, and identifies the structural regime boundary between them.

Built to support the research paper:
> *Household Infrastructure Stabilization as a Structural Economic Resilience Mechanism*
> [SSRN preprint — dx.doi.org/10.2139/ssrn.6174978](https://dx.doi.org/10.2139/ssrn.6174978)

---

## Detailed Report

The full analysis and detailed report can be found [here](paper/Coupled%20Epidemic%20and%20Economic%20Constraint%20Dynamics%3A%20A%20SER-SIR%20Framework%20with%20Market%20and%20Household%20Leakage%20Regimes.pdf).

## What This Models

Modern economies have two channels through which economic stress exits the system:

- **L_market** — stress that propagates through observable institutional channels: exchange failures, credit events, structural cascade across the sector graph. This is what standard economic measurement sees.
- **L_household** — stress absorbed through household behavioral and temporal channels: deferral, substitution, informal economy, time reallocation. This is largely invisible to standard measurement.

The central finding is that these channels are **not additive — they are competitive**. Across 200 Latin Hypercube simulation runs spanning the full plausible parameter space, the correlation between L_market and L_household is negative in every single observation (range −0.82 to −0.23). The system routes stress through one channel or the other depending on the elasticity state of the sector.

There is a **structural regime boundary** — a threshold in elasticity (E) and stress (S) space — below which stress exits through the market channel regardless of other conditions, and above which the household channel absorbs it. That boundary is empirically fitted, not assumed.

```
E ≤ 0.48                                → market dominant  (necessary condition)
E > 0.48,  S ≤ 0.27                     → household dominant
E > 0.48,  S > 0.31                     → market dominant
E > 0.48,  0.27 < S ≤ 0.31,  E > 0.61  → household dominant

Critical switching point:  S* ≈ 0.265,  E* ≈ 0.521
Classifier accuracy:  94% (logistic regression, 5-fold CV)
```

The policy implication is direct: investment that raises sectoral elasticity above the E = 0.48 threshold changes **which channel stress exits through** — not just how much stress there is. Stress routed through the household channel dissipates through time and behavior. Stress routed through the market channel propagates structurally through institutions. Same stress quantum, categorically different systemic consequences.

---

## Architecture

```
Epidemic Data (OWID/FRED)
        │
        ▼
   SIR Model  ──────────────────────────────────────────┐
   S, I, R, P_t                                         │
        │                                               │
        ▼                                               ▼
  Sector Panel                                    SER Engine
  (BEA IO + FRED)                          S_{i,t}, E_{i,t}, L_market_{i,t}
                                                        │
                                   ┌────────────────────┤
                                   ▼                    ▼
                          Connectivity C_{ij,t}   Propagation
                          W_{ij,t} = W⁰ × C       W·L_market
                                                        │
                                   ┌────────────────────┤
                                   ▼                    ▼
                           Regime Classifier    Leakage Decomposition
                           E=0.48 threshold     L_market + L_household
                                   │
                                   ▼
                              Analytics + Dashboard + Paper Figures
```

### Key Equations

**SER elasticity dynamics:**
```
E_{t+1} = (1−δ)·E_t
         + α·B_t·(1−E_t) + χ·C_t·(1−E_t)    ← diminishing-returns gains
         − β_L·(L_t + γ_dL·ΔL_t⁺)            ← leak drain
         − β_S·max(0, S_t − S_thresh)          ← stress drain
         clipped [0.01, 1.30]
```

**Leakage decomposition:**
```
L_total[i,t]   = L_market[i,t] + L_household[i,t]

A_expected     = clip((1−S)·(0.6 + 0.25·E_norm + 0.15·W_in_norm), 0, 1)
A_observed     = 0.6·(1−efficiency_loss) + 0.4·inventory_slack   ← FRED proxies
L_household    = clip(A_expected − A_observed, 0, 1)
```

**Dynamic connectivity:**
```
W_{ij,t}  = W⁰_{ij} × C_{ij,t}
C_{t+1}   = clip(ρ_c·C_t + (1−ρ_c)·G_t,  0.05, 0.95)
```

---

## Key Findings

| Finding | Result |
|---|---|
| corr(L_market, L_household) | Negative in **all** 2,000 sector-run observations |
| Range of correlation | −0.82 to −0.23 across full parameter space |
| Regime boundary classifier accuracy | 94.0% CV (logistic regression) |
| Primary regime determinant | E_mean (permutation importance 0.297) |
| Secondary determinant | S_mean (0.078); all other params <0.01 |
| Switching band | S ∈ [0.22, 0.31],  E ∈ [0.41, 0.61] |
| L_household distribution | Non-Gaussian, zero-inflated, 2–4 structural breaks per sector |
| C independence from S+E+L | 26.7% of variance unexplained (R²=0.733) |

The non-Gaussianity and zero-inflation of L_household are features, not noise. Household stress absorption is episodic and threshold-activated — consistent with a statistical exit-point process operating through behavioral and temporal channels rather than a stable in-system variable.

---

## Data Sources

All data is from **public institutional sources**. No individual-level data, no behavioral surveillance, no proprietary feeds.

| Source | Variables | Used for |
|---|---|---|
| [FRED](https://fred.stlouisfed.org) | TCU, ISRATIO, CPI components | Sector stress composite, A_obs proxy |
| [OWID / JHU](https://ourworldindata.org/coronavirus) | Cases, deaths, hospitalizations, stringency | SIR forcing |
| [BEA IO Tables](https://bea.gov) | Use table 2017 | Sector graph edge weights W⁰ |
| [Opportunity Insights](https://opportunityinsights.org) | EconomicTracker spend data | A_obs (direct activity signal) |

Synthetic fallbacks activate automatically if real data sources are unavailable. The model runs fully on synthetic data out of the box.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full simulation pipeline
python main.py run

# Launch interactive dashboard (11 tabs)
python main.py dashboard

# Export all paper figures (Figures 1–18) as interactive HTML
python main.py paper-figures

# Export specific figures
python main.py paper-figures --figs 1,15,16,17

# Export as HTML + PNG (requires kaleido)
python main.py paper-figures --fmt both

# Validate environment
python main.py check
```

---

## Paper Figures

| Figure | Title |
|---|---|
| 1 | SIR epidemic curve and sector stress forcing overlays |
| 2 | Conceptual layering of the model |
| 3 | Full system diagram |
| 4 | Data and simulation pipeline |
| 5 | Sectoral stress trajectories |
| 6 | Elasticity trajectories |
| 7 | Connectivity evolution |
| 8 | Leakage decomposition over time |
| 9 | Distribution of L_household |
| 10 | Rolling distribution diagnostics |
| 11 | Breakpoint and activation analysis |
| 12 | Correlation distribution across runs |
| 13 | Parameter sweep heatmaps |
| 14 | Stress-based regime distribution |
| 15 | Stress–elasticity phase diagram with classifier boundary |
| 16 | Logistic probability surface |
| 17 | Decision tree threshold overlay |
| 18 | Snapback illustration |

All figures are standalone interactive HTML — open directly in any browser, no server required.

---

## Project Structure

```
epi_market_sim/
├── main.py                          # CLI: run, dashboard, paper-figures, check
├── requirements.txt
├── ARCHITECTURE.md                  # Full system design and equations
├── CHANGES.md                       # Changelog from initial state
├── USAGE.md                         # Detailed usage guide
├── TRANSCRIPT.md                    # Analytical session log
├── configs/
│   ├── base_config.yaml             # SER parameters, connectivity mode
│   ├── sector_graph.yaml            # 10-sector graph topology
│   └── intervention.yaml            # Intervention scenarios
├── src/
│   ├── data_ingestion/              # FRED/OWID loaders, BEA/OI adapters
│   ├── preprocessing/               # Sector observables panel builder
│   ├── epidemic/                    # SIR model
│   ├── ser/                         # Stress–Elasticity–Reaction engine
│   ├── graph/                       # Dynamic connectivity + propagation
│   ├── simulation/                  # Pipeline runner, regime, interventions
│   ├── analytics/                   # Analysis, blobs, leakage decomposition
│   └── dashboard/                   # 11-tab Streamlit application
├── scripts/
│   ├── export_paper_figures.py      # Figures 1–18 (paper spec)
│   ├── elasticity_sweep.py          # 200-run LHS parameter sweep
│   ├── run_intervention.py          # 10-scenario intervention comparison
│   ├── elasticity_diagnostics.py    # SER diagnostics
│   └── generate_figures.py          # Legacy figure CSV export
└── outputs/
    ├── parquet/                     # Primary simulation outputs
    ├── paper_figures/               # Figures 1–18 HTML/PNG
    ├── sweep/                       # Parameter sweep results
    └── intervention/                # Scenario comparison outputs
```

---

## Dashboard

```bash
python main.py dashboard
```

11 tabs covering the full analytical surface:

| Tab | Content |
|---|---|
| Epidemic | SIR curves, pressure time series |
| Sector SER | Per-sector stress, elasticity, leakage |
| Network | Sector graph with dynamic edge weights |
| Regimes | Regime heatmap, duration table, transition matrix |
| Decomposition | Stress decomposition by forcing component |
| System Metrics | Aggregate instability and elasticity metrics |
| Interventions | 10-scenario comparison with H_t profiles |
| Parameter Sweep | Elasticity sweep heatmaps and lag classification |
| Connectivity | C_{ij,t} history, mode diagnostics |
| Sector Blobs | Activity volumes, intra/inter-sector deltas |
| Tables | Raw parquet table viewer |

---

## Configuration

Key parameters in `configs/base_config.yaml`:

```yaml
ser:
  elasticity:
    delta: 0.05        # natural decay rate
    beta_L: 0.60       # leaked-stress drain on E
    beta_S: 0.30       # high-stress drain on E
    alpha: 0.25        # buffering gain
    chi: 0.18          # clearance gain
    E_init: 0.35       # initial elasticity
    E_crit: 0.25       # critical threshold for regime flagging

graph:
  connectivity:
    mode: stress_elasticity_activity   # persistence_only | delta_deformation
    rho_c: 0.35                        # connectivity persistence
```

---

## Research Context

This simulation was built to provide the empirical architecture for:

> **Household Infrastructure Stabilization as a Structural Economic Resilience Mechanism**

The paper argues that household cost volatility — particularly in housing and energy — functions as a structural shock transmission channel. Infrastructure investment that reduces essential cost variability raises household elasticity, shifts stress routing from the market channel to the household channel, and produces non-inflationary real value by reducing structural friction in economic circulation.

The simulation demonstrates the mechanism is real, quantifiable, threshold-dependent, and detectable from existing public institutional data without individual-level surveillance. The regime boundary at E = 0.48 is not asserted — it is fitted from the parameter sweep and survives cross-validation.

The Figure 1 observation from the 2020–2022 period is illustrative: medical services stress dropped sharply when capital interventions landed in early 2021, while shelter, food, energy, transportation, and apparel stress spiked simultaneously. The intervention stabilized the monitored channel. The unmonitored household absorption layer received the redistributed burden. Standard aggregate metrics recorded a recovery. The sector graph recorded a redistribution.

---

## License

Research code — see LICENSE for terms.

---

## Citation

If you use this simulation framework, please cite the associated paper:

```
Household Infrastructure Stabilization as a Structural Economic Resilience Mechanism
SSRN preprint: https://dx.doi.org/10.2139/ssrn.6174978
```
