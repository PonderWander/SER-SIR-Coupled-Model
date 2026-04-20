# Usage Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scipy`, `plotly`, `streamlit`, `scikit-learn`, `pyarrow`, `networkx`, `pyyaml`.

For PNG figure export (optional):
```bash
pip install kaleido
```

### Data (optional — synthetic fallbacks active without these)

| Source | Local path | How to obtain |
|---|---|---|
| BEA IO Use Table (2017) | `data/raw/bea/use_table_2017.csv` | bea.gov — no registration |
| OI EconomicTracker | `data/raw/oi/affinity_national_weekly.csv` | `git clone https://github.com/OpportunityInsights/EconomicTracker` |

---

## Commands

All commands run from the project root via `main.py`:

### Full simulation
```bash
python main.py run
python main.py run --config configs/base_config.yaml
python main.py run --refresh-cache   # re-fetch raw data
```

### Dashboard
```bash
python main.py dashboard
python main.py dashboard --port 8502
```
Reads from `outputs/parquet/`. Run `main.py run` at least once first.

### Paper figures (Figures 2–18)
```bash
python main.py paper-figures                    # all figures, HTML
python main.py paper-figures --fmt both         # HTML + PNG
python main.py paper-figures --figs 5,6,7       # specific figures only
python main.py paper-figures --output-dir path  # custom output dir
```
Output: `outputs/paper_figures/figure_NN_name.html`
Open any `.html` file directly in a browser — no server required.

### Export figure-ready CSVs
```bash
python main.py figures
```

### Environment check
```bash
python main.py check
```

---

## Running scripts directly

```bash
python scripts/export_paper_figures.py             # all paper figures
python scripts/export_paper_figures.py --figs 15,16,17
python scripts/elasticity_sweep.py                 # parameter sweep
python scripts/run_intervention.py                 # intervention scenarios
python scripts/elasticity_diagnostics.py           # SER diagnostics
python scripts/generate_figures.py                 # legacy figure CSVs
```

---

## Dashboard tabs

| Tab | Content | Required files |
|---|---|---|
| 0. Epidemic | SIR curves, pressure overlay | `sir_timeseries.parquet` |
| 1. Sector SER | Stress/elasticity/leakage per sector | `sector_ser_panel.parquet` |
| 2. Network | Sector graph with propagation weights | `propagation_panel.parquet` |
| 3. Regimes | Regime heatmap, duration table | `regime_panel.parquet` |
| 4. Decomposition | Stress decomposition by component | `sector_ser_panel.parquet` |
| 5. System Metrics | Aggregate instability metrics | `sector_ser_panel.parquet` |
| 6. Interventions | 10-scenario comparison | Run `scripts/run_intervention.py` first |
| 7. Parameter Sweep | Elasticity sweep heatmaps | Run `scripts/elasticity_sweep.py` first |
| 8. Connectivity | C_{ij,t} history, mode diagnostics | `connectivity_C_history.parquet` |
| 9. Sector Blobs | Blob sizes, intra/inter deltas | `blob_sizes.parquet` |
| 10. Tables | Raw parquet table viewer | Any parquet in output dir |

---

## Paper figures reference

| Figure | Title | Data source |
|---|---|---|
| 2 | Conceptual layering | None (structural) |
| 3 | Full system diagram | None (structural) |
| 4 | Data and simulation pipeline | None (structural) |
| 5 | Sectoral stress trajectories | `sector_ser_panel.parquet` |
| 6 | Elasticity trajectories | `sector_ser_panel.parquet` |
| 7 | Connectivity evolution | `connectivity_C_history.parquet` |
| 8 | Leakage decomposition over time | `leakage_decomposition.parquet` |
| 9 | L_household distribution | `leakage_decomposition.parquet` |
| 10 | Rolling distribution diagnostics | `leakage_decomposition.parquet` |
| 11 | Breakpoint and activation analysis | `leakage_decomposition.parquet` |
| 12 | Correlation distribution across runs | `leakage_sweep.parquet` |
| 13 | Parameter sweep heatmaps | `leakage_sweep.parquet` |
| 14 | Stress-based regime distribution | `leakage_sweep.parquet` |
| 15 | Stress–elasticity phase diagram | `leakage_sweep.parquet` |
| 16 | Logistic probability surface | `leakage_sweep.parquet` |
| 17 | Decision tree boundary overlay | `leakage_sweep.parquet` |
| 18 | Snapback illustration | `sector_ser_panel.parquet` + `leakage_decomposition.parquet` |

---

## Configuration

`configs/base_config.yaml` — SER parameters, connectivity mode, run metadata.
`configs/sector_graph.yaml` — 10-sector graph topology and edge weights.
`configs/intervention.yaml` — 10 intervention scenarios and weight maps.

Key parameters:
```yaml
ser:
  elasticity:
    delta: 0.05      # natural decay rate
    beta_L: 0.60     # leaked-stress drain
    alpha: 0.25      # buffering gain
    chi: 0.18        # clearance gain
    E_init: 0.35     # initial elasticity

graph:
  connectivity:
    mode: stress_elasticity_activity  # or persistence_only / delta_deformation
    rho_c: 0.35
```

---

## Project structure

```
epi_market_sim/
├── main.py                          # entry point — run, dashboard, paper-figures, check
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── CHANGES.md
├── TRANSCRIPT.md
├── USAGE.md                         # this file
├── configs/
│   ├── base_config.yaml
│   ├── sector_graph.yaml
│   └── intervention.yaml
├── src/
│   ├── data_ingestion/              # loaders, activity adapters
│   ├── preprocessing/               # panel builder
│   ├── epidemic/                    # SIR model
│   ├── ser/                         # SER engine
│   ├── graph/                       # propagation, connectivity
│   ├── simulation/                  # runner, regime, intervention
│   ├── analytics/                   # analysis, blobs, household leakage
│   └── dashboard/                   # Streamlit app (11 tabs)
├── scripts/
│   ├── export_paper_figures.py      # Figures 2-18 export
│   ├── elasticity_sweep.py          # parameter sweep
│   ├── run_intervention.py          # intervention scenarios
│   ├── elasticity_diagnostics.py    # SER diagnostics
│   └── generate_figures.py          # legacy figure CSV export
├── outputs/
│   ├── parquet/                     # primary simulation outputs
│   ├── csv/                         # human-readable exports
│   ├── figures/                     # Plotly HTML (dashboard figures)
│   ├── paper_figures/               # Figures 2-18 (paper export)
│   ├── sweep/                       # elasticity sweep results
│   └── intervention/                # scenario comparison outputs
└── data/                            # not included — see Prerequisites
    └── raw/
        ├── bea/
        └── oi/
```
