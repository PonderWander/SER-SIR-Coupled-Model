"""
scripts/generate_figures.py
────────────────────────────
Generate static publication-quality figures from simulation outputs.
Saves PNG and interactive HTML to outputs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

PARQUET = Path("outputs/parquet")
FIGURES = Path("outputs/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

REGIME_COLORS = {
    "dispersal":    "#4CAF50",
    "accumulation": "#FF9800",
    "isolation":    "#9C27B0",
    "recovery":     "#2196F3",
    "fragmented":   "#F44336",
    "amplification":"#E91E63",
    "unknown":      "#BDBDBD",
}

SECTOR_COLORS = px.colors.qualitative.Set2


def load(name: str) -> pd.DataFrame:
    p = PARQUET / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    # Only convert index to datetime if it looks like dates
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            sample = str(df.index[0]) if len(df.index) > 0 else ""
            # Heuristic: strings with '-' and length > 6 are likely dates
            if len(sample) > 6 and "-" in sample and sample[0].isdigit():
                df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def fig1_epidemic_overview():
    """Figure 1: Epidemic trajectories and monthly pressure."""
    sir = load("sir_timeseries")
    covid = load("covid_monthly")
    if sir.empty:
        return

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=("Daily COVID-19 Cases & Hospitalizations", "Monthly Epidemic Pressure"),
        vertical_spacing=0.08,
    )

    if "new_cases_smoothed" in sir.columns:
        fig.add_trace(go.Scatter(
            x=sir.index, y=sir["new_cases_smoothed"],
            name="Daily Cases (smoothed)", fill="tozeroy",
            line=dict(color="#EF5350", width=1.5),
            fillcolor="rgba(239,83,80,0.15)"
        ), row=1, col=1)

    if "hosp_patients" in sir.columns and sir["hosp_patients"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=sir.index, y=sir["hosp_patients"],
            name="Hospitalizations", line=dict(color="#7B1FA2", width=2)
        ), row=1, col=1)

    if not covid.empty and "epidemic_pressure" in covid.columns:
        fig.add_trace(go.Bar(
            x=covid.index, y=covid["epidemic_pressure"],
            name="Monthly Pressure", marker_color="#EF9A9A",
            marker_line_color="#E53935", marker_line_width=1,
        ), row=2, col=1)

    fig.update_layout(
        title="Epidemic Layer: COVID-19 Forcing Process",
        template="plotly_white", height=600,
        legend=dict(orientation="h", y=-0.05),
        font=dict(family="Arial", size=12),
    )
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Pressure [0–1]", row=2, col=1)
    fig.write_html(str(FIGURES / "fig1_epidemic_overview.html"))
    print("  ✓ fig1_epidemic_overview.html")


def fig2_sector_stress_panel():
    """Figure 2: Sector stress time series with epidemic overlay."""
    ser = load("sector_ser_panel")
    covid = load("covid_monthly")
    if ser.empty:
        return

    sectors = ["energy", "food_at_home", "transportation_services",
               "shelter", "medical_services", "household_goods"]
    stress_cols = {s: f"{s}_stress" for s in sectors if f"{s}_stress" in ser.columns}

    fig = make_subplots(
        rows=3, cols=2, shared_xaxes=True,
        subplot_titles=list(stress_cols.keys()),
        vertical_spacing=0.08, horizontal_spacing=0.08,
    )

    ep = covid["epidemic_pressure"] if not covid.empty and "epidemic_pressure" in covid.columns else None

    positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]
    for (sec, col), (r, c), color in zip(stress_cols.items(), positions, SECTOR_COLORS):
        s = ser[col]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=sec,
            line=dict(color=color, width=2), showlegend=False,
        ), row=r, col=c)
        # Leaked stress as shaded area
        lk_col = f"{sec}_leaked_stress"
        if lk_col in ser.columns:
            lk = ser[lk_col]
            fig.add_trace(go.Scatter(
                x=lk.index, y=lk.values, name=f"{sec} leaked",
                fill="tozeroy", line=dict(color=color, width=0),
                fillcolor="rgba(180,180,180,0.2)",
                showlegend=False,
            ), row=r, col=c)
        # Epidemic pressure dotted overlay
        if ep is not None:
            ep_scaled = ep.reindex(s.index) * s.max()
            fig.add_trace(go.Scatter(
                x=ep_scaled.index, y=ep_scaled.values,
                line=dict(color="#B0BEC5", dash="dot", width=1),
                showlegend=False, name="epidemic"
            ), row=r, col=c)
        fig.update_yaxes(range=[0, 1], row=r, col=c)

    fig.update_layout(
        title="Sector Stress Time Series (solid) with Leaked Stress (shaded) and Epidemic Pressure (dotted)",
        template="plotly_white", height=700,
        font=dict(family="Arial", size=11),
    )
    fig.write_html(str(FIGURES / "fig2_sector_stress_panel.html"))
    print("  ✓ fig2_sector_stress_panel.html")


def fig3_elasticity_regime():
    """Figure 3: Elasticity dynamics with E_crit threshold and regime coloring."""
    ser = load("sector_ser_panel")
    regime = load("regime_panel")
    if ser.empty:
        return

    sectors = ["energy", "food_away_from_home", "transportation_services", "shelter"]
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        subplot_titles=sectors, vertical_spacing=0.10, horizontal_spacing=0.08,
    )

    E_CRIT = 0.25
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for sec, (r, c), color in zip(sectors, positions, SECTOR_COLORS):
        ecol = f"{sec}_elasticity"
        rcol = f"{sec}_regime"
        if ecol not in ser.columns:
            continue
        e = ser[ecol]

        # Regime background bands
        if not regime.empty and rcol in regime.columns:
            reg_s = regime[rcol]
            for t in range(len(reg_s)):
                reg_val = reg_s.iloc[t]
                clr = REGIME_COLORS.get(reg_val, "#BDBDBD")
                x0 = reg_s.index[t]
                x1 = reg_s.index[t+1] if t+1 < len(reg_s) else x0 + pd.DateOffset(months=1)
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=clr, opacity=0.12, line_width=0,
                    row=r, col=c,
                )

        fig.add_trace(go.Scatter(
            x=e.index, y=e.values, name=sec,
            line=dict(color=color, width=2), showlegend=False,
        ), row=r, col=c)
        fig.add_hline(y=E_CRIT, line_dash="dash", line_color="red",
                      annotation_text="E_crit", row=r, col=c)
        fig.update_yaxes(range=[0, 1], title_text="Elasticity", row=r, col=c)

    fig.update_layout(
        title="Sector Elasticity Dynamics with Regime Coloring (E_crit = 0.25 dashed)",
        template="plotly_white", height=600,
        font=dict(family="Arial", size=11),
    )
    fig.write_html(str(FIGURES / "fig3_elasticity_regime.html"))
    print("  ✓ fig3_elasticity_regime.html")


def fig4_regime_heatmap():
    """Figure 4: Sector × Time regime heatmap."""
    regime = load("regime_panel")
    if regime.empty:
        return

    regime_order = ["dispersal","accumulation","isolation","recovery",
                    "fragmented","amplification","unknown"]
    regime_num = {r: i for i, r in enumerate(regime_order)}

    regime_cols = [c for c in regime.columns if c.endswith("_regime")
                   and not c.endswith("_code")]
    if not regime_cols:
        return

    hm = regime[regime_cols].copy()
    hm.columns = [c.replace("_regime","") for c in regime_cols]
    hm_num = hm.map(lambda x: regime_num.get(str(x), 6))

    colorscale = [
        [0/6, "#4CAF50"],   # dispersal
        [1/6, "#FF9800"],   # accumulation
        [2/6, "#9C27B0"],   # isolation
        [3/6, "#2196F3"],   # recovery
        [4/6, "#F44336"],   # fragmented
        [5/6, "#E91E63"],   # amplification
        [6/6, "#BDBDBD"],   # unknown
    ]

    fig = go.Figure(go.Heatmap(
        z=hm_num.T.values,
        x=hm_num.index.strftime("%Y-%m"),
        y=hm_num.columns.tolist(),
        colorscale=colorscale,
        zmin=0, zmax=6,
        colorbar=dict(
            tickvals=[0,1,2,3,4,5,6],
            ticktext=regime_order,
            title="Regime",
        ),
        hovertemplate="Date: %{x}<br>Sector: %{y}<br>Regime: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Sector Regime Map: COVID-era Market States",
        xaxis_title="Date", yaxis_title="Sector",
        template="plotly_white", height=450,
        font=dict(family="Arial", size=11),
    )
    fig.write_html(str(FIGURES / "fig4_regime_heatmap.html"))
    print("  ✓ fig4_regime_heatmap.html")


def fig5_network_snapshot():
    """Figure 5: Sector network stress snapshot (peak stress month)."""
    ser = load("sector_ser_panel")
    if ser.empty:
        return

    # Find peak system stress month
    stress_cols = [c for c in ser.columns if c.endswith("_stress")
                   and "leaked" not in c and "absorbed" not in c]
    if not stress_cols:
        return
    mean_stress = ser[stress_cols].mean(axis=1)
    peak_date = mean_stress.idxmax()

    # Load graph
    try:
        from src.utils.common import load_sector_graph_config
        from src.graph.propagation import SectorGraph
        gcfg = load_sector_graph_config("configs/sector_graph.yaml")
        sg = SectorGraph(gcfg)
        G = sg.G
    except Exception as e:
        print(f"  ✗ Network snapshot: {e}")
        return

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node values at peak date
    node_stress = {}
    node_elasticity = {}
    node_leaked = {}
    for sec in G.nodes():
        for var, store in [("stress", node_stress),
                           ("elasticity", node_elasticity),
                           ("leaked_stress", node_leaked)]:
            col = f"{sec}_{var}"
            if col in ser.columns and peak_date in ser.index:
                store[sec] = float(ser[col].loc[peak_date])
            else:
                store[sec] = 0.0

    # Edges
    edge_traces = []
    for u, v, w in G.edges(data="weight"):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # Arrow direction via annotation, line for edge
        edge_traces.append(go.Scatter(
            x=[x0, (x0+x1)*0.6, None],
            y=[y0, (y0+y1)*0.6, None],
            mode="lines",
            line=dict(width=w * 5, color="rgba(100,130,180,0.5)"),
            hoverinfo="none", showlegend=False,
        ))

    nodes = list(G.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    stress_vals = [node_stress.get(n, 0) for n in nodes]
    elast_vals  = [node_elasticity.get(n, 0.5) for n in nodes]
    leaked_vals = [node_leaked.get(n, 0) for n in nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=[25 + 50 * v for v in stress_vals],
            color=stress_vals,
            colorscale="YlOrRd",
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title="Stress", thickness=15, x=1.02),
            line=dict(width=2, color="white"),
        ),
        text=nodes,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=[
            f"<b>{n}</b><br>Stress: {node_stress.get(n,0):.3f}"
            f"<br>Elasticity: {node_elasticity.get(n,0):.3f}"
            f"<br>Leaked: {node_leaked.get(n,0):.3f}"
            for n in nodes
        ],
        hoverinfo="text",
        name="Sectors",
        showlegend=False,
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f"Sector Network Snapshot — {peak_date.strftime('%Y-%m')} (Peak Stress Period)<br>"
                  f"<sup>Node size ∝ stress | Color = stress intensity | Edge width ∝ structural weight</sup>",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white", height=600,
            font=dict(family="Arial", size=12),
            margin=dict(l=20, r=80, t=80, b=20),
        )
    )
    fig.write_html(str(FIGURES / "fig5_network_snapshot.html"))
    print("  ✓ fig5_network_snapshot.html")


def fig6_stress_decomposition():
    """Figure 6: Stress decomposition for top-stress sectors."""
    ser = load("sector_ser_panel")
    if ser.empty:
        return

    sectors = ["energy", "food_away_from_home", "shelter", "transportation_services"]
    decomp_vars = [
        ("stress_price",      "Price Pressure",    "#EF5350"),
        ("stress_volatility", "Volatility",         "#FF7043"),
        ("stress_efficiency", "Efficiency Loss",    "#FFA726"),
        ("stress_covid",      "COVID Pressure",     "#AB47BC"),
        ("stress_shortage",   "Shortage Proxy",     "#5C6BC0"),
    ]

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        subplot_titles=sectors, vertical_spacing=0.10, horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for sec, (r, c) in zip(sectors, positions):
        for var, label, color in decomp_vars:
            col = f"{sec}_{var}"
            if col not in ser.columns:
                continue
            s = ser[col]
            fig.add_trace(go.Bar(
                x=s.index, y=s.values, name=label,
                marker_color=color,
                showlegend=(r == 1 and c == 1),
            ), row=r, col=c)

    fig.update_layout(
        barmode="stack",
        title="Composite Stress Decomposition by Sector",
        template="plotly_white", height=650,
        legend=dict(orientation="h", y=-0.08),
        font=dict(family="Arial", size=11),
    )
    fig.write_html(str(FIGURES / "fig6_stress_decomposition.html"))
    print("  ✓ fig6_stress_decomposition.html")


def fig7_system_metrics():
    """Figure 7: System-level aggregate metrics over time."""
    ser = load("sector_ser_panel")
    prop = load("propagation_panel")
    covid = load("covid_monthly")
    sys_reg = load("system_regime")
    if ser.empty:
        return

    def mean_var(suffix):
        cols = [c for c in ser.columns if c.endswith(suffix)
                and "absorbed" not in c]
        return ser[cols].mean(axis=1) if cols else pd.Series(dtype=float)

    stress_mean = mean_var("_stress")
    leaked_mean = mean_var("_leaked_stress")
    elast_mean  = mean_var("_elasticity")
    react_mean  = mean_var("_reaction")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Mean Stress & Leaked Stress (system average)",
            "Mean Elasticity with E_crit threshold",
            "Effective Connectivity & Epidemic Pressure",
        ],
        vertical_spacing=0.08, row_heights=[0.38, 0.32, 0.30],
    )

    # Row 1: stress + leaked
    fig.add_trace(go.Scatter(x=stress_mean.index, y=stress_mean.values,
        name="Mean Stress", line=dict(color="#E53935", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=leaked_mean.index, y=leaked_mean.values,
        name="Mean Leaked Stress", fill="tozeroy",
        line=dict(color="#EF9A9A", width=1.5),
        fillcolor="rgba(239,154,154,0.3)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=react_mean.index, y=react_mean.values.clip(0),
        name="Mean Reaction", line=dict(color="#FF7043", dash="dot")), row=1, col=1)

    # Row 2: elasticity
    fig.add_trace(go.Scatter(x=elast_mean.index, y=elast_mean.values,
        name="Mean Elasticity", line=dict(color="#43A047", width=2.5)), row=2, col=1)
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="E_crit", row=2, col=1)

    # Row 3: connectivity + epidemic pressure
    if not prop.empty:
        conn_col = "_system_effective_connectivity"
        if conn_col in prop.columns:
            fig.add_trace(go.Scatter(x=prop.index, y=prop[conn_col].values,
                name="Effective Connectivity", line=dict(color="#1E88E5", width=2)),
                row=3, col=1)
    if not covid.empty and "epidemic_pressure" in covid.columns:
        fig.add_trace(go.Scatter(x=covid.index, y=covid["epidemic_pressure"].values,
            name="Epidemic Pressure", fill="tozeroy",
            line=dict(color="#EF5350", width=1),
            fillcolor="rgba(239,83,80,0.2)"), row=3, col=1)

    # Add system regime color bands to row 1
    if not sys_reg.empty:
        col_name = sys_reg.columns[0]
        for t in range(len(sys_reg)):
            r = sys_reg[col_name].iloc[t]
            clr = REGIME_COLORS.get(r, "#BDBDBD")
            x0 = sys_reg.index[t]
            x1 = sys_reg.index[t+1] if t+1 < len(sys_reg) else x0 + pd.DateOffset(months=1)
            fig.add_vrect(x0=x0, x1=x1, fillcolor=clr, opacity=0.08,
                         line_width=0, row=1, col=1)

    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_yaxes(range=[0, 1.05], row=3, col=1)
    fig.update_layout(
        title="System-Level Aggregate Metrics with Regime Background Shading",
        template="plotly_white", height=750,
        legend=dict(orientation="h", y=-0.05),
        font=dict(family="Arial", size=12),
    )
    fig.write_html(str(FIGURES / "fig7_system_metrics.html"))
    print("  ✓ fig7_system_metrics.html")


def fig8_retention_spillover():
    """Figure 8: Retention vs spillover scatter by sector."""
    ser = load("sector_ser_panel")
    summary = load("sector_summary")
    if ser.empty:
        return

    rows = []
    for col in ser.columns:
        if col.endswith("_retention"):
            sec = col.replace("_retention", "")
            sp_col = f"{sec}_spillover_share"
            st_col = f"{sec}_stress"
            rows.append({
                "sector": sec,
                "retention": ser[col].mean(),
                "spillover":  ser[sp_col].mean() if sp_col in ser.columns else 0,
                "peak_stress": ser[st_col].max() if st_col in ser.columns else 0,
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="retention", y="spillover",
        size="peak_stress", size_max=50,
        text="sector", color="peak_stress",
        color_continuous_scale="YlOrRd",
        labels={
            "retention": "Mean Retention (absorbed/total stress)",
            "spillover": "Mean Spillover Share (leaked/total stress)",
            "peak_stress": "Peak Stress",
        },
        title="Sector Retention vs Spillover Share<br>"
              "<sup>Bubble size = peak stress | Color = peak stress intensity</sup>",
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(template="plotly_white", height=550,
                     font=dict(family="Arial", size=12))
    fig.write_html(str(FIGURES / "fig8_retention_spillover.html"))
    print("  ✓ fig8_retention_spillover.html")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_epidemic_overview()
    fig2_sector_stress_panel()
    fig3_elasticity_regime()
    fig4_regime_heatmap()
    fig5_network_snapshot()
    fig6_stress_decomposition()
    fig7_system_metrics()
    fig8_retention_spillover()
    print(f"\nAll figures saved to {FIGURES}/")
