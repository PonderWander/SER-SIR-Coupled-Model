"""
scripts/export_paper_figures.py
─────────────────
Standalone export of Figures 1–18 for the SER / leakage decomposition paper.
Reads from outputs/parquet/ and outputs/sweep/.
Writes HTML (interactive) + PNG (static) to outputs/paper_figures/.

Usage:
    python scripts/export_paper_figures.py
    python scripts/export_paper_figures.py --fmt html        # HTML only
    python scripts/export_paper_figures.py --fmt png         # PNG only
    python scripts/export_paper_figures.py --figs 1,5,6,7   # specific figures only
"""

import argparse, warnings, sys, os
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
PARQ    = ROOT / "outputs" / "parquet"
SWEEP   = ROOT / "outputs" / "sweep"
OUT     = ROOT / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Design system ─────────────────────────────────────────────────────────────
HH_COLOR  = "#1D9E75"   # household dominant — teal
MK_COLOR  = "#D85A30"   # market dominant — coral
BD_COLOR  = "#374151"   # boundary line — dark
BND_COLOR = "rgba(200,160,0,0.15)"  # switching band fill
GRID_COL  = "rgba(0,0,0,0.06)"
FONT      = "Inter, system-ui, sans-serif"
TEMPLATE  = "plotly_white"
W, H      = 900, 520    # default figure dimensions

SECTORS = [
    "food_at_home", "food_away_from_home", "energy", "gasoline",
    "electricity", "shelter", "transportation_services",
    "household_goods", "medical_services", "apparel",
]
SEC_LABELS = {s: s.replace("_"," ").title() for s in SECTORS}
PALETTE    = px.colors.qualitative.Set2

BASE_LAYOUT = dict(
    template=TEMPLATE, font=dict(family=FONT, size=12),
    margin=dict(l=60, r=40, t=60, b=50),
    legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)", borderwidth=0.5),
)

# ── Data loaders ──────────────────────────────────────────────────────────────
def load():
    d = {}
    d["ser"]   = pd.read_parquet(PARQ/"sector_ser_panel.parquet")
    d["sir"]   = pd.read_parquet(PARQ/"sir_timeseries.parquet")
    d["covid"] = pd.read_parquet(PARQ/"covid_monthly.parquet")
    d["C"]     = pd.read_parquet(PARQ/"connectivity_C_history.parquet")
    d["ld"]    = pd.read_parquet(PARQ/"leakage_decomposition.parquet")
    d["hls"]   = pd.read_csv(PARQ/"household_leakage_summary.csv")
    d["lsw"]   = pd.read_parquet(PARQ/"leakage_sweep.parquet")
    d["sw"]    = pd.read_csv(SWEEP/"sweep_results_full.csv")
    d["sw_sys"]= pd.read_csv(PARQ/"sweep_system_analysis.csv") if (PARQ/"sweep_system_analysis.csv").exists() else pd.DataFrame()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1  SIR epidemic curve + sector stress forcing overlays
# ─────────────────────────────────────────────────────────────────────────────

def fig1(d, fmt):
    """SIR epidemic curve with sector stress forcing overlays."""
    sir   = d["sir"]
    ser   = d["ser"]
    covid = d["covid"]

    # Resample SIR to monthly to align with sector panel
    sir_m = sir.resample("ME").mean()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "SIR Epidemic Dynamics",
            "Epidemic Pressure & Stringency",
            "Sector Stress Forcing Overlays",
        ],
        vertical_spacing=0.08,
        row_heights=[0.35, 0.25, 0.40],
    )

    # ── Row 1: S, I, R curves ────────────────────────────────────────────────
    sir_colors = {"S": "#3B82F6", "I": "#EF4444", "R": "#10B981"}
    sir_labels = {"S": "Susceptible (S)", "I": "Infectious (I)", "R": "Recovered (R)"}
    for col, color in sir_colors.items():
        if col in sir_m.columns:
            fig.add_trace(go.Scatter(
                x=sir_m.index, y=sir_m[col],
                name=sir_labels[col],
                line=dict(color=color, width=2),
                hovertemplate=f"{sir_labels[col]}<br>%{{x|%Y-%m}}: %{{y:.3f}}<extra></extra>",
            ), row=1, col=1)

    # ── Row 2: Epidemic pressure + stringency ────────────────────────────────
    # Use monthly covid data if available, else resample daily SIR
    if "epidemic_pressure" in covid.columns:
        ep_idx = covid.index
        ep_val = covid["epidemic_pressure"].values
    elif "pressure" in sir_m.columns:
        ep_idx = sir_m.index
        ep_val = sir_m["pressure"].values
    else:
        ep_idx, ep_val = sir_m.index, sir_m["I"].values

    fig.add_trace(go.Scatter(
        x=ep_idx, y=ep_val,
        name="Epidemic pressure P_t",
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="#EF4444", width=1.8),
        hovertemplate="Pressure: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    if "stringency_norm" in covid.columns:
        fig.add_trace(go.Scatter(
            x=covid.index, y=covid["stringency_norm"],
            name="Stringency index",
            line=dict(color="#8B5CF6", width=1.4, dash="dash"),
            hovertemplate="Stringency: %{y:.3f}<extra></extra>",
        ), row=2, col=1)
    elif "stringency_index" in sir_m.columns:
        fig.add_trace(go.Scatter(
            x=sir_m.index, y=sir_m["stringency_index"] / 100,
            name="Stringency index (norm)",
            line=dict(color="#8B5CF6", width=1.4, dash="dash"),
        ), row=2, col=1)

    # ── Row 3: Sector stress overlays ────────────────────────────────────────
    # Show 5 representative sectors — spread across sensitivity spectrum
    overlay_sectors = [
        "shelter", "food_away_from_home", "energy",
        "medical_services", "transportation_services",
    ]
    for i, sec in enumerate(overlay_sectors):
        s = ser_sector(ser, sec, "stress")
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=SEC_LABELS.get(sec, sec),
            line=dict(color=PALETTE[i], width=1.6),
            opacity=0.85,
            hovertemplate=f"{SEC_LABELS.get(sec,sec)}<br>%{{x|%Y-%m}}: %{{y:.3f}}<extra></extra>",
        ), row=3, col=1)

    # Shade peak pressure period (top quartile of epidemic pressure)
    if len(ep_val) > 4:
        thresh = np.percentile(ep_val[~np.isnan(ep_val)], 75)
        in_peak = ep_val >= thresh
        starts = [i for i in range(1, len(in_peak)) if in_peak[i] and not in_peak[i-1]]
        ends   = [i for i in range(1, len(in_peak)) if not in_peak[i] and in_peak[i-1]]
        if in_peak[0]: starts = [0] + starts
        if in_peak[-1]: ends = ends + [len(in_peak)-1]
        for s_i, e_i in zip(starts[:3], ends[:3]):   # cap at 3 bands
            x0 = ep_idx[s_i] if s_i < len(ep_idx) else ep_idx[-1]
            x1 = ep_idx[e_i] if e_i < len(ep_idx) else ep_idx[-1]
            for row in [1, 3]:
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor="rgba(239,68,68,0.07)",
                    line=dict(width=0),
                    row=row, col=1,
                )

    fig.update_layout(
        title=dict(
            text="Figure 1. SIR Epidemic Curve and Sector Stress Forcing Overlays",
            font=dict(size=14),
        ),
        legend=dict(
            orientation="v", x=1.01, y=1,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=0.5,
        ),
        height=680, width=W,
    )
    fig.update_yaxes(title_text="Compartment fraction", row=1, col=1)
    fig.update_yaxes(title_text="Normalised index",     row=2, col=1)
    fig.update_yaxes(title_text="Composite stress S",   row=3, col=1,
                     range=[0, 1])
    fig.update_xaxes(title_text="Date", row=3, col=1)

    save(fig, "figure_01_sir_and_sector_forcing", fmt)

def ser_sector(ser, sec, var):
    col = f"{sec}_{var}"
    return ser[col] if col in ser.columns else pd.Series(dtype=float)

def save(fig, name, fmt):
    fig.update_layout(**BASE_LAYOUT)
    if fmt in ("html","both"):
        fig.write_html(str(OUT/f"{name}.html"), include_plotlyjs="cdn")
        print(f"  ✓  {name}.html")
    if fmt in ("png","both"):
        try:
            fig.write_image(str(OUT/f"{name}.png"), width=W*2, height=H*2, scale=1)
            print(f"  ✓  {name}.png")
        except Exception as e:
            print(f"  ⚠  PNG failed ({e}) — HTML only")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 2–4  Structural / conceptual (SVG-based, rendered as HTML)
# ─────────────────────────────────────────────────────────────────────────────

def _arrow_trace(x0, y0, x1, y1, color="#9CA3AF"):
    """Return a scatter trace that draws a line+arrowhead between two paper-coord points."""
    dx, dy = x1-x0, y1-y0
    L = (dx**2+dy**2)**0.5 or 1e-9
    ux, uy = dx/L, dy/L
    hw, hl = 0.012, 0.022   # arrowhead half-width, length in paper coords
    # arrowhead tip at (x1,y1); base at tip - hl*(ux,uy)
    bx, by = x1-hl*ux, y1-hl*uy
    px, py = -uy*hw, ux*hw   # perpendicular
    arrow_x = [x0, x1-hl*ux, None, bx+px, x1, bx-px, bx+px]
    arrow_y = [y0, y1-hl*uy, None, by+py, y1, by-py, by+py]
    return go.Scatter(x=arrow_x, y=arrow_y, mode="lines",
                      line=dict(color=color, width=1.5),
                      showlegend=False, hoverinfo="skip")


def fig2(fmt):
    """Conceptual layering diagram — uses data coords, no paper axref."""
    layers = [
        ("Epidemic Forcing",      "SIR model → time-varying pressure P_t",                       "#3B82F6", 9),
        ("Sector Stress",         "S_{i,t} = weighted composite of price, efficiency, epidemic",  "#8B5CF6", 7),
        ("SER Dynamics",          "E_{i,t} evolves via buffering, decay, leak drain",             "#10B981", 5),
        ("Network Transmission",  "W_{ij,t} = W⁰ × C_{ij,t}  propagates leaked stress",         "#F59E0B", 3),
        ("Leakage Decomposition", "L_total = L_market + L_household",                             "#EF4444", 1),
    ]
    fig = go.Figure()
    for label, desc, color, y in layers:
        # Box
        fig.add_shape(type="rect", x0=0.5, x1=9.5, y0=y-0.7, y1=y+0.7,
                      fillcolor=color, opacity=0.13, line=dict(color=color, width=1.5))
        fig.add_trace(go.Scatter(
            x=[5], y=[y+0.22], mode="text",
            text=[f"<b>{label}</b>"], textfont=dict(size=13, color=color),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[5], y=[y-0.22], mode="text",
            text=[desc], textfont=dict(size=10, color="#6B7280"),
            showlegend=False, hoverinfo="skip"))
    # Arrows between layers
    for y_top, y_bot in [(7.7,7.3),(5.7,5.3),(3.7,3.3),(1.7,1.3)]:
        fig.add_trace(_arrow_trace(5, y_top, 5, y_bot))
    fig.update_layout(
        title="Figure 2. Conceptual Layering of the SER–SIR Model",
        xaxis=dict(visible=False, range=[0,10]),
        yaxis=dict(visible=False, range=[0,10.5]),
        height=500, width=W, paper_bgcolor="white", plot_bgcolor="white",
    )
    save(fig, "figure_02_conceptual_layering", fmt)


def fig3(fmt):
    """Full system diagram — uses data coords throughout."""
    # (x, y) in data space [0,10]×[0,10]
    nodes = {
        "COVID Data":        (1.0, 9.0),
        "SIR Model":         (1.0, 7.0),
        "Pressure P_t":      (1.0, 5.0),
        "Sector Panel":      (4.0, 7.0),
        "SER Engine":        (4.0, 5.0),
        "S / E / L_market":  (4.0, 3.0),
        "Connectivity C":    (7.0, 7.0),
        "Propagation W·L":   (7.0, 5.0),
        "L_household":       (7.0, 3.0),
        "Regime Classifier": (9.5, 5.0),
        "Outputs":           (9.5, 3.0),
    }
    edges = [
        ("COVID Data","SIR Model"),("SIR Model","Pressure P_t"),
        ("Pressure P_t","SER Engine"),("Sector Panel","SER Engine"),
        ("SER Engine","S / E / L_market"),
        ("S / E / L_market","Connectivity C"),
        ("S / E / L_market","Propagation W·L"),
        ("Connectivity C","Propagation W·L"),
        ("Propagation W·L","L_household"),
        ("S / E / L_market","Regime Classifier"),
        ("L_household","Regime Classifier"),
        ("Regime Classifier","Outputs"),
        ("Propagation W·L","Outputs"),
    ]
    node_colors = {
        "COVID Data":"#3B82F6","SIR Model":"#3B82F6","Pressure P_t":"#3B82F6",
        "Sector Panel":"#8B5CF6","SER Engine":"#8B5CF6","S / E / L_market":"#8B5CF6",
        "Connectivity C":"#F59E0B","Propagation W·L":"#F59E0B",
        "L_household":"#10B981","Regime Classifier":"#EF4444","Outputs":"#6B7280",
    }
    fig = go.Figure()
    # Edge lines
    for a,b in edges:
        x0,y0=nodes[a]; x1,y1=nodes[b]
        # offset endpoints to box edges (approx 0.7 units)
        dx,dy=x1-x0,y1-y0; L=(dx**2+dy**2)**0.5 or 1
        sx,sy=dx/L*0.75, dy/L*0.75
        fig.add_trace(_arrow_trace(x0+sx,y0+sy,x1-sx,y1-sy))
    # Node boxes + labels
    bw, bh = 1.2, 0.5
    for label,(x,y) in nodes.items():
        c = node_colors.get(label,"#6B7280")
        fig.add_shape(type="rect",x0=x-bw,x1=x+bw,y0=y-bh,y1=y+bh,
                      fillcolor=c,opacity=0.15,line=dict(color=c,width=1.5))
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="text",
            text=[f"<b>{label}</b>"],
            textfont=dict(size=9,color=c),
            showlegend=False,hoverinfo="skip"))
    fig.update_layout(
        title="Figure 3. Full System Diagram — Coupled SER–SIR Architecture",
        xaxis=dict(visible=False,range=[-0.5,11.5]),
        yaxis=dict(visible=False,range=[1.5,10.5]),
        height=520,width=W,paper_bgcolor="white",plot_bgcolor="white",
    )
    save(fig, "figure_03_system_diagram", fmt)


def fig4(fmt):
    """Data and simulation pipeline — horizontal flow."""
    steps = [
        ("Epidemic Data\n(OWID/FRED)", "#3B82F6"),
        ("SIR Process\nS, I, R, P_t",  "#3B82F6"),
        ("Sector Mapping\nBEA IO + OI", "#8B5CF6"),
        ("SER Dynamics\nS, E, L_mkt",  "#8B5CF6"),
        ("Network\nTransmission",       "#F59E0B"),
        ("Leakage\nDecomp.",            "#10B981"),
        ("Regime\nClassification",      "#EF4444"),
        ("Outputs\nParquet/HTML",       "#6B7280"),
    ]
    n = len(steps)
    xs = [1 + i*(10/(n-1)) for i in range(n)]
    fig = go.Figure()
    for i,(label,color) in enumerate(steps):
        fig.add_shape(type="rect",x0=xs[i]-0.8,x1=xs[i]+0.8,y0=3.5,y1=6.5,
                      fillcolor=color,opacity=0.15,line=dict(color=color,width=1.5))
        fig.add_trace(go.Scatter(x=[xs[i]],y=[5],mode="text",
            text=[label.replace("\n","<br>")],
            textfont=dict(size=9,color=color),
            showlegend=False,hoverinfo="skip"))
        if i < n-1:
            fig.add_trace(_arrow_trace(xs[i]+0.82,5,xs[i+1]-0.82,5))
    fig.update_layout(
        title="Figure 4. Data and Simulation Pipeline",
        xaxis=dict(visible=False,range=[0,12]),
        yaxis=dict(visible=False,range=[2,8]),
        height=260,width=W,paper_bgcolor="white",plot_bgcolor="white",
    )
    save(fig, "figure_04_pipeline", fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 5–8  Time series from SER panel
# ─────────────────────────────────────────────────────────────────────────────

REP_SECTORS = ["food_at_home","energy","shelter","transportation_services","medical_services"]

def fig5(d, fmt):
    """Sectoral stress trajectories."""
    ser = d["ser"]
    fig = go.Figure()
    for i,sec in enumerate(REP_SECTORS):
        s = ser_sector(ser, sec, "stress")
        if s.empty: continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=SEC_LABELS[sec],
            line=dict(color=PALETTE[i], width=1.8),
            hovertemplate=f"{SEC_LABELS[sec]}<br>%{{x|%Y-%m}}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Figure 5. Sectoral Stress Trajectories Under Epidemic Forcing",
        xaxis_title="Date", yaxis_title="Composite Stress S_{i,t}",
        yaxis=dict(range=[0,1]), height=H, width=W,
    )
    save(fig, "figure_05_stress_trajectories", fmt)


def fig6(d, fmt):
    """Elasticity trajectories."""
    ser = d["ser"]
    fig = go.Figure()
    for i,sec in enumerate(REP_SECTORS):
        e = ser_sector(ser, sec, "elasticity")
        if e.empty: continue
        fig.add_trace(go.Scatter(
            x=e.index, y=e.values, name=SEC_LABELS[sec],
            line=dict(color=PALETTE[i], width=1.8),
            hovertemplate=f"{SEC_LABELS[sec]}<br>%{{x|%Y-%m}}: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0.48, line=dict(color="red", dash="dash", width=1),
                  annotation_text="E* = 0.48 threshold", annotation_position="top right")
    fig.update_layout(
        title="Figure 6. Elasticity Trajectories — Depletion and Partial Recovery",
        xaxis_title="Date", yaxis_title="Elasticity E_{i,t}",
        yaxis=dict(range=[0,1.1]), height=H, width=W,
    )
    save(fig, "figure_06_elasticity_trajectories", fmt)


def fig7(d, fmt):
    """Connectivity evolution — selected edges."""
    C = d["C"]
    # Pick 5 most variable edges
    top_edges = C.std().sort_values(ascending=False).head(5).index.tolist()
    fig = go.Figure()
    for i,col in enumerate(top_edges):
        label = col.replace("_"," ").replace("  "," → ")
        fig.add_trace(go.Scatter(
            x=C.index, y=C[col].values, name=label,
            line=dict(color=PALETTE[i], width=1.8),
        ))
    fig.update_layout(
        title="Figure 7. Connectivity Evolution — Selected Edge Histories C_{ij,t}",
        xaxis_title="Date", yaxis_title="Dynamic Connectivity C_{ij,t}",
        yaxis=dict(range=[0,1]), height=H, width=W,
    )
    save(fig, "figure_07_connectivity_evolution", fmt)


def fig8(d, fmt):
    """Leakage decomposition — stacked time series."""
    ld = d["ld"]
    fig = make_subplots(rows=2, cols=3,
        subplot_titles=[SEC_LABELS[s] for s in REP_SECTORS[:5] if s in ld.sector.unique()],
        shared_yaxes=True, vertical_spacing=0.14, horizontal_spacing=0.06)
    secs_avail = [s for s in REP_SECTORS if s in ld.sector.unique()][:5]
    for pos,sec in enumerate(secs_avail):
        r,c = divmod(pos,3); r+=1; c+=1
        sub = ld[ld.sector==sec].sort_values("date")
        dates = pd.to_datetime(sub["date"])
        fig.add_trace(go.Scatter(
            x=dates, y=sub["L_market"], name="L_market",
            fill="tozeroy", fillcolor="rgba(216,90,48,0.25)",
            line=dict(color=MK_COLOR, width=1.2),
            showlegend=(pos==0),
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=dates, y=sub["L_household"], name="L_household",
            fill="tozeroy", fillcolor="rgba(29,158,117,0.25)",
            line=dict(color=HH_COLOR, width=1.2),
            showlegend=(pos==0),
        ), row=r, col=c)
    fig.update_layout(
        title="Figure 8. Leakage Decomposition — L_market and L_household Over Time",
        height=520, width=W,
    )
    save(fig, "figure_08_leakage_decomposition", fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 9–11  Distributional / breakpoint
# ─────────────────────────────────────────────────────────────────────────────

def fig9(d, fmt):
    """Histograms + KDE of L_household per sector."""
    ld  = d["ld"]
    hls = d["hls"]
    secs_avail = [s for s in SECTORS if s in ld.sector.unique()][:6]
    fig = make_subplots(rows=2, cols=3,
        subplot_titles=[SEC_LABELS[s] for s in secs_avail],
        vertical_spacing=0.14, horizontal_spacing=0.08)
    for pos,sec in enumerate(secs_avail):
        r,c = divmod(pos,3); r+=1; c+=1
        vals = ld[ld.sector==sec]["L_household"].values
        vals = vals[~np.isnan(vals)]
        # Histogram
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=20,
            marker_color=HH_COLOR, opacity=0.5,
            histnorm="probability density",
            showlegend=False,
        ), row=r, col=c)
        # KDE
        if vals.std() > 1e-6:
            kde = gaussian_kde(vals, bw_method=0.3)
            xr  = np.linspace(0, vals.max()+0.01, 100)
            fig.add_trace(go.Scatter(
                x=xr, y=kde(xr),
                line=dict(color=MK_COLOR, width=1.8),
                showlegend=False,
            ), row=r, col=c)
        # Stats annotation
        row_hls = hls[hls.sector==sec]
        if not row_hls.empty:
            sk = row_hls["skew"].values[0]
            zi = "ZI" if row_hls["zero_inflated"].values[0] else ""
            fig.add_annotation(
                x=0.95, y=0.95, xref=f"x{pos+1 if pos>0 else ''} domain",
                yref=f"y{pos+1 if pos>0 else ''} domain",
                text=f"skew={sk:.2f} {zi}",
                font=dict(size=9), showarrow=False, align="right",
            )
    fig.update_layout(
        title="Figure 9. Distribution of L_household — Histograms and KDE",
        height=520, width=W,
    )
    save(fig, "figure_09_lhh_distribution", fmt)


def fig10(d, fmt):
    """Rolling distribution diagnostics."""
    ld = d["ld"]
    rep = [s for s in ["food_at_home","energy","transportation_services","shelter"]
           if s in ld.sector.unique()]
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[SEC_LABELS[s] for s in rep],
        vertical_spacing=0.14, horizontal_spacing=0.08)
    win = 6
    for pos,sec in enumerate(rep):
        r,c = divmod(pos,2); r+=1; c+=1
        vals = ld[ld.sector==sec].sort_values("date")["L_household"].values
        # Rolling stats
        rm, rs, rsk = [],[],[]
        for i in range(win, len(vals)+1):
            seg = vals[i-win:i]
            rm.append(np.mean(seg))
            rs.append(np.std(seg))
            from scipy.stats import skew as scipy_skew
            rsk.append(scipy_skew(seg))
        x = list(range(win, len(vals)+1))
        fig.add_trace(go.Scatter(x=x,y=rm,name="Mean",
            line=dict(color=HH_COLOR,width=1.5),showlegend=(pos==0)), row=r,col=c)
        fig.add_trace(go.Scatter(x=x,y=rs,name="Std",
            line=dict(color=MK_COLOR,width=1.5,dash="dash"),showlegend=(pos==0)), row=r,col=c)
        fig.add_trace(go.Scatter(x=x,y=[abs(v)*0.05 for v in rsk],name="|Skew|×0.05",
            line=dict(color="#8B5CF6",width=1.2,dash="dot"),showlegend=(pos==0)), row=r,col=c)
    fig.update_layout(
        title="Figure 10. Rolling Distribution Diagnostics — L_household (6-period window)",
        height=500, width=W,
    )
    save(fig, "figure_10_rolling_diagnostics", fmt)


def fig11(d, fmt):
    """Breakpoint and activation analysis."""
    ld  = d["ld"]
    hls = d["hls"]
    rep = [s for s in ["food_at_home","transportation_services","electricity","apparel"]
           if s in ld.sector.unique()]
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[SEC_LABELS[s] for s in rep],
        vertical_spacing=0.14, horizontal_spacing=0.08)
    thresh = 0.005
    for pos,sec in enumerate(rep):
        r,c = divmod(pos,2); r+=1; c+=1
        sub  = ld[ld.sector==sec].sort_values("date")
        dates = pd.to_datetime(sub["date"])
        vals  = sub["L_household"].values
        fig.add_trace(go.Scatter(
            x=dates, y=vals, name=SEC_LABELS[sec],
            line=dict(color=HH_COLOR,width=1.5),
            fill="tozeroy", fillcolor="rgba(29,158,117,0.15)",
            showlegend=False,
        ), row=r, col=c)
        # Activation episodes
        in_ep = vals > thresh
        starts = [i for i in range(1,len(in_ep)) if in_ep[i] and not in_ep[i-1]]
        for s_idx in starts:
            fig.add_vline(x=dates.iloc[s_idx], line=dict(color="orange",width=1,dash="dot"),
                          row=r, col=c)
        # CUSUM changepoints from summary
        row_hls = hls[hls.sector==sec]
        if not row_hls.empty:
            n_bp = row_hls["n_changepoints"].values[0]
            fig.add_annotation(
                x=dates.iloc[0], y=vals.max()*0.9,
                text=f"{n_bp} breakpoints<br>{len(starts)} activations",
                font=dict(size=9), showarrow=False, align="left",
                xref=f"x{pos+1 if pos>0 else ''}",
                yref=f"y{pos+1 if pos>0 else ''}",
            )
    fig.update_layout(
        title="Figure 11. Breakpoint and Activation Analysis — L_household Episodes",
        height=500, width=W,
    )
    save(fig, "figure_11_breakpoint_activation", fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 12–14  Sweep results
# ─────────────────────────────────────────────────────────────────────────────

def fig12(d, fmt):
    """Correlation distribution across runs."""
    lsw = d["lsw"]
    corrs = lsw["corr_Lm_Lh"].values
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=corrs, nbinsx=30,
        marker_color=HH_COLOR, opacity=0.7,
        name="corr(L_market, L_household)",
    ))
    fig.add_vline(x=corrs.mean(), line=dict(color=MK_COLOR, dash="dash", width=2),
                  annotation_text=f"mean={corrs.mean():.3f}", annotation_position="top right")
    fig.add_vline(x=0, line=dict(color="black", width=1),
                  annotation_text="zero", annotation_position="top left")
    fig.update_layout(
        title="Figure 12. Correlation Distribution — corr(L_market, L_household) Across All Sector-Runs",
        xaxis_title="Correlation", yaxis_title="Count",
        height=H, width=W,
    )
    save(fig, "figure_12_correlation_distribution", fmt)


def fig13(d, fmt):
    """Parameter sweep heatmaps."""
    lsw = d["lsw"]
    sys_df = lsw.groupby("run_id").agg(
        Lm_mean=("Lm_mean","mean"), Lh_mean=("Lh_mean","mean"),
        corr=("corr_Lm_Lh","mean"), frac_hh=("frac_hh_gt_mk","mean"),
        delta=("delta","first"), w_covid=("w_covid","first"),
        alpha=("alpha","first"), chi=("chi","first"),
        S_mean=("S_mean","mean"), E_mean=("E_mean","mean"),
    ).reset_index()
    sys_df["hh_dominant"] = (sys_df["Lh_mean"] > sys_df["Lm_mean"]).astype(int)
    sys_df["delta_bin"]   = pd.cut(sys_df["delta"],   bins=4, labels=["δ Q1","δ Q2","δ Q3","δ Q4"])
    sys_df["wcovid_bin"]  = pd.cut(sys_df["w_covid"], bins=4, labels=["w Q1","w Q2","w Q3","w Q4"])
    sys_df["chi_bin"]     = pd.cut(sys_df["chi"],     bins=4, labels=["χ Q1","χ Q2","χ Q3","χ Q4"])
    sys_df["alpha_bin"]   = pd.cut(sys_df["alpha"],   bins=4, labels=["α Q1","α Q2","α Q3","α Q4"])

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["frac(Lh>Lm) by w_covid × delta",
                        "corr(Lm,Lh) by chi × alpha"],
        horizontal_spacing=0.12)

    pv1 = sys_df.groupby(["wcovid_bin","delta_bin"])["frac_hh"].mean().unstack()
    fig.add_trace(go.Heatmap(
        z=pv1.values, x=list(pv1.columns.astype(str)), y=list(pv1.index.astype(str)),
        colorscale=[[0,MK_COLOR],[0.5,"#FEF3C7"],[1,HH_COLOR]],
        zmid=0.5, showscale=True,
        colorbar=dict(title="frac(Lh>Lm)", x=0.44, len=0.9),
        text=np.round(pv1.values,2), texttemplate="%{text}",
    ), row=1, col=1)

    pv2 = sys_df.groupby(["chi_bin","alpha_bin"])["corr"].mean().unstack()
    fig.add_trace(go.Heatmap(
        z=pv2.values, x=list(pv2.columns.astype(str)), y=list(pv2.index.astype(str)),
        colorscale=[[0,"#1E40AF"],[0.5,"#93C5FD"],[1,"#DBEAFE"]],
        showscale=True,
        colorbar=dict(title="corr(Lm,Lh)", x=1.01, len=0.9),
        text=np.round(pv2.values,3), texttemplate="%{text}",
    ), row=1, col=2)

    fig.update_layout(
        title="Figure 13. Parameter Sweep Heatmaps — Channel Dominance Across Parameter Space",
        height=420, width=W,
    )
    save(fig, "figure_13_sweep_heatmaps", fmt)


def fig14(d, fmt):
    """Stress-based regime distribution."""
    lsw = d["lsw"]
    sys_df = lsw.groupby("run_id").agg(
        Lm_mean=("Lm_mean","mean"), Lh_mean=("Lh_mean","mean"),
        frac_hh=("frac_hh_gt_mk","mean"), S_mean=("S_mean","mean"),
    ).reset_index()
    sys_df["S_q"] = pd.qcut(sys_df["S_mean"], 4,
                             labels=["Q1 Low","Q2","Q3","Q4 High"])
    grp = sys_df.groupby("S_q")["frac_hh"].mean().reset_index()

    fig = go.Figure()
    colors = [HH_COLOR if v > 0.5 else MK_COLOR for v in grp["frac_hh"]]
    fig.add_trace(go.Bar(
        x=grp["S_q"].astype(str), y=grp["frac_hh"],
        marker_color=colors, opacity=0.8,
        text=[f"{v:.2f}" for v in grp["frac_hh"]],
        textposition="outside",
    ))
    fig.add_hline(y=0.5, line=dict(color="black", dash="dash", width=1),
                  annotation_text="Equal dominance", annotation_position="top right")
    fig.update_layout(
        title="Figure 14. Channel Dominance as a Function of Mean Stress Quartile",
        xaxis_title="Stress Quartile", yaxis_title="frac(L_household > L_market)",
        yaxis=dict(range=[0,1]), height=H, width=W,
    )
    save(fig, "figure_14_stress_regime_distribution", fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 15–17  Phase diagram / boundary
# ─────────────────────────────────────────────────────────────────────────────

def _fit_lr(lsw):
    sys_df = lsw.groupby("run_id").agg(
        Lm_mean=("Lm_mean","mean"), Lh_mean=("Lh_mean","mean"),
        S_mean=("S_mean","mean"), E_mean=("E_mean","mean"),
        delta=("delta","first"), w_covid=("w_covid","first"),
        alpha=("alpha","first"), chi=("chi","first"), rho_c=("rho_c","first"),
    ).reset_index()
    sys_df["hh_dominant"] = (sys_df["Lh_mean"] > sys_df["Lm_mean"]).astype(int)
    features = ["S_mean","E_mean","delta","w_covid","alpha","chi","rho_c"]
    X = sys_df[features].values
    y = sys_df["hh_dominant"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, random_state=42, max_iter=500)
    lr.fit(Xs, y)
    coef_raw = lr.coef_[0] / scaler.scale_
    bias_raw = lr.intercept_[0] - np.sum(lr.coef_[0]*scaler.mean_/scaler.scale_)
    coef_dict = dict(zip(features, coef_raw))
    return lr, scaler, sys_df, coef_dict, bias_raw, features


def _lr_p(s, e, coef_dict, bias_raw, sys_df, features):
    other = sum(coef_dict[f]*sys_df[f].mean() for f in features if f not in ["S_mean","E_mean"])
    z = coef_dict["S_mean"]*s + coef_dict["E_mean"]*e + other + bias_raw
    return 1/(1+np.exp(-z))


def fig15(d, fmt):
    """Stress–elasticity phase diagram with classifier boundary."""
    lsw = d["lsw"]
    lr, scaler, sys_df, coef_dict, bias_raw, features = _fit_lr(lsw)

    fig = go.Figure()
    # Points
    hh = sys_df[sys_df.hh_dominant==1]
    mk = sys_df[sys_df.hh_dominant==0]
    fig.add_trace(go.Scatter(
        x=hh["S_mean"], y=hh["E_mean"], mode="markers",
        name="Household dominant",
        marker=dict(color=HH_COLOR, size=7, opacity=0.75,
                    line=dict(width=0.5, color="white")),
        hovertemplate="Run %{customdata}<br>S=%{x:.3f} E=%{y:.3f}<extra>Household</extra>",
        customdata=hh["run_id"],
    ))
    fig.add_trace(go.Scatter(
        x=mk["S_mean"], y=mk["E_mean"], mode="markers",
        name="Market dominant",
        marker=dict(color=MK_COLOR, size=7, opacity=0.75,
                    line=dict(width=0.5, color="white")),
        hovertemplate="Run %{customdata}<br>S=%{x:.3f} E=%{y:.3f}<extra>Market</extra>",
        customdata=mk["run_id"],
    ))
    # LR boundary
    s_arr = np.linspace(0.18, 0.35, 80)
    bd_e  = []
    for s in s_arr:
        lo,hi = 0.1,0.95
        for _ in range(25):
            mid=(lo+hi)/2
            (_lr_p(s,mid,coef_dict,bias_raw,sys_df,features)>0.5 and (hi:=mid) or (lo:=mid))
        bd_e.append((lo+hi)/2)
    fig.add_trace(go.Scatter(
        x=s_arr, y=bd_e, mode="lines", name="LR boundary (p=0.5)",
        line=dict(color=BD_COLOR, width=2.2),
    ))
    # Switching band
    bd_up = []; bd_lo = []
    for s in s_arr:
        lo,hi=0.1,0.95
        for _ in range(25):
            mid=(lo+hi)/2
            (_lr_p(s,mid,coef_dict,bias_raw,sys_df,features)>0.6 and (hi:=mid) or (lo:=mid))
        bd_up.append((lo+hi)/2)
        lo,hi=0.1,0.95
        for _ in range(25):
            mid=(lo+hi)/2
            (_lr_p(s,mid,coef_dict,bias_raw,sys_df,features)>0.4 and (hi:=mid) or (lo:=mid))
        bd_lo.append((lo+hi)/2)
    fig.add_trace(go.Scatter(
        x=np.concatenate([s_arr, s_arr[::-1]]),
        y=np.concatenate([bd_up, bd_lo[::-1]]),
        fill="toself", fillcolor=BND_COLOR,
        line=dict(color="rgba(0,0,0,0)"),
        name="Switching band (p∈0.4–0.6)",
    ))
    fig.update_layout(
        title="Figure 15. Stress–Elasticity Phase Diagram with Classifier Boundary",
        xaxis_title="S_mean (stress)", yaxis_title="E_mean (elasticity)",
        xaxis=dict(range=[0.17,0.36]), yaxis=dict(range=[0.08,0.96]),
        height=H, width=W,
    )
    save(fig, "figure_15_phase_diagram", fmt)


def fig16(d, fmt):
    """Logistic probability surface — contour."""
    lsw = d["lsw"]
    lr, scaler, sys_df, coef_dict, bias_raw, features = _fit_lr(lsw)

    s_arr = np.linspace(0.18, 0.35, 60)
    e_arr = np.linspace(0.10, 0.95, 60)
    SS, EE = np.meshgrid(s_arr, e_arr)
    PP = np.vectorize(lambda s,e: _lr_p(s,e,coef_dict,bias_raw,sys_df,features))(SS,EE)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=PP, x=s_arr, y=e_arr,
        colorscale=[[0,MK_COLOR],[0.5,"#FEF3C7"],[1,HH_COLOR]],
        zmid=0.5, zmin=0, zmax=1,
        contours=dict(showlabels=True, labelfont=dict(size=10)),
        colorbar=dict(title="P(household dominant)"),
    ))
    fig.add_trace(go.Contour(
        z=PP, x=s_arr, y=e_arr,
        contours=dict(start=0.5, end=0.5, size=1,
                      showlabels=True, labelfont=dict(size=11, color="black")),
        line=dict(color=BD_COLOR, width=2.5),
        showscale=False, name="p=0.5 boundary",
    ))
    fig.update_layout(
        title="Figure 16. Logistic Probability Surface — P(Household Dominant) in S–E Space",
        xaxis_title="S_mean (stress)", yaxis_title="E_mean (elasticity)",
        height=H, width=W,
    )
    save(fig, "figure_16_probability_surface", fmt)


def fig17(d, fmt):
    """Decision tree boundary overlay."""
    lsw = d["lsw"]
    lr, scaler, sys_df, coef_dict, bias_raw, features = _fit_lr(lsw)

    fig = go.Figure()
    # Background probability
    s_arr = np.linspace(0.17, 0.36, 50)
    e_arr = np.linspace(0.08, 0.96, 50)
    SS,EE = np.meshgrid(s_arr,e_arr)
    PP = np.vectorize(lambda s,e: _lr_p(s,e,coef_dict,bias_raw,sys_df,features))(SS,EE)
    fig.add_trace(go.Contour(
        z=PP, x=s_arr, y=e_arr,
        colorscale=[[0,MK_COLOR],[0.5,"#FEF9EC"],[1,HH_COLOR]],
        opacity=0.4, zmid=0.5, showscale=False,
        contours=dict(coloring="fill"),
    ))
    # Points
    hh = sys_df[sys_df.hh_dominant==1]
    mk = sys_df[sys_df.hh_dominant==0]
    fig.add_trace(go.Scatter(x=hh["S_mean"],y=hh["E_mean"],mode="markers",
        name="Household",marker=dict(color=HH_COLOR,size=6,opacity=0.7)))
    fig.add_trace(go.Scatter(x=mk["S_mean"],y=mk["E_mean"],mode="markers",
        name="Market",marker=dict(color=MK_COLOR,size=6,opacity=0.7)))
    # LR boundary
    s_line = np.linspace(0.17,0.36,80)
    bd_e=[]
    for s in s_line:
        lo,hi=0.08,0.96
        for _ in range(25):
            mid=(lo+hi)/2
            (_lr_p(s,mid,coef_dict,bias_raw,sys_df,features)>0.5 and (hi:=mid) or (lo:=mid))
        bd_e.append((lo+hi)/2)
    fig.add_trace(go.Scatter(x=s_line,y=bd_e,mode="lines",name="LR p=0.5",
        line=dict(color=BD_COLOR,width=2.5)))
    # Tree thresholds
    fig.add_hline(y=0.48, line=dict(color="#7C3AED",dash="dash",width=1.8),
                  annotation_text="Tree: E=0.48", annotation_position="right")
    fig.add_vline(x=0.27, line=dict(color="#7C3AED",dash="dot",width=1.5),
                  annotation_text="S=0.27", annotation_position="top")
    fig.add_vline(x=0.31, line=dict(color="#7C3AED",dash="dot",width=1.5),
                  annotation_text="S=0.31", annotation_position="top")
    # Critical point
    fig.add_trace(go.Scatter(x=[0.265],y=[0.521],mode="markers",
        name="S*≈0.265, E*≈0.521",
        marker=dict(symbol="star",size=14,color="#F59E0B",
                    line=dict(width=1.5,color="white"))))
    fig.update_layout(
        title="Figure 17. Decision Tree Threshold Overlay on Continuous LR Boundary",
        xaxis_title="S_mean (stress)", yaxis_title="E_mean (elasticity)",
        xaxis=dict(range=[0.17,0.36]), yaxis=dict(range=[0.08,0.96]),
        height=H, width=W,
    )
    save(fig, "figure_17_tree_boundary_overlay", fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 18  Snapback illustration
# ─────────────────────────────────────────────────────────────────────────────

def fig18(d, fmt):
    """Snapback — apparent stabilization then delayed market-channel activation."""
    ser = d["ser"]
    ld  = d["ld"]
    # Use shelter — highest L_market mean, clear snapback potential
    sec = "shelter"
    S   = ser_sector(ser, sec, "stress")
    E   = ser_sector(ser, sec, "elasticity")
    sub = ld[ld.sector==sec].sort_values("date").set_index("date")
    Lm  = sub["L_market"]  if "L_market"  in sub.columns else pd.Series(dtype=float)
    Lh  = sub["L_household"] if "L_household" in sub.columns else pd.Series(dtype=float)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Stress S_{i,t}", "Elasticity E_{i,t}",
                        "Leakage Channels"],
        vertical_spacing=0.08, row_heights=[0.3,0.3,0.4])

    fig.add_trace(go.Scatter(x=S.index,y=S.values,
        line=dict(color="#EF4444",width=1.8),showlegend=False), row=1,col=1)

    fig.add_trace(go.Scatter(x=E.index,y=E.values,
        line=dict(color="#3B82F6",width=1.8),showlegend=False), row=2,col=1)
    fig.add_hline(y=0.48,line=dict(color="#7C3AED",dash="dash",width=1),row=2,col=1)

    if not Lm.empty:
        fig.add_trace(go.Scatter(x=Lm.index,y=Lm.values,
            name="L_market",fill="tozeroy",
            fillcolor="rgba(216,90,48,0.2)",
            line=dict(color=MK_COLOR,width=1.5)), row=3,col=1)
    if not Lh.empty:
        fig.add_trace(go.Scatter(x=Lh.index,y=Lh.values,
            name="L_household",fill="tozeroy",
            fillcolor="rgba(29,158,117,0.2)",
            line=dict(color=HH_COLOR,width=1.5)), row=3,col=1)

    # Annotate the snapback region — stress declining but L_market rising
    if not S.empty and not Lm.empty:
        # Find period where S drops but Lm rises (recovery lag)
        common_idx = S.index.intersection(Lm.index)
        if len(common_idx) > 4:
            S_al  = S.reindex(common_idx)
            Lm_al = Lm.reindex(common_idx)
            dS    = S_al.diff()
            dLm   = Lm_al.diff()
            snap  = (dS < -0.005) & (dLm > 0.001)
            if snap.any():
                t0 = snap.index[snap][0]
                fig.add_vrect(
                    x0=t0, x1=common_idx[-1],
                    fillcolor="rgba(245,158,11,0.08)",
                    line=dict(color="orange",width=1,dash="dot"),
                    annotation_text="Snapback region:<br>S declining,<br>L_market rising",
                    annotation_position="top left",
                    row=3, col=1,
                )

    fig.update_layout(
        title=f"Figure 18. Snapback Illustration — {SEC_LABELS[sec]}: "
              "Apparent Stabilization Followed by Delayed Market-Channel Activation",
        height=600, width=W,
    )
    save(fig, "figure_18_snapback", fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_FIGS = {
    1: ("fig1",   True),
    2: ("fig2",   False),
    3: ("fig3",   False),
    4: ("fig4",   False),
    5: ("fig5",   True),
    6: ("fig6",   True),
    7: ("fig7",   True),
    8: ("fig8",   True),
    9: ("fig9",   True),
    10: ("fig10", True),
    11: ("fig11", True),
    12: ("fig12", True),
    13: ("fig13", True),
    14: ("fig14", True),
    15: ("fig15", True),
    16: ("fig16", True),
    17: ("fig17", True),
    18: ("fig18", True),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt",  default="html", choices=["html","png","both"])
    parser.add_argument("--figs", default="all",
                        help="Comma-separated figure numbers e.g. 5,6,7 or 'all'")
    args = parser.parse_args()

    if args.figs == "all":
        requested = list(ALL_FIGS.keys())
    else:
        requested = [int(x) for x in args.figs.split(",")]

    needs_data = any(ALL_FIGS[n][1] for n in requested if n in ALL_FIGS)
    print("Loading data..." if needs_data else "No data needed for requested figures.")
    d = load() if needs_data else {}

    fn_map = {
        1: lambda: fig1(d, args.fmt),
        2: lambda: fig2(args.fmt),
        3: lambda: fig3(args.fmt),
        4: lambda: fig4(args.fmt),
        5: lambda: fig5(d, args.fmt),
        6: lambda: fig6(d, args.fmt),
        7: lambda: fig7(d, args.fmt),
        8: lambda: fig8(d, args.fmt),
        9: lambda: fig9(d, args.fmt),
        10: lambda: fig10(d, args.fmt),
        11: lambda: fig11(d, args.fmt),
        12: lambda: fig12(d, args.fmt),
        13: lambda: fig13(d, args.fmt),
        14: lambda: fig14(d, args.fmt),
        15: lambda: fig15(d, args.fmt),
        16: lambda: fig16(d, args.fmt),
        17: lambda: fig17(d, args.fmt),
        18: lambda: fig18(d, args.fmt),
    }

    for n in requested:
        if n not in fn_map:
            print(f"  ⚠  Figure {n} not defined — skipping")
            continue
        print(f"\nFigure {n}...")
        try:
            fn_map[n]()
        except Exception as e:
            print(f"  ✗  Error: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. Outputs in: {OUT}")
