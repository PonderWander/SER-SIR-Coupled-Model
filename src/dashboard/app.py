"""
dashboard/app.py — Rebuilt: 9 tabs, Interventions + Sweep
Launch: streamlit run src/dashboard/app.py --server.port 8501 --server.address localhost
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import networkx as nx
import yaml

st.set_page_config(
    page_title="Epidemic\u2013Market Simulation",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { width: 260px !important; }
  button[data-baseweb="tab"] { font-size: 12px !important; padding: 6px 10px !important; }
  div[data-testid="metric-container"] {
    background:#F8F9FA; border:1px solid #E0E0E0; border-radius:8px; padding:10px 14px;
  }
  h3 { margin-top:0.4rem !important; }
  .stCaption { color:#757575; }
</style>
""", unsafe_allow_html=True)

REGIME_COLORS = {
    "dispersal":"#4CAF50","accumulation":"#FF9800","isolation":"#9C27B0",
    "recovery":"#2196F3","fragmented":"#F44336","amplification":"#E91E63","unknown":"#9E9E9E",
}
REGIME_ORDER = ["dispersal","accumulation","isolation","recovery","fragmented","amplification"]
SC_COLORS = {
    "baseline":"#9E9E9E","low":"#90CAF9","medium":"#1E88E5","high":"#0D47A1",
    "food_only":"#66BB6A","filter_only":"#AB47BC","food_and_filter":"#F57F17",
    "medium_and_food":"#26A69A","medium_and_filter":"#EF5350","medium_food_filter":"#B71C1C",
}
SC_DASH = {
    "baseline":"dot","low":"dash","medium":"solid","high":"longdash",
    "food_only":"solid","filter_only":"solid","food_and_filter":"solid",
    "medium_and_food":"dash","medium_and_filter":"dash","medium_food_filter":"longdash",
}
SECTOR_PALETTE = px.colors.qualitative.Set2


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, format="ISO8601", errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_all_data(output_dir: str) -> dict:
    p = Path(output_dir); sp = p.parent/"sweep"; iv = p.parent/"intervention"
    return {
        "sir":        load_parquet(str(p/"sir_timeseries.parquet")),
        "covid":      load_parquet(str(p/"covid_monthly.parquet")),
        "ser":        load_parquet(str(p/"sector_ser_panel.parquet")),
        "prop":       load_parquet(str(p/"propagation_panel.parquet")),
        "regime":     load_parquet(str(p/"regime_panel.parquet")),
        "summary":    load_parquet(str(p/"sector_summary.parquet")),
        "system_reg": load_parquet(str(p/"system_regime.parquet")),
        "emitter":    load_parquet(str(p/"emitter_absorber_table.parquet")),
        "transitions":load_parquet(str(p/"transition_matrix.parquet")),
        "duration":   load_parquet(str(p/"regime_duration_table.parquet")),
        "breach":     load_parquet(str(p/"threshold_breach_summary.parquet")),
        "sweep_full": load_csv(str(sp/"sweep_results_full.csv")),
        "sweep_iso":  load_csv(str(sp/"isolation_variants.csv")),
        "iv_system":  load_csv(str(iv/"scenario_comparison_system.csv")),
        "iv_sector":  load_csv(str(iv/"scenario_comparison_sector.csv")),
    }


@st.cache_data
def run_iv_scenarios(ser_path: str) -> dict:
    try:
        base_cfg   = yaml.safe_load(open(ROOT/"configs/base_config.yaml"))
        interv_cfg = yaml.safe_load(open(ROOT/"configs/intervention.yaml"))["intervention"]
        from src.simulation.intervention import InterventionEngine, extract_metrics
        ser = pd.read_parquet(ser_path)
        inputs = {}
        for col in ser.columns:
            if not col.endswith("_stress") or "leaked" in col or "absorbed" in col: continue
            sec = col.replace("_stress","")
            b=f"{sec}_buffering_B"; c=f"{sec}_clearance_C"
            inputs[sec] = {"stress":ser[col].fillna(0).values,
                           "B":ser[b].fillna(0.1).values if b in ser.columns else np.full(len(ser),0.15),
                           "C":ser[c].fillna(0.4).values if c in ser.columns else np.full(len(ser),0.50),
                           "index":ser.index}
        engine = InterventionEngine(base_cfg["ser"], interv_cfg)
        all_r  = engine.run_all_scenarios(inputs, T=len(ser))
        metrics = {sc: extract_metrics(res,sc) for sc,res in all_r.items()}
        return {"results":all_r,"metrics":metrics,"inputs":inputs,"index":ser.index,"sectors":list(inputs.keys())}
    except Exception as e:
        return {"error":str(e)}


def safe_style(df, **kw):
    try: return df.reset_index(drop=False).style.background_gradient(**kw)
    except: return df


def get_sectors(ser):
    sectors = set()
    sv = {"stress","leaked","absorbed","elasticity","reaction","damping","phi",
          "retention","spillover","below","buffering","clearance"}
    for col in ser.columns:
        parts = col.split("_")
        for i,p in enumerate(parts):
            if p in sv:
                sec="_".join(parts[:i])
                if sec: sectors.add(sec)
                break
    return sorted(sectors)


def gcol(ser, sector, var):
    col=f"{sector}_{var}"
    if col in ser.columns: return ser[col]
    m=[c for c in ser.columns if c.startswith(sector) and var in c]
    return ser[m[0]] if m else pd.Series(dtype=float)


def regime_cs():
    return [[i/6,c] for i,c in enumerate(
        ["#4CAF50","#FF9800","#9C27B0","#2196F3","#F44336","#E91E63","#9E9E9E"])]


def main():
    c1,_ = st.columns([3,1])
    with c1:
        st.title("\U0001f52c Epidemic\u2013Market Simulation")
        st.caption("SIR + SER + Sector Graph  |  COVID-era Dynamics  |  Intervention Analysis")

    st.sidebar.markdown("## \u2699\ufe0f Controls")
    output_dir = st.sidebar.text_input("Output dir", value="outputs/parquet")
    data = load_all_data(output_dir)

    if data["ser"].empty:
        st.error("No data. Run `python main.py run` first."); st.stop()

    sectors = get_sectors(data["ser"])
    idx     = data["ser"].index
    dmin    = idx.min().to_pydatetime()
    dmax    = idx.max().to_pydatetime()
    dr      = st.sidebar.slider("Date range", min_value=dmin, max_value=dmax,
                                 value=(dmin,dmax), format="YYYY-MM")
    sel_sec = st.sidebar.multiselect("Sectors", sectors, default=sectors[:4])
    if not sel_sec: sel_sec = sectors[:2]
    ser_var = st.sidebar.selectbox("SER variable",
                ["stress","leaked_stress","elasticity","reaction","damping"], index=0)
    show_ep = st.sidebar.checkbox("Epidemic overlay", value=True)
    smooth  = st.sidebar.slider("Smoothing (months)", 1, 6, 1)
    st.sidebar.markdown("---")
    st.sidebar.metric("Sectors", len(sectors))
    st.sidebar.metric("Periods", len(idx))
    if not data["iv_system"].empty:
        st.sidebar.metric("IV scenarios", len(data["iv_system"]))

    mask   = (idx>=pd.Timestamp(dr[0]))&(idx<=pd.Timestamp(dr[1]))
    ser_f  = data["ser"].loc[mask]
    prop_f = data["prop"].loc[mask] if not data["prop"].empty else pd.DataFrame()
    reg_f  = data["regime"].loc[mask] if not data["regime"].empty else pd.DataFrame()

    tabs = st.tabs([
        "\U0001f4c8 Epidemic", "\U0001f3ed Sector SER", "\U0001f310 Network",
        "\U0001f5fa\ufe0f Regimes", "\U0001f52c Decomposition", "\U0001f4ca System Metrics",
        "\U0001f489 Interventions", "\U0001f9ea Parameter Sweep", "\U0001f4cb Tables",
    ])

    # ── Tab 0: Epidemic ──────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Epidemic Layer")
        co1,co2 = st.columns([2,1])
        with co1:
            sir = data["sir"]
            if not sir.empty:
                sf=sir.loc[str(dr[0]):str(dr[1])]
                fig=go.Figure()
                if "new_cases_smoothed" in sf.columns:
                    fig.add_trace(go.Scatter(x=sf.index,y=sf["new_cases_smoothed"],
                        name="Daily Cases",line=dict(color="#E53935",width=2),
                        fill="tozeroy",fillcolor="rgba(229,57,53,0.08)"))
                if "pressure" in sf.columns:
                    sc2=sf["new_cases_smoothed"].max() if "new_cases_smoothed" in sf.columns else 1
                    fig.add_trace(go.Scatter(x=sf.index,y=sf["pressure"]*sc2,
                        name="Pressure (scaled)",line=dict(color="#FF7043",dash="dot")))
                if "hosp_patients" in sf.columns and sf["hosp_patients"].sum()>0:
                    fig.add_trace(go.Scatter(x=sf.index,y=sf["hosp_patients"],
                        name="Hospitalizations",line=dict(color="#8E24AA")))
                fig.update_layout(title="COVID-19 Epidemic (Daily)",xaxis_title="Date",
                    template="plotly_white",height=400,legend=dict(orientation="h"))
                st.plotly_chart(fig,width="stretch")
        with co2:
            cv=data["covid"]
            if not cv.empty:
                cvf=cv.loc[str(dr[0]):str(dr[1])]
                fig2=go.Figure()
                if "epidemic_pressure" in cvf.columns:
                    fig2.add_trace(go.Scatter(x=cvf.index,y=cvf["epidemic_pressure"],
                        fill="tozeroy",name="Monthly Pressure",
                        line=dict(color="#EF5350",width=2),fillcolor="rgba(239,83,80,0.15)"))
                if "stringency_index" in cvf.columns:
                    fig2.add_trace(go.Scatter(x=cvf.index,y=cvf["stringency_index"]/100,
                        name="Stringency (norm)",line=dict(color="#5C6BC0",dash="dash")))
                fig2.update_layout(title="Monthly Pressure & Stringency",
                    yaxis_range=[0,1.05],template="plotly_white",height=400)
                st.plotly_chart(fig2,width="stretch")

    # ── Tab 1: Sector SER ────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Sector SER Time Series")
        iv=run_iv_scenarios(str(Path(output_dir)/"sector_ser_panel.parquet"))
        iv_ok="results" in iv and not iv.get("error")
        _,cc2=st.columns([3,1])
        with cc2:
            if iv_ok:
                sc_choices={v["label"]:k for k,v in iv["results"].items()}
                ov_lbl=st.selectbox("Scenario overlay",["None"]+list(sc_choices.keys()),index=0)
                ov_sc=sc_choices.get(ov_lbl)
            else:
                ov_sc=None; st.caption("IV engine unavailable")

        fig=go.Figure()
        for i,sec in enumerate(sel_sec):
            s=gcol(ser_f,sec,ser_var)
            if s.empty: continue
            if smooth>1: s=s.rolling(smooth,min_periods=1).mean()
            color=SECTOR_PALETTE[i%len(SECTOR_PALETTE)]
            fig.add_trace(go.Scatter(x=s.index,y=s.values,name=sec,
                line=dict(color=color,width=2.2)))
            if ov_sc and ov_sc!="baseline" and ser_var=="elasticity":
                sd=iv["results"][ov_sc]["sectors"].get(sec)
                if sd is not None:
                    oe=sd["elasticity"].reindex(s.index)
                    fig.add_trace(go.Scatter(x=oe.index,y=oe.values,
                        name=f"{sec} [{ov_lbl}]",
                        line=dict(color=color,width=1.5,dash="dot"),opacity=0.65))
        if ser_var=="elasticity":
            fig.add_hline(y=0.25,line_dash="dash",line_color="red",
                annotation_text="E_crit",annotation_position="top right")
        if show_ep and not data["covid"].empty:
            ep=data["covid"].get("epidemic_pressure")
            if ep is not None:
                epf=ep.reindex(ser_f.index)
                vals=[gcol(ser_f,s,ser_var).max() for s in sel_sec if not gcol(ser_f,s,ser_var).empty]
                if vals:
                    fig.add_trace(go.Scatter(x=epf.index,y=epf.values*max(vals),
                        name="Epidemic Pressure (scaled)",
                        line=dict(color="#B0BEC5",dash="dot",width=1),opacity=0.6))
        ttl=f"Sector {ser_var.replace('_',' ').title()} Over Time"
        if ov_sc: ttl+=f" \u2014 overlay: {ov_lbl}"
        fig.update_layout(title=ttl,xaxis_title="Date",template="plotly_white",height=440,
            legend=dict(orientation="h"))
        st.plotly_chart(fig,width="stretch")

        tc1,tc2=st.columns(2)
        with tc1:
            figs=go.Figure()
            for i,sec in enumerate(sel_sec):
                s=gcol(ser_f,sec,"stress")
                if s.empty: continue
                figs.add_trace(go.Scatter(x=s.index,y=s.values,name=sec,
                    line=dict(color=SECTOR_PALETTE[i%len(SECTOR_PALETTE)],width=2)))
            figs.update_layout(title="Stress",template="plotly_white",height=300,yaxis_range=[0,1])
            st.plotly_chart(figs,width="stretch")
        with tc2:
            fige=go.Figure()
            for i,sec in enumerate(sel_sec):
                e=gcol(ser_f,sec,"elasticity")
                if e.empty: continue
                fige.add_trace(go.Scatter(x=e.index,y=e.values,name=sec,
                    line=dict(color=SECTOR_PALETTE[i%len(SECTOR_PALETTE)],width=2)))
            fige.add_hline(y=0.25,line_dash="dash",line_color="red",annotation_text="E_crit")
            fige.update_layout(title="Elasticity",template="plotly_white",height=300,yaxis_range=[0,1])
            st.plotly_chart(fige,width="stretch")

    # ── Tab 2: Network ───────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Cross-Sector Network")
        G=None
        try:
            from src.utils.common import load_sector_graph_config
            from src.graph.propagation import SectorGraph
            gcfg=load_sector_graph_config(str(ROOT/"configs/sector_graph.yaml"))
            sg=SectorGraph(gcfg); G=sg.G
        except Exception as e:
            st.warning(f"Graph error: {e}")
        if G and not ser_f.empty:
            t_snap=st.select_slider("Snapshot date",
                options=ser_f.index.strftime("%Y-%m").tolist(),
                value=ser_f.index[len(ser_f)//2].strftime("%Y-%m"))
            ti=ser_f.index[ser_f.index.strftime("%Y-%m")==t_snap]
            if len(ti):
                tv=ti[0]
                ns={n:float(gcol(ser_f,n,"stress").loc[tv]) if tv in gcol(ser_f,n,"stress").index else 0.1 for n in G.nodes()}
                ne={n:float(gcol(ser_f,n,"elasticity").loc[tv]) if tv in gcol(ser_f,n,"elasticity").index else 0.5 for n in G.nodes()}
                pos=nx.spring_layout(G,seed=42,k=2.0)
                nc=st.radio("Node color",["stress","elasticity"],horizontal=True)
                nv=ns if nc=="stress" else ne
                et=[]
                for u,v,w in G.edges(data="weight"):
                    x0,y0=pos[u]; x1,y1=pos[v]
                    et.append(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
                        line=dict(width=w*5,color="rgba(144,164,174,0.5)"),
                        hoverinfo="none",showlegend=False))
                nodes=list(G.nodes())
                nvals=[nv.get(n,0) for n in nodes]
                nt=go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],
                    mode="markers+text",
                    marker=dict(size=[22+48*v for v in nvals],color=nvals,
                        colorscale="YlOrRd" if nc=="stress" else "RdYlGn",
                        cmin=0,cmax=1,showscale=True,
                        colorbar=dict(title=nc.title(),thickness=14,x=1.01),
                        line=dict(width=2,color="white")),
                    text=nodes,textposition="top center",textfont=dict(size=10),
                    hovertext=[f"<b>{n}</b><br>stress:{ns.get(n,0):.3f}<br>elasticity:{ne.get(n,0):.3f}" for n in nodes],
                    hoverinfo="text",showlegend=False)
                fignet=go.Figure(data=et+[nt],layout=go.Layout(
                    title=f"Sector Network \u2014 {t_snap}",showlegend=False,
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    template="plotly_white",height=560,margin=dict(l=20,r=80,t=60,b=20)))
                st.plotly_chart(fignet,width="stretch")
                if not prop_f.empty:
                    pc=[c for c in prop_f.columns if "propagation_raw" in c]
                    if pc and tv in prop_f.index:
                        sn=prop_f.loc[tv,pc].to_frame("received")
                        sn.index=[c.replace("_propagation_raw","") for c in pc]
                        st.dataframe(sn.sort_values("received",ascending=False),height=220)

    # ── Tab 3: Regimes ───────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Regime Classification")
        if not reg_f.empty:
            rc=[c for c in reg_f.columns if c.endswith("_regime") and "code" not in c]
            if rc:
                rmap={r:i for i,r in enumerate(["dispersal","accumulation","isolation","recovery","fragmented","amplification","unknown"])}
                hm=reg_f[rc].copy(); hm.columns=[c.replace("_regime","") for c in rc]
                hmn=hm.map(lambda x: rmap.get(str(x),6))
                figr=px.imshow(hmn.T,x=hmn.index.strftime("%Y-%m"),y=hmn.columns,
                    color_continuous_scale=regime_cs(),zmin=0,zmax=6,aspect="auto",title="Sector Regime Map")
                figr.update_layout(height=400,template="plotly_white",
                    coloraxis_colorbar=dict(tickvals=list(range(7)),
                        ticktext=["dispersal","accumulation","isolation","recovery","fragmented","amplification","unknown"],title=""))
                st.plotly_chart(figr,width="stretch")
            sr=data["system_reg"]
            if not sr.empty:
                srf=sr.loc[str(dr[0]):str(dr[1])]; cn=srf.columns[0]
                figsr=go.Figure()
                for r,color in REGIME_COLORS.items():
                    m=srf[cn]==r
                    if m.any():
                        figsr.add_trace(go.Scatter(x=srf.index[m],y=[r]*m.sum(),mode="markers",
                            marker=dict(color=color,size=14,symbol="square"),name=r))
                figsr.update_layout(title="System Regime",template="plotly_white",height=240,legend=dict(orientation="h"))
                st.plotly_chart(figsr,width="stretch")
            if not data["transitions"].empty:
                st.caption("Transition counts")
                st.dataframe(safe_style(data["transitions"],cmap="Blues"),width="stretch")

    # ── Tab 4: Decomposition ─────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Stress Decomposition")
        focus=st.selectbox("Sector",sectors,index=0)
        if focus:
            decomp=[("stress_price","Price Pressure","#EF5350"),("stress_volatility","Volatility","#FF7043"),
                    ("stress_efficiency","Efficiency Loss","#FB8C00"),("stress_covid","COVID Pressure","#AB47BC"),
                    ("stress_shortage","Shortage Proxy","#5C6BC0")]
            figd=go.Figure()
            for var,lbl,color in decomp:
                s=gcol(ser_f,focus,var)
                if not s.empty: figd.add_trace(go.Bar(x=s.index,y=s.values,name=lbl,marker_color=color))
            figd.update_layout(barmode="stack",title=f"Stress Decomposition \u2014 {focus}",
                template="plotly_white",height=370,legend=dict(orientation="h"))
            st.plotly_chart(figd,width="stretch")
            a1,a2,a3=st.columns(3)
            with a1:
                figb=go.Figure()
                for var,lbl,color in [("buffering_B","Buffering (B)","#26A69A"),("clearance_C","Clearance (C)","#42A5F5")]:
                    s=gcol(ser_f,focus,var)
                    if not s.empty: figb.add_trace(go.Scatter(x=s.index,y=s.values,name=lbl,line=dict(color=color)))
                figb.update_layout(title="Elasticity Drivers",template="plotly_white",height=280)
                st.plotly_chart(figb,width="stretch")
            with a2:
                phi=gcol(ser_f,focus,"phi")
                if not phi.empty:
                    figp=go.Figure()
                    figp.add_trace(go.Bar(x=phi.index,y=phi.values,
                        marker_color=phi.apply(lambda v:"#EF5350" if v>0 else "#42A5F5"),name="\u03a6"))
                    figp.update_layout(title="Discrepancy Field \u03a6",template="plotly_white",height=280)
                    st.plotly_chart(figp,width="stretch")
            with a3:
                if not prop_f.empty:
                    pr=gcol(prop_f,focus,"propagation_raw"); sl=gcol(ser_f,focus,"stress")
                    if not pr.empty and not sl.empty:
                        figpr=go.Figure()
                        figpr.add_trace(go.Scatter(x=sl.index,y=sl.values,name="Local",fill="tozeroy",line=dict(color="#90A4AE"),fillcolor="rgba(144,164,174,0.2)"))
                        figpr.add_trace(go.Scatter(x=pr.index,y=pr.values,name="Imported",fill="tozeroy",line=dict(color="#EF9A9A"),fillcolor="rgba(239,154,154,0.3)"))
                        figpr.update_layout(title="Local vs Imported",template="plotly_white",height=280)
                        st.plotly_chart(figpr,width="stretch")

    # ── Tab 5: System Metrics ────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("System-Level Aggregate Metrics")
        agg_def={"Mean Stress":("_stress","#E53935"),"Mean Leaked Stress":("_leaked_stress","#FF7043"),
                 "Mean Elasticity":("_elasticity","#43A047"),"Mean Reaction":("_reaction","#1E88E5")}
        figsys=go.Figure()
        for lbl,(suf,color) in agg_def.items():
            cols=[c for c in ser_f.columns if c.endswith(suf) and "absorbed" not in c]
            if cols:
                agg=ser_f[cols].mean(axis=1)
                if smooth>1: agg=agg.rolling(smooth,min_periods=1).mean()
                figsys.add_trace(go.Scatter(x=agg.index,y=agg.values,name=lbl,line=dict(color=color,width=2.2)))
        if not prop_f.empty:
            cc="_system_effective_connectivity"
            if cc in prop_f.columns:
                figsys.add_trace(go.Scatter(x=prop_f.index,y=prop_f[cc].values,name="Connectivity",line=dict(color="#AB47BC",dash="dash")))
        sr=data["system_reg"]
        if not sr.empty:
            srf2=sr.loc[str(dr[0]):str(dr[1])]; cn2=srf2.columns[0]
            for t in range(len(srf2)):
                r=srf2[cn2].iloc[t]; clr=REGIME_COLORS.get(r,"#BDBDBD")
                x0=srf2.index[t]; x1=srf2.index[t+1] if t+1<len(srf2) else x0+pd.DateOffset(months=1)
                figsys.add_vrect(x0=x0,x1=x1,fillcolor=clr,opacity=0.07,line_width=0)
        figsys.add_hline(y=0.25,line_dash="dot",line_color="#EF5350",annotation_text="E_crit",annotation_position="top left")
        figsys.update_layout(title="System Aggregates with Regime Background",template="plotly_white",height=450,yaxis_range=[0,1.1],legend=dict(orientation="h"))
        st.plotly_chart(figsys,width="stretch")
        if not data["summary"].empty:
            s=data["summary"]
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Peak stress", f"{s['stress_peak'].max():.3f}", delta=f"peak: {s['stress_peak'].idxmax().date()}")
            m2.metric("Min elasticity",f"{s['elasticity_min'].min():.3f}")
            m3.metric("Sectors below E_crit",int((s['below_ecrit_pct']>0).sum()))
            m4.metric("Mean retention",f"{s['retention_mean'].mean():.3f}")
            st.dataframe(safe_style(s,subset=["stress_peak","leaked_stress_peak"],cmap="Reds"),width="stretch")

    # ── Tab 6: Interventions ─────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Intervention Scenario Analysis")
        iv=run_iv_scenarios(str(Path(output_dir)/"sector_ser_panel.parquet"))
        if iv.get("error"):
            st.error(f"Engine error: {iv['error']}")
        else:
            iv_r=iv["results"]; iv_m=iv["metrics"]; iv_secs=iv["sectors"]; iv_idx=iv["index"]
            sc_names=list(iv_r.keys()); sc_lbl={k:v["label"] for k,v in iv_r.items()}
            x_iv=[str(d)[:10] for d in iv_idx]

            ic1,ic2,ic3=st.columns([2,2,1])
            with ic1:
                sel_sc=st.multiselect("Scenarios",sc_names,
                    default=["baseline","medium","food_only","filter_only","medium_food_filter"],
                    format_func=lambda k:sc_lbl[k])
            with ic2:
                foc_sec=st.selectbox("Focus sector",iv_secs,index=0)
            with ic3:
                show_band=st.checkbox("Min/max band",value=True)
            if not sel_sc: sel_sc=["baseline","medium"]

            # H_t profiles
            st.markdown("#### Inflow H_t Profiles")
            figh=go.Figure()
            for sc in sel_sc:
                if sc=="baseline": continue
                H=iv_r[sc]["H"]
                figh.add_trace(go.Scatter(x=x_iv,y=H,name=sc_lbl[sc],
                    line=dict(color=SC_COLORS.get(sc,"#333"),dash=SC_DASH.get(sc,"solid"),width=2)))
            figh.update_layout(title="H_t = A_t \u00d7 G_t",xaxis_title="Month",yaxis_title="H_t",
                template="plotly_white",height=250,legend=dict(orientation="h"))
            st.plotly_chart(figh,width="stretch")

            # Elasticity + breach bar
            st.markdown("#### Elasticity Response")
            ca,cb=st.columns([3,2])
            with ca:
                fige2=go.Figure()
                if show_band and len(sc_names)>1:
                    allE=np.stack([iv_r[sc]["sectors"][foc_sec]["elasticity"].values
                                   for sc in sc_names if foc_sec in iv_r[sc]["sectors"]])
                    fige2.add_trace(go.Scatter(x=x_iv,y=allE.max(axis=0),line=dict(width=0),showlegend=False,name="_max"))
                    fige2.add_trace(go.Scatter(x=x_iv,y=allE.min(axis=0),fill="tonexty",
                        line=dict(width=0),fillcolor="rgba(180,180,180,0.18)",name="All-scenario range"))
                for sc in sel_sc:
                    if foc_sec not in iv_r[sc]["sectors"]: continue
                    E=iv_r[sc]["sectors"][foc_sec]["elasticity"].values
                    fige2.add_trace(go.Scatter(x=x_iv,y=E,name=sc_lbl[sc],
                        line=dict(color=SC_COLORS.get(sc,"#333"),dash=SC_DASH.get(sc,"solid"),width=2.2)))
                fige2.add_hline(y=0.25,line_dash="dash",line_color="red",annotation_text="E_crit")
                fige2.update_layout(title=f"Elasticity \u2014 {foc_sec}",yaxis_range=[0,1.0],
                    template="plotly_white",height=360,legend=dict(orientation="h"))
                st.plotly_chart(fige2,width="stretch")
            with cb:
                figbar=go.Figure(go.Bar(
                    x=[sc_lbl[sc] for sc in sc_names],
                    y=[iv_m[sc]["pct_below_ecrit"] for sc in sc_names],
                    marker_color=[SC_COLORS.get(sc,"#333") for sc in sc_names],
                    text=[f"{iv_m[sc]['pct_below_ecrit']:.1f}%" for sc in sc_names],
                    textposition="outside"))
                figbar.update_layout(title="% Time Below E_crit",
                    template="plotly_white",height=360,xaxis_tickangle=-40,xaxis_tickfont=dict(size=8))
                st.plotly_chart(figbar,width="stretch")

            # Leaked stress
            st.markdown("#### Leaked Stress Comparison")
            foc_lk=st.multiselect("Sectors",iv_secs,
                default=["energy","shelter","food_at_home","medical_services"],key="iv_lk")
            if foc_lk:
                nl=len(foc_lk); nlc=min(nl,3); nlr=(nl+nlc-1)//nlc
                figlk=make_subplots(rows=nlr,cols=nlc,shared_xaxes=True,subplot_titles=foc_lk,
                    vertical_spacing=0.10,horizontal_spacing=0.07)
                for i,sec in enumerate(foc_lk):
                    r2,c2=divmod(i,nlc)
                    for sc in sel_sc:
                        if sec not in iv_r[sc]["sectors"]: continue
                        L=iv_r[sc]["sectors"][sec]["leaked_stress"].values
                        figlk.add_trace(go.Scatter(x=x_iv,y=L,name=sc_lbl[sc],
                            line=dict(color=SC_COLORS.get(sc,"#333"),dash=SC_DASH.get(sc,"solid"),width=1.8),
                            showlegend=(i==0)),row=r2+1,col=c2+1)
                figlk.update_layout(template="plotly_white",height=300 if nlr==1 else 520,
                    legend=dict(orientation="h",y=1.04))
                st.plotly_chart(figlk,width="stretch")

            # Regime maps
            st.markdown("#### Regime Maps")
            foc_rm=st.multiselect("Scenarios",sc_names,
                default=["baseline","food_only","filter_only","medium_food_filter"],
                format_func=lambda k:sc_lbl[k],key="iv_rm")
            if foc_rm:
                from src.simulation.intervention import classify_regime_simple
                rnum={r:i for i,r in enumerate(REGIME_ORDER)}
                nc3=min(len(foc_rm),2); nr3=(len(foc_rm)+1)//2
                figrm=make_subplots(rows=nr3,cols=nc3,
                    subplot_titles=[sc_lbl[s] for s in foc_rm],
                    vertical_spacing=0.12,horizontal_spacing=0.06)
                for pos,sc in enumerate(foc_rm):
                    r3,c3=divmod(pos,nc3)
                    z=np.zeros((len(iv_secs),len(iv_idx)))
                    for si,sec in enumerate(iv_secs):
                        if sec not in iv_r[sc]["sectors"]: continue
                        df2=iv_r[sc]["sectors"][sec]
                        Ev,Lv,Sv=df2["elasticity"].values,df2["leaked_stress"].values,df2["stress"].values
                        for t in range(len(Ev)):
                            reg=classify_regime_simple(Ev[t],Lv[t],Sv[t],Ev[t-1] if t>0 else None,Lv[t-1] if t>0 else None)
                            z[si,t]=rnum.get(reg,0)
                    figrm.add_trace(go.Heatmap(z=z,x=[str(d)[:7] for d in iv_idx],y=iv_secs,
                        colorscale=[[i/max(1,len(REGIME_COLORS)-1),c] for i,c in enumerate(list(REGIME_COLORS.values()))],
                        zmin=0,zmax=5,showscale=(pos==0),
                        colorbar=dict(tickvals=list(range(6)),ticktext=REGIME_ORDER,title="",x=1.01,
                            y=0.75 if pos<nc3 else 0.25,len=0.45)),row=r3+1,col=c3+1)
                figrm.update_layout(template="plotly_white",height=360 if nr3==1 else 680)
                st.plotly_chart(figrm,width="stretch")

            # Sensitivity heatmap
            st.markdown("#### Sector Intervention Sensitivity \u2014 \u0394E_mean vs Baseline")
            bl_E={sec:iv_r["baseline"]["sectors"][sec]["elasticity"].mean()
                  for sec in iv_secs if sec in iv_r["baseline"]["sectors"]}
            zs=np.zeros((len(iv_secs),len(sc_names)))
            for j,sc in enumerate(sc_names):
                for i,sec in enumerate(iv_secs):
                    if sec in iv_r[sc]["sectors"]:
                        zs[i,j]=iv_r[sc]["sectors"][sec]["elasticity"].mean()-bl_E.get(sec,0)
            figsen=go.Figure(go.Heatmap(z=zs,x=[sc_lbl[sc] for sc in sc_names],y=iv_secs,
                colorscale="RdYlGn",zmid=0,text=np.round(zs,3),texttemplate="%{text}",
                colorbar=dict(title="\u0394E_mean")))
            figsen.update_layout(title="\u0394E_mean vs Baseline \u2014 sector \u00d7 scenario",
                xaxis_tickangle=-40,xaxis_tickfont=dict(size=9),template="plotly_white",height=480)
            st.plotly_chart(figsen,width="stretch")

            if not data["iv_system"].empty:
                st.markdown("#### Scenario Comparison Table")
                dc=["label","E_mean","E_max","pct_below_ecrit","mean_run_below","L_mean","pct_amplification","pct_dispersal","H_max"]
                dsp=data["iv_system"][[c for c in dc if c in data["iv_system"].columns]]
                st.dataframe(safe_style(dsp.round(3),cmap="YlOrRd",
                    subset=["pct_below_ecrit","pct_amplification"]),width="stretch")
                st.download_button("Export scenario comparison (CSV)",
                    data["iv_system"].to_csv(index=False),"scenario_comparison.csv","text/csv")

    # ── Tab 7: Parameter Sweep ───────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Elasticity Parameter Sweep \u2014 Validation Diagnostics")
        sweep=data["sweep_full"]; iso=data["sweep_iso"]
        if sweep.empty:
            st.info("Run `python scripts/elasticity_sweep.py` to generate sweep data.")
        else:
            k1,k2,k3,k4=st.columns(4)
            k1.metric("Runs",len(sweep))
            k2.metric("In target (15\u201330%)",int(((sweep.pct_below_ecrit>=15)&(sweep.pct_below_ecrit<=30)).sum()))
            k3.metric("% Lagged",f"{(sweep.lag_class=='lagged').mean()*100:.0f}%")
            k4.metric("Max % below E_crit",f"{sweep.pct_below_ecrit.max():.1f}%")

            st.markdown("#### Lag Structure \u2014 E_min Timing vs Stress Peak")
            lc_colors={"lagged":"#1E88E5","coincident":"#43A047","leading":"#E53935"}
            figlag=go.Figure()
            for lc,color in lc_colors.items():
                sub=sweep[sweep.lag_class==lc]
                if sub.empty: continue
                figlag.add_trace(go.Scatter(x=sub["mean_lag"],y=sub["pct_below_ecrit"],mode="markers",
                    marker=dict(color=color,size=sub["E_std"]*60+5,opacity=0.75,line=dict(width=1,color="white")),
                    name=lc,
                    hovertemplate="<b>%{customdata[0]}</b><br>\u03b4=%{customdata[1]} \u03b2_L=%{customdata[2]}<br>lag=%{x:.1f} %%<E_crit=%{y:.1f}%<extra></extra>",
                    customdata=np.stack([sub["run_id"],sub["delta"],sub["beta_L"]],axis=-1)))
            figlag.add_vline(x=0,line_dash="dash",line_color="#BDBDBD")
            figlag.add_hrect(y0=15,y1=30,fillcolor="rgba(30,136,229,0.06)",line_width=0,annotation_text="target window")
            figlag.update_layout(title="Lag Classification  |  bubble size \u221d E_std",
                xaxis_title="Mean lag: E_min \u2212 S_peak (months)",yaxis_title="% below E_crit",
                template="plotly_white",height=420,legend=dict(title="Lag class"))
            st.plotly_chart(figlag,width="stretch")

            st.markdown("#### Parameter Sensitivity Heatmaps")
            h1,h2=st.columns(2)
            with h1:
                if {"beta_S","beta_L","pct_below_ecrit"}.issubset(sweep.columns):
                    pv=sweep.groupby(["beta_S","beta_L"])["pct_below_ecrit"].mean().reset_index()
                    mt=pv.pivot(index="beta_L",columns="beta_S",values="pct_below_ecrit")
                    fh1=go.Figure(go.Heatmap(z=mt.values,x=[str(v) for v in mt.columns],
                        y=[str(v) for v in mt.index],colorscale="YlOrRd",
                        text=np.round(mt.values,1),texttemplate="%{text}%",colorbar=dict(title="% < E_crit")))
                    fh1.update_layout(title="\u03b2_S \u00d7 \u03b2_L \u2192 % below E_crit",
                        xaxis_title="\u03b2_S",yaxis_title="\u03b2_L",template="plotly_white",height=340)
                    st.plotly_chart(fh1,width="stretch")
            with h2:
                if {"delta","beta_L","pct_below_ecrit"}.issubset(sweep.columns):
                    pv2=sweep.groupby(["delta","beta_L"])["pct_below_ecrit"].mean().reset_index()
                    mt2=pv2.pivot(index="beta_L",columns="delta",values="pct_below_ecrit")
                    fh2=go.Figure(go.Heatmap(z=mt2.values,x=[str(v) for v in mt2.columns],
                        y=[str(v) for v in mt2.index],colorscale="YlOrRd",
                        text=np.round(mt2.values,1),texttemplate="%{text}%",colorbar=dict(title="% < E_crit")))
                    fh2.update_layout(title="\u03b4 \u00d7 \u03b2_L \u2192 % below E_crit",
                        xaxis_title="\u03b4 (decay)",yaxis_title="\u03b2_L",template="plotly_white",height=340)
                    st.plotly_chart(fh2,width="stretch")

            if not iso.empty:
                st.markdown("#### Isolation Variants \u2014 Effect of Each Repair Component")
                iso_m=["pct_below_ecrit","pct_amplification","mean_run_below","E_mean","L_mean","corr_S_E"]
                iso_l=["% below E_crit","% Amplification","Mean breach run","Mean E","Mean L","Corr(S,E)"]
                figiso=make_subplots(rows=2,cols=3,subplot_titles=iso_l,vertical_spacing=0.18,horizontal_spacing=0.10)
                for i,(cn,lbl) in enumerate(zip(iso_m,iso_l)):
                    r4,c4=divmod(i,3)
                    if cn not in iso.columns: continue
                    figiso.add_trace(go.Bar(x=iso["run_id"].str.replace("_"," "),y=iso[cn],
                        marker_color=["#1E88E5","#43A047","#FB8C00","#E53935"],
                        showlegend=False,text=iso[cn].round(3),textposition="outside"),row=r4+1,col=c4+1)
                    figiso.update_yaxes(title_text=lbl,row=r4+1,col=c4+1)
                    figiso.update_xaxes(tickfont=dict(size=8),row=r4+1,col=c4+1)
                figiso.update_layout(
                    title="Blue=baseline \u00b7 Green=no \u03b2_S \u00b7 Orange=no \u03b3_dL \u00b7 Red=no \u03b4",
                    template="plotly_white",height=540)
                st.plotly_chart(figiso,width="stretch")

            st.markdown("#### Regime Distribution Across Sweep Runs")
            rcols=[c for c in sweep.columns if c.startswith("pct_")]
            if rcols:
                figrd=go.Figure()
                for rc2 in rcols:
                    rn2=rc2.replace("pct_",""); color=REGIME_COLORS.get(rn2,"#BDBDBD")
                    figrd.add_trace(go.Box(y=sweep[rc2],name=rn2.title(),marker_color=color,boxmean="sd"))
                figrd.update_layout(title="Regime % \u2014 distribution across 60-run sweep",
                    yaxis_title="% Time in Regime",template="plotly_white",height=380)
                st.plotly_chart(figrd,width="stretch")

    # ── Tab 8: Tables ────────────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("Analysis Tables & Export")
        choice=st.selectbox("Table",[
            "Sector Summary","Emitter/Absorber Ranking","Regime Duration",
            "Threshold Breaches","Transition Matrix","Sweep Top-10",
            "Intervention Sector Detail","Raw SER (sample)"])
        if choice=="Sector Summary" and not data["summary"].empty:
            st.dataframe(safe_style(data["summary"],subset=["stress_peak","leaked_stress_peak"],cmap="Reds"),width="stretch")
        elif choice=="Emitter/Absorber Ranking" and not data["emitter"].empty:
            st.dataframe(safe_style(data["emitter"],cmap="RdBu",axis=0),width="stretch")
        elif choice=="Regime Duration" and not data["duration"].empty:
            st.dataframe(data["duration"],width="stretch")
        elif choice=="Threshold Breaches" and not data["breach"].empty:
            st.dataframe(safe_style(data["breach"],subset=["ecrit_breaches"],cmap="Reds"),width="stretch")
        elif choice=="Transition Matrix" and not data["transitions"].empty:
            st.dataframe(safe_style(data["transitions"],cmap="Blues"),width="stretch")
        elif choice=="Sweep Top-10":
            p10=Path("outputs/sweep/summary_top10.csv")
            if p10.exists(): st.dataframe(pd.read_csv(p10),width="stretch")
            else: st.info("Run `python scripts/elasticity_sweep.py` first.")
        elif choice=="Intervention Sector Detail" and not data["iv_sector"].empty:
            st.dataframe(data["iv_sector"],width="stretch")
        elif choice=="Raw SER (sample)":
            st.dataframe(ser_f.head(24),width="stretch")

        st.markdown("---")
        e1,e2,e3=st.columns(3)
        with e1:
            st.download_button("SER panel (CSV)",ser_f.to_csv(),"ser_panel_filtered.csv","text/csv")
        with e2:
            if not data["summary"].empty:
                st.download_button("Sector summary (CSV)",data["summary"].to_csv(),"sector_summary.csv","text/csv")
        with e3:
            if not data["iv_system"].empty:
                st.download_button("Intervention comparison (CSV)",
                    data["iv_system"].to_csv(index=False),"intervention_comparison.csv","text/csv")


if __name__ == "__main__":
    main()
