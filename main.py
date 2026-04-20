#!/usr/bin/env python3
"""
main.py
────────
Command-line entrypoint for the Epidemic-Market Simulation pipeline.

Commands:
  run            - Execute full simulation pipeline
  dashboard      - Launch interactive Streamlit dashboard
  figures        - Export figure-ready CSV files
  paper-figures  - Export Figures 2-18 as interactive HTML (and optionally PNG)
  check          - Validate config and dependencies

Usage:
  python main.py run
  python main.py run --config configs/base_config.yaml
  python main.py dashboard
  python main.py figures
  python main.py paper-figures
  python main.py paper-figures --fmt both --figs 5,6,7
  python main.py check
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env from project root (silently ignored if file absent)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # dotenv not installed; fall back to environment variables only


def cmd_run(args):
    """Execute the full simulation pipeline."""
    from src.simulation.runner import SimulationRunner
    from src.analytics.analysis import SimulationAnalytics
    from src.utils.common import get_logger

    log = get_logger("main")
    log.info("Starting simulation pipeline...")

    runner = SimulationRunner(
        config_path=args.config,
        graph_config_path=args.graph_config,
        output_dir=args.output_dir,
        refresh_cache=args.refresh_cache,
    )
    results = runner.run()

    # Run analytics and export figure-ready data
    log.info("Running analytics and exporting figure-ready data...")
    analytics = SimulationAnalytics(results)
    figures_dir = Path(args.output_dir or "outputs") / "figures"
    written = analytics.export_figure_ready(figures_dir)
    log.info(f"Exported {len(written)} figure-ready files to {figures_dir}")

    print("\n✅ Simulation complete!")
    print(f"   Outputs: {args.output_dir or 'outputs'}/parquet/")
    print(f"   CSV:     {args.output_dir or 'outputs'}/csv/")
    print(f"   Figures: {figures_dir}/")
    print("\n   Launch dashboard with: python main.py dashboard")


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    print("🚀 Launching dashboard...")
    print(f"   URL: http://localhost:{args.port}")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/app.py",
            "--server.port", str(args.port),
            "--server.address", "localhost",
            "--server.headless", "false",
        ],
        cwd=str(ROOT),
    )


def cmd_figures(args):
    """Export figure-ready data from existing simulation outputs."""
    from src.analytics.analysis import SimulationAnalytics
    import pandas as pd

    parquet_dir = Path(args.output_dir or "outputs") / "parquet"
    figures_dir = Path(args.output_dir or "outputs") / "figures"

    def safe_load(name):
        p = parquet_dir / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
        return None

    results = {
        "sector_ser_panel":  safe_load("sector_ser_panel"),
        "propagation_panel": safe_load("propagation_panel"),
        "regime_panel":      safe_load("regime_panel"),
        "covid_monthly":     safe_load("covid_monthly"),
        "system_regime":     safe_load("system_regime"),
        "sector_graph":      None,
    }
    results = {k: v for k, v in results.items() if v is not None}

    analytics = SimulationAnalytics(results)
    written = analytics.export_figure_ready(figures_dir)
    print(f"✅ Exported {len(written)} figure-ready files to {figures_dir}/")
    for name, path in written.items():
        print(f"   {name}: {path.name}")



def cmd_paper_figures(args):
    """Export Figures 2-18 as interactive HTML and/or PNG for the paper."""
    import subprocess, sys
    script = ROOT / "scripts" / "export_paper_figures.py"
    if not script.exists():
        print(f"\n❌  Script not found: {script}")
        print("   Make sure scripts/export_paper_figures.py is present.")
        return
    cmd = [sys.executable, str(script), "--fmt", args.fmt]
    if args.figs != "all":
        cmd += ["--figs", args.figs]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    print(f"📊 Exporting paper figures (fmt={args.fmt}, figs={args.figs})...")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode == 0:
        out = Path(args.output_dir or "outputs") / "paper_figures"
        print(f"\n✅ Paper figures written to: {out}/")
        print("   Open any .html file directly in a browser — no server needed.")
    else:
        print("\n❌  Figure export failed — check output above for details.")


def cmd_check(args):
    """Validate configuration and check dependencies."""
    print("🔍 Checking environment...")

    # Check imports
    packages = [
        "pandas", "numpy", "scipy", "networkx", "plotly",
        "streamlit", "pyarrow", "statsmodels", "sklearn"
    ]
    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} — install with: pip install {pkg}")
            all_ok = False

    # Check configs
    from src.utils.common import load_config
    try:
        cfg = load_config(args.config)
        print(f"   ✅ Config loaded: {args.config}")
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        all_ok = False

    # Check FRED key
    import os
    key = cfg.get("data", {}).get("fred_api_key") or os.environ.get("FRED_API_KEY")
    if key:
        print("   ✅ FRED API key found")
    else:
        print("   ⚠️  No FRED API key — will use synthetic data (set FRED_API_KEY env var)")

    print("\n" + ("✅ All checks passed!" if all_ok else "⚠️  Some packages missing."))


# ─── Argument Parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Epidemic-Market Simulation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Execute full simulation")
    p_run.add_argument("--config", default="configs/base_config.yaml")
    p_run.add_argument("--graph-config", default="configs/sector_graph.yaml")
    p_run.add_argument("--output-dir", default=None)
    p_run.add_argument(
        "--refresh-cache",
        action="store_true",
        default=False,
        help="Delete cached raw data files and re-fetch from source before running",
    )
    p_run.set_defaults(func=cmd_run)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch interactive dashboard")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.add_argument("--output-dir", default="outputs")
    p_dash.set_defaults(func=cmd_dashboard)

    # figures
    p_fig = sub.add_parser("figures", help="Export figure-ready data")
    p_fig.add_argument("--output-dir", default=None)
    p_fig.set_defaults(func=cmd_figures)


    # paper-figures
    p_pfig = sub.add_parser("paper-figures", help="Export Figures 2-18 for the paper")
    p_pfig.add_argument("--fmt", default="html", choices=["html","png","both"],
                        help="Output format (default: html)")
    p_pfig.add_argument("--figs", default="all",
                        help="Comma-separated figure numbers e.g. 5,6,7 or 'all'")
    p_pfig.add_argument("--output-dir", default=None)
    p_pfig.set_defaults(func=cmd_paper_figures)

    # check
    p_check = sub.add_parser("check", help="Validate environment")
    p_check.add_argument("--config", default="configs/base_config.yaml")
    p_check.set_defaults(func=cmd_check)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
