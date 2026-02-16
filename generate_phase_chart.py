#!/usr/bin/env python3
"""
Generate interactive 1h-Candlestick charts with market phase overlays.

Shows all 4 phase indicators (JMA, KAMA, Supertrend, HTF_Crossover)
as colored background bands: Green=Up, Red=Down, Gray=Flat.

Usage:
    python generate_phase_chart.py                        # All symbols, default params
    python generate_phase_chart.py --symbol BTC/USDC      # Single symbol
    python generate_phase_chart.py --ph1-tf 24h --ph1-length 20 --ph1-factor 3.0
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import Supertrend_5Min as st

REPORT_DIR = "report_html"

# Phase colors
PHASE_UP = "rgba(39, 174, 96, 0.15)"      # green transparent
PHASE_DOWN = "rgba(231, 76, 60, 0.15)"    # red transparent
PHASE_FLAT = "rgba(127, 140, 141, 0.10)"  # gray transparent
PHASE_UP_LINE = "#27ae60"
PHASE_DOWN_LINE = "#e74c3c"
PHASE_FLAT_LINE = "#7f8c8d"


def classify_phase_jma(df_htf, length=20, phase=0, threshold=0.0010):
    """Classify phases using JMA slope on HTF data."""
    df_ind = st.compute_jma(df_htf.copy(), length=length, phase=phase)
    if "jma" not in df_ind.columns:
        return pd.Series("Flat", index=df_htf.index), df_ind.get("jma", pd.Series(dtype=float))
    jma = df_ind["jma"]
    raw_slope = jma.diff()
    slope_pct = raw_slope / jma.shift(1)
    slope_pct = slope_pct.rolling(window=5, min_periods=1).mean()
    phases = pd.Series("Flat", index=df_htf.index)
    phases[slope_pct > threshold] = "Up"
    phases[slope_pct < -threshold] = "Down"
    return phases, jma


def classify_phase_kama(df_htf, length=20, slow_length=40, threshold=0.0001):
    """Classify phases using KAMA slope on HTF data."""
    df_ind = st.compute_kama(df_htf.copy(), length=length, slow_length=slow_length)
    if "kama" not in df_ind.columns:
        return pd.Series("Flat", index=df_htf.index), pd.Series(dtype=float)
    kama = df_ind["kama"]
    raw_slope = kama.diff()
    slope_pct = raw_slope / kama.shift(1)
    slope_pct = slope_pct.rolling(window=5, min_periods=1).mean()
    phases = pd.Series("Flat", index=df_htf.index)
    phases[slope_pct > threshold] = "Up"
    phases[slope_pct < -threshold] = "Down"
    return phases, kama


def classify_phase_supertrend(df_htf, length=10, factor=3.0):
    """Classify phases using Supertrend trend direction (no Flat)."""
    df_ind = st.compute_supertrend(df_htf.copy(), length=length, factor=factor)
    if "st_trend" not in df_ind.columns:
        return pd.Series("Flat", index=df_htf.index), pd.Series(dtype=float)
    phases = pd.Series("Flat", index=df_htf.index)
    phases[df_ind["st_trend"] == 1] = "Up"
    phases[df_ind["st_trend"] == -1] = "Down"
    return phases, df_ind["supertrend"]


def classify_phase_crossover(df_htf, length=10, factor=3.0):
    """Classify phases using HTF Crossover (same as Supertrend trend, no Flat)."""
    return classify_phase_supertrend(df_htf, length=length, factor=factor)


def align_to_1h(phases_htf, index_1h):
    """Forward-fill HTF phases to 1h index."""
    return phases_htf.reindex(index_1h, method="ffill")


def add_phase_background(fig, phases_1h, row, name_prefix=""):
    """Add colored background rectangles for phase regions."""
    if phases_1h.empty:
        return

    # Find contiguous phase blocks
    phase_changes = phases_1h != phases_1h.shift(1)
    block_starts = phases_1h.index[phase_changes]

    for i, start in enumerate(block_starts):
        end = block_starts[i + 1] if i + 1 < len(block_starts) else phases_1h.index[-1]
        phase = phases_1h.loc[start]
        if phase == "Down":
            color = PHASE_DOWN
        elif phase == "Up":
            color = PHASE_UP
        else:
            color = PHASE_FLAT

        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, layer="below", line_width=0,
            row=row, col=1,
        )


def build_phase_chart(symbol, ph1_tf="24h", ph1_length=20, ph1_factor=3.0,
                      start_date=None):
    """
    Build interactive chart: 1h candles + 4 phase indicators.

    Layout (5 rows):
      Row 1 (50%): 1h Candlesticks with JMA phase background (Down=red)
      Row 2 (12%): JMA phase band
      Row 3 (12%): KAMA phase band
      Row 4 (12%): Supertrend phase band
      Row 5 (12%): HTF_Crossover phase band
    """
    # Fetch 1h data
    df_1h = st.fetch_data(symbol, "1h", st.LOOKBACK)
    if df_1h.empty:
        print(f"[Skip] No 1h data for {symbol}")
        return None

    # Filter to date range
    if start_date:
        df_1h = df_1h[df_1h.index >= start_date]
    if df_1h.empty:
        print(f"[Skip] No data after {start_date} for {symbol}")
        return None

    # Fetch ph1 timeframe data
    htf_bars = (len(df_1h) // max(1, st.timeframe_to_minutes(ph1_tf) // 60)) + 200
    df_htf = st.fetch_data(symbol, ph1_tf, htf_bars)
    if df_htf.empty:
        print(f"[Skip] No {ph1_tf} data for {symbol}")
        return None

    # Compute all 4 phase classifiers on the ph1 timeframe
    phases_jma, line_jma = classify_phase_jma(df_htf, length=ph1_length, phase=0)
    phases_kama, line_kama = classify_phase_kama(df_htf, length=ph1_length, slow_length=max(ph1_length + 10, 40))
    phases_st, line_st = classify_phase_supertrend(df_htf, length=ph1_length, factor=ph1_factor)
    phases_cross, line_cross = classify_phase_crossover(df_htf, length=ph1_length, factor=ph1_factor)

    # Align to 1h index
    idx = df_1h.index
    ph_jma = align_to_1h(phases_jma, idx)
    ph_kama = align_to_1h(phases_kama, idx)
    ph_st = align_to_1h(phases_st, idx)
    ph_cross = align_to_1h(phases_cross, idx)

    # Phase stats
    for name, ph in [("JMA", ph_jma), ("KAMA", ph_kama), ("Supertrend", ph_st), ("Crossover", ph_cross)]:
        counts = ph.value_counts().to_dict()
        total = len(ph)
        pcts = {k: f"{v/total*100:.0f}%" for k, v in counts.items()}
        print(f"  {name}: {pcts}")

    # Build figure with subplots
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.52, 0.12, 0.12, 0.12, 0.12],
        subplot_titles=[
            f"{symbol} — 1h Candlesticks (JMA-Phasen)",
            "JMA Phase", "KAMA Phase", "Supertrend Phase", "HTF Crossover Phase",
        ],
    )

    # ── Row 1: Candlesticks ──
    x_dates = [d.strftime("%Y-%m-%d %H:%M") for d in idx]

    fig.add_trace(go.Candlestick(
        x=x_dates,
        open=df_1h["open"].values.tolist(),
        high=df_1h["high"].values.tolist(),
        low=df_1h["low"].values.tolist(),
        close=df_1h["close"].values.tolist(),
        name="Price",
        increasing_line_color="#27ae60",
        increasing_fillcolor="#27ae60",
        decreasing_line_color="#e74c3c",
        decreasing_fillcolor="#e74c3c",
    ), row=1, col=1)

    # Add JMA line on price chart (aligned to 1h)
    jma_aligned = line_jma.reindex(idx, method="ffill")
    if not jma_aligned.empty:
        fig.add_trace(go.Scatter(
            x=x_dates,
            y=jma_aligned.values.tolist(),
            mode="lines",
            name=f"JMA({ph1_length}) on {ph1_tf}",
            line=dict(color="#f39c12", width=2),
        ), row=1, col=1)

    # Add JMA Down phase background on price chart
    add_phase_background(fig, ph_jma, row=1)

    # ── Rows 2-5: Phase heatmap strips ──
    phase_data = [
        ("JMA", ph_jma, "#f39c12"),
        ("KAMA", ph_kama, "#3498db"),
        ("Supertrend", ph_st, "#e74c3c"),
        ("Crossover", ph_cross, "#9b59b6"),
    ]

    phase_map = {"Down": -1, "Flat": 0, "Up": 1}

    for i, (name, phases, color) in enumerate(phase_data):
        row_num = i + 2
        # Convert phases to numeric for coloring
        numeric = phases.map(phase_map).fillna(0).values.tolist()

        # Use colored scatter with fill
        fig.add_trace(go.Bar(
            x=x_dates,
            y=numeric,
            name=name,
            marker_color=[
                PHASE_DOWN_LINE if v == -1 else PHASE_UP_LINE if v == 1 else PHASE_FLAT_LINE
                for v in numeric
            ],
            showlegend=False,
        ), row=row_num, col=1)

        fig.update_yaxes(
            range=[-1.5, 1.5],
            tickvals=[-1, 0, 1],
            ticktext=["Down", "Flat", "Up"],
            row=row_num, col=1,
            fixedrange=True,
        )

    # Layout
    fig.update_layout(
        height=1200,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=False)),
        title=dict(
            text=f"{symbol} — Marktphasen-Analyse (ph1: {ph1_tf}, L={ph1_length}, F={ph1_factor})",
            font=dict(size=18),
        ),
        margin=dict(t=80, b=30),
    )

    # Disable rangeslider on all x-axes
    for i in range(1, 6):
        fig.update_xaxes(rangeslider=dict(visible=False), row=i, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate 1h candlestick + phase charts")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol (e.g. BTC/USDC). Default: all symbols")
    parser.add_argument("--ph1-tf", type=str, default="24h",
                        help="Phase indicator timeframe (default: 24h)")
    parser.add_argument("--ph1-length", type=int, default=20,
                        help="Phase indicator length (default: 20)")
    parser.add_argument("--ph1-factor", type=float, default=3.0,
                        help="Phase indicator factor (default: 3.0)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 1 year ago)")
    args = parser.parse_args()

    # Default start: 1 year ago
    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    symbols = [args.symbol] if args.symbol else st.SYMBOLS

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Configure exchange for data fetching
    st.configure_exchange()

    print(f"Phase Chart: ph1_tf={args.ph1_tf}, L={args.ph1_length}, F={args.ph1_factor}")
    print(f"Start: {start_date}, Symbols: {len(symbols)}")
    print("=" * 60)

    html_parts = []
    for sym_idx, symbol in enumerate(symbols, 1):
        print(f"\n[{sym_idx}/{len(symbols)}] {symbol}")
        fig = build_phase_chart(
            symbol,
            ph1_tf=args.ph1_tf,
            ph1_length=args.ph1_length,
            ph1_factor=args.ph1_factor,
            start_date=start_date,
        )
        if fig is None:
            continue
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not html_parts:
        print("No charts generated.")
        return

    # Write combined HTML
    now = datetime.now(st.BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    out_path = os.path.join(REPORT_DIR, "phase_chart.html")

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Marktphasen-Chart (ph1: {args.ph1_tf}, L={args.ph1_length}, F={args.ph1_factor})</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 20px; }}
h1 {{ color: #00d2ff; }}
p.sub {{ color: #888; }}
.chart-container {{ margin-bottom: 40px; }}
</style>
</head><body>
<h1>Marktphasen-Analyse</h1>
<p class="sub">ph1: {args.ph1_tf} | Length={args.ph1_length} | Factor={args.ph1_factor} | Stand: {now}</p>
<p class="sub">Hintergrund: <span style="color:#e74c3c">Rot=Down</span> | <span style="color:#27ae60">Grün=Up</span> | <span style="color:#7f8c8d">Grau=Flat</span></p>
"""

    for part in html_parts:
        html += f'<div class="chart-container">{part}</div>\n'

    html += "</body></html>"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n{'='*60}")
    print(f"Output: {out_path} ({len(html_parts)} charts)")


if __name__ == "__main__":
    main()
