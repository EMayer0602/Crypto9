#!/usr/bin/env python3
"""
Generate sweep_dashboard.html and market_phase_report.html for Crypto9.

sweep_dashboard.html  – Übersicht aller Sweep-Ergebnisse (nur Long)
market_phase_report.html – Pro Symbol:
    Oberer Chart : 1-Tages-Candlesticks + JMA, KAMA, Supertrend
    Unterer Chart: Marktphasen (Up / Down / Flat) für alle 3 Indikatoren
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── project imports ──────────────────────────────────────────────────
import Supertrend_5Min as st

REPORT_DIR = "report_html"
SWEEP_DASHBOARD_HTML = os.path.join(REPORT_DIR, "sweep_dashboard.html")
MARKET_PHASE_HTML = os.path.join(REPORT_DIR, "market_phase_report.html")

# Default indicator parameters (from best-params or defaults)
JMA_LENGTH = 30
JMA_PHASE = 0
KAMA_LENGTH = 20
KAMA_SLOW_LENGTH = 40
SUPERTREND_LENGTH = 10
SUPERTREND_FACTOR = 3.0


# =====================================================================
#  SWEEP DASHBOARD
# =====================================================================

def generate_sweep_dashboard():
    """Generate report_html/sweep_dashboard.html from best_params_overall.csv."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    params_csv = os.path.join(REPORT_DIR, "best_params_overall.csv")
    if not os.path.exists(params_csv):
        print(f"[Dashboard] {params_csv} nicht gefunden – überspringe.")
        return

    df = pd.read_csv(params_csv, sep=";", decimal=",")
    if df.empty:
        print("[Dashboard] Keine Daten in best_params_overall.csv")
        return

    # Nur Long-Trades
    df = df[df["Direction"].str.lower() == "long"].copy()
    if df.empty:
        print("[Dashboard] Keine Long-Ergebnisse vorhanden.")
        return

    # Numerische Spalten sicherstellen
    for col in ("FinalEquity", "Trades", "WinRate", "MaxDrawdown", "AvgPnL", "ProfitFactor"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("FinalEquity", ascending=False, inplace=True)

    now = datetime.now(st.BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

    # ── Summary stats ────────────────────────────────────────────────
    total_symbols = df["Symbol"].nunique()
    avg_equity = df["FinalEquity"].mean()
    avg_winrate = df["WinRate"].mean() * 100
    total_trades = int(df["Trades"].sum())
    best_symbol = df.iloc[0]["Symbol"] if not df.empty else "-"
    best_equity = df.iloc[0]["FinalEquity"] if not df.empty else 0

    # ── Equity bar chart ─────────────────────────────────────────────
    fig_equity = go.Figure()
    colors = ["#27ae60" if eq >= st.START_EQUITY else "#e74c3c" for eq in df["FinalEquity"]]
    fig_equity.add_trace(go.Bar(
        x=df["Symbol"],
        y=df["FinalEquity"],
        marker_color=colors,
        text=[f"${v:,.0f}" for v in df["FinalEquity"]],
        textposition="outside",
    ))
    fig_equity.add_hline(y=st.START_EQUITY, line_dash="dash", line_color="gray",
                         annotation_text=f"Start ${st.START_EQUITY:,.0f}")
    fig_equity.update_layout(
        title="Final Equity pro Symbol (nur Long)",
        yaxis_title="Equity (USD)",
        height=420,
        margin=dict(t=50, b=30),
    )
    equity_html = pio.to_html(fig_equity, include_plotlyjs=False, full_html=False)

    # ── Win-Rate bar chart ───────────────────────────────────────────
    fig_wr = go.Figure()
    wr_colors = ["#2980b9" if wr >= 0.6 else "#e67e22" for wr in df["WinRate"]]
    fig_wr.add_trace(go.Bar(
        x=df["Symbol"],
        y=df["WinRate"] * 100,
        marker_color=wr_colors,
        text=[f"{v*100:.1f}%" for v in df["WinRate"]],
        textposition="outside",
    ))
    fig_wr.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig_wr.update_layout(
        title="Win-Rate pro Symbol (nur Long)",
        yaxis_title="Win-Rate (%)",
        height=380,
        margin=dict(t=50, b=30),
    )
    winrate_html = pio.to_html(fig_wr, include_plotlyjs=False, full_html=False)

    # ── Parameter table ──────────────────────────────────────────────
    display_cols = [c for c in [
        "Symbol", "Indicator", "HTF", "ParamA", "ParamB",
        "ATRStopMult", "MinHoldBars", "Trades", "WinRate",
        "FinalEquity", "MaxDrawdown", "AvgPnL", "ProfitFactor",
    ] if c in df.columns]
    table_df = df[display_cols].copy()
    if "WinRate" in table_df.columns:
        table_df["WinRate"] = table_df["WinRate"].apply(lambda v: f"{v*100:.1f}%")
    if "FinalEquity" in table_df.columns:
        table_df["FinalEquity"] = table_df["FinalEquity"].apply(lambda v: f"${v:,.2f}")
    if "MaxDrawdown" in table_df.columns:
        table_df["MaxDrawdown"] = table_df["MaxDrawdown"].apply(lambda v: f"${v:,.2f}")
    if "AvgPnL" in table_df.columns:
        table_df["AvgPnL"] = table_df["AvgPnL"].apply(lambda v: f"${v:,.2f}")
    if "ProfitFactor" in table_df.columns:
        table_df["ProfitFactor"] = table_df["ProfitFactor"].apply(lambda v: f"{v:.2f}")
    param_table_html = table_df.to_html(index=False, classes="param-table", border=0, justify="left")

    # ── Assemble HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Sweep Dashboard – Crypto9</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
  h2 {{ color: #00d4ff; margin-top: 30px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
  .summary-card {{ background: #16213e; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #0f3460; }}
  .summary-card .value {{ font-size: 28px; font-weight: bold; color: #00d4ff; }}
  .summary-card .label {{ font-size: 13px; color: #aaa; margin-top: 5px; }}
  .param-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
  .param-table th {{ background: #0f3460; padding: 10px 8px; text-align: left; border-bottom: 2px solid #00d4ff; }}
  .param-table td {{ padding: 8px; border-bottom: 1px solid #2a2a4a; }}
  .param-table tr:hover {{ background: #16213e; }}
  .chart-container {{ background: #16213e; border-radius: 10px; padding: 15px; margin: 15px 0; border: 1px solid #0f3460; }}
  .footer {{ margin-top: 40px; text-align: center; color: #666; font-size: 12px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
  .badge-long {{ background: #27ae60; color: white; }}
  .note {{ background: #16213e; border-left: 4px solid #e74c3c; padding: 10px 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
</style>
</head>
<body>

<h1>Sweep Dashboard – Crypto9</h1>
<p>Stand: {now} &nbsp; <span class="badge badge-long">NUR LONG</span> &nbsp; Max Open Positions: 10</p>

<div class="note">Short-Trades sind deaktiviert. Alle Ergebnisse zeigen ausschließlich Long-Strategien.</div>

<div class="summary-grid">
  <div class="summary-card"><div class="value">{total_symbols}</div><div class="label">Symbole</div></div>
  <div class="summary-card"><div class="value">${avg_equity:,.0f}</div><div class="label">Ø Final Equity</div></div>
  <div class="summary-card"><div class="value">{avg_winrate:.1f}%</div><div class="label">Ø Win-Rate</div></div>
  <div class="summary-card"><div class="value">{total_trades:,}</div><div class="label">Total Trades</div></div>
  <div class="summary-card"><div class="value">{best_symbol}</div><div class="label">Bestes Symbol</div></div>
  <div class="summary-card"><div class="value">${best_equity:,.0f}</div><div class="label">Beste Equity</div></div>
</div>

<h2>Final Equity pro Symbol</h2>
<div class="chart-container">{equity_html}</div>

<h2>Win-Rate pro Symbol</h2>
<div class="chart-container">{winrate_html}</div>

<h2>Optimale Parameter (Long)</h2>
{param_table_html}

<div class="footer">Crypto9 Sweep Dashboard – generiert am {now}</div>
</body>
</html>"""

    with open(SWEEP_DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Dashboard] Gespeichert: {SWEEP_DASHBOARD_HTML}")


# =====================================================================
#  MARKET PHASE REPORT
# =====================================================================

def classify_market_phase(trend_series):
    """
    Classify market phases from a trend series.
    trend == 1  → 'Up'
    trend == -1 → 'Down'
    otherwise   → 'Flat'
    """
    phases = pd.Series("Flat", index=trend_series.index)
    phases[trend_series == 1] = "Up"
    phases[trend_series == -1] = "Down"
    return phases


def compute_indicators_daily(df_daily):
    """Compute JMA, KAMA, and Supertrend on daily OHLCV data."""
    # JMA
    df_jma = st.compute_jma(df_daily.copy(), length=JMA_LENGTH, phase=JMA_PHASE)
    # KAMA
    df_kama = st.compute_kama(df_daily.copy(), length=KAMA_LENGTH, slow_length=KAMA_SLOW_LENGTH)
    # Supertrend
    df_st = st.compute_supertrend(df_daily.copy(), length=SUPERTREND_LENGTH, factor=SUPERTREND_FACTOR)

    return df_jma, df_kama, df_st


def build_market_phase_chart(symbol, df_daily, df_jma, df_kama, df_supertrend):
    """
    Build a 2-row chart:
      Row 1: 1D Candlesticks + JMA + KAMA + Supertrend lines
      Row 2: Market phases (Up/Down/Flat) for each indicator as colored bands
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f"{symbol} – 1D Candlesticks + Indikatoren",
            f"{symbol} – Marktphasen (JMA / KAMA / Supertrend)",
        ),
    )

    # ── Row 1: Candlesticks ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df_daily.index,
        open=df_daily["open"],
        high=df_daily["high"],
        low=df_daily["low"],
        close=df_daily["close"],
        name="Price",
        increasing_line_color="#27ae60",
        decreasing_line_color="#e74c3c",
    ), row=1, col=1)

    # JMA line
    fig.add_trace(go.Scatter(
        x=df_jma.index, y=df_jma["jma"],
        mode="lines", name=f"JMA({JMA_LENGTH})",
        line=dict(color="#f39c12", width=2),
    ), row=1, col=1)

    # KAMA line
    fig.add_trace(go.Scatter(
        x=df_kama.index, y=df_kama["kama"],
        mode="lines", name=f"KAMA({KAMA_LENGTH},{KAMA_SLOW_LENGTH})",
        line=dict(color="#3498db", width=2),
    ), row=1, col=1)

    # Supertrend line
    fig.add_trace(go.Scatter(
        x=df_supertrend.index, y=df_supertrend["supertrend"],
        mode="lines", name=f"Supertrend({SUPERTREND_LENGTH},{SUPERTREND_FACTOR})",
        line=dict(color="#e74c3c", width=2, dash="dot"),
    ), row=1, col=1)

    # ── Row 2: Market Phases ─────────────────────────────────────────
    phase_jma = classify_market_phase(df_jma["jma_trend"])
    phase_kama = classify_market_phase(df_kama["kama_trend"])
    phase_st = classify_market_phase(df_supertrend["st_trend"])

    color_map = {"Up": "#27ae60", "Down": "#e74c3c", "Flat": "#7f8c8d"}
    indicators = [
        ("JMA", phase_jma, 3),
        ("KAMA", phase_kama, 2),
        ("Supertrend", phase_st, 1),
    ]

    for ind_name, phases, y_level in indicators:
        # Render phase blocks as colored segments
        dates = phases.index.tolist()
        values = phases.values.tolist()

        i = 0
        while i < len(dates):
            current_phase = values[i]
            j = i
            while j < len(dates) and values[j] == current_phase:
                j += 1
            # Draw a colored rectangle for this phase segment
            x0 = dates[i]
            x1 = dates[j - 1] if j - 1 < len(dates) else dates[-1]
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=y_level - 0.35, y1=y_level + 0.35,
                fillcolor=color_map.get(current_phase, "#7f8c8d"),
                opacity=0.7,
                line_width=0,
                row=2, col=1,
            )
            i = j

    # Y-axis labels for phase chart
    fig.update_yaxes(
        tickvals=[1, 2, 3],
        ticktext=["Supertrend", "KAMA", "JMA"],
        range=[0.3, 3.7],
        row=2, col=1,
    )

    # Add legend entries for phase colors
    for phase_name, phase_color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=phase_color, symbol="square"),
            name=phase_name,
            showlegend=True,
        ), row=2, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2=dict(rangeslider=dict(visible=True, thickness=0.04), type="date"),
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Preis", row=1, col=1)
    fig.update_xaxes(title_text="Datum", row=2, col=1)

    return fig


def generate_market_phase_report():
    """Generate report_html/market_phase_report.html for all symbols."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    symbols = st.SYMBOLS
    now = datetime.now(st.BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

    chart_sections = []

    for symbol in symbols:
        print(f"[MarketPhase] Verarbeite {symbol}...")
        try:
            # Fetch daily data from cache or API
            df_daily = st.fetch_data(symbol, "1d", 365)
            if df_daily is None or df_daily.empty or len(df_daily) < 30:
                print(f"[MarketPhase] Nicht genug Daten für {symbol}, überspringe.")
                continue

            df_jma, df_kama, df_supertrend = compute_indicators_daily(df_daily)
            fig = build_market_phase_chart(symbol, df_daily, df_jma, df_kama, df_supertrend)
            fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
            chart_sections.append((symbol, fig_html))
        except Exception as exc:
            print(f"[MarketPhase] Fehler bei {symbol}: {exc}")
            continue

    if not chart_sections:
        print("[MarketPhase] Keine Charts generiert.")
        return

    # ── Build navigation ─────────────────────────────────────────────
    nav_links = " | ".join(
        f'<a href="#{sym.replace("/", "_")}">{sym}</a>' for sym, _ in chart_sections
    )

    # ── Assemble HTML ────────────────────────────────────────────────
    sections_html = []
    for sym, fig_html in chart_sections:
        anchor = sym.replace("/", "_")
        sections_html.append(f'''
        <div id="{anchor}" class="chart-section">
            <h2>{sym}</h2>
            <div class="chart-container">{fig_html}</div>
        </div>''')

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Marktphasen-Report – Crypto9</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
  h2 {{ color: #00d4ff; margin-top: 30px; }}
  .nav {{ background: #16213e; padding: 12px 20px; border-radius: 8px; margin: 15px 0; font-size: 14px; }}
  .nav a {{ color: #00d4ff; text-decoration: none; margin: 0 4px; }}
  .nav a:hover {{ text-decoration: underline; }}
  .chart-section {{ margin-bottom: 40px; }}
  .chart-container {{ background: #16213e; border-radius: 10px; padding: 15px; border: 1px solid #0f3460; }}
  .legend-box {{ display: flex; gap: 20px; margin: 10px 0; padding: 10px; background: #16213e; border-radius: 8px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 16px; height: 16px; border-radius: 3px; }}
  .footer {{ margin-top: 40px; text-align: center; color: #666; font-size: 12px; }}
  .info {{ background: #16213e; border-left: 4px solid #3498db; padding: 10px 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
</style>
</head>
<body>

<h1>Marktphasen-Report – Crypto9</h1>
<p>Stand: {now}</p>

<div class="info">
    <strong>Oberer Chart:</strong> 1-Tages-Candlesticks mit JMA({JMA_LENGTH}), KAMA({KAMA_LENGTH},{KAMA_SLOW_LENGTH}) und Supertrend({SUPERTREND_LENGTH},{SUPERTREND_FACTOR})<br>
    <strong>Unterer Chart:</strong> Marktphasen (Up / Down / Flat) für jeden Indikator
</div>

<div class="legend-box">
    <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div> Up (Aufwärtstrend)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div> Down (Abwärtstrend)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#7f8c8d"></div> Flat (Seitwärts)</div>
</div>

<div class="nav">Navigation: {nav_links}</div>

{''.join(sections_html)}

<div class="footer">Crypto9 Marktphasen-Report – generiert am {now}</div>
</body>
</html>"""

    with open(MARKET_PHASE_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[MarketPhase] Gespeichert: {MARKET_PHASE_HTML}")


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Crypto9 Report Generator")
    print("=" * 60)

    print("\n[1/2] Generiere Sweep Dashboard...")
    generate_sweep_dashboard()

    print("\n[2/2] Generiere Marktphasen-Report...")
    generate_market_phase_report()

    print("\nFertig!")
