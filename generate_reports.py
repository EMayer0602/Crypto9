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
    """Generate report_html/sweep_dashboard.html from best_params_overall.csv + Trade-CSVs."""
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

    for col in ("FinalEquity", "Trades", "WinRate", "MaxDrawdown", "AvgPnL", "ProfitFactor"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("FinalEquity", ascending=False, inplace=True)

    now = datetime.now(st.BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    stake_info = f"Equity / {st.STAKE_DIVISOR} (dynamisch)"
    max_pos = 12

    # ── Summary stats ────────────────────────────────────────────────
    total_symbols = df["Symbol"].nunique()
    avg_equity = df["FinalEquity"].mean()
    avg_winrate = df["WinRate"].mean() * 100
    total_trades = int(df["Trades"].sum())
    best_symbol = df.iloc[0]["Symbol"] if not df.empty else "-"
    best_equity = df.iloc[0]["FinalEquity"] if not df.empty else 0

    # ── Equity bar chart ─────────────────────────────────────────────
    fig_equity = go.Figure()
    colors = ["#27ae60" if eq >= st.START_EQUITY else "#e74c3c" for eq in df["FinalEquity"].values]
    fig_equity.add_trace(go.Bar(
        x=df["Symbol"].values.tolist(),
        y=df["FinalEquity"].values.tolist(),
        marker_color=colors,
        text=[f"${v:,.0f}" for v in df["FinalEquity"].values],
        textposition="outside",
    ))
    fig_equity.add_hline(y=st.START_EQUITY, line_dash="dash", line_color="gray",
                         annotation_text=f"Start ${st.START_EQUITY:,.0f}")
    fig_equity.update_layout(
        title="Final Equity pro Symbol (nur Long)",
        yaxis_title="Equity (USD)",
        height=420, margin=dict(t=50, b=30), template="plotly_dark",
    )
    equity_html = pio.to_html(fig_equity, include_plotlyjs=False, full_html=False)

    # ── Win-Rate bar chart ───────────────────────────────────────────
    fig_wr = go.Figure()
    wr_colors = ["#2980b9" if wr >= 0.6 else "#e67e22" for wr in df["WinRate"].values]
    fig_wr.add_trace(go.Bar(
        x=df["Symbol"].values.tolist(),
        y=(df["WinRate"] * 100).values.tolist(),
        marker_color=wr_colors,
        text=[f"{v*100:.1f}%" for v in df["WinRate"].values],
        textposition="outside",
    ))
    fig_wr.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig_wr.update_layout(
        title="Win-Rate pro Symbol (nur Long)",
        yaxis_title="Win-Rate (%)",
        height=380, margin=dict(t=50, b=30), template="plotly_dark",
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

    # ── Trade-Listen pro Symbol ──────────────────────────────────────
    trade_sections = []
    nav_links = []
    for _, row in df.iterrows():
        symbol = row["Symbol"]
        anchor = symbol.replace("/", "_")
        nav_links.append(f'<a href="#{anchor}">{symbol}</a>')
        trades_csv = row.get("TradesCSV", "")
        if not trades_csv or pd.isna(trades_csv) or not os.path.exists(trades_csv):
            trade_sections.append(f'<div id="{anchor}" class="symbol-section"><h3>{symbol}</h3>'
                                  f'<p class="no-trades">Keine Trade-CSV vorhanden ({trades_csv})</p></div>')
            continue

        trades = pd.read_csv(trades_csv, sep=";", decimal=",")
        if trades.empty:
            trade_sections.append(f'<div id="{anchor}" class="symbol-section"><h3>{symbol}</h3>'
                                  f'<p class="no-trades">Keine Trades</p></div>')
            continue

        # Nur Long
        if "Direction" in trades.columns:
            trades = trades[trades["Direction"].str.lower() == "long"]

        # Numerisch konvertieren
        for col in ("Stake", "Fees", "PnL (USD)", "Equity", "Entry", "ExitPreis"):
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")

        # Stats
        n_trades = len(trades)
        wins = (trades["PnL (USD)"] > 0).sum() if "PnL (USD)" in trades.columns else 0
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        total_pnl = trades["PnL (USD)"].sum() if "PnL (USD)" in trades.columns else 0
        final_eq = trades["Equity"].iloc[-1] if "Equity" in trades.columns and len(trades) > 0 else 0

        # Equity Curve
        if "Equity" in trades.columns and len(trades) > 1:
            eq_values = trades["Equity"].values.tolist()
            eq_x = list(range(len(eq_values)))
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_x, y=eq_values,
                mode="lines", fill="tozeroy",
                line=dict(color="#00d4ff", width=2),
                fillcolor="rgba(0,212,255,0.15)",
            ))
            fig_eq.update_layout(
                height=200, margin=dict(t=10, b=30, l=60, r=20),
                template="plotly_dark",
                xaxis=dict(title="Trade #"), yaxis=dict(title="Equity"),
            )
            eq_chart = pio.to_html(fig_eq, include_plotlyjs=False, full_html=False)
        else:
            eq_chart = ""

        # Trade Tabelle (formatiert)
        trade_display = trades.copy()
        fmt_cols = {
            "Entry": lambda v: f"{v:.4f}" if pd.notna(v) else "",
            "ExitPreis": lambda v: f"{v:.4f}" if pd.notna(v) else "",
            "Stake": lambda v: f"${v:,.2f}" if pd.notna(v) else "",
            "Fees": lambda v: f"${v:,.2f}" if pd.notna(v) else "",
            "PnL (USD)": lambda v: f"${v:+,.2f}" if pd.notna(v) else "",
            "Equity": lambda v: f"${v:,.2f}" if pd.notna(v) else "",
        }
        for col, fmt in fmt_cols.items():
            if col in trade_display.columns:
                trade_display[col] = trade_display[col].apply(fmt)

        show_cols = [c for c in ["Zeit", "Entry", "ExitZeit", "ExitPreis", "Stake",
                                  "ExitReason", "PnL (USD)", "Equity"] if c in trade_display.columns]
        trade_table = trade_display[show_cols].to_html(
            index=False, classes="trade-table", border=0, justify="left")

        pnl_class = "positive" if total_pnl >= 0 else "negative"
        section = f'''<div id="{anchor}" class="symbol-section">
  <h3>{symbol} <span class="badge badge-long">LONG</span>
    <span class="stats">{n_trades} Trades | WR: {wr:.1f}% |
    PnL: <span class="{pnl_class}">${total_pnl:+,.2f}</span> |
    Final: ${final_eq:,.2f}</span></h3>
  <div class="chart-container">{eq_chart}</div>
  <div class="table-scroll">{trade_table}</div>
</div>'''
        trade_sections.append(section)
        print(f"  [Dashboard] {symbol}: {n_trades} Trades, PnL=${total_pnl:+,.2f}")

    nav_html = " | ".join(nav_links)

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
  h3 {{ color: #e0e0e0; margin-top: 20px; border-bottom: 1px solid #0f3460; padding-bottom: 8px; }}
  h3 .stats {{ font-size: 14px; font-weight: normal; color: #aaa; margin-left: 10px; }}
  .nav {{ background: #16213e; padding: 12px 20px; border-radius: 8px; margin: 15px 0; font-size: 14px; }}
  .nav a {{ color: #00d4ff; text-decoration: none; margin: 0 4px; }}
  .nav a:hover {{ text-decoration: underline; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
  .summary-card {{ background: #16213e; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #0f3460; }}
  .summary-card .value {{ font-size: 26px; font-weight: bold; color: #00d4ff; }}
  .summary-card .label {{ font-size: 13px; color: #aaa; margin-top: 5px; }}
  .param-table, .trade-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
  .param-table th, .trade-table th {{ background: #0f3460; padding: 8px 6px; text-align: left; border-bottom: 2px solid #00d4ff; }}
  .param-table td, .trade-table td {{ padding: 6px; border-bottom: 1px solid #2a2a4a; }}
  .param-table tr:hover, .trade-table tr:hover {{ background: #16213e; }}
  .chart-container {{ background: #16213e; border-radius: 10px; padding: 10px; margin: 10px 0; border: 1px solid #0f3460; }}
  .symbol-section {{ margin-bottom: 30px; background: #0d1b2a; border-radius: 10px; padding: 15px; }}
  .table-scroll {{ max-height: 400px; overflow-y: auto; }}
  .footer {{ margin-top: 40px; text-align: center; color: #666; font-size: 12px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
  .badge-long {{ background: #27ae60; color: white; }}
  .note {{ background: #16213e; border-left: 4px solid #e74c3c; padding: 10px 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
  .positive {{ color: #27ae60; }}
  .negative {{ color: #e74c3c; }}
  .no-trades {{ color: #888; font-style: italic; }}
</style>
</head>
<body>

<h1>Sweep Dashboard – Crypto9</h1>
<p>Stand: {now} &nbsp; <span class="badge badge-long">NUR LONG</span> &nbsp;
  Max Open Positions: {max_pos} &nbsp; Stake: {stake_info}</p>

<div class="note">Short-Trades sind deaktiviert. Alle Ergebnisse zeigen ausschließlich Long-Strategien.
  Dynamischer Stake: Position = aktuelle Equity / {st.STAKE_DIVISOR}.</div>

<div class="summary-grid">
  <div class="summary-card"><div class="value">{total_symbols}</div><div class="label">Symbole</div></div>
  <div class="summary-card"><div class="value">${avg_equity:,.0f}</div><div class="label">&Oslash; Final Equity</div></div>
  <div class="summary-card"><div class="value">{avg_winrate:.1f}%</div><div class="label">&Oslash; Win-Rate</div></div>
  <div class="summary-card"><div class="value">{total_trades:,}</div><div class="label">Total Trades</div></div>
  <div class="summary-card"><div class="value">{best_symbol}</div><div class="label">Bestes Symbol</div></div>
  <div class="summary-card"><div class="value">${best_equity:,.0f}</div><div class="label">Beste Equity</div></div>
  <div class="summary-card"><div class="value">{max_pos}</div><div class="label">Max Positionen</div></div>
</div>

<h2>Final Equity pro Symbol</h2>
<div class="chart-container">{equity_html}</div>

<h2>Win-Rate pro Symbol</h2>
<div class="chart-container">{winrate_html}</div>

<h2>Optimale Parameter (Long)</h2>
{param_table_html}

<h2>Trade-Listen pro Symbol</h2>
<div class="nav">Navigation: {nav_html}</div>
{''.join(trade_sections)}

<div class="footer">Crypto9 Sweep Dashboard – generiert am {now}</div>
</body>
</html>"""

    with open(SWEEP_DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Dashboard] Gespeichert: {SWEEP_DASHBOARD_HTML}")


# =====================================================================
#  MARKET PHASE REPORT
# =====================================================================

def classify_phase_by_slope(indicator_series, smooth_window=5, threshold_pct=0.001):
    """
    Slope-basierte Marktphasen-Klassifikation.

    Berechnet die prozentuale Steigung der Indikator-Linie und klassifiziert:
      slope_pct >  threshold → Up  (Aufwärtstrend)
      slope_pct < -threshold → Down (Abwärtstrend)
      sonst                  → Flat (Seitwärts / Konsolidierung)

    Args:
        indicator_series: pd.Series mit Indikator-Werten (z.B. JMA, KAMA, Supertrend)
        smooth_window: Glättung der Steigung (Rolling Mean), verhindert Rauschen
        threshold_pct: Schwellwert für Up/Down (0.001 = 0.1% Änderung pro Bar)
    """
    # Steigung berechnen (erste Ableitung)
    raw_slope = indicator_series.diff()

    # Prozentuale Steigung relativ zum Preisniveau
    slope_pct = raw_slope / indicator_series.shift(1)

    # Glätten um Rauschen zu reduzieren
    if smooth_window > 1:
        slope_pct = slope_pct.rolling(window=smooth_window, min_periods=1).mean()

    # Klassifikation
    phases = pd.Series("Flat", index=indicator_series.index)
    phases[slope_pct > threshold_pct] = "Up"
    phases[slope_pct < -threshold_pct] = "Down"

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
    Build TWO separate figures (no subplots – avoids Candlestick+Heatmap conflicts):
      fig_candle: 1D Candlesticks + JMA + KAMA + Supertrend lines
      fig_phase:  Market phases (Up/Down/Flat) as go.Heatmap
    """
    # ── Figure 1: Candlestick + Indikatoren ──────────────────────────
    fig_candle = go.Figure()

    # Plotly braucht plain Python lists, keine pandas Series (sonst bdata-Bug)
    x_dates = [d.strftime("%Y-%m-%d") for d in df_daily.index]

    fig_candle.add_trace(go.Candlestick(
        x=x_dates,
        open=df_daily["open"].values.tolist(),
        high=df_daily["high"].values.tolist(),
        low=df_daily["low"].values.tolist(),
        close=df_daily["close"].values.tolist(),
        name="Price",
        increasing_line_color="#27ae60",
        increasing_fillcolor="#27ae60",
        decreasing_line_color="#e74c3c",
        decreasing_fillcolor="#e74c3c",
    ))

    fig_candle.add_trace(go.Scatter(
        x=x_dates, y=df_jma["jma"].values.tolist(),
        mode="lines", name=f"JMA({JMA_LENGTH})",
        line=dict(color="#f39c12", width=2),
    ))

    fig_candle.add_trace(go.Scatter(
        x=x_dates, y=df_kama["kama"].values.tolist(),
        mode="lines", name=f"KAMA({KAMA_LENGTH},{KAMA_SLOW_LENGTH})",
        line=dict(color="#3498db", width=2),
    ))

    fig_candle.add_trace(go.Scatter(
        x=x_dates, y=df_supertrend["supertrend"].values.tolist(),
        mode="lines", name=f"Supertrend({SUPERTREND_LENGTH},{SUPERTREND_FACTOR})",
        line=dict(color="#e74c3c", width=2, dash="dot"),
    ))

    price_min = df_daily["low"].min() * 0.98
    price_max = df_daily["high"].max() * 1.02

    fig_candle.update_layout(
        title=f"{symbol} – 1D Candlesticks + JMA / KAMA / Supertrend",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        yaxis=dict(title="Preis", range=[price_min, price_max]),
        template="plotly_dark",
    )

    # ── Figure 2: Marktphasen-Heatmap ────────────────────────────────
    phase_jma = classify_phase_by_slope(df_jma["jma"], smooth_window=5, threshold_pct=0.001)
    phase_kama = classify_phase_by_slope(df_kama["kama"], smooth_window=5, threshold_pct=0.001)
    phase_st = classify_phase_by_slope(df_supertrend["supertrend"], smooth_window=3, threshold_pct=0.002)

    phase_consensus = pd.Series("Flat", index=df_daily.index)
    for idx in df_daily.index:
        votes = [phase_jma.get(idx, "Flat"), phase_kama.get(idx, "Flat"), phase_st.get(idx, "Flat")]
        up_count = votes.count("Up")
        down_count = votes.count("Down")
        if up_count >= 2:
            phase_consensus[idx] = "Up"
        elif down_count >= 2:
            phase_consensus[idx] = "Down"

    phase_map = {"Down": 0, "Flat": 1, "Up": 2}

    indicators_data = [
        ("Consensus (2/3)", phase_consensus),
        ("Supertrend", phase_st),
        ("KAMA", phase_kama),
        ("JMA", phase_jma),
    ]

    z_matrix = []
    text_matrix = []
    y_labels = []

    for name, phases in indicators_data:
        row_z = [phase_map.get(p, 1) for p in phases.values]
        row_text = [f"{name}: {p}" for p in phases.values]
        z_matrix.append(row_z)
        text_matrix.append(row_text)
        y_labels.append(name)

    fig_phase = go.Figure()

    fig_phase.add_trace(go.Heatmap(
        x=x_dates,
        y=y_labels,
        z=z_matrix,
        text=text_matrix,
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0, "#e74c3c"],
            [0.5, "#7f8c8d"],
            [1.0, "#27ae60"],
        ],
        showscale=False,
        xgap=1,
        ygap=4,
        zmin=0,
        zmax=2,
    ))

    fig_phase.update_layout(
        title=f"{symbol} – Marktphasen + Consensus (2 von 3 = Signal)",
        height=250,
        xaxis=dict(type="date"),
        yaxis=dict(title="Indikator", fixedrange=True),
        template="plotly_dark",
        margin=dict(t=40, b=30),
    )

    return fig_candle, fig_phase


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
            fig_candle, fig_phase = build_market_phase_chart(symbol, df_daily, df_jma, df_kama, df_supertrend)
            candle_html = pio.to_html(fig_candle, include_plotlyjs=False, full_html=False)
            phase_html = pio.to_html(fig_phase, include_plotlyjs=False, full_html=False)
            chart_sections.append((symbol, candle_html + phase_html))
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
    <strong>Unterer Chart:</strong> Marktphasen (Up / Down / Flat) für jeden Indikator + <strong>Consensus</strong> (2 von 3 müssen übereinstimmen)
</div>

<div class="legend-box">
    <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div> JMA – Orange-Töne</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div> KAMA – Blau-Töne</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div> Supertrend – Rot/Grün-Töne</div>
    <div class="legend-item"><div class="legend-dot" style="background:#00ff88"></div> Consensus Up</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff3366"></div> Consensus Down</div>
    <div class="legend-item"><div class="legend-dot" style="background:#555555"></div> Kein Konsens (Flat)</div>
</div>
<div class="info" style="border-left-color:#00ff88;">
    <strong>Interpretation:</strong> Consensus = 2 von 3 Indikatoren stimmen überein → starkes Signal. Grüner Consensus-Balken = Long-Einstieg prüfen. Roter Consensus-Balken = Position schließen/meiden.
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
