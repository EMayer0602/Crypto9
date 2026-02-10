#!/usr/bin/env python3
"""
Market Phase Report - All Pairs with Indicator Charts over 2 Years.

Creates interactive charts showing:
- Price (candlesticks)
- Three indicators: Supertrend, JMA, KAMA
- Market phase detection (Up/Down/Flat) with colored regions
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from main trading module
import Supertrend_5Min as st

# Configuration
REPORT_DIR = Path("report_html")
OUTPUT_FILE = REPORT_DIR / "market_phase_report.html"
TIMEFRAME = "1h"

# 2 years of hourly data = 365 * 2 * 24 = 17520 bars
LOOKBACK_BARS = 17520

# Phase colors for background regions
PHASE_COLORS = {
    "Up": "rgba(76, 175, 80, 0.15)",      # Green with transparency
    "Down": "rgba(244, 67, 54, 0.15)",    # Red with transparency
    "Flat": "rgba(158, 158, 158, 0.10)",  # Gray with transparency
}

# Indicator line colors
INDICATOR_COLORS = {
    "supertrend": "#2196F3",  # Blue
    "jma": "#FF9800",         # Orange
    "kama": "#9C27B0",        # Purple
}

# Indicator display names
INDICATOR_NAMES = {
    "supertrend": "Supertrend",
    "jma": "JMA (Jurik MA)",
    "kama": "KAMA (Kaufman AMA)",
}


def fetch_2year_data(symbol: str) -> pd.DataFrame:
    """
    Fetch 2 years of historical data for a symbol - BIS ZUM AKTUELLEN ZEITPUNKT (now()).

    Diese Funktion stellt sicher, dass die Daten:
    1. Mindestens 2 Jahre zurückreichen (historisch)
    2. BIS ZUM AKTUELLEN ZEITPUNKT reichen (keine veralteten Daten)
    3. Cache-Lücken automatisch gefüllt werden
    """
    print(f"[Data] Fetching {symbol}...")

    now = pd.Timestamp.now(tz=st.BERLIN_TZ)
    target_start = now - pd.Timedelta(days=730)  # ~2 years ago

    # Try loading from cache first
    df = st.load_ohlcv_from_cache(symbol, TIMEFRAME)

    if df is None or df.empty:
        print(f"[Data] No cache found for {symbol}, fetching from API...")
        try:
            df = st.download_historical_ohlcv(symbol, TIMEFRAME, target_start, now)
        except Exception as e:
            print(f"[Data] Error fetching {symbol}: {e}")
            return pd.DataFrame()
    else:
        # Check if we have enough historical data (at least 1.5 years back)
        earliest = df.index.min()
        if len(df) < 13000 or target_start < earliest:
            print(f"[Data] Cache has {len(df)} bars, extending historical data...")
            try:
                extra = st.download_historical_ohlcv(symbol, TIMEFRAME, target_start, earliest)
                if not extra.empty:
                    df = pd.concat([extra, df])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
            except Exception as e:
                print(f"[Data] Could not fetch additional historical data: {e}")

        # WICHTIG: Prüfen und Füllen der Lücke bis now()
        cache_end = df.index.max()
        tf_minutes = st.timeframe_to_minutes(TIMEFRAME)
        gap_threshold = pd.Timedelta(minutes=tf_minutes * 2)
        gap = now - cache_end

        if gap > gap_threshold:
            print(f"[Data] Cache ends at {cache_end.strftime('%Y-%m-%d %H:%M')}, gap of {gap.days}d {gap.seconds//3600}h to now")
            print(f"[Data] Fetching data from {cache_end.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}...")
            try:
                # Start from cache end (plus one bar to avoid duplicates)
                start_from = cache_end + pd.Timedelta(minutes=tf_minutes)
                new_data = st.download_historical_ohlcv(symbol, TIMEFRAME, start_from, now)

                if not new_data.empty:
                    df = pd.concat([df, new_data])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                    print(f"[Data] Filled gap: now have data up to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                print(f"[Data] Could not fill gap to now: {e}")

    if not df.empty:
        # Limit to last 2 years
        df = df.tail(LOOKBACK_BARS)

        # Verifiziere, dass Daten bis (nahe) jetzt gehen
        latest = df.index.max()
        hours_behind = (now - latest).total_seconds() / 3600

        print(f"[Data] {symbol}: {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        if hours_behind > 2:
            print(f"[Data] WARNING: Data is {hours_behind:.1f} hours behind current time!")

    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all three indicators on the dataframe."""
    if df.empty:
        return df

    df = df.copy()

    # 1. Supertrend (default params: length=10, factor=3.0)
    df_st = st.compute_supertrend(df.copy(), length=10, factor=3.0)
    df["supertrend"] = df_st["supertrend"]
    df["st_trend"] = df_st["st_trend"]

    # 2. JMA (default params: length=30, phase=0)
    df_jma = st.compute_jma(df.copy(), length=30, phase=0)
    df["jma"] = df_jma["jma"]
    df["jma_trend"] = df_jma["jma_trend"]

    # 3. KAMA (default params: length=20, slow_length=40)
    df_kama = st.compute_kama(df.copy(), length=20, slow_length=40)
    df["kama"] = df_kama["kama"]
    df["kama_trend"] = df_kama["kama_trend"]

    return df


def detect_market_phases(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Detect market phase for each bar based on indicator slopes."""
    if df.empty:
        return df

    df = df.copy()

    # Calculate phase for each indicator
    for indicator in ["supertrend", "jma", "kama"]:
        col = indicator
        if col not in df.columns:
            df[f"{indicator}_phase"] = "Flat"
            continue

        # Calculate rolling slope
        slopes = []
        for i in range(len(df)):
            if i < lookback:
                slopes.append(0.0)
            else:
                values = df[col].iloc[i-lookback:i+1]
                slope = st.calculate_indicator_slope(values, lookback)
                slopes.append(slope)

        df[f"{indicator}_slope"] = slopes

        # Determine phase from slope
        phases = []
        for slope in slopes:
            phase = st.determine_phase_from_slope(slope)
            phases.append(phase)

        df[f"{indicator}_phase"] = phases

    # Overall market phase (majority voting)
    def get_majority_phase(row):
        phases = [row.get("supertrend_phase", "Flat"),
                  row.get("jma_phase", "Flat"),
                  row.get("kama_phase", "Flat")]
        up_count = phases.count("Up")
        down_count = phases.count("Down")

        if up_count >= 2:
            return "Up"
        elif down_count >= 2:
            return "Down"
        else:
            return "Flat"

    df["market_phase"] = df.apply(get_majority_phase, axis=1)

    return df


def create_phase_regions(df: pd.DataFrame, phase_col: str = "market_phase") -> list:
    """Create shape regions for phase visualization (optimized)."""
    if df.empty or phase_col not in df.columns:
        return []

    shapes = []
    phases = df[phase_col].values
    indices = df.index.values

    # Find phase change points using numpy for speed
    phase_changes = np.where(phases[:-1] != phases[1:])[0]

    # Add start and end points
    change_points = np.concatenate([[0], phase_changes + 1, [len(phases)]])

    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1] - 1
        phase = phases[start_idx]

        shapes.append({
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": indices[start_idx],
            "x1": indices[end_idx],
            "y0": 0,
            "y1": 1,
            "fillcolor": PHASE_COLORS.get(phase, PHASE_COLORS["Flat"]),
            "line": {"width": 0},
            "layer": "below",
        })

    return shapes


def create_symbol_chart(symbol: str, df: pd.DataFrame) -> go.Figure:
    """Create interactive chart for a single symbol."""

    # Resample to 4h for smoother visualization (reduces 15k to ~4k points)
    df_4h = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "supertrend": "last",
        "jma": "last",
        "kama": "last",
        "market_phase": "last",
    }).dropna()

    # Create subplots: main price chart + phase indicator
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.85, 0.15],
        subplot_titles=[f"{symbol} - Price & Indicators (4h)", "Market Phase"]
    )

    # Candlestick chart (use resampled data)
    fig.add_trace(
        go.Candlestick(
            x=df_4h.index,
            open=df_4h["open"],
            high=df_4h["high"],
            low=df_4h["low"],
            close=df_4h["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1
    )

    # Add indicator lines (use resampled data)
    for indicator, color in INDICATOR_COLORS.items():
        if indicator in df_4h.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_4h.index,
                    y=df_4h[indicator],
                    mode="lines",
                    name=INDICATOR_NAMES[indicator],
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{INDICATOR_NAMES[indicator]}: %{{y:.2f}}<extra></extra>",
                ),
                row=1, col=1
            )

    # Add phase regions to main chart (use resampled data)
    shapes = create_phase_regions(df_4h, "market_phase")
    for shape in shapes:
        shape["yref"] = "y domain"
        fig.add_shape(shape, row=1, col=1)

    # Phase indicator (bottom panel) - use scatter instead of bar for speed
    phase_map = {"Up": 1, "Flat": 0, "Down": -1}
    phase_values = df_4h["market_phase"].map(phase_map)

    # Color by phase
    colors = df_4h["market_phase"].map({
        "Up": "#4CAF50",
        "Flat": "#9E9E9E",
        "Down": "#F44336"
    })

    fig.add_trace(
        go.Scatter(
            x=df_4h.index,
            y=phase_values,
            mode="markers",
            marker=dict(color=colors.tolist(), size=4),
            name="Phase",
            showlegend=False,
            hovertemplate="Phase: %{text}<extra></extra>",
            text=df_4h["market_phase"],
        ),
        row=2, col=1
    )

    # Calculate phase statistics
    phase_counts = df["market_phase"].value_counts()
    total = len(df)
    up_pct = phase_counts.get("Up", 0) / total * 100
    down_pct = phase_counts.get("Down", 0) / total * 100
    flat_pct = phase_counts.get("Flat", 0) / total * 100

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{symbol} - Market Phases Analysis (2 Years)<br>"
                 f"<sup>Up: {up_pct:.1f}% | Down: {down_pct:.1f}% | Flat: {flat_pct:.1f}%</sup>",
            font=dict(size=16),
        ),
        template="plotly_dark",
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(
        title_text="Phase",
        tickvals=[-1, 0, 1],
        ticktext=["Down", "Flat", "Up"],
        row=2, col=1
    )

    return fig


def create_overview_chart(all_data: dict) -> go.Figure:
    """Create overview chart showing phase distribution across all symbols."""

    symbols = []
    up_pcts = []
    down_pcts = []
    flat_pcts = []

    for symbol, df in all_data.items():
        if df.empty or "market_phase" not in df.columns:
            continue

        phase_counts = df["market_phase"].value_counts()
        total = len(df)

        symbols.append(symbol.replace("/", ""))
        up_pcts.append(phase_counts.get("Up", 0) / total * 100)
        down_pcts.append(phase_counts.get("Down", 0) / total * 100)
        flat_pcts.append(phase_counts.get("Flat", 0) / total * 100)

    fig = go.Figure()

    # Stacked bar chart
    fig.add_trace(go.Bar(
        name="Up Phase",
        x=symbols,
        y=up_pcts,
        marker_color="#4CAF50",
        text=[f"{v:.1f}%" for v in up_pcts],
        textposition="inside",
    ))

    fig.add_trace(go.Bar(
        name="Flat Phase",
        x=symbols,
        y=flat_pcts,
        marker_color="#9E9E9E",
        text=[f"{v:.1f}%" for v in flat_pcts],
        textposition="inside",
    ))

    fig.add_trace(go.Bar(
        name="Down Phase",
        x=symbols,
        y=down_pcts,
        marker_color="#F44336",
        text=[f"{v:.1f}%" for v in down_pcts],
        textposition="inside",
    ))

    fig.update_layout(
        title="Market Phase Distribution by Symbol (2 Years)",
        barmode="stack",
        template="plotly_dark",
        height=500,
        yaxis_title="Percentage (%)",
        xaxis_title="Symbol",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def generate_report(symbols: list = None):
    """Generate the complete market phase report."""

    # Use default symbols if not specified
    if symbols is None:
        symbols = st.SYMBOLS

    print(f"\n{'='*60}")
    print("Market Phase Report Generator")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Lookback: ~2 years ({LOOKBACK_BARS} bars)")
    print(f"{'='*60}\n")

    # Ensure report directory exists
    REPORT_DIR.mkdir(exist_ok=True)

    all_data = {}
    charts = []

    # Process each symbol
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")

        # Fetch data
        df = fetch_2year_data(symbol)
        if df.empty:
            print(f"  Skipping {symbol} - no data available")
            continue

        # Compute indicators
        print(f"  Computing indicators...")
        df = compute_all_indicators(df)

        # Detect phases
        print(f"  Detecting market phases...")
        df = detect_market_phases(df)

        all_data[symbol] = df

        # Create chart
        print(f"  Creating chart...")
        fig = create_symbol_chart(symbol, df)
        charts.append((symbol, fig))

        # Phase summary
        phase_counts = df["market_phase"].value_counts()
        total = len(df)
        print(f"  Phases: Up={phase_counts.get('Up', 0)} ({phase_counts.get('Up', 0)/total*100:.1f}%), "
              f"Down={phase_counts.get('Down', 0)} ({phase_counts.get('Down', 0)/total*100:.1f}%), "
              f"Flat={phase_counts.get('Flat', 0)} ({phase_counts.get('Flat', 0)/total*100:.1f}%)")

    # Create overview chart
    print(f"\nCreating overview chart...")
    overview_fig = create_overview_chart(all_data)

    # Generate HTML report
    print(f"\nGenerating HTML report...")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Market Phase Report - All Pairs (2 Years)</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        .indicator-legend {{
            background: #252542;
            padding: 15px 25px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: #252542;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .chart {{
            width: 100%;
        }}
        .nav {{
            position: fixed;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            background: #252542;
            border-radius: 8px;
            padding: 10px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .nav a {{
            display: block;
            color: #888;
            text-decoration: none;
            padding: 5px 10px;
            font-size: 12px;
            border-radius: 4px;
        }}
        .nav a:hover {{
            background: #3a3a5c;
            color: #fff;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #252542;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Market Phase Report</h1>
        <p class="subtitle">All Trading Pairs - 2 Year Analysis with 3 Indicators</p>

        <div class="indicator-legend">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3;"></div>
                    <span>Supertrend (Length=10, Factor=3.0)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF9800;"></div>
                    <span>JMA - Jurik MA (Length=30, Phase=0)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9C27B0;"></div>
                    <span>KAMA - Kaufman AMA (Length=20, Slow=40)</span>
                </div>
            </div>
            <div class="legend" style="margin-top: 15px;">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(76, 175, 80, 0.5);"></div>
                    <span>Up Phase (Bullish)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(158, 158, 158, 0.5);"></div>
                    <span>Flat Phase (Sideways)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(244, 67, 54, 0.5);"></div>
                    <span>Down Phase (Bearish)</span>
                </div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(symbols)}</div>
                <div class="stat-label">Trading Pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">3</div>
                <div class="stat-label">Indicators</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">~2 Years</div>
                <div class="stat-label">Time Period</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{LOOKBACK_BARS:,}</div>
                <div class="stat-label">Data Points/Symbol</div>
            </div>
        </div>

        <div class="chart-container">
            <h3 style="margin-top: 0;">Phase Distribution Overview</h3>
            <div id="overview-chart" class="chart"></div>
        </div>
"""

    # Add individual charts
    for i, (symbol, fig) in enumerate(charts):
        safe_id = symbol.replace("/", "_")
        html_content += f"""
        <div class="chart-container" id="section-{safe_id}">
            <div id="chart-{safe_id}" class="chart"></div>
        </div>
"""

    # Add navigation
    html_content += """
        <nav class="nav">
            <a href="#" onclick="window.scrollTo(0,0); return false;">Top</a>
"""
    for symbol, _ in charts:
        safe_id = symbol.replace("/", "_")
        html_content += f'            <a href="#section-{safe_id}">{symbol}</a>\n'

    html_content += """        </nav>

        <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>

    <script>
"""

    # Add chart data
    html_content += f"        var overviewData = {overview_fig.to_json()};\n"
    html_content += "        Plotly.newPlot('overview-chart', overviewData.data, overviewData.layout);\n"

    for symbol, fig in charts:
        safe_id = symbol.replace("/", "_")
        html_content += f"        var data_{safe_id} = {fig.to_json()};\n"
        html_content += f"        Plotly.newPlot('chart-{safe_id}', data_{safe_id}.data, data_{safe_id}.layout);\n"

    html_content += """
    </script>
</body>
</html>
"""

    # Write report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n{'='*60}")
    print(f"Report generated: {OUTPUT_FILE}")
    print(f"{'='*60}")

    return OUTPUT_FILE


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Market Phase Report")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to include")

    args = parser.parse_args()

    symbols = args.symbols if args.symbols else None
    generate_report(symbols)
