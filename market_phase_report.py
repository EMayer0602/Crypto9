#!/usr/bin/env python3
"""
Market Phase Report - All Pairs with Indicator Charts on DAILY Basis.

Creates interactive charts showing:
- Row 1 (top): Daily Candlesticks with three indicators overlaid: Supertrend, JMA, KAMA
- Row 2 (bottom): Market phase as numeric values (1=Up, 0=Flat, -1=Down)

WICHTIG: Die Phasenbestimmung erfolgt auf TAGESBASIS (Daily Bars),
wodurch Phasen Wochen oder Monate dauern können.
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

# USE DAILY TIMEFRAME FOR PHASE DETECTION
DAILY_TIMEFRAME = "1d"
LOOKBACK_DAYS = 730  # ~2 years of daily data

# Phase colors for background regions (used in phase chart)
PHASE_COLORS = {
    "Up": "rgba(76, 175, 80, 0.3)",      # Green with transparency
    "Down": "rgba(244, 67, 54, 0.3)",    # Red with transparency
    "Flat": "rgba(158, 158, 158, 0.2)",  # Gray with transparency
}

# Indicator line colors (matching common trading platform colors)
INDICATOR_COLORS = {
    "supertrend": "#FF5722",  # Orange-Red (Supertrend typical)
    "jma": "#2196F3",         # Blue (JMA)
    "kama": "#9C27B0",        # Purple (KAMA)
}

# Indicator display names
INDICATOR_NAMES = {
    "supertrend": "Supertrend (10, 3.0)",
    "jma": "JMA (30, 0)",
    "kama": "KAMA (20, 40)",
}

# Phase numeric values
PHASE_VALUES = {
    "Up": 1,
    "Flat": 0,
    "Down": -1,
}


def fetch_daily_data(symbol: str) -> pd.DataFrame:
    """
    Fetch 2 years of DAILY historical data for a symbol.

    Die Phasenbestimmung erfolgt auf Tagesbasis, damit die Phasen
    Wochen oder Monate dauern können (nicht Stunden).
    """
    print(f"[Data] Fetching DAILY data for {symbol}...")

    now = pd.Timestamp.now(tz=st.BERLIN_TZ)
    target_start = now - pd.Timedelta(days=LOOKBACK_DAYS)

    # Try loading from cache first
    df = st.load_ohlcv_from_cache(symbol, DAILY_TIMEFRAME)

    if df is None or df.empty:
        print(f"[Data] No cache found for {symbol} (1d), fetching from API...")
        try:
            df = st.download_historical_ohlcv(symbol, DAILY_TIMEFRAME, target_start, now)
            if df is not None and not df.empty:
                st.save_ohlcv_to_cache(symbol, DAILY_TIMEFRAME, df)
        except Exception as e:
            print(f"[Data] Error fetching {symbol}: {e}")
            return pd.DataFrame()
    else:
        # Check if we have enough historical data
        earliest = df.index.min()
        if len(df) < 500 or target_start < earliest:
            print(f"[Data] Cache has {len(df)} bars, extending historical data...")
            try:
                extra = st.download_historical_ohlcv(symbol, DAILY_TIMEFRAME, target_start, earliest)
                if extra is not None and not extra.empty:
                    df = pd.concat([extra, df])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                    st.save_ohlcv_to_cache(symbol, DAILY_TIMEFRAME, df)
            except Exception as e:
                print(f"[Data] Could not fetch additional historical data: {e}")

        # Fill gap to current time
        cache_end = df.index.max()
        gap_threshold = pd.Timedelta(days=2)
        gap = now - cache_end

        if gap > gap_threshold:
            print(f"[Data] Filling gap from {cache_end.strftime('%Y-%m-%d')} to now...")
            try:
                start_from = cache_end + pd.Timedelta(days=1)
                new_data = st.download_historical_ohlcv(symbol, DAILY_TIMEFRAME, start_from, now)
                if new_data is not None and not new_data.empty:
                    df = pd.concat([df, new_data])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                    st.save_ohlcv_to_cache(symbol, DAILY_TIMEFRAME, df)
            except Exception as e:
                print(f"[Data] Could not fill gap: {e}")

    if not df.empty:
        # Limit to last 2 years
        df = df.tail(LOOKBACK_DAYS)
        print(f"[Data] {symbol}: {len(df)} daily bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all three indicators on the DAILY dataframe."""
    if df.empty:
        return df

    df = df.copy()

    # 1. Supertrend (default params: length=10, factor=3.0)
    try:
        df_st = st.compute_supertrend(df.copy(), length=10, factor=3.0)
        df["supertrend"] = df_st["supertrend"]
        df["st_trend"] = df_st["st_trend"]
    except Exception as e:
        print(f"[Indicator] Supertrend calculation failed: {e}")
        df["supertrend"] = np.nan
        df["st_trend"] = 0

    # 2. JMA (default params: length=30, phase=0)
    try:
        df_jma = st.compute_jma(df.copy(), length=30, phase=0)
        df["jma"] = df_jma["jma"]
        df["jma_trend"] = df_jma["jma_trend"]
    except Exception as e:
        print(f"[Indicator] JMA calculation failed: {e}")
        df["jma"] = np.nan
        df["jma_trend"] = 0

    # 3. KAMA (default params: length=20, slow_length=40)
    try:
        df_kama = st.compute_kama(df.copy(), length=20, slow_length=40)
        df["kama"] = df_kama["kama"]
        df["kama_trend"] = df_kama["kama_trend"]
    except Exception as e:
        print(f"[Indicator] KAMA calculation failed: {e}")
        df["kama"] = np.nan
        df["kama_trend"] = 0

    return df


def detect_market_phases_daily(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect market phase for each DAY based on indicator slopes.

    Phasen dauern WOCHEN oder MONATE, da die Berechnung auf Tagesbasis erfolgt.
    """
    if df.empty:
        return df

    df = df.copy()

    # Calculate phase for each indicator
    for indicator in ["supertrend", "jma", "kama"]:
        col = indicator
        if col not in df.columns:
            df[f"{indicator}_phase"] = "Flat"
            df[f"{indicator}_phase_value"] = 0
            continue

        # Calculate rolling slope (longer lookback for daily = smoother phases)
        slopes = []
        for i in range(len(df)):
            if i < lookback:
                slopes.append(0.0)
            else:
                values = df[col].iloc[i-lookback:i+1]
                slope = st.calculate_indicator_slope(values, lookback)
                slopes.append(slope)

        df[f"{indicator}_slope"] = slopes

        # Determine phase from slope (with slightly higher threshold for daily)
        phases = []
        phase_values = []
        threshold = 0.0002  # Higher threshold for daily bars

        for slope in slopes:
            phase = st.determine_phase_from_slope(slope, threshold)
            phases.append(phase)
            phase_values.append(PHASE_VALUES[phase])

        df[f"{indicator}_phase"] = phases
        df[f"{indicator}_phase_value"] = phase_values

    # Overall market phase (majority voting from all 3 indicators)
    def get_majority_phase(row):
        phases = [
            row.get("supertrend_phase", "Flat"),
            row.get("jma_phase", "Flat"),
            row.get("kama_phase", "Flat")
        ]
        up_count = phases.count("Up")
        down_count = phases.count("Down")

        if up_count >= 2:
            return "Up"
        elif down_count >= 2:
            return "Down"
        else:
            return "Flat"

    df["market_phase"] = df.apply(get_majority_phase, axis=1)
    df["market_phase_value"] = df["market_phase"].map(PHASE_VALUES)

    return df


def create_symbol_chart(symbol: str, df: pd.DataFrame) -> go.Figure:
    """
    Create interactive chart for a single symbol.

    Layout (wie Binance Chart):
    - Row 1 (top): Candlesticks + JMA, KAMA, Supertrend als Overlay
    - Row 2 (bottom): Phase als Stufendiagramm (1=Up, 0=Flat, -1=Down)
    """

    # Create subplots: candlesticks + indicators on top, phase below
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f"{symbol} - Daily Chart with Indicators",
            "Market Phase (1=Up, 0=Flat, -1=Down)"
        ]
    )

    # ===== Row 1: Candlestick chart with indicators overlaid =====

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1
    )

    # Add indicator lines on the SAME chart as candlesticks
    for indicator, color in INDICATOR_COLORS.items():
        if indicator in df.columns and not df[indicator].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    mode="lines",
                    name=INDICATOR_NAMES[indicator],
                    line=dict(color=color, width=2),
                    hovertemplate=f"{INDICATOR_NAMES[indicator]}: %{{y:.2f}}<extra></extra>",
                ),
                row=1, col=1
            )

    # ===== Row 2: Phase chart (1=Up, 0=Flat, -1=Down) =====

    # Phase as step line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["market_phase_value"],
            mode="lines",
            name="Phase",
            line=dict(color="#FFC107", width=2, shape="hv"),  # Step line
            fill="tozeroy",
            fillcolor="rgba(255, 193, 7, 0.2)",
            hovertemplate="Phase: %{y}<br>%{x}<extra></extra>",
        ),
        row=2, col=1
    )

    # Add colored background regions for phases in Row 2
    phases = df["market_phase"].values
    indices = df.index.values

    # Find phase change points
    change_points = [0]
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            change_points.append(i)
    change_points.append(len(phases))

    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = min(change_points[i + 1], len(indices) - 1)
        phase = phases[start_idx]

        fig.add_shape(
            type="rect",
            xref="x2",
            yref="y2 domain",
            x0=indices[start_idx],
            x1=indices[end_idx],
            y0=0,
            y1=1,
            fillcolor=PHASE_COLORS.get(phase, PHASE_COLORS["Flat"]),
            line=dict(width=0),
            layer="below",
            row=2, col=1
        )

    # Calculate phase statistics
    phase_counts = df["market_phase"].value_counts()
    total = len(df)
    up_pct = phase_counts.get("Up", 0) / total * 100
    down_pct = phase_counts.get("Down", 0) / total * 100
    flat_pct = phase_counts.get("Flat", 0) / total * 100

    # Current phase
    current_phase = df["market_phase"].iloc[-1] if not df.empty else "Flat"

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{symbol} - Market Phases (DAILY)<br>"
                 f"<sup>Current: {current_phase} | Up: {up_pct:.1f}% | Down: {down_pct:.1f}% | Flat: {flat_pct:.1f}%</sup>",
            font=dict(size=16),
        ),
        template="plotly_dark",
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
    )

    # Row 1: No rangeslider, disable candlestick rangeslider
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # Row 2: Add rangeslider for navigation
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        row=2, col=1
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(
        title_text="Phase",
        tickvals=[-1, 0, 1],
        ticktext=["Down (-1)", "Flat (0)", "Up (1)"],
        range=[-1.5, 1.5],
        row=2, col=1
    )

    return fig


def create_overview_chart(all_data: dict) -> go.Figure:
    """Create overview chart showing phase distribution across all symbols."""

    symbols = []
    up_pcts = []
    down_pcts = []
    flat_pcts = []
    current_phases = []

    for symbol, df in all_data.items():
        if df.empty or "market_phase" not in df.columns:
            continue

        phase_counts = df["market_phase"].value_counts()
        total = len(df)

        symbols.append(symbol.replace("/", ""))
        up_pcts.append(phase_counts.get("Up", 0) / total * 100)
        down_pcts.append(phase_counts.get("Down", 0) / total * 100)
        flat_pcts.append(phase_counts.get("Flat", 0) / total * 100)
        current_phases.append(df["market_phase"].iloc[-1])

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

    # Add annotations for current phase
    annotations = []
    for i, (sym, phase) in enumerate(zip(symbols, current_phases)):
        color = "#4CAF50" if phase == "Up" else "#F44336" if phase == "Down" else "#9E9E9E"
        annotations.append(dict(
            x=sym,
            y=105,
            text=f"Now: {phase}",
            showarrow=False,
            font=dict(size=10, color=color),
        ))

    fig.update_layout(
        title="Market Phase Distribution by Symbol (2 Years Daily Data)",
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
        annotations=annotations,
    )

    return fig


def create_current_phases_table(all_data: dict) -> str:
    """Create HTML table showing current phases for all symbols."""
    rows = []
    for symbol, df in sorted(all_data.items()):
        if df.empty or "market_phase" not in df.columns:
            continue

        current_phase = df["market_phase"].iloc[-1]
        phase_value = df["market_phase_value"].iloc[-1]
        last_date = df.index[-1].strftime("%Y-%m-%d")

        # Phase color
        if current_phase == "Up":
            color = "#4CAF50"
            icon = "&#x25B2;"  # Triangle up
        elif current_phase == "Down":
            color = "#F44336"
            icon = "&#x25BC;"  # Triangle down
        else:
            color = "#9E9E9E"
            icon = "&#x25C6;"  # Diamond

        rows.append(f"""
            <tr>
                <td>{symbol}</td>
                <td style="color: {color}; font-weight: bold;">{icon} {current_phase}</td>
                <td style="color: {color};">{phase_value}</td>
                <td>{last_date}</td>
            </tr>
        """)

    return "\n".join(rows)


def generate_report(symbols: list = None):
    """Generate the complete market phase report with DAILY data."""

    # Use default symbols if not specified
    if symbols is None:
        symbols = st.SYMBOLS

    print(f"\n{'='*60}")
    print("Market Phase Report Generator (DAILY BASIS)")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Timeframe: {DAILY_TIMEFRAME} (DAILY)")
    print(f"Lookback: ~2 years ({LOOKBACK_DAYS} days)")
    print(f"Phase detection: Based on daily indicator slopes")
    print(f"{'='*60}\n")

    # Ensure report directory exists
    REPORT_DIR.mkdir(exist_ok=True)

    all_data = {}
    charts = []

    # Process each symbol
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")

        # Fetch DAILY data
        df = fetch_daily_data(symbol)
        if df.empty:
            print(f"  Skipping {symbol} - no data available")
            continue

        # Compute indicators on daily data
        print(f"  Computing indicators on daily bars...")
        df = compute_all_indicators(df)

        # Detect phases on daily data
        print(f"  Detecting market phases (daily basis)...")
        df = detect_market_phases_daily(df, lookback=10)

        all_data[symbol] = df

        # Create chart
        print(f"  Creating chart...")
        fig = create_symbol_chart(symbol, df)
        charts.append((symbol, fig))

        # Phase summary
        phase_counts = df["market_phase"].value_counts()
        total = len(df)
        current = df["market_phase"].iloc[-1]
        print(f"  Current Phase: {current}")
        print(f"  Distribution: Up={phase_counts.get('Up', 0)} ({phase_counts.get('Up', 0)/total*100:.1f}%), "
              f"Down={phase_counts.get('Down', 0)} ({phase_counts.get('Down', 0)/total*100:.1f}%), "
              f"Flat={phase_counts.get('Flat', 0)} ({phase_counts.get('Flat', 0)/total*100:.1f}%)")

    # Create overview chart
    print(f"\nCreating overview chart...")
    overview_fig = create_overview_chart(all_data)

    # Generate HTML report
    print(f"\nGenerating HTML report...")

    current_phases_table = create_current_phases_table(all_data)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Market Phase Report - DAILY Analysis (2 Years)</title>
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
        .important-note {{
            background: #2d4a3e;
            border-left: 4px solid #4CAF50;
            padding: 15px 20px;
            margin-bottom: 30px;
            border-radius: 4px;
        }}
        .important-note h3 {{
            margin-top: 0;
            color: #4CAF50;
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
        .phase-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        .phase-table th, .phase-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #3a3a5c;
        }}
        .phase-table th {{
            background: #2a2a4a;
            color: #fff;
        }}
        .phase-table tr:hover {{
            background: #3a3a5c;
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
        <h1>Market Phase Report - DAILY Basis</h1>
        <p class="subtitle">All Trading Pairs - 2 Year Analysis with JMA, KAMA, Supertrend</p>

        <div class="important-note">
            <h3>Phasenbestimmung auf TAGESBASIS</h3>
            <p>Die Up/Down/Flat-Phasen werden auf Basis von <strong>Tages-Bars</strong> ermittelt.
            Dadurch dauern Phasen typischerweise <strong>Wochen oder Monate</strong>, nicht Stunden.</p>
            <p>Im Chart:</p>
            <ul>
                <li><strong>Oben:</strong> Tages-Candlesticks mit JMA, KAMA und Supertrend Overlay</li>
                <li><strong>Unten:</strong> Phase als numerischer Wert: <strong>1 = Up, 0 = Flat, -1 = Down</strong></li>
            </ul>
        </div>

        <div class="indicator-legend">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF5722;"></div>
                    <span>Supertrend (Length=10, Factor=3.0)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3;"></div>
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
                    <span>Phase 1 = Up (Bullish)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(158, 158, 158, 0.5);"></div>
                    <span>Phase 0 = Flat (Sideways)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(244, 67, 54, 0.5);"></div>
                    <span>Phase -1 = Down (Bearish)</span>
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
                <div class="stat-value">DAILY</div>
                <div class="stat-label">Timeframe</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">~{LOOKBACK_DAYS}</div>
                <div class="stat-label">Days Analyzed</div>
            </div>
        </div>

        <h2>Current Market Phases</h2>
        <table class="phase-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Current Phase</th>
                    <th>Phase Value</th>
                    <th>As of Date</th>
                </tr>
            </thead>
            <tbody>
                {current_phases_table}
            </tbody>
        </table>

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

    parser = argparse.ArgumentParser(description="Generate Market Phase Report (DAILY)")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to include")

    args = parser.parse_args()

    symbols = args.symbols if args.symbols else None
    generate_report(symbols)
