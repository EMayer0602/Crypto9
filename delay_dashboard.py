#!/usr/bin/env python3
"""
Delay Dashboard - Equity Curve over Time for Minute-Delay Strategies.

Creates a dashboard showing the equity curve for delay strategies (5m, 10m, 15m, 30m, 45m)
with the same time axis as the original strategy.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from existing module
import Supertrend_5Min as st

# Settings
REPORT_DIR = Path("report_html")
SOURCE_JSON = REPORT_DIR / "trading_summary.json"
START_CAPITAL = 16500.0
MAX_POSITIONS = 10

# Delay intervals to test (in minutes) - use hourly for accurate results
# Note: We only have 1h OHLCV data, so use hourly delays
DEFAULT_DELAYS = [60, 120, 180, 240, 360, 480]  # 1h, 2h, 3h, 4h, 6h, 8h

# Colors for different delay lines
DELAY_COLORS = {
    0: "#2196f3",     # Original - blue
    60: "#4caf50",    # 1h - green
    120: "#ff9800",   # 2h - orange
    180: "#9c27b0",   # 3h - purple
    240: "#f44336",   # 4h - red
    360: "#00bcd4",   # 6h - cyan
    480: "#795548",   # 8h - brown
}


def load_trades_from_json(json_path: Path, start_date: str = None) -> list:
    """Load closed trades from trading_summary.json."""
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trades = data.get("trades", [])

    # Parse start_date filter
    start_dt = None
    if start_date:
        start_dt = datetime.fromisoformat(start_date + "T00:00:00")

    filtered_trades = []
    for t in trades:
        direction = str(t.get("direction", "long")).lower()
        if direction != "long":
            continue

        entry_time = t.get("entry_time", "")
        exit_time = t.get("exit_time", "")

        # Apply start_date filter
        if start_dt and entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00").replace("+00:00", ""))
            except ValueError:
                entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            if entry_dt.replace(tzinfo=None) < start_dt:
                continue

        # Calculate original PnL percentage
        entry_price = float(t.get("entry_price", 0) or 0)
        exit_price = float(t.get("exit_price", 0) or 0)
        if entry_price > 0:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = 0

        filtered_trades.append({
            "symbol": t.get("symbol", ""),
            "direction": direction,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "original_stake": float(t.get("stake", 0) or 0),
            "original_pnl": float(t.get("pnl", 0) or 0),
            "pnl_pct": pnl_pct,
            "reason": t.get("reason", ""),
        })

    return filtered_trades


def preload_1h_cache(symbols: list, verbose: bool = False) -> dict:
    """Preload 1h OHLCV data from cache for all symbols."""
    ohlcv_cache = {}

    for symbol in symbols:
        try:
            df = st.load_ohlcv_from_cache(symbol, "1h")
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                ohlcv_cache[symbol] = df
                if verbose:
                    print(f"  Loaded {symbol}: {len(df)} bars")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {symbol}: {e}")

    return ohlcv_cache


def get_price_at_minute_offset(df: pd.DataFrame, entry_time: datetime,
                                minutes: int) -> Optional[float]:
    """Get price at entry_time + minutes offset using 1h data."""
    if df is None or df.empty:
        return None

    try:
        hours = minutes / 60.0
        target_time = entry_time + timedelta(hours=hours)

        mask = df.index >= target_time
        if mask.any():
            idx = df.index[mask][0]
            return float(df.loc[idx, "close"])
        else:
            return None
    except Exception:
        return None


def calculate_equity_curve_with_delay(trades: list, delay_minutes: int,
                                       ohlcv_cache: dict) -> tuple[list, dict]:
    """Calculate equity curve for a specific delay.

    Returns:
        Tuple of (equity_data, stats)
        equity_data: List of dicts with 'time' and 'equity'
    """
    filtered_trades = []

    for trade in trades:
        symbol = trade["symbol"]
        entry_time_str = trade["entry_time"]
        exit_time_str = trade["exit_time"]
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]

        # Parse entry time
        try:
            entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
            entry_time = entry_time.replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                entry_time = datetime.strptime(entry_time_str[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue

        # Parse exit time
        try:
            exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
            exit_time = exit_time.replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                exit_time = datetime.strptime(exit_time_str[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue

        if delay_minutes == 0:
            # Original strategy - no delay
            actual_pnl_pct = trade["pnl_pct"]
            filtered_trades.append({
                "exit_time": exit_time,
                "pnl_pct": actual_pnl_pct,
            })
        else:
            # Delayed strategy
            # Check if trade is still open at delay time
            delay_time = entry_time + timedelta(minutes=delay_minutes)
            if exit_time <= delay_time:
                # Trade already closed before delay check
                continue

            df = ohlcv_cache.get(symbol)
            if df is None or df.empty:
                continue

            price_at_delay = get_price_at_minute_offset(df, entry_time, delay_minutes)
            if price_at_delay is None:
                continue

            # Check if profitable at delay time
            if price_at_delay <= entry_price:
                continue

            # Calculate PnL from delayed entry
            actual_pnl_pct = (exit_price - price_at_delay) / price_at_delay * 100 if price_at_delay > 0 else 0

            filtered_trades.append({
                "exit_time": exit_time,
                "pnl_pct": actual_pnl_pct,
            })

    # Sort by exit time
    filtered_trades.sort(key=lambda x: x["exit_time"])

    # Calculate compound equity curve
    capital = START_CAPITAL
    equity_data = [{"time": filtered_trades[0]["exit_time"] - timedelta(hours=1), "equity": capital}] if filtered_trades else []

    wins = 0
    for t in filtered_trades:
        stake = capital / MAX_POSITIONS
        pnl = stake * (t["pnl_pct"] / 100)
        capital += pnl
        if t["pnl_pct"] > 0:
            wins += 1

        equity_data.append({
            "time": t["exit_time"],
            "equity": capital,
        })

    # Calculate stats
    total_trades = len(filtered_trades)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_return = (capital - START_CAPITAL) / START_CAPITAL * 100

    stats = {
        "delay_minutes": delay_minutes,
        "trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "final_capital": capital,
        "total_return": total_return,
    }

    return equity_data, stats


def generate_delay_dashboard(trades: list, delays: list, ohlcv_cache: dict,
                              output_path: Path) -> Path:
    """Generate HTML dashboard with equity curves for all delays."""

    print("Calculating equity curves...")

    # Calculate equity curves for all delays (including original)
    all_delays = [0] + delays
    curves = {}
    all_stats = []

    for delay in all_delays:
        delay_str = "Original" if delay == 0 else f"{delay}m"
        print(f"  Processing {delay_str}...")
        equity_data, stats = calculate_equity_curve_with_delay(trades, delay, ohlcv_cache)
        curves[delay] = equity_data
        all_stats.append(stats)

    # Create Plotly figure
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve - Delay Strategien", "Win Rate Vergleich"),
        vertical_spacing=0.12
    )

    # Add equity curves
    for delay in all_delays:
        equity_data = curves[delay]
        if not equity_data:
            continue

        times = [d["time"] for d in equity_data]
        equities = [d["equity"] for d in equity_data]

        delay_str = "Original" if delay == 0 else (f"{delay//60}h Delay" if delay >= 60 else f"{delay}m Delay")
        color = DELAY_COLORS.get(delay, "#888888")

        fig.add_trace(
            go.Scatter(
                x=times,
                y=equities,
                mode="lines",
                name=delay_str,
                line=dict(color=color, width=2 if delay == 0 else 1.5),
            ),
            row=1, col=1
        )

    # Add win rate bar chart
    delay_labels = ["Original"] + [f"{d//60}h" if d >= 60 else f"{d}m" for d in delays]
    win_rates = [s["win_rate"] for s in all_stats]
    bar_colors = [DELAY_COLORS.get(s["delay_minutes"], "#888888") for s in all_stats]

    fig.add_trace(
        go.Bar(
            x=delay_labels,
            y=win_rates,
            marker_color=bar_colors,
            text=[f"{wr:.1f}%" for wr in win_rates],
            textposition="outside",
            name="Win Rate",
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        title=dict(
            text="Momentum Delay Filter Dashboard",
            font=dict(size=24)
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Zeit", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_xaxes(title_text="Delay", row=2, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)

    # Generate HTML
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Build complete HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Delay Dashboard - Equity Curve</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #2196f3; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .concept-note {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #2196f3; }}
        .summary-boxes {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }}
        .summary-box {{ background: white; padding: 15px 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; text-align: center; }}
        .summary-box h3 {{ margin: 0 0 8px 0; color: #666; font-size: 12px; text-transform: uppercase; }}
        .summary-box .value {{ font-size: 20px; font-weight: bold; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
        th {{ background: #2196f3; color: white; text-align: center; }}
        td:first-child {{ text-align: center; font-weight: bold; }}
        .highlight {{ background: #e8f5e9; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 20px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Delay Dashboard - Equity Curve Vergleich</h1>

    <div class="concept-note">
        <strong>Konzept:</strong> Entry-Delay in Minuten testen. Wenn der Preis nach X Minuten &uuml;ber dem Signal-Preis liegt, wird eingestiegen (zum Preis bei T+X Minuten).<br>
        <strong>Kein Look-Ahead Bias:</strong> Entry-Preis = Preis bei T+Delay, nicht der Signal-Preis!
    </div>

    <h2>Zusammenfassung</h2>
    <table>
        <tr>
            <th>Delay</th>
            <th>Trades</th>
            <th>Pass Rate</th>
            <th>Win Rate</th>
            <th>vs Original</th>
            <th>Final Capital</th>
            <th>Total Return</th>
        </tr>
"""

    original_stats = all_stats[0]
    original_win_rate = original_stats["win_rate"]
    original_trades = original_stats["trades"]

    for stats in all_stats:
        delay = stats["delay_minutes"]
        delay_str = "Original" if delay == 0 else (f"{delay//60}h" if delay >= 60 else f"{delay}m")
        pass_rate = stats["trades"] / original_trades * 100 if original_trades > 0 else 0
        improvement = stats["win_rate"] - original_win_rate

        improvement_class = "positive" if improvement > 0 else ("negative" if improvement < 0 else "")
        return_class = "positive" if stats["total_return"] > 0 else "negative"

        # Highlight best win rate
        row_class = ""
        if stats == max(all_stats, key=lambda x: x["win_rate"]):
            row_class = "highlight"

        html += f"""        <tr class="{row_class}">
            <td>{delay_str}</td>
            <td>{stats['trades']}</td>
            <td>{pass_rate:.1f}%</td>
            <td>{stats['win_rate']:.1f}%</td>
            <td class="{improvement_class}">{improvement:+.1f}%</td>
            <td>${stats['final_capital']:,.0f}</td>
            <td class="{return_class}">{stats['total_return']:+.1f}%</td>
        </tr>
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html += f"""    </table>

    <div class="chart-container">
        {chart_html}
    </div>

    <p class="timestamp">Generated: {timestamp}</p>
</div>
</body>
</html>"""

    # Write to file
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Delay Dashboard with Equity Curve")
    parser.add_argument("--start", type=str, default="2024-01-31",
                        help="Start date YYYY-MM-DD (default: 2024-01-31)")
    parser.add_argument("--delays", type=str, default="5,10,15,30,45",
                        help="Comma-separated delay minutes (default: 5,10,15,30,45)")
    parser.add_argument("--output", type=str, default="report_html/delay_dashboard.html",
                        help="Output file path (default: report_html/delay_dashboard.html)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Parse delays
    delays = [int(x.strip()) for x in args.delays.split(",")]

    # Load trades
    print(f"Loading trades from {SOURCE_JSON}...")
    trades = load_trades_from_json(SOURCE_JSON, start_date=args.start)
    print(f"Loaded {len(trades)} long trades since {args.start}")

    if not trades:
        print("No trades found!")
        return

    # Pre-load 1h OHLCV cache for all symbols
    symbols = list(set(t["symbol"] for t in trades))
    print(f"Loading 1h OHLCV data for {len(symbols)} symbols...")
    ohlcv_cache = preload_1h_cache(symbols, verbose=args.verbose)
    print(f"Cache ready: {len(ohlcv_cache)} symbols loaded\n")

    # Generate dashboard
    output_path = Path(args.output)
    generate_delay_dashboard(trades, delays, ohlcv_cache, output_path)

    print(f"\nOpen with: xdg-open {output_path}")


if __name__ == "__main__":
    main()
