#!/usr/bin/env python3
"""
Delay Equity Dashboard - Generate equity curves for delay strategies.

This script generates a dashboard with equity curves comparing different
entry delay strategies. It uses minute-level OHLCV data fetched from the API.

Usage:
    python delay_equity_dashboard.py --delays 5,10,15,30,45

Requirements:
    - 5m or 1m OHLCV data from API
    - trading_summary.json with trade history
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt

# Settings
REPORT_DIR = Path("report_html")
SOURCE_JSON = REPORT_DIR / "trading_summary.json"
OUTPUT_DIR = REPORT_DIR
START_CAPITAL = 16500.0
MAX_POSITIONS = 10

# OHLCV Cache directory for 5m data
OHLCV_5M_CACHE = Path("ohlcv_cache_5m")

# Default delays (in minutes)
DEFAULT_DELAYS = [5, 10, 15, 30, 45]

# Colors for different delay lines
DELAY_COLORS = {
    0: "#2196f3",     # Original - blue
    5: "#4caf50",     # 5m - green
    10: "#ff9800",    # 10m - orange
    15: "#9c27b0",    # 15m - purple
    30: "#f44336",    # 30m - red
    45: "#00bcd4",    # 45m - cyan
    60: "#795548",    # 1h - brown
}


def load_trades_from_json(json_path: Path, start_date: str = None) -> list:
    """Load closed trades from trading_summary.json."""
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trades = data.get("trades", [])

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

        if start_dt and entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time)
            except ValueError:
                entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            if entry_dt.replace(tzinfo=None) < start_dt:
                continue

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
            "pnl_pct": pnl_pct,
            "reason": t.get("reason", ""),
        })

    return filtered_trades


def init_exchange(exchange_id: str = "hyperliquid"):
    """Initialize exchange connection."""
    if exchange_id == "hyperliquid":
        exchange = ccxt.hyperliquid({
            'enableRateLimit': True,
        })
    elif exchange_id == "binance":
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
    else:
        raise ValueError(f"Unknown exchange: {exchange_id}")

    return exchange


def fetch_ohlcv_5m(exchange, symbol: str, since: datetime,
                   limit: int = 1000, use_cache: bool = True) -> pd.DataFrame:
    """Fetch 5m OHLCV data from exchange.

    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., "BTC/USDC")
        since: Start datetime
        limit: Max candles per request
        use_cache: Use local cache if available

    Returns:
        DataFrame with OHLCV data
    """
    # Check cache first
    OHLCV_5M_CACHE.mkdir(exist_ok=True)
    cache_file = OHLCV_5M_CACHE / f"{symbol.replace('/', '_')}_5m.csv"

    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty and df.index[-1] >= since:
                print(f"  Using cached 5m data for {symbol}")
                return df
        except Exception as e:
            print(f"  Cache read error for {symbol}: {e}")

    # Fetch from API
    print(f"  Fetching 5m data for {symbol}...")

    all_data = []
    current_since = int(since.timestamp() * 1000)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '5m', since=current_since, limit=limit)
            if not ohlcv:
                break

            all_data.extend(ohlcv)

            # Check if we've reached current time
            last_time = ohlcv[-1][0]
            if last_time >= (datetime.now().timestamp() * 1000 - 300000):  # Within 5 min
                break

            current_since = last_time + 1
            time.sleep(exchange.rateLimit / 1000)  # Rate limit

        except Exception as e:
            print(f"  API error for {symbol}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # Save to cache
    try:
        df.to_csv(cache_file)
        print(f"  Cached 5m data for {symbol}: {len(df)} bars")
    except Exception as e:
        print(f"  Cache write error for {symbol}: {e}")

    return df


def get_price_at_offset(df: pd.DataFrame, entry_time: datetime,
                         minutes: int) -> Optional[float]:
    """Get price at entry_time + minutes offset.

    Uses 5m OHLCV data for precise minute-level prices.
    """
    if df is None or df.empty:
        return None

    try:
        target_time = entry_time + timedelta(minutes=minutes)

        # Make index timezone-naive if needed
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        if target_time.tzinfo:
            target_time = target_time.replace(tzinfo=None)

        # Find closest bar at or after target time
        mask = df.index >= target_time
        if mask.any():
            idx = df.index[mask][0]
            return float(df.loc[idx, "close"])
        else:
            return None

    except Exception as e:
        return None


def calculate_equity_curve_with_delay(trades: list, delay_minutes: int,
                                       ohlcv_cache: dict, verbose: bool = False) -> tuple:
    """Calculate equity curve for a specific delay.

    Returns:
        Tuple of (equity_data, stats)
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
            entry_time = datetime.fromisoformat(entry_time_str)
            entry_time = entry_time.replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                entry_time = datetime.strptime(entry_time_str[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue

        # Parse exit time
        try:
            exit_time = datetime.fromisoformat(exit_time_str)
            exit_time = exit_time.replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                exit_time = datetime.strptime(exit_time_str[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue

        if delay_minutes == 0:
            # Original strategy - no delay
            filtered_trades.append({
                "exit_time": exit_time,
                "pnl_pct": trade["pnl_pct"],
            })
        else:
            # Delayed strategy
            delay_time = entry_time + timedelta(minutes=delay_minutes)
            if exit_time <= delay_time:
                continue  # Trade closed before delay check

            df = ohlcv_cache.get(symbol)
            if df is None or df.empty:
                continue

            price_at_delay = get_price_at_offset(df, entry_time, delay_minutes)
            if price_at_delay is None:
                continue

            # Momentum check: price must be higher than entry
            if price_at_delay <= entry_price:
                continue

            # Calculate PnL from delayed entry
            actual_pnl_pct = (exit_price - price_at_delay) / price_at_delay * 100 if price_at_delay > 0 else 0

            filtered_trades.append({
                "exit_time": exit_time,
                "pnl_pct": actual_pnl_pct,
                "entry_price": entry_price,
                "delayed_price": price_at_delay,
            })

    # Sort by exit time
    filtered_trades.sort(key=lambda x: x["exit_time"])

    # Calculate compound equity curve
    if not filtered_trades:
        return [], {"delay_minutes": delay_minutes, "trades": 0, "wins": 0,
                   "win_rate": 0, "final_capital": START_CAPITAL, "total_return": 0}

    capital = START_CAPITAL
    equity_data = [{"time": filtered_trades[0]["exit_time"] - timedelta(hours=1), "equity": capital}]

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
        print(f"    Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.1f}%")

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

        delay_str = "Original" if delay == 0 else f"{delay}m Delay"
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
    delay_labels = ["Original"] + [f"{d}m" for d in delays]
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
            showlegend=False,
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
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1, range=[0, max(win_rates) * 1.2])

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
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
        th {{ background: #2196f3; color: white; text-align: center; }}
        td:first-child {{ text-align: center; font-weight: bold; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
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
        delay_str = "Original" if delay == 0 else f"{delay}m"
        pass_rate = stats["trades"] / original_trades * 100 if original_trades > 0 else 0
        improvement = stats["win_rate"] - original_win_rate

        improvement_class = "positive" if improvement > 0 else ("negative" if improvement < 0 else "")
        return_class = "positive" if stats["total_return"] > 0 else "negative"

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

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Delay Equity Dashboard")
    parser.add_argument("--start", type=str, default="2024-01-31",
                        help="Start date YYYY-MM-DD (default: 2024-01-31)")
    parser.add_argument("--delays", type=str, default="5,10,15,30,45",
                        help="Comma-separated delay minutes (default: 5,10,15,30,45)")
    parser.add_argument("--output", type=str, default="report_html/delay_equity_dashboard.html",
                        help="Output file path")
    parser.add_argument("--exchange", type=str, default="hyperliquid",
                        help="Exchange for API (hyperliquid, binance)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached OHLCV data")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Parse delays
    delays = [int(x.strip()) for x in args.delays.split(",")]
    max_delay = max(delays)

    # Load trades
    print(f"Loading trades from {SOURCE_JSON}...")
    trades = load_trades_from_json(SOURCE_JSON, start_date=args.start)
    print(f"Loaded {len(trades)} long trades since {args.start}")

    if not trades:
        print("No trades found!")
        return

    # Get unique symbols
    symbols = list(set(t["symbol"] for t in trades))
    print(f"Symbols: {len(symbols)}")

    # Determine earliest date needed
    earliest_date = min(
        datetime.fromisoformat(t["entry_time"]).replace(tzinfo=None)
        for t in trades
    ) - timedelta(days=1)

    # Initialize exchange and fetch 5m OHLCV data
    print(f"\nFetching 5m OHLCV data from {args.exchange}...")
    exchange = init_exchange(args.exchange)

    ohlcv_cache = {}
    for symbol in symbols:
        try:
            df = fetch_ohlcv_5m(exchange, symbol, earliest_date,
                               use_cache=not args.no_cache)
            if not df.empty:
                ohlcv_cache[symbol] = df
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")

    print(f"Cache ready: {len(ohlcv_cache)} symbols")

    if not ohlcv_cache:
        print("No OHLCV data available!")
        return

    # Generate dashboard
    output_path = Path(args.output)
    generate_delay_dashboard(trades, delays, ohlcv_cache, output_path)

    print(f"\nOpen with: xdg-open {output_path}")


if __name__ == "__main__":
    main()
