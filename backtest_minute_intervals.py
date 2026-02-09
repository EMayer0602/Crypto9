#!/usr/bin/env python3
"""
Backtest Momentum Filter with Minute-Based Intervals.

Tests entry delay at: 5m, 10m, 15m, 25m, 35m, 45m, 60m (1h)

Uses 5m OHLCV data to get precise prices at each interval.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import time

import pandas as pd

# Import from existing module
import Supertrend_5Min as st

# Settings
REPORT_DIR = Path("report_html")
SOURCE_JSON = REPORT_DIR / "trading_summary.json"
START_CAPITAL = 16500.0
MAX_POSITIONS = 10

# Intervals to test (in minutes) - both minute and hourly intervals
# For < 60: interpolation is used (approximation!)
# For >= 60: real hourly bar closes are used
DEFAULT_INTERVALS = [60, 120, 180, 240, 360, 480]  # 1h, 2h, 3h, 4h, 6h, 8h


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
        if direction != "long":  # Only long trades for now
            continue

        entry_time = t.get("entry_time", "")

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
            "exit_time": t.get("exit_time", ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "original_stake": float(t.get("stake", 0) or 0),
            "original_pnl": float(t.get("pnl", 0) or 0),
            "pnl_pct": pnl_pct,
            "reason": t.get("reason", ""),
        })

    return filtered_trades


def preload_1h_cache(symbols: list, verbose: bool = False) -> dict:
    """Preload 1h OHLCV data from cache for all symbols.

    Args:
        symbols: List of trading pairs
        verbose: Print progress

    Returns:
        Dict mapping symbol -> DataFrame
    """
    ohlcv_cache = {}

    for symbol in symbols:
        try:
            df = st.load_ohlcv_from_cache(symbol, "1h")
            if df is not None and not df.empty:
                # Convert index to DatetimeIndex if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Make index naive for comparison
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
    """Get price at entry_time + minutes offset.

    Uses 1h OHLCV data. For hourly intervals (60, 120, etc.),
    returns the actual bar close.

    Args:
        df: 1h OHLCV DataFrame
        entry_time: Original entry timestamp
        minutes: Minutes to add (should be multiple of 60 for accurate results)

    Returns:
        Close price at target time, or None
    """
    if df is None or df.empty:
        return None

    try:
        # Calculate target time
        hours = minutes / 60.0
        target_time = entry_time + timedelta(hours=hours)

        # Find closest bar at or after target time
        mask = df.index >= target_time
        if mask.any():
            idx = df.index[mask][0]
            return float(df.loc[idx, "close"])
        else:
            return None

    except Exception:
        return None


def backtest_minute_interval(trades: list, minutes: int,
                              ohlcv_cache: dict, verbose: bool = False) -> dict:
    """Backtest momentum filter with specific minute delay.

    Args:
        trades: List of trade dicts
        minutes: Entry delay in minutes
        ohlcv_cache: Pre-loaded 1h OHLCV data cache
        verbose: Print progress

    Returns:
        Dict with results
    """
    filtered_trades = []
    skipped_no_data = 0
    skipped_not_profitable = 0

    for i, trade in enumerate(trades):
        symbol = trade["symbol"]
        entry_time_str = trade["entry_time"]
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
                skipped_no_data += 1
                continue

        # Get pre-loaded 1h data for this symbol
        df = ohlcv_cache.get(symbol)

        if df is None or df.empty:
            skipped_no_data += 1
            continue

        # Get price at T + minutes (interpolated for < 60m)
        price_at_delay = get_price_at_minute_offset(df, entry_time, minutes)

        if price_at_delay is None:
            skipped_no_data += 1
            continue

        # Check if profitable at delay time (momentum criterion)
        is_profitable = price_at_delay > entry_price

        if not is_profitable:
            skipped_not_profitable += 1
            continue

        # Trade passes filter - calculate PnL from delayed entry
        # Entry at price_at_delay, exit at original exit_price
        actual_pnl_pct = (exit_price - price_at_delay) / price_at_delay * 100 if price_at_delay > 0 else 0

        filtered_trades.append({
            "symbol": symbol,
            "original_entry_price": entry_price,
            "delayed_entry_price": price_at_delay,
            "exit_price": exit_price,
            "pnl_pct": actual_pnl_pct,
            "original_pnl_pct": trade["pnl_pct"],
        })

    # Calculate statistics
    total_trades = len(trades)
    passed_trades = len(filtered_trades)

    if passed_trades > 0:
        wins = sum(1 for t in filtered_trades if t["pnl_pct"] > 0)
        win_rate = wins / passed_trades * 100
        avg_pnl = sum(t["pnl_pct"] for t in filtered_trades) / passed_trades

        # Calculate compound growth
        capital = START_CAPITAL
        for t in filtered_trades:
            stake = capital / MAX_POSITIONS
            pnl = stake * (t["pnl_pct"] / 100)
            capital += pnl
        total_return = (capital - START_CAPITAL) / START_CAPITAL * 100
    else:
        wins = 0
        win_rate = 0
        avg_pnl = 0
        capital = START_CAPITAL
        total_return = 0

    # Original stats for comparison
    original_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    original_win_rate = original_wins / total_trades * 100 if total_trades > 0 else 0

    return {
        "minutes": minutes,
        "total_trades": total_trades,
        "passed_trades": passed_trades,
        "pass_rate": passed_trades / total_trades * 100 if total_trades > 0 else 0,
        "wins": wins,
        "win_rate": win_rate,
        "original_win_rate": original_win_rate,
        "win_rate_improvement": win_rate - original_win_rate,
        "avg_pnl_pct": avg_pnl,
        "final_capital": capital,
        "total_return": total_return,
        "skipped_no_data": skipped_no_data,
        "skipped_not_profitable": skipped_not_profitable,
    }


def run_multi_interval_backtest(trades: list, intervals: list,
                                 verbose: bool = True) -> list:
    """Run backtest for multiple minute intervals.

    Args:
        trades: List of trades
        intervals: List of minute intervals to test
        verbose: Print progress

    Returns:
        List of result dicts
    """
    print("\n" + "=" * 70)
    print("MINUTE-BASED MOMENTUM FILTER BACKTEST")
    print("=" * 70)
    print(f"Total trades: {len(trades)}")
    print(f"Intervals to test: {intervals}")
    print("=" * 70 + "\n")

    # Pre-load 1h OHLCV cache for all symbols
    symbols = list(set(t["symbol"] for t in trades))
    print(f"Loading 1h OHLCV data for {len(symbols)} symbols...")
    ohlcv_cache = preload_1h_cache(symbols, verbose=False)
    print(f"Cache ready: {len(ohlcv_cache)} symbols loaded\n")

    results = []

    for minutes in intervals:
        delay_str = f"{minutes}m" if minutes < 60 else f"{minutes//60}h"
        print(f"\n--- Testing {delay_str} ({minutes} min) delay ---")

        result = backtest_minute_interval(
            trades, minutes, ohlcv_cache, verbose=verbose
        )
        results.append(result)

        print(f"  Passed: {result['passed_trades']}/{result['total_trades']} ({result['pass_rate']:.1f}%)")
        print(f"  Win Rate: {result['win_rate']:.1f}% (original: {result['original_win_rate']:.1f}%)")
        print(f"  Improvement: {result['win_rate_improvement']:+.1f}%")
        print(f"  Total Return: {result['total_return']:+.1f}%")

    return results


def print_comparison_table(results: list):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("COMPARISON TABLE - MINUTE INTERVALS")
    print("=" * 90)

    print(f"{'Delay':>8} | {'Trades':>7} | {'Pass%':>6} | {'Win Rate':>8} | {'Orig WR':>8} | {'Improv':>8} | {'Return':>8}")
    print("-" * 90)

    for r in results:
        delay_str = f"{r['minutes']}m" if r['minutes'] < 60 else f"{r['minutes']//60}h"
        print(f"{delay_str:>8} | {r['passed_trades']:>7} | {r['pass_rate']:>5.1f}% | "
              f"{r['win_rate']:>7.1f}% | {r['original_win_rate']:>7.1f}% | "
              f"{r['win_rate_improvement']:>+7.1f}% | {r['total_return']:>+7.1f}%")

    print("=" * 90)

    # Find best
    best_wr = max(results, key=lambda x: x['win_rate'])
    best_return = max(results, key=lambda x: x['total_return'])
    best_improvement = max(results, key=lambda x: x['win_rate_improvement'])

    print(f"\nBest Win Rate:    {best_wr['minutes']}m ({best_wr['win_rate']:.1f}%)")
    print(f"Best Improvement: {best_improvement['minutes']}m (+{best_improvement['win_rate_improvement']:.1f}%)")
    print(f"Best Return:      {best_return['minutes']}m ({best_return['total_return']:+.1f}%)")


def generate_html_report(results: list, output_path: Path):
    """Generate HTML comparison report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Minute Interval Backtest Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 2px solid #2196f3; padding-bottom: 10px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: right; }
        th { background: #2196f3; color: white; text-align: center; }
        td:first-child { text-align: center; font-weight: bold; }
        .positive { color: green; font-weight: bold; }
        .negative { color: red; font-weight: bold; }
        .highlight { background: #e3f2fd; }
        .summary { margin: 20px 0; padding: 20px; background: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .summary h2 { margin-top: 0; color: #2196f3; }
        .best { background: #c8e6c9; }
        .timestamp { color: #666; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Minute Interval Momentum Filter Backtest</h1>

    <div class="summary">
        <h2>Konzept</h2>
        <p>Entry-Delay in Minuten testen: Wenn der Preis nach X Minuten über dem Signal-Preis liegt,
        wird eingestiegen (zum Preis bei T+X Minuten).</p>
        <p><strong>Kein Look-Ahead Bias:</strong> Entry-Preis = Preis bei T+Delay, nicht der Signal-Preis!</p>
    </div>

    <table>
        <tr>
            <th>Delay</th>
            <th>Trades</th>
            <th>Pass Rate</th>
            <th>Win Rate</th>
            <th>Original WR</th>
            <th>Improvement</th>
            <th>Avg PnL%</th>
            <th>Total Return</th>
        </tr>
"""

    # Find best values for highlighting
    best_wr = max(results, key=lambda x: x['win_rate'])
    best_return = max(results, key=lambda x: x['total_return'])

    for r in results:
        delay_str = f"{r['minutes']}m" if r['minutes'] < 60 else f"1h"

        wr_class = "best" if r == best_wr else ""
        return_class = "best" if r == best_return else ""
        improvement_class = "positive" if r['win_rate_improvement'] > 0 else "negative"
        return_value_class = "positive" if r['total_return'] > 0 else "negative"

        html += f"""        <tr class="{wr_class if wr_class else return_class}">
            <td>{delay_str}</td>
            <td>{r['passed_trades']}</td>
            <td>{r['pass_rate']:.1f}%</td>
            <td>{r['win_rate']:.1f}%</td>
            <td>{r['original_win_rate']:.1f}%</td>
            <td class="{improvement_class}">{r['win_rate_improvement']:+.1f}%</td>
            <td>{r['avg_pnl_pct']:.2f}%</td>
            <td class="{return_value_class}">{r['total_return']:+.1f}%</td>
        </tr>
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>

    <div class="summary">
        <h2>Beste Ergebnisse</h2>
        <p><strong>Beste Win Rate:</strong> {best_wr['minutes']}m mit {best_wr['win_rate']:.1f}%</p>
        <p><strong>Bester Return:</strong> {best_return['minutes']}m mit {best_return['total_return']:+.1f}%</p>
    </div>

    <p class="timestamp">Generated: {timestamp}</p>
</div>
</body>
</html>"""

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Minute-based Momentum Filter Backtest")
    parser.add_argument("--start", type=str, default="2024-01-31",
                        help="Start date YYYY-MM-DD (default: 2024-01-31)")
    parser.add_argument("--intervals", type=str, default="5,10,15,25,35,45,60",
                        help="Comma-separated minute intervals (default: 5,10,15,25,35,45,60)")
    parser.add_argument("--output-dir", type=str, default="report_html",
                        help="Output directory (default: report_html)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Parse intervals
    intervals = [int(x.strip()) for x in args.intervals.split(",")]

    # Load trades
    print(f"Loading trades from {SOURCE_JSON}...")
    trades = load_trades_from_json(SOURCE_JSON, start_date=args.start)
    print(f"Loaded {len(trades)} long trades since {args.start}")

    if not trades:
        print("No trades found!")
        return

    # Run backtest
    results = run_multi_interval_backtest(trades, intervals, verbose=args.verbose)

    # Print comparison
    print_comparison_table(results)

    # Generate HTML report
    output_dir = Path(args.output_dir)
    html_path = output_dir / "backtest_minute_intervals.html"
    generate_html_report(results, html_path)

    print(f"\nOpen with: xdg-open {html_path}")


if __name__ == "__main__":
    main()
