#!/usr/bin/env python3
"""
Minuten-Delay Backtest - Interpolierte Preise mit 1h-Daten.

Testet: 5, 10, 15, 30, 45 Minuten Verzögerung
Bedingung: Preis muss nach X Minuten gestiegen sein -> dann Entry

Hinweis: Verwendet lineare Interpolation zwischen 1h-Kerzen.
         Das ist eine Approximation - echte Minutendaten wären genauer.

Usage:
    python test_minute_delay.py                          # Standard-Intervalle
    python test_minute_delay.py --minutes 5,10,15,30,45  # Spezifische Intervalle
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import time

import pandas as pd
import numpy as np

import Supertrend_5Min as st

# Settings
SOURCE_JSON = Path("report_html/trading_summary.json")
START_CAPITAL = 16500.0
MAX_POSITIONS = 10

# Default intervals in minutes
DEFAULT_MINUTES = [5, 10, 15, 30, 45]

# Cache for 1h data per symbol
_hourly_cache: Dict[str, pd.DataFrame] = {}


def load_trades(start_date: str = "2024-01-31") -> list:
    """Load long trades from JSON."""
    with open(SOURCE_JSON, "r") as f:
        data = json.load(f)

    start_dt = datetime.fromisoformat(start_date + "T00:00:00")
    trades = []

    for t in data.get("trades", []):
        if str(t.get("direction", "")).lower() != "long":
            continue

        entry_time = t.get("entry_time", "")
        try:
            entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            entry_dt = entry_dt.replace(tzinfo=None)
        except:
            try:
                entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            except:
                continue

        if entry_dt < start_dt:
            continue

        entry_price = float(t.get("entry_price", 0) or 0)
        exit_price = float(t.get("exit_price", 0) or 0)

        if entry_price <= 0:
            continue

        trades.append({
            "symbol": t.get("symbol", ""),
            "entry_time": entry_dt,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": (exit_price - entry_price) / entry_price * 100
        })

    return trades


def load_hourly_cache(symbols: list) -> dict:
    """Load 1h OHLCV data from cache for all symbols."""
    global _hourly_cache

    cache = {}

    for symbol in symbols:
        if symbol in _hourly_cache:
            cache[symbol] = _hourly_cache[symbol]
            continue

        try:
            df = st.load_ohlcv_from_cache(symbol, "1h")
            if df is not None and not df.empty:
                # Convert index to DatetimeIndex if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Make index naive for comparison
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                cache[symbol] = df
                _hourly_cache[symbol] = df
        except Exception as e:
            pass

    return cache


def get_interpolated_price_at_delay(df: pd.DataFrame, entry_time: datetime, minutes: int) -> Optional[float]:
    """Get interpolated price at entry_time + minutes using 1h data.

    Uses linear interpolation between hourly bars.
    This is an approximation - real minute data would be more accurate.
    """
    if df is None or df.empty:
        return None

    target_time = entry_time + timedelta(minutes=minutes)

    try:
        # Find the bars surrounding target_time
        mask_before = df.index <= target_time
        mask_after = df.index > target_time

        if not mask_before.any():
            return None

        # Get the bar at or before target
        idx_before = df.index[mask_before][-1]
        price_before = float(df.loc[idx_before, "close"])

        # If target_time is exactly on a bar, return that price
        if idx_before == target_time:
            return price_before

        # Get the bar after target (if exists)
        if mask_after.any():
            idx_after = df.index[mask_after][0]
            price_after = float(df.loc[idx_after, "open"])  # Use open of next bar

            # Linear interpolation
            time_span = (idx_after - idx_before).total_seconds()
            time_offset = (target_time - idx_before).total_seconds()

            if time_span > 0:
                fraction = time_offset / time_span
                interpolated_price = price_before + (price_after - price_before) * fraction
                return interpolated_price

        # If no bar after, use the last known price
        return price_before

    except Exception:
        return None


def backtest_delay_minutes(trades: list, delay_minutes: int, hourly_cache: dict) -> list:
    """Backtest with interpolated price at T+delay_minutes using 1h data."""

    passed = []
    skipped_no_data = 0

    for t in trades:
        symbol = t["symbol"]
        df = hourly_cache.get(symbol)

        if df is None or df.empty:
            skipped_no_data += 1
            continue

        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        entry_time = t["entry_time"]

        # Interpolierter Preis nach delay_minutes Minuten
        price_at_delay = get_interpolated_price_at_delay(df, entry_time, delay_minutes)

        if price_at_delay is None:
            skipped_no_data += 1
            continue

        # Momentum-Check: Preis muss gestiegen sein
        if price_at_delay > entry_price:
            # Entry zum interpolierten verzögerten Preis, Exit zum Original-Exit
            new_pnl_pct = (exit_price - price_at_delay) / price_at_delay * 100
            passed.append({
                "symbol": symbol,
                "entry_time": entry_time,
                "pnl_pct": new_pnl_pct,
                "price_at_delay": price_at_delay,
                "original_entry": entry_price,
                "original_pnl_pct": t["pnl_pct"],
                "price_change_pct": (price_at_delay - entry_price) / entry_price * 100
            })

    return passed


def calculate_stats(trades: list, passed: list) -> dict:
    """Calculate statistics."""
    total = len(trades)
    filtered = len(passed)

    if filtered == 0:
        return {
            "total": total,
            "filtered": filtered,
            "pass_rate": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "total_return": 0,
            "avg_price_change": 0
        }

    wins = sum(1 for t in passed if t["pnl_pct"] > 0)
    win_rate = wins / filtered * 100
    avg_pnl = sum(t["pnl_pct"] for t in passed) / filtered
    avg_price_change = sum(t["price_change_pct"] for t in passed) / filtered

    # Compound growth
    capital = START_CAPITAL
    for t in passed:
        stake = capital / MAX_POSITIONS
        pnl = stake * (t["pnl_pct"] / 100)
        capital += pnl

    total_return = (capital - START_CAPITAL) / START_CAPITAL * 100

    return {
        "total": total,
        "filtered": filtered,
        "pass_rate": filtered / total * 100,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_return": total_return,
        "avg_price_change": avg_price_change
    }


def generate_html_report(results: list, orig_win_rate: float, output_path: Path):
    """Generate HTML comparison report."""

    best_wr = max(results, key=lambda x: x["win_rate"]) if results else None
    best_ret = max(results, key=lambda x: x["total_return"]) if results else None

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Minuten-Delay Backtest Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        .info {{ background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .info h3 {{ color: #00d4ff; margin-top: 0; }}
        table {{ border-collapse: collapse; width: 100%; background: #16213e; border-radius: 8px; overflow: hidden; }}
        th {{ background: #0f3460; color: #00d4ff; padding: 12px; text-align: center; }}
        td {{ padding: 10px; text-align: right; border-bottom: 1px solid #0f3460; }}
        td:first-child {{ text-align: center; font-weight: bold; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4757; }}
        .best {{ background: #0f3460; }}
        .summary {{ margin-top: 20px; background: #16213e; padding: 15px; border-radius: 8px; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .chart {{ margin: 20px 0; padding: 20px; background: #16213e; border-radius: 8px; }}
        .bar {{ display: inline-block; background: linear-gradient(90deg, #00d4ff, #00ff88); height: 20px; margin: 2px 0; border-radius: 4px; }}
        .bar-label {{ display: inline-block; width: 50px; color: #00d4ff; font-weight: bold; }}
        .bar-container {{ margin: 5px 0; }}
        .bar-value {{ color: #aaa; margin-left: 10px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Minuten-Delay Momentum Filter Backtest</h1>

    <div class="info">
        <h3>Konzept</h3>
        <p><strong>Bedingung:</strong> Wenn der Preis nach X Minuten <span class="positive">gestiegen</span> ist,
        wird zum verzögerten Preis eingestiegen.</p>
        <p><strong>Kein Look-Ahead Bias:</strong> Entry-Preis = Preis bei T+Delay (nicht Signal-Preis)</p>
        <p>Original Win Rate: <strong>{orig_win_rate:.1f}%</strong></p>
    </div>

    <table>
        <tr>
            <th>Delay</th>
            <th>Trades</th>
            <th>Pass Rate</th>
            <th>Win Rate</th>
            <th>vs Original</th>
            <th>Avg Price Δ</th>
            <th>Avg PnL</th>
            <th>Total Return</th>
        </tr>
"""

    for r in results:
        improvement = r["win_rate"] - orig_win_rate
        imp_class = "positive" if improvement > 0 else "negative"
        ret_class = "positive" if r["total_return"] > 0 else "negative"
        row_class = "best" if r == best_wr or r == best_ret else ""

        html += f"""        <tr class="{row_class}">
            <td>{r['minutes']}m</td>
            <td>{r['filtered']}</td>
            <td>{r['pass_rate']:.1f}%</td>
            <td>{r['win_rate']:.1f}%</td>
            <td class="{imp_class}">{improvement:+.1f}%</td>
            <td class="positive">+{r['avg_price_change']:.2f}%</td>
            <td>{r['avg_pnl']:.2f}%</td>
            <td class="{ret_class}">{r['total_return']:+.1f}%</td>
        </tr>
"""

    # Pass Rate Chart
    html += """    </table>

    <div class="chart">
        <h3 style="color: #00d4ff; margin-top: 0;">Pass Rate Visualisierung</h3>
"""

    max_pass_rate = max(r['pass_rate'] for r in results) if results else 100
    if max_pass_rate == 0:
        max_pass_rate = 100  # Avoid division by zero
    for r in results:
        bar_width = int(r['pass_rate'] / max_pass_rate * 300)
        html += f"""        <div class="bar-container">
            <span class="bar-label">{r['minutes']}m</span>
            <span class="bar" style="width: {bar_width}px;"></span>
            <span class="bar-value">{r['pass_rate']:.1f}% ({r['filtered']} Trades)</span>
        </div>
"""

    html += """    </div>
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if best_wr and best_ret:
        html += f"""
    <div class="summary">
        <h3 style="color: #00d4ff; margin-top: 0;">Beste Ergebnisse</h3>
        <p>Beste Win Rate: <strong>{best_wr['minutes']}m</strong> mit {best_wr['win_rate']:.1f}%</p>
        <p>Bester Return: <strong>{best_ret['minutes']}m</strong> mit {best_ret['total_return']:+.1f}%</p>
    </div>
"""

    html += f"""
    <p class="timestamp">Generiert: {timestamp}</p>
</div>
</body>
</html>"""

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML Report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Minuten-Delay Backtest mit interpolierten 1h-Daten")
    parser.add_argument("--start", default="2024-01-31", help="Start date YYYY-MM-DD")
    parser.add_argument("--minutes", default="5,10,15,30,45",
                        help="Comma-separated minutes to test")
    args = parser.parse_args()

    minutes_list = [int(m) for m in args.minutes.split(",")]

    print(f"\n{'='*70}")
    print("MINUTEN-DELAY BACKTEST (mit interpolierten 1h-Daten)")
    print(f"{'='*70}")
    print(f"Intervalle: {minutes_list} Minuten")
    print("Hinweis: Verwendet lineare Interpolation zwischen 1h-Kerzen")

    print(f"\nLade Trades seit {args.start}...")
    trades = load_trades(args.start)
    print(f"  {len(trades)} Long-Trades geladen")

    if not trades:
        print("Keine Trades gefunden!")
        return

    # Original stats
    orig_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    orig_win_rate = orig_wins / len(trades) * 100
    print(f"  Original Win Rate: {orig_win_rate:.1f}%")

    # Load 1h data for all symbols from cache
    symbols = list(set(t["symbol"] for t in trades))
    print(f"\nLade 1h-Daten aus Cache...")
    print(f"  {len(symbols)} Symbole zu laden...")

    hourly_cache = load_hourly_cache(symbols)
    print(f"  {len(hourly_cache)} Symbole mit 1h-Daten geladen")

    # Run backtests
    print(f"\n{'='*80}")
    print(f"{'Delay':>6} | {'Trades':>7} | {'Pass%':>7} | {'WinRate':>8} | "
          f"{'vs Orig':>8} | {'Δ Preis':>8} | {'Avg PnL':>8} | {'Return':>8}")
    print("-" * 80)

    results = []

    for minutes in minutes_list:
        passed = backtest_delay_minutes(trades, minutes, hourly_cache)
        stats = calculate_stats(trades, passed)

        improvement = stats["win_rate"] - orig_win_rate

        print(f"{minutes:>5}m | {stats['filtered']:>7} | {stats['pass_rate']:>6.1f}% | "
              f"{stats['win_rate']:>7.1f}% | {improvement:>+7.1f}% | "
              f"+{stats['avg_price_change']:>6.2f}% | {stats['avg_pnl']:>+7.2f}% | {stats['total_return']:>+7.1f}%")

        results.append({
            "minutes": minutes,
            **stats,
            "improvement": improvement
        })

    print("-" * 80)

    # Best results
    if results:
        best_wr = max(results, key=lambda x: x["win_rate"])
        best_ret = max(results, key=lambda x: x["total_return"])

        print(f"\nBeste Win Rate:  {best_wr['minutes']}m mit {best_wr['win_rate']:.1f}%")
        print(f"Bester Return:   {best_ret['minutes']}m mit {best_ret['total_return']:+.1f}%")

        # Generate HTML report
        html_path = Path("report_html/backtest_minute_delay.html")
        generate_html_report(results, orig_win_rate, html_path)

        print(f"\nÖffne mit: xdg-open {html_path}")


if __name__ == "__main__":
    main()
