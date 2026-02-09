#!/usr/bin/env python3
"""
Simple Momentum Delay Backtest - Test different entry delays.

Usage:
    python test_momentum_delay.py                    # Test 1h to 12h
    python test_momentum_delay.py --hours 1,2,4,8   # Test specific hours
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

import Supertrend_5Min as st

# Settings
SOURCE_JSON = Path("report_html/trading_summary.json")
START_CAPITAL = 16500.0
MAX_POSITIONS = 10


def load_trades(start_date="2024-01-31"):
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
            entry_dt = entry_dt.replace(tzinfo=None)  # Make naive
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


def load_ohlcv_cache(symbols):
    """Load 1h OHLCV data for all symbols."""
    cache = {}
    for symbol in symbols:
        df = st.load_ohlcv_from_cache(symbol, "1h")
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            cache[symbol] = df
    return cache


def get_price_at_delay(df, entry_time, hours):
    """Get close price at entry_time + hours."""
    if df is None or df.empty:
        return None

    target_time = entry_time + timedelta(hours=hours)
    mask = df.index >= target_time

    if mask.any():
        idx = df.index[mask][0]
        row = df.loc[idx]
        if isinstance(row, pd.DataFrame):
            return float(row["close"].iloc[0])
        return float(row["close"])
    return None


def backtest_delay(trades, ohlcv_cache, delay_hours):
    """Backtest with specific delay in hours."""

    passed = []

    for t in trades:
        symbol = t["symbol"]
        df = ohlcv_cache.get(symbol)
        if df is None:
            continue

        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        entry_time = t["entry_time"]

        # Preis nach delay_hours Stunden
        price_at_delay = get_price_at_delay(df, entry_time, delay_hours)

        if price_at_delay is None:
            continue

        # Momentum-Check: Preis muss gestiegen sein
        if price_at_delay > entry_price:
            # Entry zum verzögerten Preis, Exit zum Original-Exit
            new_pnl_pct = (exit_price - price_at_delay) / price_at_delay * 100
            passed.append({
                "symbol": symbol,
                "pnl_pct": new_pnl_pct,
                "price_at_delay": price_at_delay,
                "original_pnl_pct": t["pnl_pct"]
            })

    return passed


def calculate_stats(trades, passed):
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
            "total_return": 0
        }

    wins = sum(1 for t in passed if t["pnl_pct"] > 0)
    win_rate = wins / filtered * 100
    avg_pnl = sum(t["pnl_pct"] for t in passed) / filtered

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
        "total_return": total_return
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-31", help="Start date YYYY-MM-DD")
    parser.add_argument("--hours", default="1,2,3,4,5,6,7,8,9,10,11,12",
                        help="Comma-separated hours to test")
    args = parser.parse_args()

    hours_list = [int(h) for h in args.hours.split(",")]

    print(f"\nLade Trades seit {args.start}...")
    trades = load_trades(args.start)
    print(f"  {len(trades)} Long-Trades geladen")

    # Original stats
    orig_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    orig_win_rate = orig_wins / len(trades) * 100
    print(f"  Original Win Rate: {orig_win_rate:.1f}%")

    print("\nLade OHLCV-Daten...")
    symbols = list(set(t["symbol"] for t in trades))
    ohlcv_cache = load_ohlcv_cache(symbols)
    print(f"  {len(ohlcv_cache)} Symbole im Cache")

    print("\n" + "=" * 80)
    print("MOMENTUM FILTER BACKTEST - Entry-Delay Vergleich")
    print("=" * 80)
    print(f"\n{'Delay':>6} | {'Trades':>7} | {'Pass%':>7} | {'WinRate':>8} | "
          f"{'vs Orig':>8} | {'Avg PnL':>8} | {'Return':>8}")
    print("-" * 80)

    results = []

    for hours in hours_list:
        passed = backtest_delay(trades, ohlcv_cache, hours)
        stats = calculate_stats(trades, passed)

        improvement = stats["win_rate"] - orig_win_rate

        print(f"{hours:>5}h | {stats['filtered']:>7} | {stats['pass_rate']:>6.1f}% | "
              f"{stats['win_rate']:>7.1f}% | {improvement:>+7.1f}% | "
              f"{stats['avg_pnl']:>+7.2f}% | {stats['total_return']:>+7.1f}%")

        results.append({
            "hours": hours,
            **stats,
            "improvement": improvement
        })

    print("-" * 80)

    # Best results
    if results:
        best_wr = max(results, key=lambda x: x["win_rate"])
        best_ret = max(results, key=lambda x: x["total_return"])

        print(f"\nBeste Win Rate:  {best_wr['hours']}h mit {best_wr['win_rate']:.1f}%")
        print(f"Bester Return:   {best_ret['hours']}h mit {best_ret['total_return']:+.1f}%")


if __name__ == "__main__":
    main()
