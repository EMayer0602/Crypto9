#!/usr/bin/env python3
"""
Fetch minute-level OHLCV data around trading times.

This script fetches 1m or 5m data for symbols around their trade entry/exit times
and saves to the ohlcv_cache directory. Does NOT modify any existing functionality.

Usage:
    python fetch_minute_data.py                    # Fetch 5m data for all trades
    python fetch_minute_data.py --timeframe 1m    # Fetch 1m data
    python fetch_minute_data.py --symbol BTC/USDC # Fetch for specific symbol
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Use existing infrastructure from Supertrend_5Min
import Supertrend_5Min as st


# Settings
SOURCE_JSON = Path("report_html/trading_summary.json")

# Buffer around trading times (hours before entry, hours after exit)
BUFFER_BEFORE_HOURS = 2
BUFFER_AFTER_HOURS = 2


def load_trades(json_path: Path) -> list:
    """Load trades from trading_summary.json."""
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("trades", [])


def get_trading_windows(trades: list, symbol: str = None) -> dict:
    """Get time windows for each symbol based on their trades.

    Returns dict: {symbol: [(start_time, end_time), ...]}
    """
    windows = {}

    for trade in trades:
        sym = trade.get("symbol", "")
        if symbol and sym != symbol:
            continue

        entry_time_str = trade.get("entry_time", "")
        exit_time_str = trade.get("exit_time", "")

        if not entry_time_str:
            continue

        try:
            entry_time = datetime.fromisoformat(entry_time_str).replace(tzinfo=None)
        except ValueError:
            try:
                entry_time = datetime.strptime(entry_time_str[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

        if exit_time_str:
            try:
                exit_time = datetime.fromisoformat(exit_time_str).replace(tzinfo=None)
            except ValueError:
                try:
                    exit_time = datetime.strptime(exit_time_str[:19], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    exit_time = entry_time + timedelta(hours=24)
        else:
            exit_time = datetime.now()

        # Add buffer
        start = entry_time - timedelta(hours=BUFFER_BEFORE_HOURS)
        end = exit_time + timedelta(hours=BUFFER_AFTER_HOURS)

        if sym not in windows:
            windows[sym] = []
        windows[sym].append((start, end))

    # Merge overlapping windows
    for sym in windows:
        windows[sym] = merge_windows(windows[sym])

    return windows


def merge_windows(windows: list) -> list:
    """Merge overlapping time windows."""
    if not windows:
        return []

    # Sort by start time
    sorted_windows = sorted(windows, key=lambda x: x[0])
    merged = [sorted_windows[0]]

    for start, end in sorted_windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping - merge
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def fetch_minute_data(symbol: str, start: datetime, end: datetime,
                      timeframe: str = "5m") -> pd.DataFrame:
    """Fetch minute OHLCV data using existing infrastructure."""
    # Use the existing download_historical_ohlcv function
    try:
        df = st.download_historical_ohlcv(symbol, timeframe, start, end)
        return df
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Fetch minute data around trading times")
    parser.add_argument("--timeframe", "-t", type=str, default="5m",
                        help="Timeframe: 1m or 5m (default: 5m)")
    parser.add_argument("--symbol", "-s", type=str, default=None,
                        help="Specific symbol (e.g., BTC/USDC)")
    parser.add_argument("--json", type=str, default=str(SOURCE_JSON),
                        help="Path to trading_summary.json")
    args = parser.parse_args()

    print(f"Fetching {args.timeframe} data around trading times...")

    # Load trades
    trades = load_trades(Path(args.json))
    if not trades:
        print("No trades found!")
        return

    print(f"Loaded {len(trades)} trades")

    # Get trading windows per symbol
    windows = get_trading_windows(trades, symbol=args.symbol)
    print(f"Found {len(windows)} symbols with trading windows")

    # Fetch data for each symbol
    for symbol, time_windows in windows.items():
        print(f"\n{symbol}: {len(time_windows)} time windows")

        all_dfs = []
        for i, (start, end) in enumerate(time_windows):
            print(f"  Window {i+1}: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")

            df = fetch_minute_data(symbol, start, end, args.timeframe)
            if not df.empty:
                all_dfs.append(df)
                print(f"    Fetched {len(df)} bars")
            else:
                print(f"    No data")

        if all_dfs:
            # Combine all windows
            combined_df = pd.concat(all_dfs)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)

            # Save to cache using existing function
            st.save_ohlcv_to_cache(symbol, args.timeframe, combined_df)
            print(f"  Saved {len(combined_df)} bars to ohlcv_cache/")

    print("\nDone!")


if __name__ == "__main__":
    main()
