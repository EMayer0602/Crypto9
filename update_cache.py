#!/usr/bin/env python3
"""
Update OHLCV cache for all symbols and timeframes.
Usage: python update_cache.py [--start YYYY-MM-DD]

Default start date: 2024-01-01
Downloads/updates cache incrementally (only fetches new bars since last cached bar).
"""

import argparse
import sys
import time

import pandas as pd

import Supertrend_5Min as st


# Timeframes needed for simulation (1h for signals, HTF for trend)
TIMEFRAMES = ["1h", "8h", "12h"]


def update_cache(start_date: str = "2024-01-01"):
    symbols = st.SYMBOLS
    now = pd.Timestamp.now(tz=st.BERLIN_TZ)
    start = pd.Timestamp(start_date, tz=st.BERLIN_TZ)

    total = len(symbols) * len(TIMEFRAMES)
    done = 0

    print(f"=== OHLCV Cache Update ===")
    print(f"Symbols: {len(symbols)}, Timeframes: {TIMEFRAMES}")
    print(f"Period: {start_date} to {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Total downloads: {total}")
    print()

    for symbol in symbols:
        for tf in TIMEFRAMES:
            done += 1
            cached = st.load_ohlcv_from_cache(symbol, tf)

            if cached.empty:
                print(f"[{done}/{total}] {symbol} {tf}: No cache - full download from {start_date}")
            else:
                earliest = cached.index.min()
                latest = cached.index.max()
                print(f"[{done}/{total}] {symbol} {tf}: {len(cached)} bars, last: {latest.strftime('%Y-%m-%d %H:%M')} - updating...")

            try:
                st.download_historical_ohlcv(symbol, tf, start, now)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # Verify
            updated = st.load_ohlcv_from_cache(symbol, tf)
            if not updated.empty:
                print(f"  OK: {len(updated)} bars, {updated.index.min().strftime('%Y-%m-%d')} to {updated.index.max().strftime('%Y-%m-%d %H:%M')}")
            print()

            # Small delay between symbols to avoid rate limits
            time.sleep(0.3)

    print(f"=== Cache update complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update OHLCV cache for all symbols")
    parser.add_argument("--start", default="2024-01-01", help="Start date (default: 2024-01-01)")
    args = parser.parse_args()

    update_cache(args.start)
