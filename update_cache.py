#!/usr/bin/env python3
"""
Update OHLCV cache for all symbols and timeframes.
Usage: python update_cache.py [--start YYYY-MM-DD]

Default start date: 2024-01-01
Only fills gaps: downloads missing history at the start and new bars at the end.
"""

import argparse
import time

import pandas as pd

import Supertrend_5Min as st


# Timeframes needed for simulation (1h for signals, HTF for trend, 1d for phases)
TIMEFRAMES = ["1h", "8h", "12h", "1d"]


def update_cache(start_date: str = "2024-01-01"):
    symbols = st.SYMBOLS
    now = pd.Timestamp.now(tz=st.BERLIN_TZ)
    start = pd.Timestamp(start_date, tz=st.BERLIN_TZ)

    total = len(symbols) * len(TIMEFRAMES)
    done = 0
    skipped = 0
    updated = 0

    print(f"=== OHLCV Cache Update (gap-fill) ===")
    print(f"Symbols: {len(symbols)}, Timeframes: {TIMEFRAMES}")
    print(f"Period: {start_date} to {now.strftime('%Y-%m-%d %H:%M')}")
    print()

    for symbol in symbols:
        for tf in TIMEFRAMES:
            done += 1
            cached = st.load_ohlcv_from_cache(symbol, tf)

            if cached.empty:
                # No cache at all - full download
                print(f"[{done}/{total}] {symbol} {tf}: No cache - downloading from {start_date}...")
                try:
                    st.download_historical_ohlcv(symbol, tf, start, now)
                    updated += 1
                except Exception as e:
                    print(f"  ERROR: {e}")
                continue

            earliest = cached.index.min()
            latest = cached.index.max()
            downloads_needed = []

            # Gap at the beginning?
            if earliest > start + pd.Timedelta(days=1):
                downloads_needed.append(("start", start, earliest))

            # Gap at the end? (more than 1 bar behind)
            tf_minutes = st.timeframe_to_minutes(tf)
            if latest < now - pd.Timedelta(minutes=tf_minutes * 2):
                downloads_needed.append(("end", latest, now))

            if not downloads_needed:
                print(f"[{done}/{total}] {symbol} {tf}: {len(cached)} bars, up to {latest.strftime('%Y-%m-%d %H:%M')} - OK")
                skipped += 1
                continue

            gaps = ", ".join(f"{g[0]}: {g[1].strftime('%Y-%m-%d')}..{g[2].strftime('%Y-%m-%d')}" for g in downloads_needed)
            print(f"[{done}/{total}] {symbol} {tf}: {len(cached)} bars, filling gaps ({gaps})")

            for _, dl_start, dl_end in downloads_needed:
                try:
                    st.download_historical_ohlcv(symbol, tf, dl_start, dl_end)
                except Exception as e:
                    print(f"  ERROR: {e}")

            updated += 1
            time.sleep(0.3)

    print()
    print(f"=== Done: {updated} updated, {skipped} already current, {total - updated - skipped} errors ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update OHLCV cache for all symbols")
    parser.add_argument("--start", default="2024-01-01", help="Start date (default: 2024-01-01)")
    args = parser.parse_args()

    update_cache(args.start)
