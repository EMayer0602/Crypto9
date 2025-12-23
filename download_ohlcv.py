"""Download historical OHLCV data for all symbols.

Usage:
    python download_ohlcv.py --start 2025-01-01
"""
import argparse
import sys
import Supertrend_5Min as st

def main():
    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), defaults to now")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (default: all)")
    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = st.SYMBOLS

    print(f"Downloading OHLCV data from {args.start} to {args.end or 'now'}")
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {st.TIMEFRAME}")
    print()

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")

        try:
            # Download main timeframe
            df = st.download_historical_ohlcv(symbol, st.TIMEFRAME, args.start, args.end)
            if not df.empty:
                st.save_ohlcv_to_cache(symbol, st.TIMEFRAME, df)
                print(f"[OK] {symbol} {st.TIMEFRAME}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            else:
                print(f"[WARN] {symbol} {st.TIMEFRAME}: No data")

            # Also download higher timeframe if different
            if st.HIGHER_TIMEFRAME != st.TIMEFRAME:
                df_htf = st.download_historical_ohlcv(symbol, st.HIGHER_TIMEFRAME, args.start, args.end)
                if not df_htf.empty:
                    st.save_ohlcv_to_cache(symbol, st.HIGHER_TIMEFRAME, df_htf)
                    print(f"[OK] {symbol} {st.HIGHER_TIMEFRAME}: {len(df_htf)} bars")

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

    print("\n" + "="*60)
    print("Download complete!")
    print(f"Data saved to: {st.OHLCV_CACHE_DIR}/")

if __name__ == "__main__":
    main()
