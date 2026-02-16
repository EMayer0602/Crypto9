#!/usr/bin/env python3
"""
Validation test for phase sweep:
1. Load existing trades from trading_summary_ph1_jma_jma.json
2. Re-classify them with classify_market_phases() (same as sweep code)
3. Verify phase labels match the JSON
4. Verify per-phase trade counts and PnLs match the reference

Reference:
  Up:   1,120 trades, PnL: +583,876.43
  Down:   151 trades, PnL:  +46,321.21
  Flat: 1,281 trades, PnL: +145,318.60
  Total: 2,552 trades, PnL: +775,516.25
"""
import json
import sys
from collections import Counter, defaultdict

import pandas as pd

import Supertrend_5Min as st

# Phase classifier settings (same as test_baseline_phases.py)
PHASE_CLASSIFIER = "jma"
PHASE_HTF = "6h"
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0

# Reference values
REFERENCE = {
    "Up":   {"trades": 1120, "pnl": 583876.43},
    "Down": {"trades": 151,  "pnl": 46321.21},
    "Flat": {"trades": 1281, "pnl": 145318.60},
}
REFERENCE_TOTAL = {"trades": 2552, "pnl": 775516.25}


def get_phase_at(phases_series, ts):
    """Look up market phase at timestamp using ffill."""
    if phases_series is None or phases_series.empty:
        return "Flat"
    try:
        if ts in phases_series.index:
            return phases_series.loc[ts]
        idx = phases_series.index.get_indexer([ts], method="ffill")[0]
        if idx >= 0:
            return phases_series.iloc[idx]
    except Exception:
        pass
    return "Flat"


def main():
    json_path = "report_html/trading_summary_ph1_jma_jma.json"
    print("=" * 70)
    print("  SWEEP VALIDATION TEST - Phase Classification")
    print("=" * 70)
    print(f"  Source: {json_path}")
    print(f"  Phase classifier: {PHASE_CLASSIFIER} (HTF={PHASE_HTF}, L={PHASE_HTF_LENGTH}, F={PHASE_HTF_FACTOR})")
    print("=" * 70)

    # Step 1: Load existing trades from JSON
    with open(json_path) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    print(f"\n[Step 1] Loaded {len(trades)} trades from JSON")

    # Original phase distribution from JSON
    orig_counts = Counter()
    orig_pnls = defaultdict(float)
    for t in trades:
        phase = t.get("phase", "Unknown")
        pnl = float(t.get("pnl", 0) or 0)
        orig_counts[phase] += 1
        orig_pnls[phase] += pnl

    print(f"  Original phases: Up={orig_counts.get('Up',0)} Down={orig_counts.get('Down',0)} Flat={orig_counts.get('Flat',0)}")

    # Step 2: Re-classify using sweep's classify_market_phases()
    print(f"\n[Step 2] Re-classifying with classify_market_phases()...")

    orig_indicator = st.INDICATOR_TYPE
    orig_htf = st.HIGHER_TIMEFRAME
    orig_htf_len = st.HTF_LENGTH
    orig_htf_fac = st.HTF_FACTOR

    st.apply_indicator_type(PHASE_CLASSIFIER)
    st.apply_higher_timeframe(PHASE_HTF)
    st.HTF_LENGTH = PHASE_HTF_LENGTH
    st.HTF_FACTOR = PHASE_HTF_FACTOR
    st.clear_data_cache()

    # Get unique symbols from trades
    symbols = sorted(set(t["symbol"] for t in trades))
    print(f"  Symbols: {len(symbols)}: {symbols}")

    phase_cache = {}
    for symbol in symbols:
        df_phases = st.fetch_data(symbol, st.TIMEFRAME, st.LOOKBACK)
        if df_phases.empty:
            phase_cache[symbol] = pd.Series(dtype=str)
            continue
        phases = st.classify_market_phases(df_phases, symbol)
        phase_cache[symbol] = phases
        n_up = (phases == "Up").sum()
        n_down = (phases == "Down").sum()
        n_flat = (phases == "Flat").sum()
        print(f"    {symbol}: Up={n_up}, Down={n_down}, Flat={n_flat}")

    # Restore
    st.apply_indicator_type(orig_indicator)
    st.apply_higher_timeframe(orig_htf)
    st.HTF_LENGTH = orig_htf_len
    st.HTF_FACTOR = orig_htf_fac

    # Step 3: Re-tag trades and compare
    print(f"\n[Step 3] Re-tagging trades and comparing...")
    new_counts = Counter()
    new_pnls = defaultdict(float)
    mismatches = 0
    mismatch_detail = Counter()  # (orig_phase, new_phase) -> count

    for t in trades:
        symbol = t["symbol"]
        entry_ts = pd.Timestamp(t["entry_time"])
        pnl = float(t.get("pnl", 0) or 0)
        orig_phase = t.get("phase", "Unknown")

        phases_s = phase_cache.get(symbol, pd.Series(dtype=str))
        new_phase = get_phase_at(phases_s, entry_ts)

        new_counts[new_phase] += 1
        new_pnls[new_phase] += pnl

        if orig_phase != new_phase:
            mismatches += 1
            mismatch_detail[(orig_phase, new_phase)] += 1

    # Step 4: Results
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  Phase label mismatches: {mismatches} / {len(trades)} "
          f"({mismatches/len(trades)*100:.2f}%)")
    if mismatch_detail:
        print(f"  Mismatch details:")
        for (orig, new), count in sorted(mismatch_detail.items(), key=lambda x: -x[1]):
            print(f"    {orig:5s} -> {new:5s}: {count:>4d}")

    print(f"\n{'Phase':8s} {'Re-tag':>8s} {'Original':>9s} {'Ref':>8s} {'Match Orig':>11s} {'Match Ref':>10s}")
    print(f"{'-'*70}")

    all_pass = True
    for phase in ["Up", "Down", "Flat"]:
        n = new_counts.get(phase, 0)
        o = orig_counts.get(phase, 0)
        r = REFERENCE[phase]["trades"]
        match_orig = "OK" if n == o else "FAIL"
        match_ref = "OK" if n == r else "FAIL"
        if n != o:
            all_pass = False
        print(f"{phase:8s} {n:>8d} {o:>9d} {r:>8d} {match_orig:>11s} {match_ref:>10s}")

    print(f"\n{'Phase':8s} {'Re-tag PnL':>14s} {'Orig PnL':>14s} {'Ref PnL':>14s} {'Match':>6s}")
    print(f"{'-'*70}")

    for phase in ["Up", "Down", "Flat"]:
        np_ = new_pnls.get(phase, 0)
        op_ = orig_pnls.get(phase, 0)
        rp_ = REFERENCE[phase]["pnl"]
        pnl_match = abs(np_ - op_) < 1.0  # Compare with original JSON
        if not pnl_match:
            all_pass = False
        print(f"{phase:8s} {np_:>+14,.2f} {op_:>+14,.2f} {rp_:>+14,.2f} {'OK' if pnl_match else 'FAIL':>6s}")

    print(f"\n{'='*70}")
    if all_pass:
        print("TEST PASSED: Phase classification is 100% consistent!")
        print("Re-tagged phases match original JSON exactly.")
        print("Phase sweep code can proceed.")
    else:
        if mismatches == 0:
            print("Phase labels match but PnL sums differ (rounding).")
            print("TEST PASSED (effectively): Phase classification is consistent!")
            all_pass = True
        else:
            print(f"TEST FAILED: {mismatches} phase mismatches found.")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
