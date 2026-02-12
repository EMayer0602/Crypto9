#!/usr/bin/env python3
"""
Phase Test: ONE trading strategy (JMA + time-based exit), classified by
4 different phase methods (supertrend, htf_crossover, jma, kama).

Architecture:
1. Run ONE JMA backtest per symbol with MaxHoldBars (time-based exit)
2. Classify those SAME trades with 4 different phase methods
3. Generate 4 output sets (same trades, different phase tags)

This isolates the phase classification question: which phase method
gives the best basis for per-phase optimization?

Generates per phase classifier:
  - trading_summary_ph1_{classifier}.json  (JMA trades with phases)
  - dashboard_ph1_{classifier}.html        (ab --start, default 2025-12-01)
  - trading_summary_ph1_{classifier}.html  (ab --summary-start, default 2024-12-01)

Compound growth: Summary AND Dashboard both start independently at 16500.

Usage:
    python test_baseline_phases.py
    python test_baseline_phases.py --max-hold 4
    python test_baseline_phases.py --max-hold 8 --start 2025-12-01 --summary-start 2024-12-01
"""
import argparse
import json
import os

import pandas as pd

import Supertrend_5Min as st

# ── CONFIG ──
START_CAPITAL = 16500.0
MAX_POSITIONS = 8

# Phase classification: FIXED params (independent of trading params)
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0
PHASE_HTF = "6h"

# Backtest HTF filter params
BACKTEST_HTF_LENGTH = 20
BACKTEST_HTF_FACTOR = 3.0
BACKTEST_HTF = "6h"

# Trading strategy: JMA with time-based exit (like BCK)
TRADING_INDICATOR = "jma"
TRADING_PARAM_A = 30       # JMA Length (default)
TRADING_PARAM_B = 0        # JMA Phase (default)
TRADING_ATR_STOP = None    # No ATR stop (BCK uses it for <5% of trades)
TRADING_MIN_HOLD = 0       # No minimum hold

# Phase classifiers: the 4 methods to tag the same trades
PHASE_CLASSIFIERS = ["supertrend", "htf_crossover", "jma", "kama"]

CLASSIFIER_DISPLAY = {
    "supertrend": "Supertrend-Phasen",
    "htf_crossover": "HTF-Crossover-Phasen",
    "jma": "JMA-Phasen",
    "kama": "KAMA-Phasen",
}


def get_phase_at_entry(phases, entry_ts):
    """Look up market phase at entry timestamp using ffill."""
    if phases is None or phases.empty:
        return "Flat"
    try:
        if entry_ts in phases.index:
            return phases.loc[entry_ts]
        idx = phases.index.get_indexer([entry_ts], method="ffill")[0]
        if idx >= 0:
            return phases.iloc[idx]
    except Exception:
        pass
    return "Flat"


def main():
    parser = argparse.ArgumentParser(description="JMA + time-based exit with 4 phase classifiers")
    parser.add_argument("--start", type=str, default="2025-12-01",
                        help="Dashboard start date (default: 2025-12-01)")
    parser.add_argument("--summary-start", type=str, default="2024-12-01",
                        help="Trading summary start date (default: 2024-12-01)")
    parser.add_argument("--max-hold", type=int, default=4,
                        help="MaxHoldBars for time-based exit (default: 4, like BCK)")
    args = parser.parse_args()

    dashboard_start = args.start
    summary_start = args.summary_start
    max_hold_bars = args.max_hold

    print(f"=== JMA + Time-Based Exit (MaxHold={max_hold_bars}) ===")
    print(f"Trading: JMA L={TRADING_PARAM_A}, Phase={TRADING_PARAM_B}, MaxHold={max_hold_bars}")
    print(f"Capital: {START_CAPITAL:,.0f} | Max Positions: {MAX_POSITIONS}")
    print(f"Phase classification: L={PHASE_HTF_LENGTH}, F={PHASE_HTF_FACTOR}, HTF={PHASE_HTF}")
    print(f"Backtest HTF filter: L={BACKTEST_HTF_LENGTH}, F={BACKTEST_HTF_FACTOR}, HTF={BACKTEST_HTF}")
    print(f"Dashboard ab {dashboard_start} | Summary ab {summary_start}")

    symbols = st.SYMBOLS

    # Populate OHLCV cache once
    st.ensure_cache_populated(symbols, st.TIMEFRAME, st.LOOKBACK)

    # Save original globals
    orig_backtest_start = st.BACKTEST_START_DATE
    orig_htf_length = st.HTF_LENGTH
    orig_htf_factor = st.HTF_FACTOR

    # Disable date filter — we want ALL data, filtering in HTML later
    st.BACKTEST_START_DATE = ""

    # ══════════════════════════════════════════════════════════════
    # Step 1: Run ONE JMA backtest per symbol (same for all phases)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 1: JMA Backtest (MaxHold={max_hold_bars})...")

    st.apply_indicator_type(TRADING_INDICATOR)
    st.apply_higher_timeframe(BACKTEST_HTF)
    st.HTF_LENGTH = BACKTEST_HTF_LENGTH
    st.HTF_FACTOR = BACKTEST_HTF_FACTOR
    st.clear_data_cache()

    # raw_trades: list of dicts with entry/exit info (no phase yet)
    raw_trades = []

    for symbol in symbols:
        sym_short = symbol.replace("/USDC", "").replace("/USDT", "")

        df = st.prepare_symbol_dataframe(symbol)
        if df.empty:
            print(f"  {sym_short}: NO DATA")
            continue

        df_ind = st.compute_indicator(df, TRADING_PARAM_A, TRADING_PARAM_B)

        # JMA uses backtest_supertrend (trend-flip entry/exit)
        trades_df = st.backtest_supertrend(
            df_ind, atr_stop_mult=TRADING_ATR_STOP, direction="long",
            min_hold_bars=TRADING_MIN_HOLD, max_hold_bars=max_hold_bars,
        )

        if trades_df.empty:
            print(f"  {sym_short}: 0 trades")
            continue

        count = 0
        for _, row in trades_df.iterrows():
            raw_trades.append({
                "symbol": symbol,
                "direction": "long",
                "indicator": "JMA",
                "htf": BACKTEST_HTF,
                "entry_time": str(row["Zeit"]),
                "exit_time": str(row["ExitZeit"]),
                "entry_price": float(row["Entry"]),
                "exit_price": float(row["ExitPreis"]),
                "reason": row["ExitReason"],
            })
            count += 1

        print(f"  {sym_short}: {count} trades")

    raw_trades.sort(key=lambda t: t["entry_time"])

    closed = [t for t in raw_trades if t["reason"] != "Final bar"]
    opened = [t for t in raw_trades if t["reason"] == "Final bar"]
    print(f"\nTotal: {len(closed)} closed + {len(opened)} open = {len(raw_trades)} trades")

    # Quick compound check over all trades
    capital = START_CAPITAL
    for t in raw_trades:
        stake = capital / MAX_POSITIONS
        ep, xp = t["entry_price"], t["exit_price"]
        pnl_pct = (xp - ep) / ep if ep else 0
        pnl_net = pnl_pct * stake - stake * st.FEE_RATE * 2.0
        capital += pnl_net
    print(f"Compound Growth: {START_CAPITAL:,.2f} -> {capital:,.2f} | PnL: {capital - START_CAPITAL:+,.2f}")

    # Exit reasons breakdown
    from collections import Counter
    reasons = Counter(t["reason"] for t in raw_trades)
    print(f"\nExit reasons:")
    for reason, cnt in reasons.most_common():
        print(f"  {reason}: {cnt}")

    # ══════════════════════════════════════════════════════════════
    # Step 2: Classify with 4 phase methods, generate 4 output sets
    # ══════════════════════════════════════════════════════════════
    for classifier in PHASE_CLASSIFIERS:
        cls_display = CLASSIFIER_DISPLAY[classifier]
        print(f"\n{'='*60}")
        print(f"Step 2: Phase classification → {cls_display}")

        # Set indicator type for phase classification
        st.apply_indicator_type(classifier)
        st.apply_higher_timeframe(PHASE_HTF)
        st.HTF_LENGTH = PHASE_HTF_LENGTH
        st.HTF_FACTOR = PHASE_HTF_FACTOR
        st.clear_data_cache()

        # Classify phases per symbol
        phase_cache = {}
        for symbol in symbols:
            sym_short = symbol.replace("/USDC", "").replace("/USDT", "")
            df_for_phases = st.fetch_data(symbol, st.TIMEFRAME, st.LOOKBACK)
            if df_for_phases.empty:
                phase_cache[symbol] = pd.Series(dtype=str)
                continue
            phases = st.classify_market_phases(df_for_phases, symbol)
            phase_counts = phases.value_counts().to_dict()
            print(f"  {sym_short}: {phase_counts}")
            phase_cache[symbol] = phases

        # Tag trades with phases
        tagged_trades = []
        for t in raw_trades:
            phases = phase_cache.get(t["symbol"], pd.Series(dtype=str))
            entry_ts = pd.Timestamp(t["entry_time"])
            phase = get_phase_at_entry(phases, entry_ts)
            tc = dict(t)
            tc["phase"] = phase
            # Label shows trading strategy + phase classifier
            tc["indicator"] = f"JMA ({cls_display})"
            tagged_trades.append(tc)

        # Phase distribution
        phase_dist = Counter(t["phase"] for t in tagged_trades)
        print(f"  Phase distribution: {dict(phase_dist)}")

        # ── Write JSON ──
        json_out = {
            "trading_strategy": TRADING_INDICATOR,
            "trading_params": {
                "param_a": TRADING_PARAM_A,
                "param_b": TRADING_PARAM_B,
                "max_hold_bars": max_hold_bars,
                "atr_stop": TRADING_ATR_STOP,
            },
            "phase_classifier": classifier,
            "phase_classifier_display": cls_display,
            "indicator_display": f"JMA + MaxHold={max_hold_bars}",
            "start_capital": START_CAPITAL,
            "max_positions": MAX_POSITIONS,
            "phase_config": {
                "HTF_LENGTH": PHASE_HTF_LENGTH,
                "HTF_FACTOR": PHASE_HTF_FACTOR,
                "HTF": PHASE_HTF,
            },
            "backtest_config": {
                "HTF_LENGTH": BACKTEST_HTF_LENGTH,
                "HTF_FACTOR": BACKTEST_HTF_FACTOR,
                "HTF": BACKTEST_HTF,
            },
            "trades": [],
            "open_positions_data": [],
        }

        for t in tagged_trades:
            entry = {
                "symbol": t["symbol"],
                "direction": t["direction"],
                "indicator": t["indicator"],
                "htf": t["htf"],
                "entry_time": t["entry_time"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "phase": t["phase"],
                "reason": t["reason"],
            }
            if t["reason"] == "Final bar":
                entry["last_price"] = t["exit_price"]
                json_out["open_positions_data"].append(entry)
            else:
                entry["exit_time"] = t["exit_time"]
                json_out["trades"].append(entry)

        json_path = os.path.join(st.BASE_OUT_DIR, f"trading_summary_ph1_{classifier}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_out, jf, indent=2, default=str)
        print(f"  -> {json_path} ({len(json_out['trades'])} closed, {len(json_out['open_positions_data'])} open)")

        # ── Generate HTML ──
        prefix = f"ph1_{classifier}"
        st._generate_phase_dashboard(
            tagged_trades,
            dashboard_start=dashboard_start,
            summary_start=summary_start,
            stake_divisor=MAX_POSITIONS,
            indicator_label=f"JMA + MaxHold={max_hold_bars} ({cls_display})",
            output_prefix=prefix,
        )

    # Restore globals
    st.BACKTEST_START_DATE = orig_backtest_start
    st.HTF_LENGTH = orig_htf_length
    st.HTF_FACTOR = orig_htf_factor

    print(f"\n{'='*60}")
    print(f"Done! Trading strategy: JMA L={TRADING_PARAM_A}, MaxHold={max_hold_bars}")
    print(f"Same {len(raw_trades)} trades, tagged with 4 phase classifiers:")
    for classifier in PHASE_CLASSIFIERS:
        cls_display = CLASSIFIER_DISPLAY[classifier]
        print(f"  {cls_display}:")
        print(f"    - trading_summary_ph1_{classifier}.json")
        print(f"    - trading_summary_ph1_{classifier}.html  (ab {summary_start}, start=16.500)")
        print(f"    - dashboard_ph1_{classifier}.html        (ab {dashboard_start}, start=16.500)")


if __name__ == "__main__":
    main()
