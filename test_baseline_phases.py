#!/usr/bin/env python3
"""
Phase Baseline: Read params from best_params_overall.csv, run backtests
with EXACT classic params (supertrend), classify with 4 phase methods.

CSV has 36 rows (9 symbols × 4 phase classifiers). Each row contains:
- Per-symbol trading params from best_params_overall_bck.csv
- PhaseClassifier column (supertrend, htf_crossover, jma, kama)

Architecture:
1. Read params from CSV
2. Run ONE supertrend backtest per symbol (exact classic params)
3. Tag same trades with 4 phase classifiers
4. Generate 4 output sets (same trades, different phase tags)

Generates per phase classifier:
  - trading_summary_ph1_{classifier}.json
  - dashboard_ph1_{classifier}.html
  - trading_summary_ph1_{classifier}.html

Compound growth: Summary AND Dashboard both start independently at 16500.

Usage:
    python test_baseline_phases.py
    python test_baseline_phases.py --start 2025-12-01 --summary-start 2024-12-01
"""
import argparse
import csv
import json
import os
from collections import Counter

import pandas as pd

import Supertrend_5Min as st

# ── CONFIG ──
START_CAPITAL = 16500.0
MAX_POSITIONS = 8
PARAMS_CSV = os.path.join("report_html", "best_params_overall.csv")

# Phase classification: FIXED params (independent of trading params)
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0
PHASE_HTF = "6h"

# Backtest HTF filter params (same as classic)
BACKTEST_HTF_LENGTH = 20
BACKTEST_HTF_FACTOR = 3.0
BACKTEST_HTF = "6h"

CLASSIFIER_DISPLAY = {
    "supertrend": "Supertrend-Phasen",
    "htf_crossover": "HTF-Crossover-Phasen",
    "jma": "JMA-Phasen",
    "kama": "KAMA-Phasen",
}


def parse_german_float(s):
    """Parse German number format: '3,0' -> 3.0, '15087,73' -> 15087.73"""
    if not s or s == "None" or s == "nan":
        return None
    return float(s.replace(",", "."))


def read_params_csv(csv_path):
    """Read best_params_overall.csv, return dict keyed by (symbol, phase_classifier)."""
    params = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            symbol = row["Symbol"]
            classifier = row["PhaseClassifier"]
            param_a = parse_german_float(row["ParamA"])
            param_b = parse_german_float(row["ParamB"])
            atr_str = row.get("ATRStopMult", "None")
            atr_stop = None if atr_str in ("None", "", "nan") else parse_german_float(atr_str)
            min_hold = int(row.get("MinHoldBars", "0") or "0")
            indicator = row.get("Indicator", "supertrend")
            htf = row.get("HTF", "6h")

            params[(symbol, classifier)] = {
                "param_a": param_a,
                "param_b": param_b,
                "atr_stop": atr_stop,
                "min_hold": min_hold,
                "indicator": indicator,
                "htf": htf,
            }
    return params


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
    parser = argparse.ArgumentParser(description="Phase baseline from CSV params")
    parser.add_argument("--start", type=str, default="2025-12-01",
                        help="Dashboard start date (default: 2025-12-01)")
    parser.add_argument("--summary-start", type=str, default="2024-12-01",
                        help="Trading summary start date (default: 2024-12-01)")
    args = parser.parse_args()

    dashboard_start = args.start
    summary_start = args.summary_start

    # ── Read CSV params ──
    if not os.path.exists(PARAMS_CSV):
        print(f"ERROR: {PARAMS_CSV} not found!")
        return
    csv_params = read_params_csv(PARAMS_CSV)
    symbols_in_csv = sorted(set(s for s, _ in csv_params.keys()))
    classifiers_in_csv = sorted(set(c for _, c in csv_params.keys()))

    print(f"=== Phase Baseline (from {PARAMS_CSV}) ===")
    print(f"Symbols: {len(symbols_in_csv)} | Classifiers: {classifiers_in_csv}")
    print(f"Capital: {START_CAPITAL:,.0f} | Max Positions: {MAX_POSITIONS}")
    print(f"Phase classification: L={PHASE_HTF_LENGTH}, F={PHASE_HTF_FACTOR}, HTF={PHASE_HTF}")
    print(f"Backtest HTF filter: L={BACKTEST_HTF_LENGTH}, F={BACKTEST_HTF_FACTOR}, HTF={BACKTEST_HTF}")
    print(f"Dashboard ab {dashboard_start} | Summary ab {summary_start}")

    # Show per-symbol params
    print(f"\nPer-symbol params (from CSV):")
    for sym in symbols_in_csv:
        # All classifiers have same params, just pick first
        p = csv_params.get((sym, classifiers_in_csv[0]), {})
        sym_short = sym.replace("/USDC", "").replace("/USDT", "")
        atr_label = p.get('atr_stop', None)
        atr_str = f"ATR={atr_label}" if atr_label else "ATR=None"
        print(f"  {sym_short}: A={p.get('param_a')}, B={p.get('param_b')}, "
              f"{atr_str}, MinHold={p.get('min_hold')}")

    # Use all symbols from SYMBOLS list (CSV may not have all)
    symbols = st.SYMBOLS

    # Populate OHLCV cache once
    st.ensure_cache_populated(symbols, st.TIMEFRAME, st.LOOKBACK)

    # Save + override globals
    orig_backtest_start = st.BACKTEST_START_DATE
    orig_htf_length = st.HTF_LENGTH
    orig_htf_factor = st.HTF_FACTOR
    orig_trailing_stop = st.USE_TRAILING_STOP
    orig_profit_target = st.USE_PROFIT_TARGET

    st.BACKTEST_START_DATE = ""
    st.USE_TRAILING_STOP = False
    st.USE_PROFIT_TARGET = False

    # ══════════════════════════════════════════════════════════════
    # Step 1: Run ONE backtest per symbol with EXACT CSV params
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 1: Backtest with exact CSV params...")

    # Set indicator type from CSV (all rows are supertrend)
    indicator_type = "supertrend"
    st.apply_indicator_type(indicator_type)
    st.apply_higher_timeframe(BACKTEST_HTF)
    st.HTF_LENGTH = BACKTEST_HTF_LENGTH
    st.HTF_FACTOR = BACKTEST_HTF_FACTOR
    st.clear_data_cache()

    raw_trades = []

    for symbol in symbols:
        sym_short = symbol.replace("/USDC", "").replace("/USDT", "")

        # Get params from CSV (use first classifier since params are same)
        p = csv_params.get((symbol, classifiers_in_csv[0]), None)
        if p is None:
            # Symbol not in CSV — use defaults
            preset = st.INDICATOR_PRESETS[indicator_type]
            param_a = preset["default_a"]
            param_b = preset["default_b"]
            atr_stop = None
            min_hold = 0
        else:
            param_a = p["param_a"]
            param_b = p["param_b"]
            atr_stop = p["atr_stop"]
            min_hold = p["min_hold"]

        df = st.prepare_symbol_dataframe(symbol)
        if df.empty:
            print(f"  {sym_short}: NO DATA")
            continue

        df_ind = st.compute_indicator(df, param_a, param_b)

        trades_df = st.backtest_supertrend(
            df_ind, atr_stop_mult=atr_stop, direction="long",
            min_hold_bars=min_hold, max_hold_bars=0,
        )

        if trades_df.empty:
            print(f"  {sym_short}: 0 trades")
            continue

        count = 0
        for _, row in trades_df.iterrows():
            raw_trades.append({
                "symbol": symbol,
                "direction": "long",
                "indicator": "Supertrend",
                "htf": BACKTEST_HTF,
                "entry_time": str(row["Zeit"]),
                "exit_time": str(row["ExitZeit"]),
                "entry_price": float(row["Entry"]),
                "exit_price": float(row["ExitPreis"]),
                "reason": row["ExitReason"],
            })
            count += 1

        atr_label = f"ATR={atr_stop}" if atr_stop else "ATR=None"
        print(f"  {sym_short}: {count} trades (A={param_a}, B={param_b}, {atr_label}, MinHold={min_hold})")

    raw_trades.sort(key=lambda t: t["entry_time"])

    closed = [t for t in raw_trades if t["reason"] != "Final bar"]
    opened = [t for t in raw_trades if t["reason"] == "Final bar"]
    print(f"\nTotal: {len(closed)} closed + {len(opened)} open = {len(raw_trades)} trades")

    # Compound growth check
    capital = START_CAPITAL
    for t in raw_trades:
        stake = capital / MAX_POSITIONS
        ep, xp = t["entry_price"], t["exit_price"]
        pnl_pct = (xp - ep) / ep if ep else 0
        pnl_net = pnl_pct * stake - stake * st.FEE_RATE * 2.0
        capital += pnl_net
    print(f"Compound Growth: {START_CAPITAL:,.2f} -> {capital:,.2f} | PnL: {capital - START_CAPITAL:+,.2f}")

    # Exit reasons
    reasons = Counter(t["reason"] for t in raw_trades)
    print(f"\nExit reasons:")
    for reason, cnt in reasons.most_common():
        print(f"  {reason}: {cnt}")

    # Restore exit strategy globals
    st.USE_TRAILING_STOP = orig_trailing_stop
    st.USE_PROFIT_TARGET = orig_profit_target

    # ══════════════════════════════════════════════════════════════
    # Step 2: Classify with 4 phase methods, generate 4 output sets
    # ══════════════════════════════════════════════════════════════
    for classifier in classifiers_in_csv:
        cls_display = CLASSIFIER_DISPLAY.get(classifier, classifier)
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
            tc["indicator"] = f"Supertrend ({cls_display})"
            tagged_trades.append(tc)

        # Phase distribution
        phase_dist = Counter(t["phase"] for t in tagged_trades)
        print(f"  Phase distribution: {dict(phase_dist)}")

        # ── Write JSON ──
        json_out = {
            "trading_strategy": indicator_type,
            "phase_classifier": classifier,
            "phase_classifier_display": cls_display,
            "indicator_display": f"Supertrend ({cls_display})",
            "start_capital": START_CAPITAL,
            "max_positions": MAX_POSITIONS,
            "params_csv": PARAMS_CSV,
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
            indicator_label=f"Supertrend ({cls_display})",
            output_prefix=prefix,
        )

    # Restore globals
    st.BACKTEST_START_DATE = orig_backtest_start
    st.HTF_LENGTH = orig_htf_length
    st.HTF_FACTOR = orig_htf_factor

    print(f"\n{'='*60}")
    print(f"Done! Params from {PARAMS_CSV}")
    print(f"Same {len(raw_trades)} trades, tagged with {len(classifiers_in_csv)} phase classifiers:")
    for classifier in classifiers_in_csv:
        cls_display = CLASSIFIER_DISPLAY.get(classifier, classifier)
        print(f"  {cls_display}:")
        print(f"    - trading_summary_ph1_{classifier}.json")
        print(f"    - trading_summary_ph1_{classifier}.html  (ab {summary_start}, start=16.500)")
        print(f"    - dashboard_ph1_{classifier}.html        (ab {dashboard_start}, start=16.500)")


if __name__ == "__main__":
    main()
