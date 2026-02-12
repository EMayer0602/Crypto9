#!/usr/bin/env python3
"""
Phase Baseline: Read params from best_params_overall.csv, run backtests
with EXACT per-indicator per-symbol params, classify with 4 phase methods.

CSV has 208 rows (13 symbols × 4 indicators × 4 phase classifiers).
Each row contains per-symbol optimized params + PhaseClassifier column.

Architecture:
1. Read params from CSV (per indicator, per symbol: ParamA, ParamB, ATR, MinHold, HTF)
2. For each trading indicator (supertrend, htf_crossover, jma, kama):
   Run backtests with exact per-symbol params
3. Tag same trades with 4 phase classifiers
4. Generate output sets

Generates per (trading indicator, phase classifier):
  - trading_summary_ph1_{indicator}_{classifier}.json
  - dashboard_ph1_{indicator}_{classifier}.html
  - trading_summary_ph1_{indicator}_{classifier}.html

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

TRADING_INDICATORS = ["supertrend", "htf_crossover", "jma", "kama"]

INDICATOR_DISPLAY = {
    "supertrend": "Supertrend",
    "htf_crossover": "HTF Crossover",
    "jma": "JMA",
    "kama": "KAMA",
}

CLASSIFIER_DISPLAY = {
    "supertrend": "ST-Phasen",
    "htf_crossover": "HTF-Phasen",
    "jma": "JMA-Phasen",
    "kama": "KAMA-Phasen",
}


def parse_german_float(s):
    """Parse German number format: '3,0' -> 3.0"""
    if not s or s == "None" or s == "nan":
        return None
    return float(s.replace(",", "."))


def read_params_csv(csv_path):
    """Read best_params_overall.csv, return dict keyed by (symbol, indicator, classifier)."""
    params = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            symbol = row["Symbol"]
            indicator = row["Indicator"]
            classifier = row["PhaseClassifier"]

            param_a = parse_german_float(row["ParamA"])
            param_b = parse_german_float(row["ParamB"])
            atr_str = row.get("ATRStopMult", "None")
            atr_stop = None if atr_str in ("None", "", "nan") else parse_german_float(atr_str)
            min_hold = int(row.get("MinHoldBars", "0") or "0")
            htf = row.get("HTF", "6h")

            params[(symbol, indicator, classifier)] = {
                "param_a": param_a,
                "param_b": param_b,
                "atr_stop": atr_stop,
                "min_hold": min_hold,
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
    symbols_in_csv = sorted(set(s for s, _, _ in csv_params.keys()))
    classifiers_in_csv = sorted(set(c for _, _, c in csv_params.keys()))

    print(f"=== Phase Baseline (from {PARAMS_CSV}) ===")
    print(f"Symbols: {len(symbols_in_csv)} | Indicators: {TRADING_INDICATORS}")
    print(f"Phase classifiers: {classifiers_in_csv}")
    print(f"Capital: {START_CAPITAL:,.0f} | Max Positions: {MAX_POSITIONS}")
    print(f"Phase classification: L={PHASE_HTF_LENGTH}, F={PHASE_HTF_FACTOR}")
    print(f"Dashboard ab {dashboard_start} | Summary ab {summary_start}")

    symbols = symbols_in_csv

    # Populate OHLCV cache once
    st.ensure_cache_populated(symbols, st.TIMEFRAME, st.LOOKBACK)

    # Save original globals
    orig_backtest_start = st.BACKTEST_START_DATE
    orig_htf_length = st.HTF_LENGTH
    orig_htf_factor = st.HTF_FACTOR
    orig_trailing_stop = st.USE_TRAILING_STOP
    orig_profit_target = st.USE_PROFIT_TARGET

    st.BACKTEST_START_DATE = ""
    st.USE_TRAILING_STOP = False
    st.USE_PROFIT_TARGET = False

    # ══════════════════════════════════════════════════════════════
    # For each trading indicator: backtest, then tag with 4 classifiers
    # ══════════════════════════════════════════════════════════════
    for indicator in TRADING_INDICATORS:
        ind_display = INDICATOR_DISPLAY[indicator]

        print(f"\n{'='*60}")
        print(f"Trading indicator: {ind_display}")
        print(f"{'='*60}")

        # ── Step 1: Backtest per symbol with exact CSV params ──
        st.apply_indicator_type(indicator)

        raw_trades = []

        for symbol in symbols:
            sym_short = symbol.replace("/USDC", "").replace("/USDT", "")

            # Get params (use first classifier since trading params are same)
            p = csv_params.get((symbol, indicator, classifiers_in_csv[0]), None)
            if p is None:
                print(f"  {sym_short}: no params in CSV, skipping")
                continue

            param_a = p["param_a"]
            param_b = p["param_b"]
            atr_stop = p["atr_stop"]
            min_hold = p["min_hold"]
            htf = p["htf"]

            # Set HTF for this symbol's backtest
            st.apply_higher_timeframe(htf)
            # HTF filter uses same Length/Factor as the indicator params
            st.HTF_LENGTH = int(param_a) if param_a else 10
            st.HTF_FACTOR = float(param_b) if param_b else 3.0
            st.clear_data_cache()

            df = st.prepare_symbol_dataframe(symbol)
            if df.empty:
                print(f"  {sym_short}: NO DATA")
                continue

            df_ind = st.compute_indicator(df, param_a, param_b)

            # Route to correct backtest function
            if indicator == "htf_crossover":
                trades_df = st.backtest_htf_crossover(
                    df_ind, atr_stop_mult=atr_stop, direction="long",
                    min_hold_bars=min_hold, max_hold_bars=0,
                )
            else:
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
                    "indicator": ind_display,
                    "htf": htf,
                    "entry_time": str(row["Zeit"]),
                    "exit_time": str(row["ExitZeit"]),
                    "entry_price": float(row["Entry"]),
                    "exit_price": float(row["ExitPreis"]),
                    "reason": row["ExitReason"],
                })
                count += 1

            atr_label = f"ATR={atr_stop}" if atr_stop else "ATR=None"
            print(f"  {sym_short}: {count} trades (A={param_a}, B={param_b}, {atr_label}, MinHold={min_hold}, HTF={htf})")

        raw_trades.sort(key=lambda t: t["entry_time"])

        closed = [t for t in raw_trades if t["reason"] != "Final bar"]
        opened = [t for t in raw_trades if t["reason"] == "Final bar"]
        print(f"\n  {ind_display} total: {len(closed)} closed + {len(opened)} open = {len(raw_trades)}")

        # Compound check
        capital = START_CAPITAL
        for t in raw_trades:
            stake = capital / MAX_POSITIONS
            ep, xp = t["entry_price"], t["exit_price"]
            pnl_pct = (xp - ep) / ep if ep else 0
            pnl_net = pnl_pct * stake - stake * st.FEE_RATE * 2.0
            capital += pnl_net
        print(f"  Compound: {START_CAPITAL:,.2f} -> {capital:,.2f} | PnL: {capital - START_CAPITAL:+,.2f}")

        # Exit reasons
        reasons = Counter(t["reason"] for t in raw_trades)
        print(f"  Exits: {dict(reasons.most_common(5))}")

        # ── Step 2: Tag with 4 phase classifiers ──
        for classifier in classifiers_in_csv:
            cls_display = CLASSIFIER_DISPLAY.get(classifier, classifier)

            print(f"\n  Phase: {cls_display}")

            # Set phase classifier
            st.apply_indicator_type(classifier)
            st.apply_higher_timeframe(PHASE_HTF)
            st.HTF_LENGTH = PHASE_HTF_LENGTH
            st.HTF_FACTOR = PHASE_HTF_FACTOR
            st.clear_data_cache()

            # Classify phases per symbol
            phase_cache = {}
            for symbol in symbols:
                df_for_phases = st.fetch_data(symbol, st.TIMEFRAME, st.LOOKBACK)
                if df_for_phases.empty:
                    phase_cache[symbol] = pd.Series(dtype=str)
                    continue
                phases = st.classify_market_phases(df_for_phases, symbol)
                phase_cache[symbol] = phases

            # Tag trades
            tagged_trades = []
            for t in raw_trades:
                phases = phase_cache.get(t["symbol"], pd.Series(dtype=str))
                entry_ts = pd.Timestamp(t["entry_time"])
                phase = get_phase_at_entry(phases, entry_ts)
                tc = dict(t)
                tc["phase"] = phase
                tagged_trades.append(tc)

            phase_dist = Counter(t["phase"] for t in tagged_trades)
            print(f"    Distribution: {dict(phase_dist)}")

            # ── Write JSON ──
            json_out = {
                "trading_strategy": indicator,
                "trading_indicator_display": ind_display,
                "phase_classifier": classifier,
                "phase_classifier_display": cls_display,
                "start_capital": START_CAPITAL,
                "max_positions": MAX_POSITIONS,
                "params_csv": PARAMS_CSV,
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

            prefix = f"ph1_{indicator}_{classifier}"
            json_path = os.path.join(st.BASE_OUT_DIR, f"trading_summary_{prefix}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(json_out, jf, indent=2, default=str)
            print(f"    -> {json_path} ({len(json_out['trades'])} closed, {len(json_out['open_positions_data'])} open)")

            # ── Generate HTML ──
            label = f"{ind_display} ({cls_display})"
            st._generate_phase_dashboard(
                tagged_trades,
                dashboard_start=dashboard_start,
                summary_start=summary_start,
                stake_divisor=MAX_POSITIONS,
                indicator_label=label,
                output_prefix=prefix,
            )

    # Restore globals
    st.BACKTEST_START_DATE = orig_backtest_start
    st.HTF_LENGTH = orig_htf_length
    st.HTF_FACTOR = orig_htf_factor
    st.USE_TRAILING_STOP = orig_trailing_stop
    st.USE_PROFIT_TARGET = orig_profit_target

    print(f"\n{'='*60}")
    print(f"Done! {len(TRADING_INDICATORS)} indicators × {len(classifiers_in_csv)} classifiers = {len(TRADING_INDICATORS) * len(classifiers_in_csv)} output sets")


if __name__ == "__main__":
    main()
