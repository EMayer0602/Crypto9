#!/usr/bin/env python3
"""
Phase Baseline: Run paper_trader simulation per indicator, tag with 4 phase classifiers.

Uses paper_trader.run_simulation() directly to get IDENTICAL trades as the BCK.
Then tags each trade with 4 different phase classifiers.

Generates per (trading indicator, phase classifier):
  - trading_summary_ph1_{indicator}_{classifier}.json
  - dashboard_ph1_{indicator}_{classifier}.html
  - trading_summary_ph1_{indicator}_{classifier}.html

Usage:
    python test_baseline_phases.py
    python test_baseline_phases.py --start 2025-12-01 --summary-start 2024-12-01
"""
import argparse
import json
import os
from collections import Counter

import pandas as pd

import Supertrend_5Min as st
import paper_trader as pt

# ── CONFIG ──
START_CAPITAL = 16500.0
MAX_POSITIONS = 8
STAKE_DIVISOR = 10

# Phase classification: FIXED params (independent of trading params)
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0
PHASE_HTF = "6h"

TRADING_INDICATORS = ["jma"]  # TODO: add supertrend, htf_crossover, kama after sweep

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


def trades_to_dicts(trade_results):
    """Convert paper_trader TradeResult objects to plain dicts."""
    trades = []
    for t in trade_results:
        trades.append({
            "symbol": t.symbol,
            "direction": t.direction,
            "indicator": INDICATOR_DISPLAY.get(t.indicator, t.indicator),
            "htf": t.htf,
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "reason": t.reason,
        })
    return trades


def main():
    parser = argparse.ArgumentParser(description="Phase baseline using paper_trader simulation")
    parser.add_argument("--start", type=str, default="2025-12-01",
                        help="Dashboard start date (default: 2025-12-01)")
    parser.add_argument("--summary-start", type=str, default="2024-12-01",
                        help="Trading summary start date (default: 2024-12-01)")
    parser.add_argument("--sim-start", type=str, default="2024-01-31",
                        help="Simulation start date (default: 2024-01-31)")
    args = parser.parse_args()

    dashboard_start = args.start
    summary_start = args.summary_start
    sim_start = pd.Timestamp(args.sim_start, tz=st.BERLIN_TZ)
    sim_end = pd.Timestamp.now(tz=st.BERLIN_TZ)

    symbols = st.SYMBOLS

    print(f"=== Phase Baseline (via paper_trader.run_simulation) ===")
    print(f"Indicators: {TRADING_INDICATORS}")
    print(f"Capital: {START_CAPITAL:,.0f} | Max Positions: {MAX_POSITIONS} | Stake Divisor: {STAKE_DIVISOR}")
    print(f"Simulation: {args.sim_start} to now")
    print(f"Dashboard ab {dashboard_start} | Summary ab {summary_start}")

    # Save original globals
    orig_htf_length = st.HTF_LENGTH
    orig_htf_factor = st.HTF_FACTOR

    for indicator in TRADING_INDICATORS:
        ind_display = INDICATOR_DISPLAY[indicator]

        print(f"\n{'='*60}")
        print(f"Trading indicator: {ind_display}")
        print(f"{'='*60}")

        # ── Step 1: Run simulation with this indicator only ──
        trade_results, sim_state = pt.run_simulation(
            start_ts=sim_start,
            end_ts=sim_end,
            use_saved_state=False,
            allowed_indicators=[indicator],
            use_testnet=False,
            reset_state=True,
        )

        accepted_trades = trades_to_dicts(trade_results)
        accepted_trades.sort(key=lambda t: t["entry_time"])

        closed = [t for t in accepted_trades if t.get("reason") != "Final bar"]
        opened = [t for t in accepted_trades if t.get("reason") == "Final bar"]
        sym_counts = Counter(t["symbol"] for t in accepted_trades)

        # Compound growth
        capital = START_CAPITAL
        for t in accepted_trades:
            stake = capital / STAKE_DIVISOR
            ep, xp = t["entry_price"], t["exit_price"]
            pnl_pct = (xp - ep) / ep if ep else 0
            pnl = pnl_pct * stake - stake * st.FEE_RATE * 2
            capital += pnl

        print(f"  {ind_display}: {len(closed)} closed + {len(opened)} open = {len(accepted_trades)}")
        print(f"  Compound: {START_CAPITAL:,.2f} -> {capital:,.2f} | PnL: {capital - START_CAPITAL:+,.2f}")
        print(f"  Symbols: {len(sym_counts)} | {dict(sym_counts.most_common())}")

        # ── Step 2: Tag with 4 phase classifiers ──
        for classifier in sorted(CLASSIFIER_DISPLAY.keys()):
            cls_display = CLASSIFIER_DISPLAY[classifier]

            print(f"\n  Phase: {cls_display}")

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
            for t in accepted_trades:
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
                "stake_divisor": STAKE_DIVISOR,
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
                if t.get("reason") == "Final bar":
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
                stake_divisor=STAKE_DIVISOR,
                indicator_label=label,
                output_prefix=prefix,
            )

    # Restore globals
    st.HTF_LENGTH = orig_htf_length
    st.HTF_FACTOR = orig_htf_factor

    print(f"\n{'='*60}")
    print(f"Done! {len(TRADING_INDICATORS)} indicators × 4 classifiers = {len(TRADING_INDICATORS) * 4} output sets")


if __name__ == "__main__":
    main()
