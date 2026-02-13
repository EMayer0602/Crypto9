#!/usr/bin/env python3
"""
Phase Baseline: Run paper_trader simulation per indicator, tag with 4 phase classifiers.

Uses paper_trader.run_simulation() directly to get IDENTICAL trades as the BCK.
Then tags each trade with 4 different phase classifiers.
Uses paper_trader.build_summary_payload() + write_summary_json() for output
(same functions that generate the BCK), ensuring identical format and PnL.

Generates per (trading indicator, phase classifier):
  - trading_summary_ph1_{indicator}_{classifier}.json
  - trading_summary_ph1_{indicator}_{classifier}.html

Usage:
    python test_baseline_phases.py
    python test_baseline_phases.py --sim-start 2024-01-31
"""
import argparse
import os
from collections import Counter

import pandas as pd

import Supertrend_5Min as st
import paper_trader as pt

# Phase classification: FIXED params (independent of trading params)
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0
PHASE_HTF = "6h"

TRADING_INDICATORS = ["jma"]  # TODO: add supertrend, htf_crossover, kama after sweep

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


def main():
    parser = argparse.ArgumentParser(description="Phase baseline using paper_trader simulation")
    parser.add_argument("--sim-start", type=str, default="2024-01-31",
                        help="Simulation start date (default: 2024-01-31)")
    args = parser.parse_args()

    sim_start = pd.Timestamp(args.sim_start, tz=st.BERLIN_TZ)
    sim_end = pd.Timestamp.now(tz=st.BERLIN_TZ)
    symbols = st.SYMBOLS

    print(f"=== Phase Baseline (via paper_trader) ===")
    print(f"Indicators: {TRADING_INDICATORS}")
    print(f"MAX_OPEN_POSITIONS: {pt.MAX_OPEN_POSITIONS} | STAKE_DIVISOR: {pt.STAKE_DIVISOR}")
    print(f"Simulation: {args.sim_start} to now")

    # Save original globals
    orig_htf_length = st.HTF_LENGTH
    orig_htf_factor = st.HTF_FACTOR

    for indicator in TRADING_INDICATORS:
        print(f"\n{'='*60}")
        print(f"Trading indicator: {indicator}")
        print(f"{'='*60}")

        # ── Step 1: Run simulation (same engine as BCK) ──
        trade_results, sim_state = pt.run_simulation(
            start_ts=sim_start,
            end_ts=sim_end,
            use_saved_state=False,
            allowed_indicators=[indicator],
            use_testnet=False,
            reset_state=True,
        )

        # Convert to DataFrame using paper_trader's own function
        trades_df = pt.trades_to_dataframe(trade_results)

        # Open positions from sim_state
        open_positions = sim_state.get("positions", [])
        open_df = pt.open_positions_to_dataframe(open_positions)

        n_closed = len(trades_df)
        n_open = len(open_df)
        print(f"  Closed: {n_closed} | Open: {n_open}")
        if not trades_df.empty:
            sym_counts = Counter(trades_df["symbol"].values)
            pnl_sum = float(trades_df["pnl"].sum())
            print(f"  PnL: {pnl_sum:+,.2f} | Symbols: {len(sym_counts)}")

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

            # Tag trades with phase
            tagged_df = trades_df.copy()
            phase_tags = []
            for _, row in tagged_df.iterrows():
                phases_s = phase_cache.get(row["symbol"], pd.Series(dtype=str))
                entry_ts = pd.Timestamp(row["entry_time"])
                phase = get_phase_at_entry(phases_s, entry_ts)
                phase_tags.append(phase)
            tagged_df["phase"] = phase_tags

            phase_dist = Counter(phase_tags)
            print(f"    Distribution: {dict(phase_dist)}")

            # Build summary using paper_trader's own function (same as BCK)
            summary = pt.build_summary_payload(
                tagged_df, open_df, sim_state, sim_start, sim_end
            )

            # Recalculate summary stats from compound-recalculated trade PnLs
            # (build_summary_payload recalculates stake/pnl in the export,
            #  but uses original simulation PnLs for summary stats - fix that)
            exported_trades = summary.get("trades", [])
            if exported_trades:
                compound_pnl = sum(float(t.get("pnl", 0) or 0) for t in exported_trades)
                compound_winners = sum(1 for t in exported_trades if float(t.get("pnl", 0) or 0) > 0)
                compound_losers = sum(1 for t in exported_trades if float(t.get("pnl", 0) or 0) < 0)
                n_trades = len(exported_trades)
                compound_wr = (compound_winners / n_trades * 100) if n_trades else 0
                open_eq = summary.get("open_equity", 0)
                summary["closed_pnl"] = round(compound_pnl, 6)
                summary["avg_trade_pnl"] = round(compound_pnl / n_trades, 6) if n_trades else 0
                summary["win_rate_pct"] = round(compound_wr, 4)
                summary["winners"] = compound_winners
                summary["losers"] = compound_losers
                summary["final_capital"] = round(st.START_EQUITY + compound_pnl + open_eq, 6)
                # Long stats (all trades are long)
                summary["long_pnl"] = summary["closed_pnl"]
                summary["long_avg_pnl"] = summary["avg_trade_pnl"]
                summary["long_win_rate"] = summary["win_rate_pct"]
                summary["long_winners"] = summary["winners"]
                summary["long_losers"] = summary["losers"]

            # Write JSON + HTML using paper_trader's own function (same as BCK)
            prefix = f"ph1_{indicator}_{classifier}"
            json_path = os.path.join(st.BASE_OUT_DIR, f"trading_summary_{prefix}.json")
            pt.write_summary_json(summary, json_path)

            print(f"    -> {json_path} ({summary['closed_trades']} closed, PnL: {summary['closed_pnl']:+,.2f})")

    # Restore globals
    st.HTF_LENGTH = orig_htf_length
    st.HTF_FACTOR = orig_htf_factor

    print(f"\n{'='*60}")
    print(f"Done! {len(TRADING_INDICATORS)} indicators × 4 classifiers = {len(TRADING_INDICATORS) * 4} output sets")


if __name__ == "__main__":
    main()
