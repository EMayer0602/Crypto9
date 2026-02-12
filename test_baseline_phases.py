#!/usr/bin/env python3
"""
Baseline Phase Test: Run actual backtests per indicator, classify phases,
generate per-indicator JSON and HTML dashboards.

For each indicator (supertrend, htf_crossover, jma, kama):
1. Classify phases: HTF_LENGTH=10, HTF_FACTOR=3.0, HTF=6h (per indicator type)
2. Backtest with correct params: HTF_LENGTH=20, HTF_FACTOR=3.0, HTF=6h
   - supertrend: per-symbol optimized params from best_params_overall_bck.csv
   - htf_crossover / jma / kama: default params from INDICATOR_PRESETS
3. Tag trades with phases
4. Write per-indicator JSON (raw trades, no PnL/stake)
5. Generate HTML via _generate_phase_dashboard() (compound growth from 16500)

Generates per indicator:
  - trading_summary_ph1_{indicator}.json  (raw trades with phases)
  - dashboard_ph1_{indicator}.html        (ab --start, default 2025-12-01)
  - trading_summary_ph1_{indicator}.html  (ab --summary-start, default 2024-12-01)

Compound growth: Summary AND Dashboard both start independently at 16500.

Usage:
    python test_baseline_phases.py
    python test_baseline_phases.py --start 2025-12-01 --summary-start 2024-12-01
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

# Backtest HTF filter params (same as classic)
BACKTEST_HTF_LENGTH = 20
BACKTEST_HTF_FACTOR = 3.0
BACKTEST_HTF = "6h"

# Classic per-symbol params from best_params_overall_bck.csv (supertrend only)
# Symbols NOT listed here use default supertrend params (10, 3.0)
CLASSIC_PARAMS = {
    "BTC/USDC":  {"param_a": 7,  "param_b": 4.0, "atr_stop_mult": 1.0, "min_hold_bars": 12},
    "ETH/USDC":  {"param_a": 10, "param_b": 3.0, "atr_stop_mult": None, "min_hold_bars": 24},
    "SOL/USDC":  {"param_a": 14, "param_b": 4.0, "atr_stop_mult": 1.5,  "min_hold_bars": 0},
    "XRP/USDC":  {"param_a": 14, "param_b": 3.0, "atr_stop_mult": None, "min_hold_bars": 24},
    "LINK/USDC": {"param_a": 10, "param_b": 3.0, "atr_stop_mult": None, "min_hold_bars": 24},
    "SUI/USDC":  {"param_a": 10, "param_b": 4.0, "atr_stop_mult": 2.0,  "min_hold_bars": 0},
    "ADA/USDC":  {"param_a": 7,  "param_b": 4.0, "atr_stop_mult": None, "min_hold_bars": 24},
    "ICP/USDC":  {"param_a": 14, "param_b": 2.0, "atr_stop_mult": 2.0,  "min_hold_bars": 12},
    "TNSR/USDC": {"param_a": 7,  "param_b": 4.0, "atr_stop_mult": 2.0,  "min_hold_bars": 0},
}

INDICATORS = ["supertrend", "htf_crossover", "jma", "kama"]

INDICATOR_DISPLAY = {
    "supertrend": "Supertrend",
    "htf_crossover": "HTF Crossover",
    "jma": "JMA",
    "kama": "KAMA",
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
    parser = argparse.ArgumentParser(description="Generate per-indicator phase dashboards via backtests")
    parser.add_argument("--start", type=str, default="2025-12-01",
                        help="Dashboard start date (default: 2025-12-01)")
    parser.add_argument("--summary-start", type=str, default="2024-12-01",
                        help="Trading summary start date (default: 2024-12-01)")
    args = parser.parse_args()

    dashboard_start = args.start
    summary_start = args.summary_start

    print(f"=== Baseline Phase Test (Backtests) ===")
    print(f"Capital: {START_CAPITAL:,.0f} | Max Positions: {MAX_POSITIONS}")
    print(f"Phase classification: L={PHASE_HTF_LENGTH}, F={PHASE_HTF_FACTOR}, HTF={PHASE_HTF}")
    print(f"Backtest HTF filter: L={BACKTEST_HTF_LENGTH}, F={BACKTEST_HTF_FACTOR}, HTF={BACKTEST_HTF}")
    print(f"Dashboard ab {dashboard_start} | Summary ab {summary_start}")

    symbols = st.SYMBOLS

    # Populate OHLCV cache once for all symbols
    st.ensure_cache_populated(symbols, st.TIMEFRAME, st.LOOKBACK)

    # Save original globals to restore later
    orig_backtest_start = st.BACKTEST_START_DATE
    orig_htf_length = st.HTF_LENGTH
    orig_htf_factor = st.HTF_FACTOR

    # Disable BACKTEST_START_DATE filter — we want ALL data, filtering in HTML later
    st.BACKTEST_START_DATE = ""

    for indicator in INDICATORS:
        ind_display = INDICATOR_DISPLAY[indicator]
        print(f"\n{'='*60}")
        print(f"Processing {ind_display}...")

        # ── Step 1: Phase classification ──
        # Set indicator type + phase HTF params
        st.apply_indicator_type(indicator)
        st.apply_higher_timeframe(PHASE_HTF)
        st.HTF_LENGTH = PHASE_HTF_LENGTH
        st.HTF_FACTOR = PHASE_HTF_FACTOR
        st.clear_data_cache()

        phase_cache = {}
        for symbol in symbols:
            sym_short = symbol.replace("/USDC", "").replace("/USDT", "")
            # fetch_data gives us a df with the right index for phase alignment
            df_for_phases = st.fetch_data(symbol, st.TIMEFRAME, st.LOOKBACK)
            if df_for_phases.empty:
                print(f"  Phase {sym_short}: NO DATA")
                phase_cache[symbol] = pd.Series(dtype=str)
                continue

            phases = st.classify_market_phases(df_for_phases, symbol)
            phase_counts = phases.value_counts().to_dict()
            print(f"  Phase {sym_short}: {phase_counts}")
            phase_cache[symbol] = phases

        # ── Step 2: Backtest with correct params ──
        # Set backtest HTF params (same as classic: L=20, F=3.0)
        st.HTF_LENGTH = BACKTEST_HTF_LENGTH
        st.HTF_FACTOR = BACKTEST_HTF_FACTOR
        st.apply_higher_timeframe(BACKTEST_HTF)
        st.clear_data_cache()

        all_trades = []

        for symbol in symbols:
            sym_short = symbol.replace("/USDC", "").replace("/USDT", "")

            # Get indicator/backtest params
            if indicator == "supertrend" and symbol in CLASSIC_PARAMS:
                p = CLASSIC_PARAMS[symbol]
                param_a = p["param_a"]
                param_b = p["param_b"]
                atr_stop_mult = p["atr_stop_mult"]
                min_hold_bars = p["min_hold_bars"]
            else:
                preset = st.INDICATOR_PRESETS[indicator]
                param_a = preset["default_a"]
                param_b = preset["default_b"]
                atr_stop_mult = None
                min_hold_bars = 0

            max_hold_bars = 0

            # Prepare dataframe: fetch data + attach HTF trend + filters
            df = st.prepare_symbol_dataframe(symbol)
            if df.empty:
                print(f"  Backtest {sym_short}: NO DATA")
                continue

            # Compute main indicator (supertrend/jma/kama on main TF)
            df_ind = st.compute_indicator(df, param_a, param_b)

            # Run backtest (htf_crossover uses different entry/exit logic)
            if indicator == "htf_crossover":
                trades_df = st.backtest_htf_crossover(
                    df_ind, atr_stop_mult=atr_stop_mult, direction="long",
                    min_hold_bars=min_hold_bars, max_hold_bars=max_hold_bars,
                )
            else:
                trades_df = st.backtest_supertrend(
                    df_ind, atr_stop_mult=atr_stop_mult, direction="long",
                    min_hold_bars=min_hold_bars, max_hold_bars=max_hold_bars,
                )

            if trades_df.empty:
                print(f"  Backtest {sym_short}: 0 trades")
                continue

            # Tag each trade with its phase and convert to dict
            phases = phase_cache.get(symbol, pd.Series(dtype=str))
            symbol_trade_count = 0

            for _, row in trades_df.iterrows():
                entry_ts = row["Zeit"]
                exit_ts = row["ExitZeit"]
                phase = get_phase_at_entry(phases, entry_ts)

                trade_dict = {
                    "symbol": symbol,
                    "direction": "long",
                    "indicator": ind_display,
                    "htf": BACKTEST_HTF,
                    "entry_time": str(entry_ts),
                    "exit_time": str(exit_ts),
                    "entry_price": float(row["Entry"]),
                    "exit_price": float(row["ExitPreis"]),
                    "phase": phase,
                    "reason": row["ExitReason"],
                }
                all_trades.append(trade_dict)
                symbol_trade_count += 1

            atr_label = "None" if atr_stop_mult is None else atr_stop_mult
            print(f"  Backtest {sym_short}: {symbol_trade_count} trades "
                  f"(A={param_a}, B={param_b}, ATR={atr_label}, MinHold={min_hold_bars})")

        # Sort all trades by entry time
        all_trades.sort(key=lambda t: t["entry_time"])

        # Console summary: compound growth over all trades
        capital = START_CAPITAL
        for t in all_trades:
            stake = capital / MAX_POSITIONS
            ep = t["entry_price"]
            xp = t["exit_price"]
            pnl_pct = (xp - ep) / ep if ep else 0
            pnl_gross = pnl_pct * stake
            fees = stake * st.FEE_RATE * 2.0
            pnl_net = pnl_gross - fees
            capital += pnl_net

        total_pnl = capital - START_CAPITAL
        closed = [t for t in all_trades if t["reason"] != "Final bar"]
        opened = [t for t in all_trades if t["reason"] == "Final bar"]
        print(f"\n  {ind_display}: {len(closed)} closed + {len(opened)} open = {len(all_trades)} total")
        print(f"  Compound Growth: {START_CAPITAL:,.2f} -> {capital:,.2f} | PnL: {total_pnl:+,.2f}")

        # ── Step 3: Write JSON (raw trades, no PnL/stake) ──
        json_out = {
            "indicator": indicator,
            "indicator_display": ind_display,
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

        for t in all_trades:
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

        json_path = os.path.join(st.BASE_OUT_DIR, f"trading_summary_ph1_{indicator}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_out, jf, indent=2, default=str)
        print(f"  -> {json_path} ({len(json_out['trades'])} closed, {len(json_out['open_positions_data'])} open)")

        # ── Step 4: Generate HTML (dashboard + summary) ──
        prefix = f"ph1_{indicator}"
        st._generate_phase_dashboard(
            all_trades,
            dashboard_start=dashboard_start,
            summary_start=summary_start,
            stake_divisor=MAX_POSITIONS,
            indicator_label=ind_display,
            output_prefix=prefix,
        )

    # Restore original globals
    st.BACKTEST_START_DATE = orig_backtest_start
    st.HTF_LENGTH = orig_htf_length
    st.HTF_FACTOR = orig_htf_factor

    print(f"\n{'='*60}")
    print("Done! Generated files:")
    for indicator in INDICATORS:
        ind_display = INDICATOR_DISPLAY[indicator]
        print(f"  {ind_display}:")
        print(f"    - trading_summary_ph1_{indicator}.json")
        print(f"    - trading_summary_ph1_{indicator}.html  (ab {summary_start}, start=16.500)")
        print(f"    - dashboard_ph1_{indicator}.html        (ab {dashboard_start}, start=16.500)")


if __name__ == "__main__":
    main()
