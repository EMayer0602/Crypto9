#!/usr/bin/env python3
"""
Baseline Phase Test: Run old strategy params (best_params_overall_bck.csv)
through the NEW phase classification (st_trend for supertrend).
Generates: trading_summary_ph1.html + dashboard_ph1.html

Usage:
    python test_baseline_phases.py
"""
import csv
import os
from datetime import datetime

import pandas as pd
import numpy as np

import Supertrend_5Min as st

# Phase classification uses FIXED params (independent of trading params)
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0

CSV_PATH = "report_html/best_params_overall_bck.csv"
print(f"=== Baseline Phase Test ===")
print(f"Phase classification: st_trend L={PHASE_HTF_LENGTH}, F={PHASE_HTF_FACTOR}")
print(f"Reading old params from {CSV_PATH}")

params_list = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        if not row.get("Symbol"):
            continue
        params_list.append(row)

print(f"Found {len(params_list)} symbol configs")

st.ensure_cache_populated(st.SYMBOLS, st.TIMEFRAME, st.LOOKBACK)

all_trades = []

for p in params_list:
    symbol = p["Symbol"]
    indicator = p["Indicator"]
    param_a = p["ParamA"]
    param_b = p["ParamB"].replace(",", ".")
    htf = p.get("HTF", "6h")
    length = int(p.get("Length", param_a))
    factor = float(p.get("Factor", param_b).replace(",", "."))
    atr_mult_str = p.get("ATRStopMultValue", "").replace(",", ".")
    atr_mult = float(atr_mult_str) if atr_mult_str else None
    min_hold = int(p.get("MinHoldBars", 0))
    direction = p.get("Direction", "Long").lower()

    st.apply_indicator_type(indicator)
    st.apply_higher_timeframe(htf)

    # Phase classification with FIXED params
    st.HTF_LENGTH = PHASE_HTF_LENGTH
    st.HTF_FACTOR = PHASE_HTF_FACTOR

    sym_short = symbol.replace("/USDC", "").replace("/USDT", "")
    print(f"\n--- {sym_short}: {indicator} L={length} F={factor} ATR={atr_mult} MinH={min_hold} ---")

    df_raw = st.prepare_symbol_dataframe(symbol)
    if df_raw.empty:
        print(f"  NO DATA for {symbol}")
        continue

    phase_labels = st.classify_market_phases(df_raw, symbol)
    phase_counts = phase_labels.value_counts().to_dict()
    print(f"  Phases: {phase_counts}")

    # Restore trading params for backtest
    st.HTF_LENGTH = length
    st.HTF_FACTOR = factor
    st.clear_data_cache()
    df_raw = st.prepare_symbol_dataframe(symbol)
    if df_raw.empty:
        continue

    df_ind = st.compute_indicator(df_raw, int(param_a), float(param_b))
    for col in ("htf_trend", "htf_indicator", "momentum"):
        if col in df_raw.columns:
            df_ind[col] = df_raw[col]

    if indicator == "htf_crossover":
        trades = st.backtest_htf_crossover(df_ind, atr_stop_mult=atr_mult,
            direction=direction, min_hold_bars=min_hold)
    else:
        trades = st.backtest_supertrend(df_ind, atr_stop_mult=atr_mult,
            direction=direction, min_hold_bars=min_hold)

    if trades.empty:
        print(f"  No trades")
        continue

    print(f"  Trades: {len(trades)}")

    for _, trade in trades.iterrows():
        entry_time = trade["Zeit"]
        if entry_time in phase_labels.index:
            trade_phase = phase_labels.loc[entry_time]
        else:
            idx = phase_labels.index.get_indexer([entry_time], method="ffill")[0]
            trade_phase = phase_labels.iloc[idx] if idx >= 0 else "Flat"

        all_trades.append({
            "symbol": symbol,
            "indicator": st.INDICATOR_DISPLAY_NAME,
            "indicator_key": indicator,
            "htf": htf,
            "direction": direction,
            "phase": trade_phase,
            "entry_time": str(entry_time),
            "exit_time": str(trade["ExitZeit"]),
            "entry_price": float(trade["Entry"]),
            "exit_price": float(trade["ExitPreis"]),
            "reason": str(trade.get("ExitReason", "")),
        })

all_trades.sort(key=lambda t: t["entry_time"])

# Compound growth
capital = st.START_EQUITY
for t in all_trades:
    stake = capital / st.PHASE_STAKE_DIVISOR
    ep, xp = t["entry_price"], t["exit_price"]
    pnl_pct = (xp - ep) / ep if t["direction"] == "long" else (ep - xp) / ep
    pnl_net = pnl_pct * stake - stake * st.FEE_RATE * 2.0
    capital += pnl_net
    t["stake"] = stake
    t["pnl"] = pnl_net
    t["pnl_pct"] = pnl_pct
    t["fees"] = stake * st.FEE_RATE * 2.0
    t["equity_after"] = capital

# Print summary
total_pnl = sum(t["pnl"] for t in all_trades)
winners = sum(1 for t in all_trades if t["pnl"] > 0)
print(f"\n{'='*60}")
print(f"Total: {len(all_trades)} trades | PnL: {total_pnl:+,.2f} | WR: {winners/len(all_trades)*100:.1f}%")
print(f"Start: {st.START_EQUITY:,.2f} → Final: {capital:,.2f}")

for phase in ["Up", "Down", "Flat"]:
    pt = [t for t in all_trades if t["phase"] == phase]
    if pt:
        pp = sum(t["pnl"] for t in pt)
        pw = sum(1 for t in pt if t["pnl"] > 0)
        print(f"  {phase:5s}: {len(pt):4d}t  PnL={pp:+10,.2f}  WR={pw/len(pt)*100:.1f}%")

# Generate HTML
st._generate_phase_dashboard(all_trades, dashboard_start="2024-01-31")

# Rename to _ph1
base = st.BASE_OUT_DIR
for old, new in [("dashboard_Ph.html", "dashboard_ph1.html"),
                 ("trading_summary_Ph.html", "trading_summary_ph1.html")]:
    old_path, new_path = os.path.join(base, old), os.path.join(base, new)
    if os.path.exists(old_path):
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(old_path, new_path)
        print(f"  Generated: {new_path}")

print("\nDone!")
