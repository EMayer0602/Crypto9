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
    length_str = p.get("Length", "").replace(",", ".").strip()
    length = int(float(length_str)) if length_str else int(float(param_a.replace(",", ".")))
    factor_str = p.get("Factor", "").replace(",", ".").strip()
    factor = float(factor_str) if factor_str else float(param_b)
    atr_mult_str = p.get("ATRStopMultValue", "").replace(",", ".").strip()
    atr_mult = float(atr_mult_str) if atr_mult_str else None
    min_hold_str = str(p.get("MinHoldBars", "0")).replace(",", ".").strip()
    min_hold = int(float(min_hold_str)) if min_hold_str else 0
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

        ep = float(trade["Entry"])
        xp = float(trade["ExitPreis"])
        if direction == "long":
            pnl_pct = (xp - ep) / ep if ep else 0
        else:
            pnl_pct = (ep - xp) / ep if ep else 0

        all_trades.append({
            "symbol": symbol,
            "indicator": st.INDICATOR_DISPLAY_NAME,
            "indicator_key": indicator,
            "htf": htf,
            "direction": direction,
            "phase": trade_phase,
            "entry_time": str(entry_time),
            "exit_time": str(trade["ExitZeit"]),
            "entry_price": ep,
            "exit_price": xp,
            "reason": str(trade.get("ExitReason", "")),
            # Internal backtest PnL for verification against classic dashboard
            "bt_pnl": float(trade["PnL (USD)"]),
            "bt_stake": float(trade["Stake"]),
            "bt_equity": float(trade["Equity"]),
        })

all_trades.sort(key=lambda t: t["entry_time"])

# Group by indicator
from collections import defaultdict
trades_by_indicator = defaultdict(list)
for t in all_trades:
    trades_by_indicator[t["indicator_key"]].append(t)

# Process each indicator separately
for ind_key, ind_trades in trades_by_indicator.items():
    ind_trades.sort(key=lambda t: t["entry_time"])
    total = len(ind_trades)
    ind_display = ind_trades[0]["indicator"] if ind_trades else ind_key

    # ── VERIFICATION: per-symbol backtest PnL (must match classic dashboard) ──
    print(f"\n{'='*60}")
    print(f"{ind_display}: {total} trades")
    print(f"  VERIFICATION (per-symbol, wie klassisches Dashboard):")
    symbols_seen = {}
    for t in ind_trades:
        sym = t["symbol"]
        if sym not in symbols_seen:
            symbols_seen[sym] = {"count": 0, "bt_pnl": 0, "last_eq": 0}
        symbols_seen[sym]["count"] += 1
        symbols_seen[sym]["bt_pnl"] += t["bt_pnl"]
        symbols_seen[sym]["last_eq"] = t["bt_equity"]
    classic_total_pnl = 0
    for sym, info in symbols_seen.items():
        sym_short = sym.replace("/USDC", "").replace("/USDT", "")
        classic_total_pnl += info["bt_pnl"]
        print(f"    {sym_short:6s}: {info['count']:3d}t  PnL={info['bt_pnl']:+10,.2f}  FinalEq={info['last_eq']:,.2f}")
    bt_winners = sum(1 for t in ind_trades if t["bt_pnl"] > 0)
    print(f"  SUMME: {total}t  PnL={classic_total_pnl:+,.2f}  WR={bt_winners/total*100:.1f}%")

    # ── SHARED EQUITY: one equity across all symbols (preferred model) ──
    capital = st.START_EQUITY
    for t in ind_trades:
        stake = capital / st.STAKE_DIVISOR
        ep, xp = t["entry_price"], t["exit_price"]
        pnl_pct = (xp - ep) / ep if t["direction"] == "long" else (ep - xp) / ep
        pnl_net = pnl_pct * stake - stake * st.FEE_RATE * 2.0
        capital += pnl_net
        t["stake"] = stake
        t["pnl"] = pnl_net
        t["pnl_pct"] = pnl_pct
        t["fees"] = stake * st.FEE_RATE * 2.0
        t["equity_after"] = capital

    total_pnl = sum(t["pnl"] for t in ind_trades)
    winners = sum(1 for t in ind_trades if t["pnl"] > 0)
    print(f"  SHARED EQUITY (equity/{st.STAKE_DIVISOR}):")
    print(f"    {total}t  PnL={total_pnl:+,.2f}  WR={winners/total*100:.1f}%")
    print(f"    Start: {st.START_EQUITY:,.2f} → Final: {capital:,.2f}")

    for phase in ["Up", "Down", "Flat"]:
        pt = [t for t in ind_trades if t["phase"] == phase]
        if pt:
            pp = sum(t["pnl"] for t in pt)
            pw = sum(1 for t in pt if t["pnl"] > 0)
            print(f"      {phase:5s}: {len(pt):4d}t  PnL={pp:+10,.2f}  WR={pw/len(pt)*100:.1f}%")

    # Generate dashboard (ab 2025-12-01) + summary (ab 2024-01-31) with shared equity
    st._generate_phase_dashboard(ind_trades, dashboard_start="2025-12-01",
                                  summary_start="2024-01-31",
                                  stake_divisor=st.STAKE_DIVISOR, use_precomputed=True)

    # Rename to indicator-specific filenames
    base = st.BASE_OUT_DIR
    for src, dst in [
        ("dashboard_ph1.html", f"dashboard_ph1_{ind_key}.html"),
        ("trading_summary_ph1.html", f"trading_summary_ph1_{ind_key}.html"),
    ]:
        src_path = os.path.join(base, src)
        dst_path = os.path.join(base, dst)
        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.rename(src_path, dst_path)
            print(f"  Generated: {dst}")

# Also generate combined report (all indicators) – shared equity
print(f"\n{'='*60}")
print(f"COMBINED (all indicators) – SHARED EQUITY")
capital = st.START_EQUITY
for t in all_trades:
    stake = capital / st.STAKE_DIVISOR
    ep, xp = t["entry_price"], t["exit_price"]
    pnl_pct = (xp - ep) / ep if t["direction"] == "long" else (ep - xp) / ep
    pnl_net = pnl_pct * stake - stake * st.FEE_RATE * 2.0
    capital += pnl_net
    t["stake"] = stake
    t["pnl"] = pnl_net
    t["pnl_pct"] = pnl_pct
    t["fees"] = stake * st.FEE_RATE * 2.0
    t["equity_after"] = capital

total_pnl = sum(t["pnl"] for t in all_trades)
winners = sum(1 for t in all_trades if t["pnl"] > 0)
print(f"Total: {len(all_trades)} trades | PnL: {total_pnl:+,.2f} | WR: {winners/len(all_trades)*100:.1f}%")
print(f"Start: {st.START_EQUITY:,.2f} → Final: {capital:,.2f}")
st._generate_phase_dashboard(all_trades, dashboard_start="2025-12-01",
                              summary_start="2024-01-31",
                              stake_divisor=st.STAKE_DIVISOR, use_precomputed=True)

print("\nDone!")
