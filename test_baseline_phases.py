#!/usr/bin/env python3
"""
Baseline Phase Test: Read trades from trading_summary_bck.json (same source
as classic dashboard), apply phase classification, recalculate with
shared equity (START=16500, MAX_POSITIONS=8).

Generates per indicator:
  - dashboard_ph1_{indicator}.html  (ab 2025-12-01)
  - trading_summary_ph1_{indicator}.html (ab 2024-01-31)
Plus combined:
  - dashboard_ph1.html / trading_summary_ph1.html

Usage:
    python test_baseline_phases.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd

import Supertrend_5Min as st

# ── CONFIG ──
START_CAPITAL = 16500.0
MAX_POSITIONS = 8
JSON_PATH = os.path.join("report_html", "trading_summary_bck.json")

# Phase classification uses FIXED params (independent of trading params)
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0
PHASE_HTF = "6h"

print(f"=== Baseline Phase Test (from JSON) ===")
print(f"Source: {JSON_PATH}")
print(f"Capital: {START_CAPITAL:,.0f} | Max Positions: {MAX_POSITIONS}")
print(f"Phase classification: L={PHASE_HTF_LENGTH}, F={PHASE_HTF_FACTOR}, HTF={PHASE_HTF}")

# ── 1. READ TRADES FROM JSON ──
if not os.path.exists(JSON_PATH):
    print(f"ERROR: {JSON_PATH} not found!")
    print("This file is on your local PC. Copy it to report_html/ first.")
    exit(1)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

raw_trades = data.get("trades", [])
raw_open = data.get("open_positions_data", [])
print(f"JSON: {len(raw_trades)} closed trades, {len(raw_open)} open positions")

# Filter to Long only (shorts never enabled)
trades_list = []
for t in raw_trades:
    direction = str(t.get("direction", "long")).lower()
    if direction != "long":
        continue
    entry_price = float(t.get("entry_price", 0) or 0)
    exit_price = float(t.get("exit_price", 0) or 0)
    if entry_price <= 0:
        continue
    pnl_pct = (exit_price - entry_price) / entry_price
    trades_list.append({
        "symbol": t.get("symbol", ""),
        "indicator": t.get("indicator", ""),
        "htf": t.get("htf", ""),
        "entry_time": t.get("entry_time", ""),
        "exit_time": t.get("exit_time", ""),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "reason": t.get("reason", ""),
        "direction": "long",
    })

# Open positions
open_list = []
for p in raw_open:
    direction = str(p.get("direction", "long")).lower()
    if direction != "long":
        continue
    entry_price = float(p.get("entry_price", 0) or 0)
    last_price = float(p.get("last_price", 0) or 0)
    if entry_price <= 0:
        continue
    pnl_pct = (last_price - entry_price) / entry_price
    open_list.append({
        "symbol": p.get("symbol", ""),
        "indicator": p.get("indicator", ""),
        "htf": p.get("htf", ""),
        "entry_time": p.get("entry_time", ""),
        "exit_time": "",
        "entry_price": entry_price,
        "exit_price": last_price,
        "pnl_pct": pnl_pct,
        "reason": "Final bar",
        "direction": "long",
    })

print(f"Long trades: {len(trades_list)} closed + {len(open_list)} open")

# ── 2. BUILD PHASE LABELS PER SYMBOL/INDICATOR ──
# Group trades by (symbol, indicator) to know which combos we need
symbol_indicator_pairs = set()
for t in trades_list + open_list:
    symbol = t["symbol"]
    indicator = t["indicator"]
    if symbol and indicator:
        symbol_indicator_pairs.add((symbol, indicator))

print(f"\nBuilding phase labels for {len(symbol_indicator_pairs)} symbol/indicator pairs...")

# Populate cache once
symbols_needed = list(set(s for s, _ in symbol_indicator_pairs))
st.ensure_cache_populated(symbols_needed, st.TIMEFRAME, st.LOOKBACK)

# For each (symbol, indicator), compute phase labels
phase_cache = {}  # (symbol, indicator) -> pd.Series of phase labels
for symbol, indicator in sorted(symbol_indicator_pairs):
    sym_short = symbol.replace("/USDC", "").replace("/USDT", "")

    # Set indicator type for phase classification
    st.apply_indicator_type(indicator)
    st.apply_higher_timeframe(PHASE_HTF)
    st.HTF_LENGTH = PHASE_HTF_LENGTH
    st.HTF_FACTOR = PHASE_HTF_FACTOR
    st.clear_data_cache()

    df_raw = st.prepare_symbol_dataframe(symbol)
    if df_raw.empty:
        print(f"  {sym_short}/{indicator}: NO DATA - all trades will be 'Flat'")
        phase_cache[(symbol, indicator)] = pd.Series(dtype=str)
        continue

    phases = st.classify_market_phases(df_raw, symbol)
    phase_counts = phases.value_counts().to_dict()
    print(f"  {sym_short}/{indicator}: {phase_counts}")
    phase_cache[(symbol, indicator)] = phases

# ── 3. TAG TRADES WITH PHASES ──
def get_phase(trade):
    """Look up phase for a trade's entry time."""
    key = (trade["symbol"], trade["indicator"])
    phases = phase_cache.get(key)
    if phases is None or phases.empty:
        return "Flat"
    entry_str = trade["entry_time"]
    try:
        entry_ts = pd.Timestamp(entry_str)
        if entry_ts in phases.index:
            return phases.loc[entry_ts]
        idx = phases.index.get_indexer([entry_ts], method="ffill")[0]
        if idx >= 0:
            return phases.iloc[idx]
    except Exception:
        pass
    return "Flat"

all_trades = []
for t in trades_list + open_list:
    t["phase"] = get_phase(t)
    # Normalize indicator key (lowercase)
    t["indicator_key"] = t["indicator"].lower().replace(" ", "_")
    all_trades.append(t)

# Sort chronologically
all_trades.sort(key=lambda t: t["entry_time"])

print(f"\nTotal tagged: {len(all_trades)} trades")

# ── 4. GROUP BY INDICATOR, APPLY SHARED EQUITY ──
trades_by_indicator = defaultdict(list)
for t in all_trades:
    trades_by_indicator[t["indicator_key"]].append(t)

for ind_key, ind_trades in trades_by_indicator.items():
    ind_trades.sort(key=lambda t: t["entry_time"])
    total = len(ind_trades)
    ind_display = ind_key.title()

    # Shared equity compound growth: START=16500, stake=equity/8
    capital = START_CAPITAL
    for t in ind_trades:
        stake = capital / MAX_POSITIONS
        pnl_net = t["pnl_pct"] * stake
        capital += pnl_net
        t["stake"] = stake
        t["pnl"] = pnl_net
        t["fees"] = 0  # fees already baked into pnl_pct from prices
        t["equity_after"] = capital

    total_pnl = sum(t["pnl"] for t in ind_trades)
    winners = sum(1 for t in ind_trades if t["pnl"] > 0)

    print(f"\n{'='*60}")
    print(f"{ind_display}: {total} trades | PnL: {total_pnl:+,.2f} | WR: {winners/total*100:.1f}%")
    print(f"  Start: {START_CAPITAL:,.2f} → Final: {capital:,.2f} (equity/{MAX_POSITIONS})")

    for phase in ["Up", "Down", "Flat"]:
        pt = [t for t in ind_trades if t["phase"] == phase]
        if pt:
            pp = sum(t["pnl"] for t in pt)
            pw = sum(1 for t in pt if t["pnl"] > 0)
            print(f"    {phase:5s}: {len(pt):4d}t  PnL={pp:+10,.2f}  WR={pw/len(pt)*100:.1f}%")

    # Generate dashboard (ab 2025-12-01) + summary (ab 2024-01-31)
    st._generate_phase_dashboard(ind_trades, dashboard_start="2025-12-01",
                                  summary_start="2024-01-31",
                                  stake_divisor=MAX_POSITIONS, use_precomputed=True)

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
            print(f"  → {dst}")

# ── 5. COMBINED REPORT (all indicators) ──
print(f"\n{'='*60}")
print(f"COMBINED (all indicators)")
capital = START_CAPITAL
for t in all_trades:
    stake = capital / MAX_POSITIONS
    pnl_net = t["pnl_pct"] * stake
    capital += pnl_net
    t["stake"] = stake
    t["pnl"] = pnl_net
    t["fees"] = 0
    t["equity_after"] = capital

total_pnl = sum(t["pnl"] for t in all_trades)
winners = sum(1 for t in all_trades if t["pnl"] > 0)
print(f"Total: {len(all_trades)} trades | PnL: {total_pnl:+,.2f} | WR: {winners/len(all_trades)*100:.1f}%")
print(f"Start: {START_CAPITAL:,.2f} → Final: {capital:,.2f}")
st._generate_phase_dashboard(all_trades, dashboard_start="2025-12-01",
                              summary_start="2024-01-31",
                              stake_divisor=MAX_POSITIONS, use_precomputed=True)

print("\nDone!")
