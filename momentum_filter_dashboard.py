#!/usr/bin/env python3
"""Momentum Filter Dashboard - Filter trades by early profit criterion.

Concept: If a trade signal shows profit after dT bars (parameter to optimize),
it has a higher probability of being a winning trade overall.

Entry Logic (NO Look-Ahead Bias):
1. Original signal comes at time T0 with price P0
2. Wait dT bars (e.g., 8 hours for dT=8)
3. At T0+dT, check if price P_dT > P0 (for long)
4. If yes: ENTER at price P_dT (NOT P0!)
5. Exit at original exit time/price
6. PnL = (exit_price - P_dT) / P_dT  (NOT based on P0!)

This ensures realistic backtesting - we can only decide to enter AFTER
seeing the momentum, so our entry price must be the price at that moment.
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Import OHLCV fetching from existing module
import Supertrend_5Min as st

# Settings
REPORT_DIR = Path("report_html")
OHLCV_CACHE_DIR = "ohlcv_cache"
SOURCE_JSON = REPORT_DIR / "trading_summary.json"
OUTPUT_DIR = REPORT_DIR
START_CAPITAL = 16500.0
MAX_POSITIONS = 10
TIMEFRAME = "1h"  # Base timeframe for bars


def load_trades_from_json(json_path: Path, start_date: str = None) -> list:
    """Load closed trades from trading_summary.json."""
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trades = data.get("trades", [])

    # Parse start_date filter
    start_dt = None
    if start_date:
        start_dt = datetime.fromisoformat(start_date + "T00:00:00")

    filtered_trades = []
    for t in trades:
        direction = str(t.get("direction", "long")).lower()
        if direction != "long":  # Only long trades for now
            continue

        entry_time = t.get("entry_time", "")

        # Apply start_date filter
        if start_dt and entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00").replace("+00:00", ""))
            except ValueError:
                entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            if entry_dt.replace(tzinfo=None) < start_dt:
                continue

        # Calculate original PnL percentage
        entry_price = float(t.get("entry_price", 0) or 0)
        exit_price = float(t.get("exit_price", 0) or 0)
        if entry_price > 0:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = 0

        filtered_trades.append({
            "symbol": t.get("symbol", ""),
            "direction": direction,
            "entry_time": entry_time,
            "exit_time": t.get("exit_time", ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "original_stake": float(t.get("stake", 0) or 0),
            "original_pnl": float(t.get("pnl", 0) or 0),
            "pnl_pct": pnl_pct,
            "reason": t.get("reason", ""),
            "indicator": t.get("indicator", ""),
            "htf": t.get("htf", ""),
        })

    return filtered_trades


def load_open_positions(json_path: Path) -> list:
    """Load open positions from trading_summary.json."""
    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("open_positions_data", [])


def get_current_price(symbol: str, ohlcv_cache: dict) -> Optional[float]:
    """Get the most recent price for a symbol from OHLCV cache.

    Args:
        symbol: Trading pair like "BTC/USDC"
        ohlcv_cache: Pre-loaded cache for OHLCV data

    Returns:
        Most recent close price, or None if not available
    """
    df = ohlcv_cache.get(symbol)
    if df is None or df.empty:
        return None

    try:
        return float(df.iloc[-1]["close"])
    except Exception:
        return None


def filter_open_positions_by_momentum(positions: list, dt_bars: int,
                                       ohlcv_cache: dict = None,
                                       verbose: bool = True) -> list:
    """Filter open positions by the same momentum criterion as closed trades.

    A position passes if it was profitable after dt_bars from entry.

    Args:
        positions: List of open position dictionaries
        dt_bars: Number of bars to check for early profit
        ohlcv_cache: Pre-loaded OHLCV data
        verbose: Print progress

    Returns:
        List of filtered positions with additional computed fields
    """
    if not positions:
        return []

    # Preload cache if not provided
    if ohlcv_cache is None:
        symbols = list(set(p["symbol"] for p in positions))
        if verbose:
            print(f"  Preloading OHLCV data for {len(symbols)} open position symbols...")
        ohlcv_cache = preload_ohlcv_cache(symbols, verbose=verbose)

    filtered = []
    now = datetime.now()

    for pos in positions:
        symbol = pos.get("symbol", "")
        entry_time = pos.get("entry_time", "")
        entry_price = float(pos.get("entry_price", 0) or 0)
        direction = str(pos.get("direction", "long")).lower()
        stake = float(pos.get("stake", 0) or 0)

        # Skip if no OHLCV data
        if symbol not in ohlcv_cache:
            continue

        # Get price after dt_bars
        price_at_dt = get_price_after_dt_bars(symbol, entry_time, dt_bars, ohlcv_cache)

        if price_at_dt is None:
            # Position may be too new - skip momentum filter but include if recent
            continue

        # Check if profitable at dt_bars (same criterion as closed trades)
        if direction == "long":
            is_profitable_at_dt = price_at_dt > entry_price
        else:
            is_profitable_at_dt = price_at_dt < entry_price

        if not is_profitable_at_dt:
            continue

        # Position passed momentum filter - compute additional fields
        # CRITICAL FIX: No look-ahead bias!
        # Our ACTUAL entry is at T+dT, not the original entry time

        current_price = get_current_price(symbol, ohlcv_cache)
        if current_price is None:
            current_price = price_at_dt  # Fallback

        # Calculate ACTUAL entry time (original + dT bars)
        try:
            entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            entry_dt = entry_dt.replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                entry_dt = now

        # ACTUAL entry is at dT bars after original signal
        actual_entry_dt = entry_dt + timedelta(hours=dt_bars)
        actual_entry_time = actual_entry_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Bars since ACTUAL entry (not original)
        bars = int((now - actual_entry_dt).total_seconds() / 3600)  # 1h bars
        if bars < 0:
            bars = 0

        # Calculate amount, PnL from ACTUAL entry price (price_at_dt)
        # This is our real entry price, not the original!
        actual_entry_price = price_at_dt
        if actual_entry_price > 0:
            amount = stake / actual_entry_price
            if direction == "long":
                pnl = (current_price - actual_entry_price) * amount
                pnl_pct = (current_price - actual_entry_price) / actual_entry_price * 100
            else:
                pnl = (actual_entry_price - current_price) * amount
                pnl_pct = (actual_entry_price - current_price) / actual_entry_price * 100
        else:
            amount = 0
            pnl = 0
            pnl_pct = 0

        filtered_pos = dict(pos)
        # Store original values for reference
        filtered_pos["original_entry_time"] = entry_time
        filtered_pos["original_entry_price"] = entry_price
        # Set ACTUAL entry values (delayed by dT bars)
        filtered_pos["entry_time"] = actual_entry_time
        filtered_pos["entry_price"] = actual_entry_price  # This is price_at_dt!
        filtered_pos["last_price"] = current_price
        filtered_pos["amount"] = amount
        filtered_pos["bars"] = bars
        filtered_pos["pnl"] = pnl
        filtered_pos["pnl_pct"] = pnl_pct
        filtered_pos["status"] = "OPEN"
        filtered_pos["price_at_dt"] = price_at_dt
        filtered_pos["pnl_at_dt_pct"] = (price_at_dt - entry_price) / entry_price * 100 if entry_price > 0 else 0
        filtered.append(filtered_pos)

    if verbose:
        print(f"  Open positions: {len(positions)} total, {len(filtered)} passed momentum filter")

    return filtered


def get_price_after_dt_bars(symbol: str, entry_time: str, dt_bars: int,
                            ohlcv_cache: dict) -> Optional[float]:
    """Get the price dt_bars after entry time.

    Args:
        symbol: Trading pair like "BTC/USDC"
        entry_time: Entry timestamp in ISO format
        dt_bars: Number of bars to look ahead
        ohlcv_cache: Pre-loaded cache for OHLCV data

    Returns:
        Close price after dt_bars, or None if not available
    """
    # Get DataFrame from cache
    df = ohlcv_cache.get(symbol)
    if df is None or df.empty:
        return None

    # Parse entry time
    try:
        entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        entry_dt = entry_dt.replace(tzinfo=None)  # Make naive for comparison
    except (ValueError, TypeError):
        try:
            entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return None

    # Calculate target time (entry + dt_bars hours for 1h timeframe)
    target_dt = entry_dt + timedelta(hours=dt_bars)

    # Find closest candle to target time
    try:
        # Look for exact match or next available candle
        mask = df.index >= target_dt
        if mask.any():
            idx = df.index[mask][0]
            return float(df.loc[idx, "close"])
        else:
            # Target is after last available data
            return None
    except Exception:
        return None


def preload_ohlcv_cache(symbols: list, verbose: bool = True) -> dict:
    """Preload OHLCV data for all symbols from cache.

    Args:
        symbols: List of trading pairs
        verbose: Print progress

    Returns:
        Dict mapping symbol -> DataFrame
    """
    ohlcv_cache = {}
    failed_symbols = set()

    for symbol in symbols:
        try:
            df = st.load_ohlcv_from_cache(symbol, TIMEFRAME)
            if df is not None and not df.empty:
                # Convert index to DatetimeIndex if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Make index naive for comparison
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                ohlcv_cache[symbol] = df
                if verbose:
                    print(f"  Loaded {symbol}: {len(df)} bars")
            else:
                failed_symbols.add(symbol)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {symbol}: {e}")
            failed_symbols.add(symbol)

    if verbose and failed_symbols:
        print(f"  Symbols without cache data: {', '.join(sorted(failed_symbols))}")

    return ohlcv_cache


def filter_trades_by_momentum(trades: list, dt_bars: int, verbose: bool = True,
                               ohlcv_cache: dict = None) -> tuple[list, dict]:
    """Filter trades based on early momentum criterion.

    A trade passes the filter if it's profitable after dt_bars.

    Args:
        trades: List of trade dictionaries
        dt_bars: Number of bars to check for early profit
        verbose: Print progress
        ohlcv_cache: Pre-loaded OHLCV data (optional)

    Returns:
        Tuple of (filtered_trades, statistics)
    """
    if verbose:
        print(f"\n=== Filtering trades with dT = {dt_bars} bars ===")

    # Preload cache if not provided
    if ohlcv_cache is None:
        symbols = list(set(t["symbol"] for t in trades))
        if verbose:
            print(f"  Preloading OHLCV data for {len(symbols)} symbols...")
        ohlcv_cache = preload_ohlcv_cache(symbols, verbose=verbose)

    filtered = []
    stats = {
        "total_trades": len(trades),
        "filtered_trades": 0,
        "skipped_no_data": 0,
        "skipped_not_profitable_at_dt": 0,
    }

    for i, trade in enumerate(trades):
        symbol = trade["symbol"]
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        direction = trade["direction"]

        if verbose and (i + 1) % 500 == 0:
            print(f"  Processing trade {i + 1}/{len(trades)}...")

        # Skip if no OHLCV data for this symbol
        if symbol not in ohlcv_cache:
            stats["skipped_no_data"] += 1
            continue

        # Get price after dt_bars
        price_at_dt = get_price_after_dt_bars(symbol, entry_time, dt_bars, ohlcv_cache)

        if price_at_dt is None:
            stats["skipped_no_data"] += 1
            continue

        # Check if profitable at dt_bars
        if direction == "long":
            is_profitable_at_dt = price_at_dt > entry_price
        else:  # short
            is_profitable_at_dt = price_at_dt < entry_price

        if is_profitable_at_dt:
            # CRITICAL FIX: No look-ahead bias!
            # When momentum filter triggers at T+dT, that's when we actually enter
            # So entry_price = price_at_dt, not the original entry_price

            # Calculate the ACTUAL entry time (original + dT bars)
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                entry_dt = entry_dt.replace(tzinfo=None)
            except (ValueError, TypeError):
                try:
                    entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    entry_dt = None

            if entry_dt:
                actual_entry_dt = entry_dt + timedelta(hours=dt_bars)
                actual_entry_time = actual_entry_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                actual_entry_time = entry_time  # Fallback

            # Calculate ACTUAL PnL from the delayed entry price
            exit_price = trade["exit_price"]
            if direction == "long":
                actual_pnl_pct = (exit_price - price_at_dt) / price_at_dt * 100 if price_at_dt > 0 else 0
            else:  # short
                actual_pnl_pct = (price_at_dt - exit_price) / price_at_dt * 100 if price_at_dt > 0 else 0

            trade_copy = dict(trade)
            # Store original values for reference
            trade_copy["original_entry_time"] = entry_time
            trade_copy["original_entry_price"] = entry_price
            trade_copy["original_pnl_pct"] = trade["pnl_pct"]
            # Set ACTUAL entry values (delayed by dT bars)
            trade_copy["entry_time"] = actual_entry_time
            trade_copy["entry_price"] = price_at_dt  # This is now our real entry price!
            trade_copy["pnl_pct"] = actual_pnl_pct  # Recalculated from delayed entry
            trade_copy["price_at_dt"] = price_at_dt
            trade_copy["pnl_at_dt_pct"] = (price_at_dt - entry_price) / entry_price * 100 if entry_price > 0 else 0
            filtered.append(trade_copy)
            stats["filtered_trades"] += 1
        else:
            stats["skipped_not_profitable_at_dt"] += 1

    # Calculate win rates
    if trades:
        original_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        stats["original_win_rate"] = original_wins / len(trades) * 100
    else:
        stats["original_win_rate"] = 0

    if filtered:
        filtered_wins = sum(1 for t in filtered if t["pnl_pct"] > 0)
        stats["filtered_win_rate"] = filtered_wins / len(filtered) * 100
    else:
        stats["filtered_win_rate"] = 0

    if verbose:
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Passed momentum filter: {stats['filtered_trades']}")
        print(f"  Skipped (no data): {stats['skipped_no_data']}")
        print(f"  Skipped (not profitable at dT): {stats['skipped_not_profitable_at_dt']}")
        print(f"  Original win rate: {stats['original_win_rate']:.1f}%")
        print(f"  Filtered win rate: {stats['filtered_win_rate']:.1f}%")

    return filtered, stats


def recalculate_with_compound_growth(trades: list) -> tuple[list, float]:
    """Recalculate trades with compound growth starting from START_CAPITAL."""
    # Sort chronologically
    def get_entry_time(t):
        entry_time = t.get("entry_time", "")
        try:
            return datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            try:
                return datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return datetime.min

    trades_chrono = sorted(trades, key=get_entry_time, reverse=False)

    capital = START_CAPITAL
    recalculated = []

    for t in trades_chrono:
        stake = capital / MAX_POSITIONS
        pnl_pct = t.get("pnl_pct", 0) / 100  # Convert from percentage

        pnl = pnl_pct * stake
        capital += pnl

        recalc = dict(t)
        recalc["stake"] = stake
        recalc["pnl"] = pnl
        recalc["equity_after"] = capital
        recalculated.append(recalc)

    return recalculated, capital


def fmt_de(value: float) -> str:
    """Format number in German style."""
    if abs(value) >= 1000:
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:.2f}".replace(".", ",")


def generate_momentum_dashboard(trades: list, stats: dict, dt_bars: int,
                                 output_dir: Path, original_stats: dict = None,
                                 open_positions: list = None):
    """Generate HTML dashboard for momentum-filtered trades."""

    # Recalculate with compound growth
    recalculated, final_capital = recalculate_with_compound_growth(trades)

    # Ensure open_positions is a list
    if open_positions is None:
        open_positions = []

    # Sort by entry time (newest first) for display
    def get_entry_time(t):
        entry_time = t.get("entry_time", "")
        try:
            return datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            try:
                return datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return datetime.min

    recalculated.sort(key=get_entry_time, reverse=True)

    # Calculate statistics
    total_pnl = sum(t.get("pnl", 0) for t in recalculated)
    wins = sum(1 for t in recalculated if t.get("pnl", 0) > 0)
    losses = sum(1 for t in recalculated if t.get("pnl", 0) <= 0)
    win_rate = wins / len(recalculated) * 100 if recalculated else 0

    # Format prices
    def fmt_price(price):
        if price < 0.0001:
            return f"{price:.8f}"
        elif price < 1:
            return f"{price:.6f}"
        elif price < 100:
            return f"{price:.4f}"
        else:
            return fmt_de(price)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="60">
    <title>Momentum Filter Dashboard (dT={dt_bars})</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #9c27b0; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background: #9c27b0; color: white; text-align: center; }}
        td:first-child {{ text-align: left; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
        .summary-box {{ display: inline-block; background: white; padding: 20px; margin: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 150px; text-align: center; }}
        .summary-box h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .summary-box .value {{ font-size: 24px; font-weight: bold; }}
        .comparison-box {{ background: #e8f5e9; border: 2px solid #4caf50; }}
        .improvement {{ color: #4caf50; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .timestamp-header {{ background: #333; color: #fff; padding: 10px 20px; border-radius: 5px; display: inline-block; margin-bottom: 20px; font-size: 14px; }}
        .timestamp-header .time {{ font-weight: bold; color: #9c27b0; }}
        .open-positions {{ background: #fff3e0; border: 2px solid #ff9800; }}
        .open-positions th {{ background: #ff9800 !important; }}
        .section {{ margin-bottom: 30px; }}
        .concept-note {{ background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #ff9800; }}
        .filter-header {{ background: #9c27b0 !important; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Momentum Filter Dashboard (dT = {dt_bars} bars)</h1>

    <div class="timestamp-header">
        Letztes Update: <span class="time">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
        &nbsp;|&nbsp; Auto-Refresh: 60s
    </div>

    <div class="concept-note">
        <strong>Konzept:</strong> Nur Trades aufnehmen, die nach {dt_bars} Bar(s) ({dt_bars}h) im Gewinn sind.<br>
        <strong>Entry-Logik:</strong> Signal kommt bei T0 → Warten bis T0+{dt_bars}h → Prüfen ob im Plus → Wenn ja: Entry zum Preis bei T0+{dt_bars}h<br>
        <strong>Kein Look-Ahead Bias:</strong> Entry-Preis = Preis bei dT, nicht Original-Signal-Preis!
    </div>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>Start Capital</h3>
            <div class="value">{fmt_de(START_CAPITAL)}</div>
        </div>
        <div class="summary-box">
            <h3>Final Capital</h3>
            <div class="value {'positive' if final_capital >= START_CAPITAL else 'negative'}">{fmt_de(final_capital)}</div>
        </div>
        <div class="summary-box">
            <h3>Filtered Trades</h3>
            <div class="value">{len(recalculated)}</div>
        </div>
        <div class="summary-box">
            <h3>Total PnL</h3>
            <div class="value {'positive' if total_pnl >= 0 else 'negative'}">{fmt_de(total_pnl)}</div>
        </div>
        <div class="summary-box">
            <h3>Win Rate</h3>
            <div class="value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div>
        </div>
        <div class="summary-box">
            <h3>Wins / Losses</h3>
            <div class="value">{wins} / {losses}</div>
        </div>
    </div>
"""

    # Comparison section if original stats available
    if original_stats:
        orig_win_rate = original_stats.get("original_win_rate", 0)
        improvement = win_rate - orig_win_rate
        html += f"""
    <div class="section">
    <h2>Vergleich: Original vs. Momentum-Filter</h2>
    <table>
        <tr><th>Metrik</th><th>Original</th><th>Momentum-Filter (dT={dt_bars})</th><th>Differenz</th></tr>
        <tr>
            <td>Anzahl Trades</td>
            <td>{stats['total_trades']}</td>
            <td>{len(recalculated)}</td>
            <td class="negative">-{stats['total_trades'] - len(recalculated)}</td>
        </tr>
        <tr>
            <td>Win Rate</td>
            <td>{orig_win_rate:.1f}%</td>
            <td class="{'positive' if win_rate > orig_win_rate else ''}">{win_rate:.1f}%</td>
            <td class="{'improvement' if improvement > 0 else 'negative'}">{improvement:+.1f}%</td>
        </tr>
        <tr>
            <td>Filter-Quote</td>
            <td>100%</td>
            <td>{len(recalculated) / stats['total_trades'] * 100:.1f}%</td>
            <td>-</td>
        </tr>
    </table>
    </div>
"""

    # Open Positions section (filtered by momentum criterion)
    # Recalculate open position stakes based on compound growth capital
    if open_positions:
        current_stake = final_capital / MAX_POSITIONS
        for pos in open_positions:
            entry_price = pos.get("entry_price", 0)
            if entry_price > 0:
                pos["stake"] = current_stake
                pos["amount"] = current_stake / entry_price
                last_price = pos.get("last_price", entry_price)
                direction = pos.get("direction", "long")
                if direction == "long":
                    pos["pnl"] = (last_price - entry_price) * pos["amount"]
                    pos["pnl_pct"] = (last_price - entry_price) / entry_price * 100
                else:
                    pos["pnl"] = (entry_price - last_price) * pos["amount"]
                    pos["pnl_pct"] = (entry_price - last_price) / entry_price * 100

        html += f"""
    <div class="section">
    <h2>Offene Positionen - Momentum Filter ({len(open_positions)})</h2>
    <p style="color: #666; font-size: 12px;">Signal = Original-Zeitpunkt | Entry = Tatsächlicher Einstieg nach {dt_bars}h Momentum-Check</p>
    <table class="open-positions">
        <tr>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Indicator</th>
            <th>HTF</th>
            <th>Signal Time</th>
            <th>Signal Price</th>
            <th>Entry Time<br/>(+{dt_bars}h)</th>
            <th>Entry Price</th>
            <th>Last Price</th>
            <th>Stake</th>
            <th>Bars</th>
            <th>PnL</th>
            <th>PnL%</th>
            <th>Status</th>
        </tr>
"""
        for pos in open_positions:
            # ACTUAL entry time (delayed by dT bars)
            entry_time = pos.get("entry_time", "")
            # ORIGINAL signal time
            original_entry_time = pos.get("original_entry_time", entry_time)

            try:
                orig_entry_dt = datetime.fromisoformat(original_entry_time.replace("Z", "+00:00"))
                orig_entry_str = orig_entry_dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError, AttributeError):
                orig_entry_str = original_entry_time[:16] if original_entry_time else "N/A"

            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                entry_str = entry_dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError, AttributeError):
                entry_str = entry_time[:16] if entry_time else "N/A"

            pnl = pos.get("pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            pnl_class = "positive" if pnl >= 0 else "negative"

            # Original signal price and actual entry price
            original_entry_price = pos.get("original_entry_price", pos.get("entry_price", 0))
            actual_entry_price = pos.get("entry_price", 0)  # This is now price_at_dt

            html += f"""        <tr>
            <td>{pos.get('symbol', 'N/A')}</td>
            <td>{pos.get('direction', 'N/A')}</td>
            <td>{pos.get('indicator', 'N/A')}</td>
            <td>{pos.get('htf', 'N/A')}</td>
            <td>{orig_entry_str}</td>
            <td>{fmt_price(original_entry_price)}</td>
            <td>{entry_str}</td>
            <td>{fmt_price(actual_entry_price)}</td>
            <td>{fmt_price(pos.get('last_price', 0))}</td>
            <td>{fmt_de(pos.get('stake', 0))}</td>
            <td>{pos.get('bars', 0)}</td>
            <td class="{pnl_class}">{fmt_de(pnl)}</td>
            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            <td>{pos.get('status', 'OPEN')}</td>
        </tr>
"""
        html += """    </table>
    </div>
"""

    # Trades table
    html += f"""
    <div class="section">
    <h2>Gefilterte Trades ({len(recalculated)})</h2>
    <p style="color: #666; font-size: 12px;">Signal = Original-Zeitpunkt | Entry = Tatsächlicher Einstieg nach {dt_bars}h Momentum-Check</p>
    <table>
        <tr class="filter-header">
            <th>Symbol</th>
            <th>Signal Time</th>
            <th>Signal Price</th>
            <th>Entry Time<br/>(+{dt_bars}h)</th>
            <th>Entry Price</th>
            <th>Exit Time</th>
            <th>Exit Price</th>
            <th>Stake</th>
            <th>PnL</th>
            <th>PnL%</th>
            <th>Reason</th>
        </tr>
"""

    for t in recalculated:
        # ACTUAL entry time (delayed by dT bars)
        entry_time = t.get("entry_time", "")
        # ORIGINAL signal time
        original_entry_time = t.get("original_entry_time", entry_time)
        exit_time = t.get("exit_time", "")

        try:
            orig_entry_dt = datetime.fromisoformat(original_entry_time.replace("Z", "+00:00"))
            orig_entry_str = orig_entry_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, AttributeError):
            orig_entry_str = original_entry_time[:16] if original_entry_time else "N/A"

        try:
            entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            entry_str = entry_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, AttributeError):
            entry_str = entry_time[:16] if entry_time else "N/A"

        try:
            exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
            exit_str = exit_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, AttributeError):
            exit_str = exit_time[:16] if exit_time else "N/A"

        pnl = t.get("pnl", 0)
        pnl_pct = t.get("pnl_pct", 0)
        pnl_class = "positive" if pnl >= 0 else "negative"

        # Original signal price and actual entry price (= price_at_dt)
        original_entry_price = t.get("original_entry_price", t.get("entry_price", 0))
        actual_entry_price = t.get("entry_price", 0)  # This is now price_at_dt

        html += f"""        <tr>
            <td>{t.get('symbol', 'N/A')}</td>
            <td>{orig_entry_str}</td>
            <td>{fmt_price(original_entry_price)}</td>
            <td>{entry_str}</td>
            <td>{fmt_price(actual_entry_price)}</td>
            <td>{exit_str}</td>
            <td>{fmt_price(t.get('exit_price', 0))}</td>
            <td>{fmt_de(t.get('stake', 0))}</td>
            <td class="{pnl_class}">{fmt_de(pnl)}</td>
            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            <td>{t.get('reason', 'N/A')}</td>
        </tr>
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">Generated: {timestamp}</p>
</div>
</body>
</html>"""

    # Write to file
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"dashboard_momentum_dt{dt_bars}.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to: {output_path}")
    return output_path


def optimize_dt(trades: list, dt_range: range, verbose: bool = True) -> dict:
    """Test multiple dT values and find the optimal one.

    Args:
        trades: List of trade dictionaries
        dt_range: Range of dT values to test (e.g., range(1, 13))
        verbose: Print progress

    Returns:
        Dict with optimization results
    """
    print("\n" + "="*60)
    print("MOMENTUM FILTER OPTIMIZATION")
    print("="*60)

    # Preload OHLCV cache once for all dT values
    symbols = list(set(t["symbol"] for t in trades))
    print(f"Preloading OHLCV data for {len(symbols)} symbols...")
    ohlcv_cache = preload_ohlcv_cache(symbols, verbose=True)
    print(f"Cache ready: {len(ohlcv_cache)} symbols loaded\n")

    results = []

    for dt in dt_range:
        filtered, stats = filter_trades_by_momentum(trades, dt, verbose=False, ohlcv_cache=ohlcv_cache)

        # Calculate compound growth for filtered trades
        if filtered:
            _, final_capital = recalculate_with_compound_growth(filtered)
            total_return = (final_capital - START_CAPITAL) / START_CAPITAL * 100
        else:
            final_capital = START_CAPITAL
            total_return = 0

        result = {
            "dt_bars": dt,
            "filtered_trades": len(filtered),
            "win_rate": stats["filtered_win_rate"],
            "original_win_rate": stats["original_win_rate"],
            "win_rate_improvement": stats["filtered_win_rate"] - stats["original_win_rate"],
            "filter_rate": len(filtered) / len(trades) * 100 if trades else 0,
            "final_capital": final_capital,
            "total_return": total_return,
        }
        results.append(result)

        if verbose:
            print(f"dT={dt:2d}: {len(filtered):4d} trades | Win: {stats['filtered_win_rate']:5.1f}% | "
                  f"Improvement: {result['win_rate_improvement']:+5.1f}% | "
                  f"Return: {total_return:+.1f}%")

    # Find best dT by win rate improvement
    best_by_winrate = max(results, key=lambda x: x["win_rate"])
    best_by_improvement = max(results, key=lambda x: x["win_rate_improvement"])
    best_by_return = max(results, key=lambda x: x["total_return"])

    print("\n" + "-"*60)
    print(f"Best by Win Rate:    dT={best_by_winrate['dt_bars']} ({best_by_winrate['win_rate']:.1f}%)")
    print(f"Best by Improvement: dT={best_by_improvement['dt_bars']} (+{best_by_improvement['win_rate_improvement']:.1f}%)")
    print(f"Best by Return:      dT={best_by_return['dt_bars']} ({best_by_return['total_return']:.1f}%)")
    print("-"*60)

    return {
        "results": results,
        "best_by_winrate": best_by_winrate,
        "best_by_improvement": best_by_improvement,
        "best_by_return": best_by_return,
    }


def run_dashboard_cycle(dt_bars: int, start_date: str, output_dir: Path,
                         verbose: bool = True) -> Optional[Path]:
    """Run a single dashboard generation cycle.

    Args:
        dt_bars: Number of bars for momentum filter
        start_date: Start date filter (YYYY-MM-DD)
        output_dir: Output directory for HTML
        verbose: Print progress

    Returns:
        Path to generated dashboard or None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        print(f"\n[{timestamp}] Running dashboard cycle...")

    # Load trades
    trades = load_trades_from_json(SOURCE_JSON, start_date=start_date)
    if not trades:
        if verbose:
            print("  No trades found.")
        return None

    # Load open positions
    open_positions = load_open_positions(SOURCE_JSON)

    if verbose:
        print(f"  Loaded {len(trades)} trades, {len(open_positions)} open positions")

    # Preload OHLCV cache for all symbols (trades + open positions)
    all_symbols = list(set(t["symbol"] for t in trades))
    open_symbols = [p.get("symbol") for p in open_positions if p.get("symbol")]
    all_symbols = list(set(all_symbols + open_symbols))
    ohlcv_cache = preload_ohlcv_cache(all_symbols, verbose=False)

    # Filter trades
    filtered, stats = filter_trades_by_momentum(trades, dt_bars, verbose=False, ohlcv_cache=ohlcv_cache)

    if not filtered:
        if verbose:
            print("  No trades passed momentum filter.")
        return None

    # Filter open positions by momentum criterion (same as closed trades)
    filtered_open_positions = filter_open_positions_by_momentum(
        open_positions, dt_bars, ohlcv_cache=ohlcv_cache, verbose=verbose
    )

    # Generate dashboard
    path = generate_momentum_dashboard(
        filtered, stats, dt_bars, output_dir, stats, filtered_open_positions
    )

    if verbose:
        wins = sum(1 for t in filtered if t.get("pnl_pct", 0) > 0)
        win_rate = wins / len(filtered) * 100 if filtered else 0
        print(f"  Filtered: {len(filtered)} trades, Win rate: {win_rate:.1f}%")
        print(f"  Dashboard: {path}")

    return path


def monitor_loop(dt_bars: int, start_date: str, output_dir: Path,
                  interval_min: float = 5.0) -> None:
    """Continuous monitor loop that regenerates dashboard periodically.

    Args:
        dt_bars: Number of bars for momentum filter
        start_date: Start date filter (YYYY-MM-DD)
        output_dir: Output directory for HTML
        interval_min: Minutes between updates (default: 5)
    """
    print("="*60)
    print("MOMENTUM FILTER DASHBOARD - MONITOR MODE")
    print("="*60)
    print(f"  dT = {dt_bars} bars")
    print(f"  Start date: {start_date}")
    print(f"  Update interval: {interval_min} minutes")
    print(f"  Output: {output_dir}")
    print("="*60)
    print("Press Ctrl+C to stop\n")

    interval_sec = interval_min * 60.0

    try:
        while True:
            run_dashboard_cycle(dt_bars, start_date, output_dir, verbose=True)

            # Wait for next cycle
            next_update = datetime.now() + timedelta(seconds=interval_sec)
            print(f"  Next update: {next_update.strftime('%H:%M:%S')}")

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Momentum Filter Dashboard")
    parser.add_argument("--dt", type=int, default=2,
                        help="Number of bars to check for early profit (default: 2)")
    parser.add_argument("--start", type=str, default="2024-01-31",
                        help="Start date YYYY-MM-DD (default: 2024-01-31)")
    parser.add_argument("--optimize", action="store_true",
                        help="Test multiple dT values and find optimal")
    parser.add_argument("--dt-min", type=int, default=1,
                        help="Minimum dT for optimization (default: 1)")
    parser.add_argument("--dt-max", type=int, default=12,
                        help="Maximum dT for optimization (default: 12)")
    parser.add_argument("--output-dir", type=str, default="report_html",
                        help="Output directory (default: report_html)")
    parser.add_argument("--monitor", action="store_true",
                        help="Enable continuous monitor mode with periodic updates")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Minutes between dashboard updates in monitor mode (default: 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Monitor mode - continuous updates
    if args.monitor:
        monitor_loop(args.dt, args.start, output_dir, args.interval)
        return

    print(f"Loading trades from {SOURCE_JSON} (start_date={args.start})...")
    trades = load_trades_from_json(SOURCE_JSON, start_date=args.start)
    print(f"Loaded {len(trades)} trades")

    # Load open positions
    open_positions = load_open_positions(SOURCE_JSON)
    print(f"Open positions: {len(open_positions)}")

    if not trades:
        print("No trades found. Check the source file.")
        return

    # Calculate original statistics
    original_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    original_win_rate = original_wins / len(trades) * 100
    print(f"Original win rate: {original_win_rate:.1f}%")

    # Preload OHLCV cache for all symbols (trades + open positions)
    all_symbols = list(set(t["symbol"] for t in trades))
    open_symbols = [p.get("symbol") for p in open_positions if p.get("symbol")]
    all_symbols = list(set(all_symbols + open_symbols))
    print(f"Preloading OHLCV cache for {len(all_symbols)} symbols...")
    ohlcv_cache = preload_ohlcv_cache(all_symbols, verbose=False)
    print(f"Cache ready: {len(ohlcv_cache)} symbols loaded")

    if args.optimize:
        # Optimization mode
        opt_results = optimize_dt(trades, range(args.dt_min, args.dt_max + 1))

        # Generate dashboards for top 3 dT values
        print("\nGenerating dashboards for best dT values...")
        for key in ["best_by_winrate", "best_by_improvement", "best_by_return"]:
            best = opt_results[key]
            dt = best["dt_bars"]
            filtered, stats = filter_trades_by_momentum(trades, dt, verbose=False, ohlcv_cache=ohlcv_cache)
            # Filter open positions by momentum criterion
            filtered_open_positions = filter_open_positions_by_momentum(
                open_positions, dt, ohlcv_cache=ohlcv_cache, verbose=False
            )
            generate_momentum_dashboard(filtered, stats, dt, output_dir, stats, filtered_open_positions)
    else:
        # Single dT mode
        filtered, stats = filter_trades_by_momentum(trades, args.dt, verbose=True, ohlcv_cache=ohlcv_cache)

        if filtered:
            # Filter open positions by momentum criterion (same as closed trades)
            filtered_open_positions = filter_open_positions_by_momentum(
                open_positions, args.dt, ohlcv_cache=ohlcv_cache, verbose=True
            )
            path = generate_momentum_dashboard(filtered, stats, args.dt, output_dir, stats, filtered_open_positions)
            print(f"\nOpen with: xdg-open {path}")
        else:
            print("No trades passed the momentum filter.")


if __name__ == "__main__":
    main()
