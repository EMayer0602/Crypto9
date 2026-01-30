#!/usr/bin/env python3
"""Generate HTML dashboard for Crypto9 Testnet trading (local tracking)."""

import os
import json
import time
import hmac
import hashlib
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Local Crypto9 tracking files
CRYPTO9_POSITIONS_FILE = "crypto9_testnet_positions.json"
CRYPTO9_CLOSED_TRADES_FILE = "crypto9_testnet_closed_trades.json"
PAPER_TRADING_STATE_FILE = "paper_trading_state.json"
PAPER_TRADING_SIMULATION_LOG = "paper_trading_simulation_log.json"
PAPER_TRADING_OPEN_POSITIONS = "paper_trading_actual_trades.json"

# Fee rate for PnL correction
FEE_RATE = 0.00075

# Start capital for equity calculation (must match paper_trader.py)
START_TOTAL_CAPITAL = 16_500.0


def fmt_de(n, decimals=2):
    """Format number in German format (period as thousands sep, comma as decimal).

    Examples:
        1234.56 → "1.234,56"
        1234567.89 → "1.234.567,89"
    """
    if n is None or n == 0:
        return "-"
    # Format with specified decimals
    formatted = f"{n:,.{decimals}f}"
    # Swap separators: US (1,234.56) → German (1.234,56)
    # First replace comma with temp, then period with comma, then temp with period
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted


def fmt_de_currency(n, decimals=2):
    """Format currency in German format with $ prefix."""
    if n is None:
        return "-"
    return f"${fmt_de(n, decimals)}"


def fmt_en(n, decimals=2):
    """Format number in English/US format (comma as thousands sep, period as decimal).

    Examples:
        1234.56 → "1,234.56"
        1234567.89 → "1,234,567.89"
    """
    if n is None or n == 0:
        return "-"
    return f"{n:,.{decimals}f}"


def fmt_en_currency(n, decimals=2):
    """Format currency in English/US format with $ prefix."""
    if n is None:
        return "-"
    return f"${fmt_en(n, decimals)}"


def correct_trades_pnl(json_path: str) -> int:
    """Correct PnL for trades using: size_units = stake/entry, pnl = size_units * diff - fees."""
    import math
    print(f"[PnL-Fix] Checking {json_path}...")
    if not os.path.exists(json_path):
        print(f"[PnL-Fix] File not found: {json_path}")
        return 0
    try:
        with open(json_path, 'r') as f:
            trades = json.load(f)
        if not isinstance(trades, list):
            print(f"[PnL-Fix] Not a list in {json_path}")
            return 0
        print(f"[PnL-Fix] Found {len(trades)} trades in {json_path}")
        corrected = 0
        for t in trades:
            entry_price = float(t.get('entry_price', 0) or 0)
            exit_price = float(t.get('exit_price', 0) or 0)
            stake = float(t.get('stake', 0) or 0)
            if entry_price > 0 and exit_price > 0 and stake > 0:
                size_units = stake / entry_price
                fees = (entry_price + exit_price) * size_units * FEE_RATE
                direction = str(t.get('direction', 'Long')).lower()
                if direction == 'long':
                    new_pnl = size_units * (exit_price - entry_price) - fees
                else:
                    new_pnl = size_units * (entry_price - exit_price) - fees
                old_pnl = float(t.get('pnl', 0) or 0)
                if abs(new_pnl - old_pnl) > 0.001:
                    t['pnl'] = round(new_pnl, 8)
                    t['fees'] = round(fees, 8)
                    t['size_units'] = size_units
                    corrected += 1
        if corrected > 0:
            with open(json_path, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
            print(f"[PnL-Fix] Corrected {corrected} trades in {json_path}")
        else:
            print(f"[PnL-Fix] No corrections needed in {json_path}")
        return corrected
    except Exception as e:
        print(f"[PnL-Fix] Error: {e}")
        return 0

# Spot Testnet API keys
SPOT_API_KEY = os.getenv("BINANCE_API_KEY_SPOT")
SPOT_API_SECRET = os.getenv("BINANCE_API_SECRET_SPOT")

# Futures Testnet API keys
FUTURES_API_KEY = os.getenv("BINANCE_API_KEY_TEST_F")
FUTURES_API_SECRET = os.getenv("BINANCE_API_SECRET_TEST_F")

# API Base URLs
SPOT_BASE_URL = "https://testnet.binance.vision"
FUTURES_BASE_URL = "https://testnet.binancefuture.com"

RECV_WINDOW_MS = 5_000

# Trading symbols - Testnet only supports USDT pairs
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LINKUSDT", "SOLUSDT", "BNBUSDT",
           "SUIUSDT", "ZECUSDT", "LUNCUSDT", "TNSRUSDT", "ADAUSDT", "ICPUSDT"]

# Base currencies (don't count as positions)
BASE_CURRENCIES = {"USDT", "USDC", "BUSD", "BTC", "TUSD"}

# Relevant trading assets (filter out testnet junk)
RELEVANT_ASSETS = {"BTC", "ETH", "SOL", "XRP", "LINK", "BNB", "SUI", "ZEC", "LUNC", "TNSR", "ADA", "ICP"}

# Only show trades from this date onwards (set to None to show all)
# Show ALL trades to capture full profit history (Long + Short = $6,535 profit)
TRADES_SINCE_DATE = None

OUTPUT_DIR = Path("report_testnet")


def sign_request(params: dict, secret: str) -> str:
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + "&signature=" + signature


# ============================================================================
# SPOT TESTNET API
# ============================================================================

def get_spot_balances() -> list:
    """Get all spot account balances."""
    # Check if Spot API credentials are available
    if not SPOT_API_KEY or not SPOT_API_SECRET:
        print("  Spot API keys not configured, skipping spot balances")
        return []
    endpoint = "/api/v3/account"
    url = SPOT_BASE_URL + endpoint
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, SPOT_API_SECRET)
    headers = {"X-MBX-APIKEY": SPOT_API_KEY}
    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("balances", [])
    except Exception as e:
        print(f"Error fetching spot balances: {e}")
        return []


def get_spot_orders(symbol: str) -> list:
    """Get all spot orders for a symbol."""
    # Check if Spot API credentials are available
    if not SPOT_API_KEY or not SPOT_API_SECRET:
        return []
    endpoint = "/api/v3/allOrders"
    url = SPOT_BASE_URL + endpoint
    params = {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, SPOT_API_SECRET)
    headers = {"X-MBX-APIKEY": SPOT_API_KEY}
    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


# ============================================================================
# FUTURES TESTNET API
# ============================================================================

def get_futures_balance() -> float:
    """Get futures USDT balance."""
    endpoint = "/fapi/v2/balance"
    url = FUTURES_BASE_URL + endpoint
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, FUTURES_API_SECRET)
    headers = {"X-MBX-APIKEY": FUTURES_API_KEY}
    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        if response.status_code == 200:
            for asset in response.json():
                if asset["asset"] == "USDT":
                    return float(asset["availableBalance"])
        else:
            print(f"Futures balance error: {response.status_code} - {response.text[:100]}")
        return 0.0
    except Exception as e:
        print(f"Error fetching futures balance: {e}")
        return 0.0


def get_futures_positions() -> list:
    """Get all open futures positions."""
    endpoint = "/fapi/v2/positionRisk"
    url = FUTURES_BASE_URL + endpoint
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, FUTURES_API_SECRET)
    headers = {"X-MBX-APIKEY": FUTURES_API_KEY}
    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        if response.status_code == 200:
            positions = []
            for p in response.json():
                amt = float(p.get("positionAmt", 0))
                if amt != 0:
                    positions.append({
                        "symbol": p["symbol"],
                        "side": "LONG" if amt > 0 else "SHORT",
                        "amount": abs(amt),
                        "entry_price": float(p.get("entryPrice", 0)),
                        "mark_price": float(p.get("markPrice", 0)),
                        "unrealized_pnl": float(p.get("unRealizedProfit", 0)),
                        "leverage": p.get("leverage", "1"),
                    })
            return positions
        else:
            print(f"Futures positions error: {response.status_code}")
        return []
    except Exception as e:
        print(f"Error fetching futures positions: {e}")
        return []


def get_futures_trades(symbol: str) -> list:
    """Get futures trade history for a symbol."""
    endpoint = "/fapi/v1/userTrades"
    url = FUTURES_BASE_URL + endpoint
    params = {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, FUTURES_API_SECRET)
    headers = {"X-MBX-APIKEY": FUTURES_API_KEY}
    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


# ============================================================================
# LOCAL CRYPTO9 TRACKING
# ============================================================================

def fetch_current_prices(symbols: list) -> dict:
    """Fetch current prices for given symbols from Binance public API."""
    prices = {}
    try:
        # Use production API for prices (testnet prices are unrealistic)
        url = "https://api.binance.com/api/v3/ticker/price"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            all_prices = {item["symbol"]: float(item["price"]) for item in response.json()}
            for sym in symbols:
                # Normalize symbol (remove /)
                clean_sym = sym.replace("/", "")
                if clean_sym in all_prices:
                    prices[clean_sym] = all_prices[clean_sym]
                # Try USDT variant if USDC
                elif clean_sym.replace("USDC", "USDT") in all_prices:
                    prices[clean_sym] = all_prices[clean_sym.replace("USDC", "USDT")]
    except Exception as e:
        print(f"  Warning: Could not fetch prices: {e}")
    return prices


def load_crypto9_positions() -> list:
    """Load Crypto9's open positions from local tracking file."""
    try:
        if os.path.exists(CRYPTO9_POSITIONS_FILE):
            with open(CRYPTO9_POSITIONS_FILE, "r") as f:
                positions = json.load(f)
                print(f"  Loaded {len(positions)} Crypto9 positions from local file")
                return positions
    except Exception as e:
        print(f"Error loading Crypto9 positions: {e}")
    return []


def load_crypto9_closed_trades() -> list:
    """Load Crypto9's closed trades from local tracking file."""
    try:
        if os.path.exists(CRYPTO9_CLOSED_TRADES_FILE):
            with open(CRYPTO9_CLOSED_TRADES_FILE, "r") as f:
                trades = json.load(f)
                # Handle double-encoded JSON (string instead of list)
                if isinstance(trades, str):
                    try:
                        trades = json.loads(trades)
                    except:
                        trades = []
                if not isinstance(trades, list):
                    trades = []
                print(f"  Loaded {len(trades)} Crypto9 closed trades from local file")
                return trades
    except Exception as e:
        print(f"Error loading Crypto9 closed trades: {e}")
    return []


def load_paper_trading_state() -> dict:
    """Load paper trading state (capital, positions) from local file."""
    try:
        if os.path.exists(PAPER_TRADING_STATE_FILE):
            with open(PAPER_TRADING_STATE_FILE, "r") as f:
                state = json.load(f)
                print(f"  Loaded paper trading state: {state.get('total_capital', 0):.2f} USDT capital")
                return state
    except Exception as e:
        print(f"Error loading paper trading state: {e}")
    return {"total_capital": 0, "positions": [], "symbol_trade_counts": {}}


def load_simulation_open_positions() -> list:
    """Load open positions from simulation (paper_trading_actual_trades.json)."""
    try:
        if os.path.exists(PAPER_TRADING_OPEN_POSITIONS):
            with open(PAPER_TRADING_OPEN_POSITIONS, "r") as f:
                positions = json.load(f)
                print(f"  Loaded {len(positions)} simulation open positions")
                return positions
    except Exception as e:
        print(f"Error loading simulation open positions: {e}")
    return []


def load_simulation_trades(days_back: int = None) -> list:
    """Load trades from simulation log.

    Args:
        days_back: If None, load ALL trades (recommended for stable history).
                   If set, filter to last N days.
    """
    try:
        if os.path.exists(PAPER_TRADING_SIMULATION_LOG):
            with open(PAPER_TRADING_SIMULATION_LOG, "r") as f:
                all_trades = json.load(f)

                # If no filter, return all trades (stable history)
                if days_back is None:
                    print(f"  Loaded {len(all_trades)} simulation trades (full history)")
                    return all_trades

                # Filter for recent trades
                cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days_back)
                recent_trades = []
                for trade in all_trades:
                    exit_time_str = trade.get("exit_time") or trade.get("ExitZeit")
                    if exit_time_str:
                        try:
                            exit_time = pd.to_datetime(exit_time_str)
                            if exit_time.tzinfo is None:
                                exit_time = exit_time.tz_localize('Europe/Berlin')
                            if exit_time >= cutoff:
                                recent_trades.append(trade)
                        except:
                            pass
                print(f"  Loaded {len(recent_trades)} simulation trades from last {days_back} days (of {len(all_trades)} total)")
                return recent_trades
    except Exception as e:
        print(f"Error loading simulation trades: {e}")
    return []


# ============================================================================
# TRADE MATCHING
# ============================================================================

def match_spot_trades(orders: list, symbol: str) -> list:
    """Match BUY/SELL spot orders into round-trip trades (LONG only)."""
    filled = [o for o in orders if o.get("status") == "FILLED"]
    filled.sort(key=lambda x: x.get("time", 0))

    matched = []
    position_qty = 0.0
    position_cost = 0.0
    entry_time = None

    for o in filled:
        side = o.get("side")
        qty = float(o.get("executedQty", 0))
        quote = float(o.get("cummulativeQuoteQty", 0))
        ts = o.get("time", 0)
        order_time = datetime.fromtimestamp(ts/1000, tz=timezone.utc) if ts else None

        if position_qty == 0 and side == "BUY":
            # Opening long position
            position_qty = qty
            position_cost = quote
            entry_time = order_time
        elif position_qty > 0 and side == "SELL":
            # Closing long position
            pnl = quote - position_cost
            matched.append({
                "symbol": symbol,
                "direction": "LONG",
                "source": "SPOT",
                "entry_time": entry_time,
                "exit_time": order_time,
                "entry_value": position_cost,
                "exit_value": quote,
                "qty": position_qty,
                "pnl": pnl,
                "pnl_pct": (pnl / position_cost * 100) if position_cost > 0 else 0
            })
            position_qty = 0
            position_cost = 0
            entry_time = None
        elif position_qty > 0 and side == "BUY":
            # Adding to long
            position_qty += qty
            position_cost += quote

    return matched


def match_futures_trades(trades: list, symbol: str) -> list:
    """Match futures trades into round-trips (LONG and SHORT)."""
    if not trades:
        return []

    trades = sorted(trades, key=lambda x: x.get("time", 0))

    matched = []
    position_qty = 0.0
    position_cost = 0.0
    entry_time = None
    entry_side = None  # "LONG" or "SHORT"

    for t in trades:
        side = t.get("side")  # BUY or SELL
        qty = float(t.get("qty", 0))
        quote = float(t.get("quoteQty", 0))
        ts = t.get("time", 0)
        trade_time = datetime.fromtimestamp(ts/1000, tz=timezone.utc) if ts else None

        if position_qty == 0:
            # Opening new position
            position_qty = qty
            position_cost = quote
            entry_time = trade_time
            entry_side = "LONG" if side == "BUY" else "SHORT"
        elif entry_side == "LONG" and side == "SELL":
            # Closing long
            pnl = quote - position_cost
            matched.append({
                "symbol": symbol,
                "direction": "LONG",
                "source": "FUTURES",
                "entry_time": entry_time,
                "exit_time": trade_time,
                "entry_value": position_cost,
                "exit_value": quote,
                "qty": position_qty,
                "pnl": pnl,
                "pnl_pct": (pnl / position_cost * 100) if position_cost > 0 else 0
            })
            position_qty = 0
            position_cost = 0
            entry_time = None
            entry_side = None
        elif entry_side == "SHORT" and side == "BUY":
            # Closing short
            pnl = position_cost - quote
            matched.append({
                "symbol": symbol,
                "direction": "SHORT",
                "source": "FUTURES",
                "entry_time": entry_time,
                "exit_time": trade_time,
                "entry_value": position_cost,
                "exit_value": quote,
                "qty": position_qty,
                "pnl": pnl,
                "pnl_pct": (pnl / position_cost * 100) if position_cost > 0 else 0
            })
            position_qty = 0
            position_cost = 0
            entry_time = None
            entry_side = None
        else:
            # Adding to position
            position_qty += qty
            position_cost += quote

    return matched


# ============================================================================
# VARIABLE STAKE RECALCULATION
# ============================================================================

def recalculate_trades_with_variable_stake(
    trades: list,
    start_capital: float = START_TOTAL_CAPITAL,
    max_positions: int = 10,
    filter_start_date: str = None,
) -> tuple:
    """
    Filter trades by start date and recalculate with variable stake.

    Args:
        trades: List of trade dicts
        start_capital: Starting capital (default 16500)
        max_positions: Max positions for stake calculation (default 10)
        filter_start_date: Filter trades from this date (format: "2025-12-01")

    Returns:
        (filtered_trades, final_capital, filtered_count)
    """
    if not trades:
        return [], start_capital, 0

    # Helper to parse entry time (handles Unix timestamps and ISO strings)
    def parse_timestamp(val):
        """Parse timestamp from various formats including Unix ms/s."""
        if not val:
            return pd.Timestamp.min
        # Handle Unix timestamps (int or numeric string)
        try:
            if isinstance(val, (int, float)):
                ts = int(val)
                if ts > 1e12:  # milliseconds
                    return pd.Timestamp(ts, unit='ms', tz='UTC').tz_convert('Europe/Berlin').tz_localize(None)
                else:  # seconds
                    return pd.Timestamp(ts, unit='s', tz='UTC').tz_convert('Europe/Berlin').tz_localize(None)
            if isinstance(val, str) and val.isdigit():
                ts = int(val)
                if ts > 1e12:  # milliseconds
                    return pd.Timestamp(ts, unit='ms', tz='UTC').tz_convert('Europe/Berlin').tz_localize(None)
                else:  # seconds
                    return pd.Timestamp(ts, unit='s', tz='UTC').tz_convert('Europe/Berlin').tz_localize(None)
        except:
            pass
        # Try normal parsing
        try:
            ts = pd.to_datetime(val)
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            return ts
        except:
            return pd.Timestamp.min

    # Debug: show sample of entry times
    if trades:
        sample_times = [t.get("entry_time") or t.get("Zeit") for t in trades[:3]]
        print(f"  [Debug] Sample entry times: {sample_times}")
        # Also show parsed values
        sample_parsed = [parse_timestamp(t.get("entry_time") or t.get("Zeit")) for t in trades[:3]]
        print(f"  [Debug] Parsed as: {sample_parsed}")

    # Sort trades by entry time (parse dates properly)
    sorted_trades = sorted(trades, key=lambda t: parse_timestamp(t.get("entry_time") or t.get("Zeit") or ""))

    # Filter by start date if provided
    if filter_start_date:
        filter_ts = pd.to_datetime(filter_start_date)
        filtered = []
        for t in sorted_trades:
            entry_time = t.get("entry_time") or t.get("Zeit") or ""
            entry_ts = parse_timestamp(entry_time)
            if entry_ts >= filter_ts:
                filtered.append(t)
        sorted_trades = filtered
        print(f"  [Filter] Trades from {filter_start_date}: {len(sorted_trades)}")

    if not sorted_trades:
        return [], start_capital, 0

    # Recalculate each trade with variable stake
    current_capital = start_capital
    recalculated = []

    for t in sorted_trades:
        t_copy = t.copy()
        entry_price = float(t.get("entry_price") or t.get("Entry") or 0)
        exit_price = float(t.get("exit_price") or t.get("ExitPreis") or 0)
        direction = str(t.get("direction") or "long").lower()

        if entry_price <= 0 or exit_price <= 0:
            recalculated.append(t_copy)
            continue

        # Variable stake based on current capital
        stake = current_capital / max_positions
        size_units = stake / entry_price

        # Fees: (entry + exit) * size_units * fee_rate
        fees = (entry_price + exit_price) * size_units * FEE_RATE

        # PnL calculation
        if direction == "long":
            pnl_raw = size_units * (exit_price - entry_price)
        else:
            pnl_raw = size_units * (entry_price - exit_price)

        net_pnl = pnl_raw - fees
        current_capital += net_pnl

        # Update trade with recalculated values
        t_copy["stake"] = stake
        t_copy["size_units"] = size_units
        t_copy["fees"] = fees
        t_copy["pnl"] = net_pnl
        t_copy["equity_after"] = current_capital

        recalculated.append(t_copy)

    return recalculated, current_capital, len(recalculated)


def recalculate_open_positions_with_variable_stake(
    positions: list,
    closed_trades: list,
    start_capital: float = START_TOTAL_CAPITAL,
    max_positions: int = 10,
) -> list:
    """
    Recalculate open positions with variable stake based on capital at entry time.
    """
    if not positions:
        return []

    # Calculate capital at each point in time from closed trades
    recalculated = []

    for pos in positions:
        pos_copy = pos.copy()
        entry_time = pos.get("entry_time") or ""
        entry_price = float(pos.get("entry_price") or 0)
        current_price = float(pos.get("current_price") or pos.get("last_price") or entry_price)
        direction = str(pos.get("direction") or "long").lower()

        if entry_price <= 0:
            recalculated.append(pos_copy)
            continue

        # Calculate capital at entry time (start + PnL from prior closed trades)
        capital_at_entry = start_capital
        for t in closed_trades:
            exit_time = t.get("exit_time") or t.get("ExitZeit") or ""
            if str(exit_time) < str(entry_time):
                capital_at_entry += float(t.get("pnl") or 0)

        # Recalculate stake and size
        new_stake = capital_at_entry / max_positions
        new_size_units = new_stake / entry_price

        # Recalculate unrealized PnL (without fees - only charged at close)
        if direction == "long":
            unrealized_pnl = new_size_units * (current_price - entry_price)
        else:
            unrealized_pnl = new_size_units * (entry_price - current_price)

        pos_copy["stake"] = new_stake
        pos_copy["size_units"] = new_size_units
        pos_copy["amount"] = new_size_units
        pos_copy["unrealized_pnl"] = unrealized_pnl
        if entry_price > 0:
            pct = ((current_price - entry_price) / entry_price) * 100 if direction == "long" else ((entry_price - current_price) / entry_price) * 100
            pos_copy["unrealized_pct"] = pct

        recalculated.append(pos_copy)

    return recalculated


# ============================================================================
# DASHBOARD GENERATION
# ============================================================================

def generate_dashboard(german_format=False, filter_start_date: str = None):
    """Generate HTML dashboard with Crypto9 local tracking - Long-only SPOT mode.

    Args:
        german_format: If True, use German number formatting (1.234,56)
                      If False, use English/US formatting (1,234.56)
        filter_start_date: Filter trades from this date (format: "2025-12-01")
                          If set, recalculates stakes from START_TOTAL_CAPITAL
    """
    fmt_suffix = "_de" if german_format else ""
    print(f"Fetching Crypto9 testnet data (Long-only SPOT mode, {'German' if german_format else 'English'} format)...")

    # ========== PAPER TRADING STATE (CAPITAL + POSITIONS) ==========
    print("  Loading paper trading state...")
    paper_state = load_paper_trading_state()
    paper_capital = paper_state.get("total_capital", 0)
    paper_positions = paper_state.get("positions", [])
    print(f"  Found {len(paper_positions)} positions in paper trading state")

    # ========== SIMULATION OPEN POSITIONS ==========
    print("  Loading simulation open positions...")
    simulation_positions = load_simulation_open_positions()

    # ========== LOCAL CRYPTO9 POSITIONS (fallback) ==========
    print("  Loading Crypto9 local positions...")
    crypto9_positions = load_crypto9_positions()

    # Combine all position sources (prefer paper state, add simulation, fallback to crypto9)
    # Use a dict to deduplicate by key
    positions_by_key = {}
    for pos in crypto9_positions:
        key = f"{pos.get('symbol', '')}|{pos.get('direction', '')}"
        positions_by_key[key] = pos
    for pos in simulation_positions:
        key = f"{pos.get('symbol', '')}|{pos.get('direction', '')}"
        if key not in positions_by_key:
            positions_by_key[key] = pos
    for pos in paper_positions:
        key = f"{pos.get('symbol', '')}|{pos.get('direction', '')}"
        positions_by_key[key] = pos  # Paper positions override

    source_positions = list(positions_by_key.values())

    # ========== SIMULATION TRADES (base history) ==========
    print("  Loading simulation trades...")
    simulation_trades = load_simulation_trades(days_back=None)

    # Find the latest exit time in simulation trades
    latest_sim_exit = None
    for trade in simulation_trades:
        exit_time_str = trade.get("exit_time") or trade.get("ExitZeit")
        if exit_time_str:
            try:
                exit_time = pd.to_datetime(exit_time_str)
                if latest_sim_exit is None or exit_time > latest_sim_exit:
                    latest_sim_exit = exit_time
            except:
                pass
    print(f"  Latest simulation exit: {latest_sim_exit}")

    # ========== CRYPTO9 CLOSED TRADES (recent live trades) ==========
    print("  Loading Crypto9 closed trades...")
    crypto9_closed_trades = load_crypto9_closed_trades()

    # Only use crypto9 trades that are NEWER than the latest simulation trade
    # This prevents "ghost trades" from the past while capturing recent live trades
    recent_crypto9_trades = []
    if latest_sim_exit and crypto9_closed_trades:
        for trade in crypto9_closed_trades:
            exit_time_str = trade.get("exit_time") or trade.get("ExitZeit")
            if exit_time_str:
                try:
                    exit_time = pd.to_datetime(exit_time_str)
                    if exit_time.tzinfo is None:
                        exit_time = exit_time.tz_localize('Europe/Berlin')
                    # Only include if AFTER the last simulation trade
                    if exit_time > latest_sim_exit:
                        recent_crypto9_trades.append(trade)
                except:
                    pass
        print(f"  Added {len(recent_crypto9_trades)} recent crypto9 trades (after {latest_sim_exit})")

    # Combine: simulation trades + recent crypto9 trades
    all_closed_trades_raw = simulation_trades + recent_crypto9_trades

    # Filter trades by TRADES_SINCE_DATE (paper trading start date)
    if TRADES_SINCE_DATE:
        filtered_trades = []
        for trade in all_closed_trades_raw:
            exit_time_str = trade.get("exit_time") or trade.get("ExitZeit")
            if exit_time_str:
                try:
                    exit_time = pd.to_datetime(exit_time_str)
                    if exit_time.tzinfo is None:
                        exit_time = exit_time.tz_localize('Europe/Berlin')
                    if exit_time >= TRADES_SINCE_DATE:
                        filtered_trades.append(trade)
                except:
                    filtered_trades.append(trade)  # Keep if can't parse
            else:
                filtered_trades.append(trade)  # Keep if no exit time
        print(f"  Filtered to {len(filtered_trades)} trades since {TRADES_SINCE_DATE.strftime('%Y-%m-%d')} (from {len(all_closed_trades_raw)} total)")
        all_closed_trades_raw = filtered_trades

    def normalize_time(t):
        """Normalize timestamp to comparable format (first 16 chars: YYYY-MM-DDTHH:MM)."""
        if not t:
            return ""
        t_str = str(t).replace(" ", "T")[:16]  # "2026-01-11 13:00" -> "2026-01-11T13:00"
        return t_str

    def normalize_symbol(s):
        """Normalize symbol to comparable format."""
        if not s:
            return ""
        return str(s).replace("/", "").replace("USDC", "USDT").upper()

    # Deduplicate trades by unique key (symbol + entry_time + exit_time + indicator)
    seen_trades = set()
    all_closed_trades = []
    for trade in all_closed_trades_raw:
        # Create unique key from trade details (normalized)
        trade_key = (
            normalize_symbol(trade.get("symbol", "")),
            normalize_time(trade.get("entry_time", "")),
            normalize_time(trade.get("exit_time", "")),
            str(trade.get("indicator", "")).lower(),
            str(trade.get("htf", "")).lower(),
        )
        if trade_key not in seen_trades:
            seen_trades.add(trade_key)
            all_closed_trades.append(trade)

    if len(all_closed_trades_raw) != len(all_closed_trades):
        print(f"  Removed {len(all_closed_trades_raw) - len(all_closed_trades)} duplicate trades")

    # ========== FILTER AND RECALCULATE WITH VARIABLE STAKE ==========
    final_capital = START_TOTAL_CAPITAL
    if filter_start_date:
        print(f"  Filtering and recalculating trades from {filter_start_date}...")
        all_closed_trades, final_capital, trade_count = recalculate_trades_with_variable_stake(
            all_closed_trades,
            start_capital=START_TOTAL_CAPITAL,
            max_positions=10,
            filter_start_date=filter_start_date,
        )
        total_pnl = sum(t.get("pnl", 0) for t in all_closed_trades)
        print(f"  Recalculated {trade_count} trades: PnL={total_pnl:,.2f}, Capital={final_capital:,.2f}")

    # ========== PROCESS POSITIONS (Long only) ==========
    # First collect all symbols to fetch prices
    position_symbols = []
    for pos in source_positions:
        symbol = pos.get("symbol", "").replace("/", "")
        if symbol:
            position_symbols.append(symbol)

    # Fetch current prices for all position symbols
    print("  Fetching current prices...")
    current_prices = fetch_current_prices(position_symbols)

    all_open_positions = []
    for pos in source_positions:
        symbol = pos.get("symbol", "").replace("/", "")
        direction = pos.get("direction", "long").upper()
        # Long-only mode: skip any short positions
        if direction == "SHORT":
            continue
        entry_price = pos.get("entry_price", 0) or pos.get("entry_price_live", 0)
        stake = pos.get("stake", 0)
        size_units = pos.get("size_units", 0)

        # Calculate unrealized PnL using current price
        current_price = current_prices.get(symbol, 0)
        # Calculate fees: (entry_price + current_price) * size_units * fee_rate
        fee_rate = 0.00075  # VIP Level 1
        fees = (entry_price + current_price) * size_units * fee_rate if entry_price and current_price and size_units else 0
        if current_price and entry_price and size_units:
            unrealized_pnl = (current_price - entry_price) * size_units - fees
        else:
            unrealized_pnl = pos.get("unrealized_pnl", 0)

        all_open_positions.append({
            "asset": symbol.replace("USDT", "").replace("USDC", "").replace("EUR", ""),
            "symbol": symbol,
            "amount": size_units,
            "source": "SPOT",
            "side": "LONG",
            "entry_price": entry_price,
            "current_price": current_price,
            "stake": stake,
            "fees": fees,
            "unrealized_pnl": unrealized_pnl,
            "entry_time": pos.get("entry_time", ""),
        })

    # Recalculate open positions with variable stake if filter is active
    if filter_start_date and all_open_positions:
        # Filter positions by entry time
        filtered_positions = [
            p for p in all_open_positions
            if str(p.get("entry_time", ""))[:10] >= filter_start_date
        ]
        # Recalculate with variable stake
        recalc_positions = recalculate_open_positions_with_variable_stake(
            filtered_positions, all_closed_trades, start_capital=START_TOTAL_CAPITAL, max_positions=10
        )
        all_open_positions = recalc_positions
        print(f"  Recalculated {len(all_open_positions)} open positions with variable stake")

    # ========== PROCESS ALL CLOSED TRADES (Long + Short) ==========
    all_trades_list = []
    for trade in all_closed_trades:
        direction = trade.get("direction", "long").upper()
        # Include both Long and Short trades for full profit
        stake = trade.get("stake", 0)
        pnl = trade.get("pnl", 0)
        # Calculate exit_value: stake + pnl
        exit_value = trade.get("exit_value") or (stake + pnl)
        # Calculate pnl_pct if not provided
        pnl_pct = trade.get("pnl_pct") or ((pnl / stake * 100) if stake else 0)
        # Get reason with multiple fallbacks
        reason = trade.get("reason") or trade.get("Reason") or trade.get("exit_reason") or "-"
        if not reason or reason == "":
            reason = "-"
        # Get or calculate fees
        entry_price_val = trade.get("entry_price", 0) or 0
        exit_price_val = trade.get("exit_price", 0) or 0
        amount_val = trade.get("size_units") or trade.get("amount") or (stake / entry_price_val if entry_price_val else 0)
        fees = trade.get("fees") or trade.get("Fees") or ((entry_price_val + exit_price_val) * amount_val * 0.00075 if entry_price_val and exit_price_val and amount_val else 0)
        # Convert Unix timestamps to ISO format for display
        def convert_timestamp(val):
            if not val:
                return val
            try:
                if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
                    ts = int(val)
                    if ts > 1e12:  # milliseconds
                        ts = ts / 1000
                    dt = datetime.fromtimestamp(ts)
                    return dt.strftime("%Y-%m-%dT%H:%M:%S")
            except:
                pass
            return val

        raw_entry = trade.get("entry_time") or trade.get("Zeit")
        raw_exit = trade.get("exit_time") or trade.get("ExitZeit") or trade.get("closed_at")

        trade_data = {
            "symbol": trade.get("symbol", "").replace("/", ""),
            "direction": direction,
            "source": "SPOT",
            "entry_time": convert_timestamp(raw_entry),
            "exit_time": convert_timestamp(raw_exit),
            "entry_price": entry_price_val,
            "exit_price": exit_price_val,
            "amount": amount_val,
            "entry_value": stake,
            "exit_value": exit_value,
            "fees": fees,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "indicator": trade.get("indicator", ""),
            "htf": trade.get("htf", ""),
            "reason": reason,
        }
        all_trades_list.append(trade_data)

    # Debug: count trades with/without reason
    with_reason = sum(1 for t in all_trades_list if t["reason"] != "-")
    print(f"  Processed {len(all_trades_list)} closed trades ({with_reason} with exit reason)")

    # Sort by exit time (most recent first)
    all_trades_list.sort(key=lambda t: t.get("exit_time") or "", reverse=True)

    total_volume = sum(t["entry_value"] for t in all_trades_list) if all_trades_list else 1
    total_realized_pnl = sum(t["pnl"] for t in all_trades_list)
    total_closed_trades = len(all_trades_list)
    long_pnl = total_realized_pnl
    long_wins = sum(1 for t in all_trades_list if t["pnl"] > 0)

    # DEBUG: Verify PnL calculation
    print(f"  DEBUG PnL: total_realized_pnl = ${total_realized_pnl:,.2f}")
    print(f"  DEBUG PnL: total_closed_trades = {total_closed_trades}")
    print(f"  DEBUG PnL: First 5 PnLs = {[round(t['pnl'], 2) for t in all_trades_list[:5]]}")
    print(f"  DEBUG PnL: Last 5 PnLs = {[round(t['pnl'], 2) for t in all_trades_list[-5:]]}")

    # Calculate total unrealized PnL from open positions
    total_unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in all_open_positions)

    # Calculate total equity: START_TOTAL_CAPITAL + realized PnL
    # This ensures dashboard shows correct value even if state file is out of sync
    total_usdt = START_TOTAL_CAPITAL + total_realized_pnl

    # ========== GENERATE HTML ==========
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="60">
    <title>Crypto9 Testnet Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        h2 {{ color: #aaa; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: #16213e; }}
        th, td {{ border: 1px solid #333; padding: 8px; text-align: right; }}
        th {{ background: #0f3460; color: #00d4ff; text-align: center; }}
        td:first-child {{ text-align: left; }}
        .positive {{ color: #00ff88; font-weight: bold; }}
        .negative {{ color: #ff4757; font-weight: bold; }}
        .summary-boxes {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }}
        .summary-box {{ background: #16213e; padding: 20px; border-radius: 10px; min-width: 140px; text-align: center; border: 1px solid #333; }}
        .summary-box h3 {{ margin: 0 0 10px 0; color: #888; font-size: 12px; text-transform: uppercase; }}
        .summary-box .value {{ font-size: 24px; font-weight: bold; color: #fff; }}
        .long-header {{ background: #1e5631 !important; }}
        .short-header {{ background: #5c1e1e !important; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .source-spot {{ color: #ffa502; }}
        .source-futures {{ color: #ff6b81; }}
        .badge {{ padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-left: 5px; }}
        .badge-spot {{ background: #ffa502; color: #000; }}
        .badge-futures {{ background: #ff6b81; color: #fff; }}
        .badge-long {{ background: #1e5631; color: #0f0; }}
        .badge-short {{ background: #5c1e1e; color: #f66; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Crypto9 Testnet Dashboard</h1>
    <p style="color:#666;">SPOT Only - Long-only Mode | Auto-refresh: 60s</p>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>Paper Trading Capital</h3>
            <div class="value">{fmt_de_currency(total_usdt) if german_format else f"${total_usdt:,.2f}"}</div>
        </div>
        <div class="summary-box">
            <h3>Open Positions</h3>
            <div class="value">{len(all_open_positions)}</div>
        </div>
        <div class="summary-box">
            <h3>Unrealized PnL</h3>
            <div class="value {'positive' if total_unrealized_pnl >= 0 else 'negative'}">{fmt_de_currency(total_unrealized_pnl) if german_format else f"${total_unrealized_pnl:,.2f}"}</div>
        </div>
        <div class="summary-box">
            <h3>Closed Trades</h3>
            <div class="value">{total_closed_trades}</div>
        </div>
        <div class="summary-box">
            <h3>Realized PnL</h3>
            <div class="value {'positive' if total_realized_pnl >= 0 else 'negative'}">{fmt_de_currency(total_realized_pnl) if german_format else f"${total_realized_pnl:,.2f}"}</div>
        </div>
        <div class="summary-box">
            <h3>Win Rate</h3>
            <div class="value">{fmt_de((long_wins / len(all_trades_list) * 100) if all_trades_list else 0, 1) if german_format else f"{(long_wins / len(all_trades_list) * 100) if all_trades_list else 0:.1f}"}%</div>
        </div>
    </div>

"""
    # Calculate open position PnL
    open_long_pnl = sum(p.get("unrealized_pnl", 0) for p in all_open_positions)

    def format_entry_time(t):
        """Format entry time for display as yyyy-mm-dd HH:MM."""
        if not t:
            return "-"
        t_str = str(t)
        # Handle ISO format: 2026-01-10T14:00:00+01:00 -> 2026-01-10 14:00
        if "T" in t_str:
            try:
                return t_str[:16].replace("T", " ")  # "2026-01-10 14:00"
            except:
                return t_str[:16]
        # Handle space format: 2026-01-10 14:00:00 -> 2026-01-10 14:00
        if " " in t_str:
            try:
                return t_str[:16]  # "2026-01-10 14:00"
            except:
                return t_str[:16]
        return t_str[:16]

    def position_table_rows(positions):
        if not positions:
            return "<tr><td colspan='9'>No positions</td></tr>\n"
        # Select formatting functions based on german_format flag
        fmt_num = fmt_de if german_format else fmt_en
        fmt_curr = fmt_de_currency if german_format else fmt_en_currency
        rows = ""
        for pos in positions:
            source_class = "badge-spot" if pos["source"] == "SPOT" else "badge-futures"
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", 0)
            stake = pos.get("stake", 0)
            fees = pos.get("fees", 0)
            # Format prices based on magnitude
            def fmt_price(p):
                if not p:
                    return "-"
                if p >= 1000:
                    return fmt_curr(p, 2)
                elif p >= 1:
                    return fmt_curr(p, 4)
                else:
                    return fmt_curr(p, 6)
            entry_str = fmt_price(entry_price)
            current_str = fmt_price(current_price)
            entry_time = format_entry_time(pos.get("entry_time", ""))
            upnl = pos.get("unrealized_pnl", 0)
            upnl_class = "positive" if upnl >= 0 else "negative"
            upnl_str = fmt_curr(upnl, 2) if upnl != 0 else "-"
            stake_str = fmt_curr(stake, 2) if stake else "-"
            fees_str = fmt_curr(fees, 2) if fees else "-"
            rows += f"""        <tr>
            <td><span class='badge {source_class}'>{pos['source']}</span></td>
            <td>{pos['asset']}</td>
            <td>{fmt_num(pos['amount'], 6)}</td>
            <td>{entry_time}</td>
            <td>{entry_str}</td>
            <td>{current_str}</td>
            <td>{stake_str}</td>
            <td>{fees_str}</td>
            <td class='{upnl_class}'>{upnl_str}</td>
        </tr>\n"""
        return rows

    # Open Positions table (Long only)
    html += f"""
    <h2>Open Positions ({len(all_open_positions)}, PnL: <span class="{'positive' if open_long_pnl >= 0 else 'negative'}">{fmt_de_currency(open_long_pnl) if german_format else f"${open_long_pnl:,.2f}"}</span>)</h2>
    <table>
        <tr class="long-header"><th>Source</th><th>Asset</th><th>Amount</th><th>Entry Time</th><th>Entry Price</th><th>Actual Price</th><th>Stake</th><th>Fees</th><th>Unrealized PnL</th></tr>
"""
    html += position_table_rows(all_open_positions)
    html += "    </table>\n"

    # Trade table helper
    def format_time(t):
        if not t:
            return "N/A"
        # Handle Unix timestamps (integer or numeric string like "1769526000000")
        try:
            if isinstance(t, (int, float)):
                ts = int(t)
                if ts > 1e12:  # milliseconds
                    ts = ts / 1000
                dt = datetime.fromtimestamp(ts)
                return dt.strftime("%Y-%m-%d %H:%M")
            if isinstance(t, str) and t.isdigit():
                ts = int(t)
                if ts > 1e12:  # milliseconds
                    ts = ts / 1000
                dt = datetime.fromtimestamp(ts)
                return dt.strftime("%Y-%m-%d %H:%M")
        except:
            pass
        if isinstance(t, str):
            try:
                return t[:16].replace("T", " ")  # "2026-01-10T12:00" -> "2026-01-10 12:00"
            except:
                return t
        return t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16]

    def trade_table_rows(trades):
        if not trades:
            return "<tr><td colspan='12'>No trades</td></tr>\n"
        # Select formatting functions based on german_format flag
        fmt_num = fmt_de if german_format else fmt_en
        fmt_curr = fmt_de_currency if german_format else fmt_en_currency
        rows = ""
        for t in trades:
            pnl_class = "positive" if t["pnl"] >= 0 else "negative"
            entry_str = format_time(t.get("entry_time"))
            exit_str = format_time(t.get("exit_time")) if t.get("exit_time") else "Open"
            indicator = t.get("indicator", "-")
            htf = t.get("htf", "-")
            reason = t.get("reason", "-")
            entry_price = t.get("entry_price", 0)
            exit_price = t.get("exit_price", 0)
            amount = t.get("amount", 0)
            stake = t.get("entry_value", 0)  # stake is stored as entry_value
            fees = t.get("fees", 0)
            # Shorten reason for display
            if reason and len(reason) > 25:
                reason = reason[:22] + "..."
            # Format prices based on magnitude
            def fmt_price(p):
                if p >= 1000:
                    return fmt_curr(p, 2)
                elif p >= 1:
                    return fmt_curr(p, 4)
                elif p >= 0.0001:
                    return fmt_curr(p, 6)
                else:
                    return fmt_curr(p, 8)  # For very small prices like LUNC
            # Format amount based on magnitude
            def fmt_amount(a):
                if a >= 1:
                    return fmt_num(a, 4)
                else:
                    return fmt_num(a, 6)
            stake_str = fmt_curr(stake, 2) if stake else "-"
            fees_str = fmt_curr(fees, 2) if fees else "-"
            pnl_str = fmt_curr(t['pnl'], 2)
            pnl_pct_str = fmt_num(t['pnl_pct'], 2)
            # Add + sign for positive pnl_pct
            if t['pnl_pct'] > 0:
                pnl_pct_str = "+" + pnl_pct_str
            rows += f"""        <tr>
            <td>{t['symbol']}</td>
            <td>{indicator}/{htf}</td>
            <td>{entry_str}</td>
            <td>{fmt_price(entry_price)}</td>
            <td>{exit_str}</td>
            <td>{fmt_price(exit_price)}</td>
            <td>{fmt_amount(amount)}</td>
            <td>{stake_str}</td>
            <td>{fees_str}</td>
            <td class="{pnl_class}">{pnl_str}</td>
            <td class="{pnl_class}">{pnl_pct_str}%</td>
            <td>{reason}</td>
        </tr>\n"""
        return rows

    # Closed Trades section (Long only)
    html += f"""
    <div class="section">
    <h2>Closed Trades ({len(all_trades_list)} trades, PnL: <span class="{'positive' if long_pnl >= 0 else 'negative'}">{fmt_de_currency(long_pnl) if german_format else f"${long_pnl:,.2f}"}</span>)</h2>
    <table>
        <tr class="long-header"><th>Symbol</th><th>Strategy</th><th>Entry Time</th><th>Entry Price</th><th>Exit Time</th><th>Exit Price</th><th>Amount</th><th>Stake</th><th>Fees</th><th>PnL</th><th>PnL %</th><th>Reason</th></tr>
"""
    html += trade_table_rows(all_trades_list)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate verification sums
    table_pnl_sum = sum(t["pnl"] for t in all_trades_list)

    html += f"""    </table>
    </div>

    <div class="section" style="background: #2d2d44; padding: 15px; border-radius: 5px; margin-top: 20px; border: 1px solid #444;">
        <h3 style="margin-top: 0; color: #ffa;">PnL Verification (Debug)</h3>
        <table style="width: auto; background: #1a1a2e;">
            <tr><td style="text-align: left;">Sum of all PnLs in table:</td><td style="text-align: right;">{fmt_de_currency(table_pnl_sum) if german_format else f"${table_pnl_sum:,.2f}"}</td></tr>
            <tr><td style="text-align: left;">Displayed Total (header):</td><td style="text-align: right;">{fmt_de_currency(total_realized_pnl) if german_format else f"${total_realized_pnl:,.2f}"}</td></tr>
            <tr><td style="text-align: left;">Difference:</td><td style="text-align: right; color: {'#ff4757' if abs(table_pnl_sum - total_realized_pnl) > 0.01 else '#00ff88'};">{fmt_de_currency(table_pnl_sum - total_realized_pnl) if german_format else f"${table_pnl_sum - total_realized_pnl:,.2f}"}</td></tr>
            <tr><td style="text-align: left;">Trade count:</td><td style="text-align: right;">{len(all_trades_list)}</td></tr>
        </table>
    </div>

    <p class="timestamp">Generated: {timestamp} | Paper Trading Mode (Long-only SPOT) | {'German' if german_format else 'English'} Number Format</p>
</div>
</body>
</html>"""

    # Write to file
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_filename = f"dashboard{fmt_suffix}.html"
    output_path = OUTPUT_DIR / output_filename
    output_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop mode")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    args = parser.parse_args()

    # PnL correction disabled - paper_trader.py already calculates correct PnL with lot sizes
    # correct_trades_pnl(PAPER_TRADING_SIMULATION_LOG)
    # correct_trades_pnl(CRYPTO9_CLOSED_TRADES_FILE)

    missing = []
    if not SPOT_API_KEY or not SPOT_API_SECRET:
        missing.append("BINANCE_API_KEY_SPOT / BINANCE_API_SECRET_SPOT (Spot)")
    if not FUTURES_API_KEY or not FUTURES_API_SECRET:
        missing.append("BINANCE_API_KEY_TEST_F / BINANCE_API_SECRET_TEST_F (Futures)")
    if missing:
        print("Warning: Missing API keys in .env:")
        for m in missing:
            print(f"  - {m}")

    if args.loop:
        print(f"Running dashboard loop (refresh every {args.interval}s). Press Ctrl+C to stop.")
        while True:
            try:
                # Generate both English and German dashboards
                path_en = generate_dashboard(german_format=False)
                path_de = generate_dashboard(german_format=True)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboards updated: {path_en}, {path_de}")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        # Generate both English and German dashboards
        path_en = generate_dashboard(german_format=False)
        path_de = generate_dashboard(german_format=True)
        print(f"Open English: start {path_en}")
        print(f"Open German:  start {path_de}")
