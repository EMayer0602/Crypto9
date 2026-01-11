#!/usr/bin/env python3
"""Generate HTML dashboard for Crypto9 Testnet trading (local tracking)."""

import os
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Local Crypto9 tracking files
CRYPTO9_POSITIONS_FILE = "crypto9_testnet_positions.json"
CRYPTO9_CLOSED_TRADES_FILE = "crypto9_testnet_closed_trades.json"
PAPER_TRADING_STATE_FILE = "paper_trading_state.json"

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
TRADES_SINCE_DATE = datetime(2026, 1, 2, tzinfo=timezone.utc)

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
# DASHBOARD GENERATION
# ============================================================================

def generate_dashboard():
    """Generate HTML dashboard with Crypto9 local tracking - Long-only SPOT mode."""
    print("Fetching Crypto9 testnet data (Long-only SPOT mode)...")

    # ========== PAPER TRADING STATE (CAPITAL + POSITIONS) ==========
    print("  Loading paper trading state...")
    paper_state = load_paper_trading_state()
    paper_capital = paper_state.get("total_capital", 0)
    paper_positions = paper_state.get("positions", [])
    print(f"  Found {len(paper_positions)} positions in paper trading state")

    # ========== LOCAL CRYPTO9 POSITIONS (fallback) ==========
    print("  Loading Crypto9 local positions...")
    crypto9_positions = load_crypto9_positions()

    # Use paper trading positions if available, otherwise fallback to crypto9 file
    source_positions = paper_positions if paper_positions else crypto9_positions

    # ========== LOCAL CRYPTO9 CLOSED TRADES ==========
    print("  Loading Crypto9 closed trades...")
    crypto9_closed_trades = load_crypto9_closed_trades()

    # ========== PROCESS POSITIONS (Long only) ==========
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

        all_open_positions.append({
            "asset": symbol.replace("USDT", "").replace("USDC", "").replace("EUR", ""),
            "symbol": symbol,
            "amount": size_units,
            "source": "SPOT",
            "side": "LONG",
            "entry_price": entry_price,
            "stake": stake,
            "unrealized_pnl": pos.get("unrealized_pnl", 0),
            "entry_time": pos.get("entry_time", ""),
        })

    # ========== PROCESS CRYPTO9 CLOSED TRADES (Long only) ==========
    long_trades = []
    for trade in crypto9_closed_trades:
        direction = trade.get("direction", "long").upper()
        # Long-only mode: skip short trades
        if direction == "SHORT":
            continue
        stake = trade.get("stake", 0)
        pnl = trade.get("pnl", 0)
        # Calculate exit_value: stake + pnl
        exit_value = trade.get("exit_value") or (stake + pnl)
        trade_data = {
            "symbol": trade.get("symbol", "").replace("/", ""),
            "direction": "LONG",
            "source": "SPOT",
            "entry_time": trade.get("entry_time"),
            "exit_time": trade.get("exit_time") or trade.get("closed_at"),
            "entry_value": stake,
            "exit_value": exit_value,
            "pnl": pnl,
            "pnl_pct": trade.get("pnl_pct", 0),
        }
        long_trades.append(trade_data)

    total_volume = sum(t["entry_value"] for t in long_trades) if long_trades else 1
    total_realized_pnl = sum(t["pnl"] for t in long_trades)
    total_closed_trades = len(long_trades)
    long_pnl = total_realized_pnl
    long_wins = sum(1 for t in long_trades if t["pnl"] > 0)

    # Calculate total unrealized PnL from open positions
    total_unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in all_open_positions)

    # Capital from paper trading state
    total_usdt = paper_capital

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
            <div class="value">${total_usdt:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>Open Positions</h3>
            <div class="value">{len(all_open_positions)}</div>
        </div>
        <div class="summary-box">
            <h3>Unrealized PnL</h3>
            <div class="value {'positive' if total_unrealized_pnl >= 0 else 'negative'}">${total_unrealized_pnl:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>Closed Trades</h3>
            <div class="value">{total_closed_trades}</div>
        </div>
        <div class="summary-box">
            <h3>Realized PnL</h3>
            <div class="value {'positive' if total_realized_pnl >= 0 else 'negative'}">${total_realized_pnl:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>Win Rate</h3>
            <div class="value">{(long_wins / len(long_trades) * 100) if long_trades else 0:.1f}%</div>
        </div>
    </div>

"""
    # Calculate open position PnL
    open_long_pnl = sum(p.get("unrealized_pnl", 0) for p in all_open_positions)

    def position_table_rows(positions):
        if not positions:
            return "<tr><td colspan='5'>No positions</td></tr>\n"
        rows = ""
        for pos in positions:
            source_class = "badge-spot" if pos["source"] == "SPOT" else "badge-futures"
            entry_price = pos.get("entry_price", "-")
            entry_str = f"${entry_price:,.2f}" if isinstance(entry_price, (int, float)) else entry_price
            upnl = pos.get("unrealized_pnl", 0)
            upnl_class = "positive" if upnl >= 0 else "negative"
            upnl_str = f"${upnl:,.2f}" if upnl != 0 else "-"
            rows += f"""        <tr>
            <td><span class='badge {source_class}'>{pos['source']}</span></td>
            <td>{pos['asset']}</td>
            <td>{pos['amount']:,.6f}</td>
            <td>{entry_str}</td>
            <td class='{upnl_class}'>{upnl_str}</td>
        </tr>\n"""
        return rows

    # Open Positions table (Long only)
    html += f"""
    <h2>Open Positions ({len(all_open_positions)}, PnL: <span class="{'positive' if open_long_pnl >= 0 else 'negative'}">${open_long_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="long-header"><th>Source</th><th>Asset</th><th>Amount</th><th>Entry Price</th><th>Unrealized PnL</th></tr>
"""
    html += position_table_rows(all_open_positions)
    html += "    </table>\n"

    # Trade table helper
    def format_time(t):
        if not t:
            return "N/A"
        if isinstance(t, str):
            try:
                return t[:16].replace("T", " ")  # "2026-01-10T12:00" -> "2026-01-10 12:00"
            except:
                return t
        return t.strftime("%m-%d %H:%M") if hasattr(t, "strftime") else str(t)

    def trade_table_rows(trades):
        if not trades:
            return "<tr><td colspan='9'>No trades</td></tr>\n"
        rows = ""
        for t in trades:
            pnl_class = "positive" if t["pnl"] >= 0 else "negative"
            entry_str = format_time(t.get("entry_time"))
            exit_str = format_time(t.get("exit_time")) if t.get("exit_time") else "Open"
            source_class = "badge-spot" if t["source"] == "SPOT" else "badge-futures"
            share = (t["entry_value"] / total_volume * 100) if total_volume > 0 else 0
            rows += f"""        <tr>
            <td><span class='badge {source_class}'>{t['source']}</span></td>
            <td>{t['symbol']}</td>
            <td>{entry_str}</td>
            <td>{exit_str}</td>
            <td>${t['entry_value']:,.2f}</td>
            <td>${t['exit_value']:,.2f}</td>
            <td class="{pnl_class}">${t['pnl']:,.2f}</td>
            <td class="{pnl_class}">{t['pnl_pct']:+.2f}%</td>
            <td>{share:.1f}%</td>
        </tr>\n"""
        return rows

    # Closed Trades section (Long only)
    html += f"""
    <div class="section">
    <h2>Closed Trades ({len(long_trades)} trades, PnL: <span class="{'positive' if long_pnl >= 0 else 'negative'}">${long_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="long-header"><th>Source</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>PnL</th><th>PnL %</th><th>Share</th></tr>
"""
    html += trade_table_rows(long_trades)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">Generated: {timestamp} | Paper Trading Mode (Long-only SPOT)</p>
</div>
</body>
</html>"""

    # Write to file
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    missing = []
    if not SPOT_API_KEY or not SPOT_API_SECRET:
        missing.append("BINANCE_API_KEY_SPOT / BINANCE_API_SECRET_SPOT (Spot)")
    if not FUTURES_API_KEY or not FUTURES_API_SECRET:
        missing.append("BINANCE_API_KEY_TEST_F / BINANCE_API_SECRET_TEST_F (Futures)")
    if missing:
        print("Warning: Missing API keys in .env:")
        for m in missing:
            print(f"  - {m}")
    path = generate_dashboard()
    print(f"Open with: start {path}")
