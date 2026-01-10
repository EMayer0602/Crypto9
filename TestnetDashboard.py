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

# Spot Testnet API keys
SPOT_API_KEY = os.getenv("BINANCE_API_KEY_TEST")
SPOT_API_SECRET = os.getenv("BINANCE_API_SECRET_TEST")

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
    """Generate HTML dashboard with Crypto9 local tracking + API balances."""
    print("Fetching Crypto9 testnet data...")

    # ========== LOCAL CRYPTO9 POSITIONS ==========
    print("  Loading Crypto9 local positions...")
    crypto9_positions = load_crypto9_positions()

    # ========== LOCAL CRYPTO9 CLOSED TRADES ==========
    print("  Loading Crypto9 closed trades...")
    crypto9_closed_trades = load_crypto9_closed_trades()

    # ========== API: BALANCES ONLY ==========
    print("  Fetching Spot balances (API)...")
    spot_balances = get_spot_balances()
    spot_base_balances = []
    for bal in spot_balances:
        asset = bal["asset"]
        free = float(bal["free"])
        locked = float(bal["locked"])
        total = free + locked
        if total > 0 and asset in BASE_CURRENCIES:
            spot_base_balances.append({"asset": asset, "amount": total, "source": "SPOT"})

    print("  Fetching Futures balance (API)...")
    futures_usdt = get_futures_balance()

    # ========== PROCESS CRYPTO9 POSITIONS ==========
    all_open_positions = []
    for pos in crypto9_positions:
        symbol = pos.get("symbol", "").replace("/", "")
        direction = pos.get("direction", "long").upper()
        source = "FUTURES" if direction == "SHORT" else "SPOT"
        entry_price = pos.get("entry_price", 0) or pos.get("entry_price_live", 0)
        stake = pos.get("stake", 0)
        size_units = pos.get("size_units", 0)

        all_open_positions.append({
            "asset": symbol.replace("USDT", "").replace("EUR", ""),
            "symbol": symbol,
            "amount": size_units,
            "source": source,
            "side": direction,
            "entry_price": entry_price,
            "stake": stake,
            "unrealized_pnl": pos.get("unrealized_pnl", 0),
            "entry_time": pos.get("entry_time", ""),
        })

    # ========== PROCESS CRYPTO9 CLOSED TRADES ==========
    long_trades = []
    short_trades = []
    for trade in crypto9_closed_trades:
        direction = trade.get("direction", "long").upper()
        source = "FUTURES" if direction == "SHORT" else "SPOT"
        trade_data = {
            "symbol": trade.get("symbol", "").replace("/", ""),
            "direction": direction,
            "source": source,
            "entry_time": trade.get("entry_time"),
            "exit_time": trade.get("exit_time") or trade.get("closed_at"),
            "entry_value": trade.get("stake", 0),
            "exit_value": trade.get("exit_value", 0),
            "pnl": trade.get("pnl", 0),
            "pnl_pct": trade.get("pnl_pct", 0),
        }
        if direction == "LONG":
            long_trades.append(trade_data)
        else:
            short_trades.append(trade_data)

    all_matched_trades = long_trades + short_trades
    total_volume = sum(t["entry_value"] for t in all_matched_trades) if all_matched_trades else 1
    total_realized_pnl = sum(t["pnl"] for t in all_matched_trades)
    total_closed_trades = len(all_matched_trades)
    long_pnl = sum(t["pnl"] for t in long_trades)
    short_pnl = sum(t["pnl"] for t in short_trades)
    long_wins = sum(1 for t in long_trades if t["pnl"] > 0)
    short_wins = sum(1 for t in short_trades if t["pnl"] > 0)

    # Calculate total unrealized PnL from Crypto9 positions
    total_unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in all_open_positions)

    # Total USDT (Spot + Futures)
    spot_usdt = sum(b["amount"] for b in spot_base_balances if b["asset"] == "USDT")
    total_usdt = spot_usdt + futures_usdt

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
    <p style="color:#666;">Spot (Longs) + Futures (Shorts) | Auto-refresh: 60s</p>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>Total USDT</h3>
            <div class="value">${total_usdt:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>Spot USDT</h3>
            <div class="value">${spot_usdt:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>Futures USDT</h3>
            <div class="value">${futures_usdt:,.2f}</div>
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
            <h3>Long ({len(long_trades)})</h3>
            <div class="value {'positive' if long_pnl >= 0 else 'negative'}">${long_pnl:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>Short ({len(short_trades)})</h3>
            <div class="value {'positive' if short_pnl >= 0 else 'negative'}">${short_pnl:,.2f}</div>
        </div>
    </div>

    <h2>Base Currency Balances</h2>
    <table>
        <tr><th>Source</th><th>Currency</th><th>Amount</th></tr>
"""
    # Add Spot base balances
    for bal in sorted(spot_base_balances, key=lambda x: x["amount"], reverse=True):
        html += f"        <tr><td><span class='badge badge-spot'>SPOT</span></td><td>{bal['asset']}</td><td>{bal['amount']:,.2f}</td></tr>\n"
    # Add Futures USDT balance
    if futures_usdt > 0:
        html += f"        <tr><td><span class='badge badge-futures'>FUTURES</span></td><td>USDT</td><td>{futures_usdt:,.2f}</td></tr>\n"
    if not spot_base_balances and futures_usdt == 0:
        html += "        <tr><td colspan='3'>No balances</td></tr>\n"

    html += """    </table>

    <h2>Open Positions</h2>
    <table>
        <tr><th>Source</th><th>Asset</th><th>Side</th><th>Amount</th><th>Entry Price</th><th>Unrealized PnL</th></tr>
"""
    if all_open_positions:
        for pos in all_open_positions:
            source_class = "badge-spot" if pos["source"] == "SPOT" else "badge-futures"
            side_class = "badge-long" if pos["side"] == "LONG" else "badge-short"
            entry_price = pos.get("entry_price", "-")
            entry_str = f"${entry_price:,.2f}" if isinstance(entry_price, (int, float)) else entry_price
            upnl = pos.get("unrealized_pnl", 0)
            upnl_class = "positive" if upnl >= 0 else "negative"
            upnl_str = f"${upnl:,.2f}" if upnl != 0 else "-"
            html += f"""        <tr>
            <td><span class='badge {source_class}'>{pos['source']}</span></td>
            <td>{pos['asset']}</td>
            <td><span class='badge {side_class}'>{pos['side']}</span></td>
            <td>{pos['amount']:,.6f}</td>
            <td>{entry_str}</td>
            <td class='{upnl_class}'>{upnl_str}</td>
        </tr>\n"""
    else:
        html += "        <tr><td colspan='6'>No open positions</td></tr>\n"

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

    # Long Trades section
    html += f"""    </table>

    <div class="section">
    <h2>Long Trades ({len(long_trades)} closed, PnL: <span class="{'positive' if long_pnl >= 0 else 'negative'}">${long_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="long-header"><th>Source</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>PnL</th><th>PnL %</th><th>Share</th></tr>
"""
    html += trade_table_rows(long_trades)

    # Short Trades section
    html += f"""    </table>
    </div>

    <div class="section">
    <h2>Short Trades ({len(short_trades)} closed, PnL: <span class="{'positive' if short_pnl >= 0 else 'negative'}">${short_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="short-header"><th>Source</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>PnL</th><th>PnL %</th><th>Share</th></tr>
"""
    html += trade_table_rows(short_trades)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">Generated: {timestamp} | Spot: testnet.binance.vision | Futures: testnet.binancefuture.com</p>
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
        missing.append("BINANCE_API_KEY_TEST / BINANCE_API_SECRET_TEST (Spot)")
    if not FUTURES_API_KEY or not FUTURES_API_SECRET:
        missing.append("BINANCE_API_KEY_TEST_F / BINANCE_API_SECRET_TEST_F (Futures)")
    if missing:
        print("Warning: Missing API keys in .env:")
        for m in missing:
            print(f"  - {m}")
    path = generate_dashboard()
    print(f"Open with: start {path}")
