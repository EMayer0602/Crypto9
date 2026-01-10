#!/usr/bin/env python3
"""Generate HTML dashboard for Binance Testnet trading (Spot + Futures)."""

import os
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY_TEST")
API_SECRET = os.getenv("BINANCE_API_SECRET_TEST")

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
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
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
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
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
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
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
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
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
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
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
    """Generate HTML dashboard with Spot + Futures data."""
    print("Fetching testnet data...")

    # ========== SPOT DATA ==========
    print("  Fetching Spot balances...")
    spot_balances = get_spot_balances()

    spot_base_balances = []
    spot_positions = []
    for bal in spot_balances:
        asset = bal["asset"]
        free = float(bal["free"])
        locked = float(bal["locked"])
        total = free + locked
        if total > 0:
            if asset in BASE_CURRENCIES:
                spot_base_balances.append({"asset": asset, "amount": total, "source": "SPOT"})
            elif asset in RELEVANT_ASSETS:
                spot_positions.append({"asset": asset, "amount": total, "source": "SPOT", "side": "LONG"})

    # Spot trades
    print("  Fetching Spot orders...")
    spot_matched_trades = []
    for sym in SYMBOLS:
        orders = get_spot_orders(sym)
        if orders:
            if TRADES_SINCE_DATE:
                orders = [o for o in orders if o.get("time", 0) and
                         datetime.fromtimestamp(o["time"]/1000, tz=timezone.utc) >= TRADES_SINCE_DATE]
            matched = match_spot_trades(orders, sym)
            spot_matched_trades.extend(matched)

    # ========== FUTURES DATA ==========
    print("  Fetching Futures balance...")
    futures_usdt = get_futures_balance()

    print("  Fetching Futures positions...")
    futures_positions = get_futures_positions()

    print("  Fetching Futures trades...")
    futures_matched_trades = []
    for sym in SYMBOLS:
        trades = get_futures_trades(sym)
        if trades:
            if TRADES_SINCE_DATE:
                trades = [t for t in trades if t.get("time", 0) and
                         datetime.fromtimestamp(t["time"]/1000, tz=timezone.utc) >= TRADES_SINCE_DATE]
            matched = match_futures_trades(trades, sym)
            futures_matched_trades.extend(matched)

    # ========== COMBINE DATA ==========
    all_matched_trades = spot_matched_trades + futures_matched_trades
    all_matched_trades.sort(key=lambda x: x["exit_time"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    long_trades = [t for t in all_matched_trades if t["direction"] == "LONG"]
    short_trades = [t for t in all_matched_trades if t["direction"] == "SHORT"]

    total_volume = sum(t["entry_value"] for t in all_matched_trades) if all_matched_trades else 1
    total_realized_pnl = sum(t["pnl"] for t in all_matched_trades)
    total_closed_trades = len(all_matched_trades)
    long_pnl = sum(t["pnl"] for t in long_trades)
    short_pnl = sum(t["pnl"] for t in short_trades)
    long_wins = sum(1 for t in long_trades if t["pnl"] > 0)
    short_wins = sum(1 for t in short_trades if t["pnl"] > 0)

    # Combined open positions
    all_open_positions = spot_positions.copy()
    for fp in futures_positions:
        all_open_positions.append({
            "asset": fp["symbol"].replace("USDT", ""),
            "amount": fp["amount"],
            "source": "FUTURES",
            "side": fp["side"],
            "entry_price": fp["entry_price"],
            "unrealized_pnl": fp["unrealized_pnl"]
        })

    # Calculate total unrealized PnL
    total_unrealized_pnl = sum(fp["unrealized_pnl"] for fp in futures_positions)

    # Total USDT (Spot + Futures)
    spot_usdt = sum(b["amount"] for b in spot_base_balances if b["asset"] == "USDT")
    total_usdt = spot_usdt + futures_usdt

    # ========== GENERATE HTML ==========
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="60">
    <title>Crypto9 Testnet Dashboard (Spot + Futures)</title>
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
    def trade_table_rows(trades):
        if not trades:
            return "<tr><td colspan='9'>No trades</td></tr>\n"
        rows = ""
        for t in trades:
            pnl_class = "positive" if t["pnl"] >= 0 else "negative"
            entry_str = t["entry_time"].strftime("%m-%d %H:%M") if t["entry_time"] else "N/A"
            exit_str = t["exit_time"].strftime("%m-%d %H:%M") if t["exit_time"] else "Open"
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
    if not API_KEY or not API_SECRET:
        print("Error: BINANCE_API_KEY_TEST or BINANCE_API_SECRET_TEST not set in .env")
    else:
        path = generate_dashboard()
        print(f"Open with: start {path}")
