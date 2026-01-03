#!/usr/bin/env python3
"""Generate HTML dashboard for Binance Testnet trading."""

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
BASE_URL = "https://testnet.binance.vision"
RECV_WINDOW_MS = 5_000

# Trading symbols - Testnet only supports USDT pairs
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LINKUSDT", "SOLUSDT", "BNBUSDT",
           "SUIUSDT", "ZECUSDT", "LUNCUSDT", "TNSRUSDT"]

# Base currencies (don't count as positions)
BASE_CURRENCIES = {"USDT", "USDC", "BUSD", "BTC", "TUSD"}

# Relevant trading assets (filter out testnet junk)
RELEVANT_ASSETS = {"BTC", "ETH", "SOL", "XRP", "LINK", "BNB", "SUI", "ZEC", "LUNC", "TNSR"}

# Only show trades from this date onwards (set to None to show all)
TRADES_SINCE_DATE = datetime(2026, 1, 2, tzinfo=timezone.utc)  # Today

OUTPUT_DIR = Path("report_testnet")


def sign_request(params: dict, secret: str) -> str:
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + "&signature=" + signature


def get_account_balances() -> list:
    """Get all account balances."""
    endpoint = "/api/v3/account"
    url = BASE_URL + endpoint
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
        print(f"Error fetching balances: {e}")
        return []


def get_all_orders(symbol: str) -> list:
    """Get all orders for a symbol."""
    endpoint = "/api/v3/allOrders"
    url = BASE_URL + endpoint
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


def calculate_trade_stats(orders: list) -> dict:
    """Calculate trading statistics from orders."""
    if not orders:
        return {"trades": 0, "buys": 0, "sells": 0, "volume": 0, "pnl": 0}

    buys = []
    sells = []

    for o in orders:
        if o.get("status") != "FILLED":
            continue
        side = o.get("side")
        qty = float(o.get("executedQty", 0))
        quote = float(o.get("cummulativeQuoteQty", 0))

        if side == "BUY":
            buys.append({"qty": qty, "quote": quote})
        elif side == "SELL":
            sells.append({"qty": qty, "quote": quote})

    total_buy_quote = sum(b["quote"] for b in buys)
    total_sell_quote = sum(s["quote"] for s in sells)

    return {
        "trades": len(buys) + len(sells),
        "buys": len(buys),
        "sells": len(sells),
        "buy_volume": total_buy_quote,
        "sell_volume": total_sell_quote,
        "realized_pnl": total_sell_quote - total_buy_quote if sells else 0
    }


def match_trades(orders: list, symbol: str) -> list:
    """Match BUY/SELL orders into round-trip trades.

    Long trade: BUY entry -> SELL exit
    Short trade: SELL entry -> BUY exit (on margin/futures, rare on spot)
    """
    filled = [o for o in orders if o.get("status") == "FILLED"]
    filled.sort(key=lambda x: x.get("time", 0))

    matched = []
    position_qty = 0.0
    position_cost = 0.0
    entry_time = None
    entry_side = None

    for o in filled:
        side = o.get("side")
        qty = float(o.get("executedQty", 0))
        quote = float(o.get("cummulativeQuoteQty", 0))
        ts = o.get("time", 0)
        order_time = datetime.fromtimestamp(ts/1000, tz=timezone.utc) if ts else None

        if position_qty == 0:
            # Opening new position
            position_qty = qty
            position_cost = quote
            entry_time = order_time
            entry_side = side
        elif entry_side == "BUY" and side == "SELL":
            # Closing long position
            pnl = quote - position_cost
            matched.append({
                "symbol": symbol,
                "direction": "LONG",
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
            entry_side = None
        elif entry_side == "SELL" and side == "BUY":
            # Closing short position
            pnl = position_cost - quote
            matched.append({
                "symbol": symbol,
                "direction": "SHORT",
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
            entry_side = None
        else:
            # Adding to position (same side)
            position_qty += qty
            position_cost += quote

    return matched


def generate_dashboard():
    """Generate HTML dashboard."""
    print("Fetching testnet data...")

    # Get balances
    balances = get_account_balances()

    # Filter relevant balances (ignore testnet junk tokens)
    open_positions = []
    base_balances = []
    for bal in balances:
        asset = bal["asset"]
        free = float(bal["free"])
        locked = float(bal["locked"])
        total = free + locked
        if total > 0:
            if asset in BASE_CURRENCIES:
                base_balances.append({"asset": asset, "amount": total})
            elif asset in RELEVANT_ASSETS:
                open_positions.append({"asset": asset, "amount": total})

    # Get order history for each symbol (filtered by date)
    all_matched_trades = []
    symbol_stats = {}
    for sym in SYMBOLS:
        orders = get_all_orders(sym)
        if orders:
            # Filter orders by date if TRADES_SINCE_DATE is set
            if TRADES_SINCE_DATE:
                filtered_orders = []
                for o in orders:
                    ts = o.get("time", 0)
                    if ts:
                        order_time = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                        if order_time >= TRADES_SINCE_DATE:
                            filtered_orders.append(o)
                orders = filtered_orders

            stats = calculate_trade_stats(orders)
            if stats["trades"] > 0:
                symbol_stats[sym] = stats

            # Match trades into round-trips
            matched = match_trades(orders, sym)
            all_matched_trades.extend(matched)

    # Sort matched trades by exit time (most recent first)
    all_matched_trades.sort(key=lambda x: x["exit_time"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    # Separate long and short trades
    long_trades = [t for t in all_matched_trades if t["direction"] == "LONG"]
    short_trades = [t for t in all_matched_trades if t["direction"] == "SHORT"]

    # Calculate total volume for share calculation
    total_volume = sum(t["entry_value"] for t in all_matched_trades) if all_matched_trades else 1

    # Calculate totals
    total_realized_pnl = sum(t["pnl"] for t in all_matched_trades)
    total_closed_trades = len(all_matched_trades)
    long_pnl = sum(t["pnl"] for t in long_trades)
    short_pnl = sum(t["pnl"] for t in short_trades)
    long_wins = sum(1 for t in long_trades if t["pnl"] > 0)
    short_wins = sum(1 for t in short_trades if t["pnl"] > 0)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Testnet Trading Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background: #007bff; color: white; text-align: center; }}
        td:first-child {{ text-align: left; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
        .summary-box {{ display: inline-block; background: white; padding: 20px; margin: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 150px; text-align: center; }}
        .summary-box h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .summary-box .value {{ font-size: 24px; font-weight: bold; }}
        .long-header {{ background: #28a745 !important; }}
        .short-header {{ background: #dc3545 !important; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .section {{ margin-bottom: 30px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Binance Testnet Dashboard</h1>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>Closed Trades</h3>
            <div class="value">{total_closed_trades}</div>
        </div>
        <div class="summary-box">
            <h3>Realized PnL</h3>
            <div class="value {'positive' if total_realized_pnl >= 0 else 'negative'}">{total_realized_pnl:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>Long Trades</h3>
            <div class="value">{len(long_trades)} ({long_wins}W)</div>
        </div>
        <div class="summary-box">
            <h3>Short Trades</h3>
            <div class="value">{len(short_trades)} ({short_wins}W)</div>
        </div>
        <div class="summary-box">
            <h3>Open Positions</h3>
            <div class="value">{len(open_positions)}</div>
        </div>
    </div>

    <h2>Base Currency Balances</h2>
    <table>
        <tr><th>Currency</th><th>Amount</th></tr>
"""
    for bal in sorted(base_balances, key=lambda x: x["amount"], reverse=True):
        html += f"        <tr><td>{bal['asset']}</td><td>{bal['amount']:,.2f}</td></tr>\n"

    html += """    </table>

    <h2>Open Positions</h2>
    <table>
        <tr><th>Asset</th><th>Amount</th></tr>
"""
    if open_positions:
        for pos in sorted(open_positions, key=lambda x: x["amount"], reverse=True)[:20]:
            html += f"        <tr><td>{pos['asset']}</td><td>{pos['amount']:,.4f}</td></tr>\n"
    else:
        html += "        <tr><td colspan='2'>No open positions</td></tr>\n"

    # Helper function to generate trade table rows
    def trade_table_rows(trades, header_class=""):
        if not trades:
            return "<tr><td colspan='8'>No trades</td></tr>\n"
        rows = ""
        for t in trades:
            pnl_class = "positive" if t["pnl"] >= 0 else "negative"
            entry_str = t["entry_time"].strftime("%Y-%m-%d %H:%M") if t["entry_time"] else "N/A"
            exit_str = t["exit_time"].strftime("%Y-%m-%d %H:%M") if t["exit_time"] else "N/A"
            share = (t["entry_value"] / total_volume * 100) if total_volume > 0 else 0
            rows += f"""        <tr>
            <td>{t['symbol']}</td>
            <td>{entry_str}</td>
            <td>{exit_str}</td>
            <td>{t['entry_value']:,.2f}</td>
            <td>{t['exit_value']:,.2f}</td>
            <td class="{pnl_class}">{t['pnl']:,.2f}</td>
            <td class="{pnl_class}">{t['pnl_pct']:+.2f}%</td>
            <td>{share:.1f}%</td>
        </tr>\n"""
        return rows

    # Long Trades section
    html += f"""    </table>

    <div class="section">
    <h2>Long Trades ({len(long_trades)} closed, PnL: <span class="{'positive' if long_pnl >= 0 else 'negative'}">{long_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="long-header"><th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry Value</th><th>Exit Value</th><th>PnL</th><th>PnL %</th><th>Share</th></tr>
"""
    html += trade_table_rows(long_trades)

    # Short Trades section
    html += f"""    </table>
    </div>

    <div class="section">
    <h2>Short Trades ({len(short_trades)} closed, PnL: <span class="{'positive' if short_pnl >= 0 else 'negative'}">{short_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="short-header"><th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry Value</th><th>Exit Value</th><th>PnL</th><th>PnL %</th><th>Share</th></tr>
"""
    html += trade_table_rows(short_trades)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">Generated: {timestamp}</p>
</div>
</body>
</html>"""

    # Write to file
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print("Error: BINANCE_API_KEY_TEST or BINANCE_API_SECRET_TEST not set")
    else:
        path = generate_dashboard()
        print(f"\nOpen with: start {path}")
