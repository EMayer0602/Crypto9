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

# Trading symbols
SYMBOLS = ["BTCEUR", "ETHEUR", "SUIEUR", "SOLEUR", "XRPEUR", "LINKEUR",
           "ZECUSDC", "LUNCUSDT", "TNSRUSDC"]

# Base currencies (don't count as positions)
BASE_CURRENCIES = {"EUR", "USDT", "USDC", "BUSD", "BTC", "TUSD"}

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


def generate_dashboard():
    """Generate HTML dashboard."""
    print("Fetching testnet data...")

    # Get balances
    balances = get_account_balances()

    # Filter relevant balances
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
            else:
                open_positions.append({"asset": asset, "amount": total})

    # Get order history for each symbol
    all_trades = []
    symbol_stats = {}
    for sym in SYMBOLS:
        orders = get_all_orders(sym)
        if orders:
            stats = calculate_trade_stats(orders)
            if stats["trades"] > 0:
                symbol_stats[sym] = stats
            for o in orders:
                if o.get("status") == "FILLED":
                    ts = o.get("time", 0)
                    all_trades.append({
                        "symbol": sym,
                        "side": o.get("side"),
                        "qty": float(o.get("executedQty", 0)),
                        "quote": float(o.get("cummulativeQuoteQty", 0)),
                        "time": datetime.fromtimestamp(ts/1000, tz=timezone.utc) if ts else None
                    })

    # Sort trades by time
    all_trades.sort(key=lambda x: x["time"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    # Calculate totals
    total_realized_pnl = sum(s.get("realized_pnl", 0) for s in symbol_stats.values())
    total_trades = sum(s.get("trades", 0) for s in symbol_stats.values())

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Testnet Trading Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
        th {{ background: #007bff; color: white; text-align: center; }}
        td:first-child {{ text-align: left; font-weight: bold; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .summary-box {{ display: inline-block; background: white; padding: 20px; margin: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 150px; text-align: center; }}
        .summary-box h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .summary-box .value {{ font-size: 24px; font-weight: bold; }}
        .buy {{ background: #d4edda; }}
        .sell {{ background: #f8d7da; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Binance Testnet Dashboard</h1>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>Total Trades</h3>
            <div class="value">{total_trades}</div>
        </div>
        <div class="summary-box">
            <h3>Realized PnL</h3>
            <div class="value {'positive' if total_realized_pnl >= 0 else 'negative'}">{total_realized_pnl:,.2f}</div>
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

    html += """    </table>

    <h2>Statistics by Symbol</h2>
    <table>
        <tr><th>Symbol</th><th>Trades</th><th>Buys</th><th>Sells</th><th>Buy Volume</th><th>Sell Volume</th><th>Realized PnL</th></tr>
"""
    for sym, stats in sorted(symbol_stats.items()):
        pnl_class = "positive" if stats["realized_pnl"] >= 0 else "negative"
        html += f"""        <tr>
            <td>{sym}</td>
            <td>{stats['trades']}</td>
            <td>{stats['buys']}</td>
            <td>{stats['sells']}</td>
            <td>{stats['buy_volume']:,.2f}</td>
            <td>{stats['sell_volume']:,.2f}</td>
            <td class="{pnl_class}">{stats['realized_pnl']:,.2f}</td>
        </tr>\n"""

    html += """    </table>

    <h2>Recent Trades (Last 50)</h2>
    <table>
        <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Quantity</th><th>Quote Amount</th></tr>
"""
    for trade in all_trades[:50]:
        side_class = "buy" if trade["side"] == "BUY" else "sell"
        time_str = trade["time"].strftime("%Y-%m-%d %H:%M") if trade["time"] else "N/A"
        html += f"""        <tr class="{side_class}">
            <td>{time_str}</td>
            <td>{trade['symbol']}</td>
            <td>{trade['side']}</td>
            <td>{trade['qty']:,.4f}</td>
            <td>{trade['quote']:,.2f}</td>
        </tr>\n"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>

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
