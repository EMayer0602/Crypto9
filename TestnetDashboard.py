#!/usr/bin/env python3
"""Generate HTML dashboard for Binance Testnet trading."""

import argparse
import json
import os
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Import backfill functions from paper_trader
try:
    import pandas as pd
    from paper_trader import (
        get_last_processed_timestamp,
        run_simulation,
        trades_to_dataframe,
        write_closed_trades_report,
        write_open_positions_report,
        SIMULATION_LOG_FILE,
        SIMULATION_LOG_JSON,
        SIMULATION_OPEN_POSITIONS_FILE,
        SIMULATION_OPEN_POSITIONS_JSON,
    )
    import paper_trader as pt  # For setting global variables
    import Supertrend_5Min as st
    BACKFILL_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Backfill not available: {e}")
    BACKFILL_AVAILABLE = False

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


def load_simulation_data(trades_since: datetime = None, start_capital: float = 16500.0, max_positions: int = 10):
    """Load trades from trading_summary.html and recalculate with proper capital management.

    Args:
        trades_since: Only include trades from this date onwards (by entry_time)
        start_capital: Starting capital for simulation
        max_positions: Maximum open positions (stake = capital / max_positions)

    Returns:
        Tuple of (closed_trades_list, open_positions_list, final_capital)
    """
    FEE_RATE = 0.001  # 0.1% per trade (entry + exit)

    closed_trades = []
    open_positions = []
    raw_trades = []
    raw_open_positions = []

    # Load from trading_summary.html
    summary_path = Path("report_html/trading_summary.html")
    if not summary_path.exists():
        print(f"[Warning] trading_summary.html not found at {summary_path}")
        return closed_trades, open_positions, start_capital

    try:
        # Read HTML tables using pandas
        tables = pd.read_html(summary_path, encoding="utf-8")

        # Find the Long Trades table (has columns: symbol, direction, indicator, htf, entry_time, etc.)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if "symbol" in cols and "pnl" in cols and "exit_time" in cols:
                # This is the trades table
                for _, row in df.iterrows():
                    direction = str(row.get("direction", "long")).upper()
                    # Only include LONG trades
                    if direction != "LONG":
                        continue

                    entry_time_str = str(row.get("entry_time", ""))
                    exit_time_str = str(row.get("exit_time", ""))
                    if not entry_time_str or entry_time_str == "nan":
                        continue
                    if not exit_time_str or exit_time_str == "nan":
                        continue

                    # Parse entry_time for filtering
                    try:
                        entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                        if trades_since and entry_time < trades_since:
                            continue
                    except Exception:
                        continue

                    raw_trades.append({
                        "symbol": str(row.get("symbol", "?")),
                        "direction": direction,
                        "entry_time": entry_time_str,
                        "entry_time_dt": entry_time,
                        "exit_time": exit_time_str,
                        "entry_price": float(row.get("entry_price", 0) or 0),
                        "exit_price": float(row.get("exit_price", 0) or 0),
                        "reason": str(row.get("reason", "")),
                    })
                break

        # Find the Open Positions table
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if "symbol" in cols and "unrealized_pnl" in cols and "last_price" in cols:
                # This is the open positions table
                for _, row in df.iterrows():
                    direction = str(row.get("direction", "long")).upper()
                    # Only include LONG positions
                    if direction != "LONG":
                        continue

                    entry_time_str = str(row.get("entry_time", ""))
                    try:
                        entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                        if trades_since and entry_time < trades_since:
                            continue
                    except Exception:
                        continue

                    raw_open_positions.append({
                        "symbol": str(row.get("symbol", "?")),
                        "direction": direction,
                        "entry_time": entry_time_str,
                        "entry_time_dt": entry_time,
                        "entry_price": float(row.get("entry_price", 0) or 0),
                        "last_price": float(row.get("last_price", 0) or 0),
                        "bars_held": int(row.get("bars_held", 0) or 0),
                    })
                break

    except Exception as e:
        print(f"[Warning] Failed to load trading_summary.html: {e}")
        return closed_trades, open_positions, start_capital

    # Sort raw trades chronologically by entry_time
    raw_trades.sort(key=lambda x: x["entry_time_dt"])

    # Recalculate trades with proper capital management
    capital = start_capital
    for t in raw_trades:
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]

        if entry_price <= 0:
            continue

        # Calculate stake based on current capital
        stake = capital / max_positions
        amount = stake / entry_price

        # Calculate fees (entry + exit)
        entry_fee = stake * FEE_RATE
        exit_value = amount * exit_price
        exit_fee = exit_value * FEE_RATE
        total_fees = entry_fee + exit_fee

        # Calculate PnL
        gross_pnl = exit_value - stake
        pnl = gross_pnl - total_fees
        pnl_pct = (pnl / stake * 100) if stake > 0 else 0

        # Update capital
        capital += pnl

        closed_trades.append({
            "symbol": t["symbol"],
            "direction": t["direction"],
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stake": stake,
            "amount": amount,
            "fees": total_fees,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": t["reason"],
        })

    # Recalculate open positions with current capital
    for p in raw_open_positions:
        entry_price = p["entry_price"]
        last_price = p["last_price"]

        if entry_price <= 0:
            continue

        # Calculate stake based on current capital
        stake = capital / max_positions
        amount = stake / entry_price

        # Calculate unrealized PnL (only entry fee paid so far)
        entry_fee = stake * FEE_RATE
        current_value = amount * last_price
        unrealized_pnl = current_value - stake - entry_fee
        unrealized_pct = (unrealized_pnl / stake * 100) if stake > 0 else 0

        open_positions.append({
            "symbol": p["symbol"],
            "direction": p["direction"],
            "entry_time": p["entry_time"],
            "entry_price": entry_price,
            "last_price": last_price,
            "stake": stake,
            "amount": amount,
            "fees": entry_fee,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pct": unrealized_pct,
            "bars_held": p["bars_held"],
        })

    print(f"[Data] Loaded {len(closed_trades)} closed trades, {len(open_positions)} open positions")
    print(f"[Data] Capital: {start_capital:,.2f} -> {capital:,.2f} (PnL: {capital - start_capital:,.2f})")

    return closed_trades, open_positions, capital


def generate_dashboard(trades_since: datetime = None, start_capital: float = 16500.0, max_positions: int = 10, lang: str = "en"):
    """Generate HTML dashboard from trading_summary.html with recalculated capital.

    Args:
        trades_since: Only show trades from this date onwards. If None, uses TRADES_SINCE_DATE.
        start_capital: Starting capital for simulation
        max_positions: Maximum open positions
        lang: Language for dashboard ("en" or "de")
    """
    if trades_since is None:
        trades_since = TRADES_SINCE_DATE
    print(f"Loading simulation data (lang={lang})...")

    # Load data from trading_summary.html with recalculated capital
    closed_trades, open_positions, final_capital = load_simulation_data(
        trades_since, start_capital, max_positions
    )

    # Calculate totals
    total_realized_pnl = sum(t["pnl"] for t in closed_trades)
    total_closed_trades = len(closed_trades)
    total_unrealized_pnl = sum(p["unrealized_pnl"] for p in open_positions)
    wins = sum(1 for t in closed_trades if t["pnl"] > 0)
    win_rate = wins / total_closed_trades * 100 if total_closed_trades > 0 else 0

    # Labels based on language
    labels = {
        "en": {
            "title": "Paper Trading Dashboard",
            "subtitle": f"SPOT Only - Long-only Mode | Start: ${start_capital:,.2f} | Max Positions: {max_positions}",
            "start_capital": "Start Capital",
            "current_capital": "Current Capital",
            "closed_trades": "Closed Trades",
            "realized_pnl": "Realized PnL",
            "win_rate": "Win Rate",
            "open_positions": "Open Positions",
            "unrealized_pnl": "Unrealized PnL",
            "symbol": "Symbol",
            "entry_time": "Entry Time",
            "exit_time": "Exit Time",
            "entry_price": "Entry Price",
            "exit_price": "Exit Price",
            "last_price": "Last Price",
            "stake": "Stake",
            "amount": "Amount",
            "fees": "Fees",
            "pnl": "PnL",
            "pnl_pct": "PnL %",
            "reason": "Reason",
            "no_positions": "No open positions",
            "no_trades": "No closed trades",
            "generated": "Generated",
            "data_source": "Data from trading_summary.html (Long only, recalculated)",
        },
        "de": {
            "title": "Paper Trading Dashboard",
            "subtitle": f"Nur SPOT - Nur Long | Start: {start_capital:,.2f}€ | Max Positionen: {max_positions}",
            "start_capital": "Startkapital",
            "current_capital": "Aktuelles Kapital",
            "closed_trades": "Geschlossene Trades",
            "realized_pnl": "Realisierter PnL",
            "win_rate": "Gewinnrate",
            "open_positions": "Offene Positionen",
            "unrealized_pnl": "Unrealisierter PnL",
            "symbol": "Symbol",
            "entry_time": "Einstiegszeit",
            "exit_time": "Ausstiegszeit",
            "entry_price": "Einstiegspreis",
            "exit_price": "Ausstiegspreis",
            "last_price": "Aktueller Preis",
            "stake": "Einsatz",
            "amount": "Menge",
            "fees": "Gebühren",
            "pnl": "PnL",
            "pnl_pct": "PnL %",
            "reason": "Grund",
            "no_positions": "Keine offenen Positionen",
            "no_trades": "Keine geschlossenen Trades",
            "generated": "Erstellt",
            "data_source": "Daten aus trading_summary.html (Nur Long, neu berechnet)",
        },
    }
    L = labels.get(lang, labels["en"])

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{L['title']}</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .subtitle {{ color: #666; font-size: 14px; margin-top: -10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; font-size: 13px; }}
        th {{ background: #007bff; color: white; text-align: center; }}
        td:first-child {{ text-align: left; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
        .summary-box {{ display: inline-block; background: white; padding: 15px; margin: 8px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 130px; text-align: center; }}
        .summary-box h3 {{ margin: 0 0 8px 0; color: #666; font-size: 12px; }}
        .summary-box .value {{ font-size: 20px; font-weight: bold; }}
        .long-header {{ background: #28a745 !important; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .section {{ margin-bottom: 30px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>{L['title']}</h1>
    <p class="subtitle">{L['subtitle']}</p>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>{L['start_capital']}</h3>
            <div class="value">{start_capital:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>{L['current_capital']}</h3>
            <div class="value {'positive' if final_capital >= start_capital else 'negative'}">{final_capital:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>{L['closed_trades']}</h3>
            <div class="value">{total_closed_trades}</div>
        </div>
        <div class="summary-box">
            <h3>{L['realized_pnl']}</h3>
            <div class="value {'positive' if total_realized_pnl >= 0 else 'negative'}">{total_realized_pnl:,.2f}</div>
        </div>
        <div class="summary-box">
            <h3>{L['win_rate']}</h3>
            <div class="value">{win_rate:.1f}%</div>
        </div>
        <div class="summary-box">
            <h3>{L['open_positions']}</h3>
            <div class="value">{len(open_positions)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['unrealized_pnl']}</h3>
            <div class="value {'positive' if total_unrealized_pnl >= 0 else 'negative'}">{total_unrealized_pnl:,.2f}</div>
        </div>
    </div>

    <h2>{L['open_positions']} ({len(open_positions)})</h2>
    <table>
        <tr><th>{L['symbol']}</th><th>{L['entry_time']}</th><th>{L['entry_price']}</th><th>{L['last_price']}</th><th>{L['stake']}</th><th>{L['amount']}</th><th>{L['fees']}</th><th>{L['unrealized_pnl']}</th><th>{L['pnl_pct']}</th></tr>
"""
    if open_positions:
        for pos in open_positions:
            pnl_class = "positive" if pos["unrealized_pnl"] >= 0 else "negative"
            entry_time = pos.get("entry_time", "N/A")
            html += f"""        <tr>
            <td>{pos['symbol']}</td>
            <td>{entry_time}</td>
            <td>{pos['entry_price']:.8f}</td>
            <td>{pos['last_price']:.8f}</td>
            <td>{pos['stake']:,.2f}</td>
            <td>{pos['amount']:,.6f}</td>
            <td>{pos['fees']:,.2f}</td>
            <td class="{pnl_class}">{pos['unrealized_pnl']:,.2f}</td>
            <td class="{pnl_class}">{pos['unrealized_pct']:+.2f}%</td>
        </tr>\n"""
    else:
        html += f"        <tr><td colspan='9'>{L['no_positions']}</td></tr>\n"

    # Closed Trades section (Long only)
    html += f"""    </table>

    <div class="section">
    <h2>{L['closed_trades']} ({total_closed_trades}, PnL: <span class="{'positive' if total_realized_pnl >= 0 else 'negative'}">{total_realized_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="long-header"><th>{L['symbol']}</th><th>{L['entry_time']}</th><th>{L['exit_time']}</th><th>{L['entry_price']}</th><th>{L['exit_price']}</th><th>{L['stake']}</th><th>{L['amount']}</th><th>{L['fees']}</th><th>{L['pnl']}</th><th>{L['pnl_pct']}</th><th>{L['reason']}</th></tr>
"""
    if closed_trades:
        for t in closed_trades:
            pnl_class = "positive" if t["pnl"] >= 0 else "negative"
            html += f"""        <tr>
            <td>{t['symbol']}</td>
            <td>{t['entry_time']}</td>
            <td>{t['exit_time']}</td>
            <td>{t['entry_price']:.8f}</td>
            <td>{t['exit_price']:.8f}</td>
            <td>{t['stake']:,.2f}</td>
            <td>{t['amount']:,.6f}</td>
            <td>{t['fees']:,.2f}</td>
            <td class="{pnl_class}">{t['pnl']:,.2f}</td>
            <td class="{pnl_class}">{t['pnl_pct']:+.2f}%</td>
            <td>{t['reason']}</td>
        </tr>\n"""
    else:
        html += f"        <tr><td colspan='11'>{L['no_trades']}</td></tr>\n"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">{L['generated']}: {timestamp} | {L['data_source']}</p>
</div>
</body>
</html>"""

    # Write to file
    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = "_de" if lang == "de" else ""
    output_path = OUTPUT_DIR / f"dashboard{suffix}.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard ({lang}) saved to: {output_path}")
    return output_path


def run_backfill_simulation(trades_since: datetime = None, start_capital: float = 16500.0, max_positions: int = 10) -> bool:
    """Run backfill simulation to fill gaps since last processed time.

    Args:
        trades_since: Filter trades from this date onwards
        start_capital: Starting capital for simulation (default: 16500)
        max_positions: Maximum open positions (default: 10)

    Returns True if backfill was performed, False otherwise.
    """
    if not BACKFILL_AVAILABLE:
        print("[Backfill] Not available - paper_trader import failed")
        return False

    # Set global variables in paper_trader module
    pt.START_TOTAL_CAPITAL = start_capital
    pt.MAX_OPEN_POSITIONS = max_positions

    last_ts = get_last_processed_timestamp()
    now_ts = pd.Timestamp.now(tz=st.BERLIN_TZ)

    if last_ts is None:
        print("[Backfill] No previous state found - skipping")
        return False

    gap_hours = (now_ts - last_ts).total_seconds() / 3600
    if gap_hours <= 1:
        print(f"[Backfill] Gap is only {gap_hours:.1f} hours - skipping")
        return False

    print(f"[Backfill] Detected gap of {gap_hours:.1f} hours since {last_ts.strftime('%Y-%m-%d %H:%M')}")
    print(f"[Backfill] Running simulation (capital={start_capital}, max_pos={max_positions})...")

    try:
        backfill_trades, backfill_state = run_simulation(
            last_ts,
            now_ts,
            use_saved_state=True,
            emit_entry_log=False,
            allowed_symbols=None,
            allowed_indicators=None,
            fixed_stake=None,
            use_testnet=True,
            refresh_params=False,
            reset_state=False,
            clear_outputs=False,
        )
        print(f"[Backfill] Completed: {len(backfill_trades)} trades simulated")

        # Write results to logs
        if backfill_trades:
            trades_df = trades_to_dataframe(backfill_trades)
            write_closed_trades_report(trades_df, SIMULATION_LOG_FILE, SIMULATION_LOG_JSON)
        open_positions = backfill_state.get("positions", [])
        write_open_positions_report(open_positions, SIMULATION_OPEN_POSITIONS_FILE, SIMULATION_OPEN_POSITIONS_JSON)
        return True
    except Exception as e:
        print(f"[Backfill] Error: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard for Binance Testnet trading")
    parser.add_argument("--start", type=str, default=None,
                        help="Only show trades from this date onwards (YYYY-MM-DD)")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds when using --loop (default: 60)")
    parser.add_argument("--start-capital", type=float, default=16500.0,
                        help="Starting capital for backfill simulation (default: 16500)")
    parser.add_argument("--max-positions", type=int, default=10,
                        help="Maximum open positions for backfill simulation (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not API_KEY or not API_SECRET:
        print("Error: BINANCE_API_KEY_TEST or BINANCE_API_SECRET_TEST not set")
    else:
        # Parse --start date if provided
        trades_since = None
        if args.start:
            try:
                trades_since = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                print(f"Filtering trades since: {trades_since.strftime('%Y-%m-%d')}")
            except ValueError:
                print(f"Warning: Invalid date format '{args.start}', using default. Use YYYY-MM-DD.")

        if args.loop:
            print(f"Running in loop mode, refreshing every {args.interval} seconds. Press Ctrl+C to stop.")
            print(f"[Config] Start capital: {args.start_capital}, Max positions: {args.max_positions}")
            try:
                while True:
                    # Run backfill simulation on every refresh to get latest trades
                    run_backfill_simulation(trades_since, args.start_capital, args.max_positions)
                    # Generate both English and German dashboards
                    path_en = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="en")
                    path_de = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="de")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboards updated: {path_en}, {path_de}")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
        else:
            # Run backfill simulation to get latest trades
            run_backfill_simulation(trades_since, args.start_capital, args.max_positions)
            # Generate both English and German dashboards
            path_en = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="en")
            path_de = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="de")
            print(f"\nOpen with: start {path_en}")
            print(f"           start {path_de}")
