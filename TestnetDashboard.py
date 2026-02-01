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


def load_simulation_data(trades_since: datetime = None):
    """Load closed trades and open positions from trading_summary.html.

    Args:
        trades_since: Only include trades from this date onwards

    Returns:
        Tuple of (closed_trades_list, open_positions_list)
    """
    closed_trades = []
    open_positions = []

    # Load from trading_summary.html
    summary_path = Path("report_html/trading_summary.html")
    if summary_path.exists():
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

                        exit_time_str = str(row.get("exit_time", ""))
                        if not exit_time_str or exit_time_str == "nan":
                            continue

                        # Parse exit_time for filtering
                        try:
                            exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
                            if trades_since and exit_time < trades_since:
                                continue
                        except Exception:
                            continue

                        closed_trades.append({
                            "symbol": str(row.get("symbol", "?")),
                            "direction": direction,
                            "entry_time": str(row.get("entry_time", "")),
                            "exit_time": exit_time_str,
                            "entry_price": float(row.get("entry_price", 0) or 0),
                            "exit_price": float(row.get("exit_price", 0) or 0),
                            "stake": float(row.get("stake", 0) or 0),
                            "pnl": float(row.get("pnl", 0) or 0),
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

                        open_positions.append({
                            "symbol": str(row.get("symbol", "?")),
                            "direction": direction,
                            "entry_time": str(row.get("entry_time", "")),
                            "entry_price": float(row.get("entry_price", 0) or 0),
                            "last_price": float(row.get("last_price", 0) or 0),
                            "stake": float(row.get("stake", 0) or 0),
                            "unrealized_pnl": float(row.get("unrealized_pnl", 0) or 0),
                            "unrealized_pct": float(row.get("unrealized_pct", 0) or 0),
                            "bars_held": int(row.get("bars_held", 0) or 0),
                        })
                    break

            print(f"[Data] Loaded {len(closed_trades)} closed trades, {len(open_positions)} open positions from trading_summary.html")
        except Exception as e:
            print(f"[Warning] Failed to load trading_summary.html: {e}")
    else:
        print(f"[Warning] trading_summary.html not found at {summary_path}")

    return closed_trades, open_positions


def generate_dashboard(trades_since: datetime = None):
    """Generate HTML dashboard from simulation logs.

    Args:
        trades_since: Only show trades from this date onwards. If None, uses TRADES_SINCE_DATE.
    """
    if trades_since is None:
        trades_since = TRADES_SINCE_DATE
    print("Loading simulation data...")

    # Load data from trading_summary.html (only LONG trades)
    closed_trades, open_positions = load_simulation_data(trades_since)

    # Calculate totals
    total_realized_pnl = sum(t["pnl"] for t in closed_trades)
    total_closed_trades = len(closed_trades)
    total_unrealized_pnl = sum(p["unrealized_pnl"] for p in open_positions)
    wins = sum(1 for t in closed_trades if t["pnl"] > 0)
    win_rate = wins / total_closed_trades * 100 if total_closed_trades > 0 else 0

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Paper Trading Dashboard</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
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
    <h1>Paper Trading Dashboard</h1>

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
            <h3>Win Rate</h3>
            <div class="value">{win_rate:.1f}%</div>
        </div>
        <div class="summary-box">
            <h3>Open Positions</h3>
            <div class="value">{len(open_positions)}</div>
        </div>
        <div class="summary-box">
            <h3>Unrealized PnL</h3>
            <div class="value {'positive' if total_unrealized_pnl >= 0 else 'negative'}">{total_unrealized_pnl:,.2f}</div>
        </div>
    </div>

    <h2>Open Positions ({len(open_positions)})</h2>
    <table>
        <tr><th>Symbol</th><th>Direction</th><th>Entry Time</th><th>Entry Price</th><th>Last Price</th><th>Stake</th><th>Unrealized PnL</th><th>PnL %</th></tr>
"""
    if open_positions:
        for pos in open_positions:
            pnl_class = "positive" if pos["unrealized_pnl"] >= 0 else "negative"
            entry_time = pos.get("entry_time", "N/A")
            html += f"""        <tr>
            <td>{pos['symbol']}</td>
            <td>{pos['direction']}</td>
            <td>{entry_time}</td>
            <td>{pos['entry_price']:.8f}</td>
            <td>{pos['last_price']:.8f}</td>
            <td>{pos['stake']:,.2f}</td>
            <td class="{pnl_class}">{pos['unrealized_pnl']:,.2f}</td>
            <td class="{pnl_class}">{pos['unrealized_pct']:+.2f}%</td>
        </tr>\n"""
    else:
        html += "        <tr><td colspan='8'>No open positions</td></tr>\n"

    # Closed Trades section (Long only)
    html += f"""    </table>

    <div class="section">
    <h2>Closed Trades ({total_closed_trades} trades, PnL: <span class="{'positive' if total_realized_pnl >= 0 else 'negative'}">{total_realized_pnl:,.2f}</span>)</h2>
    <table>
        <tr class="long-header"><th>Symbol</th><th>Entry Time</th><th>Exit Time</th><th>Entry Price</th><th>Exit Price</th><th>Stake</th><th>PnL</th><th>Reason</th></tr>
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
            <td class="{pnl_class}">{t['pnl']:,.2f}</td>
            <td>{t['reason']}</td>
        </tr>\n"""
    else:
        html += "        <tr><td colspan='8'>No closed trades</td></tr>\n"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">Generated: {timestamp} | Data from trading_summary.html (Long only)</p>
</div>
</body>
</html>"""

    # Write to file
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to: {output_path}")
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
                    path = generate_dashboard(trades_since)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard updated: {path}")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
        else:
            # Run backfill simulation to get latest trades
            run_backfill_simulation(trades_since, args.start_capital, args.max_positions)
            path = generate_dashboard(trades_since)
            print(f"\nOpen with: start {path}")
