#!/usr/bin/env python3
"""
TestnetDashboard - READ-ONLY Dashboard für Paper Trading

Liest Daten von:
- report_html/trading_summary.json (historische + neue Trades)
- paper_trading_state.json (offene Positionen)

Holt Live-Preise von Binance für offene Positionen.
Schreibt NUR die Dashboard-HTML Dateien (report_html/dashboard.html, dashboard_de.html)
"""

import argparse
import json
import os
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Binance API for live prices (real, not testnet)
BINANCE_PUBLIC_URL = "https://api.binance.com"

# Default input files (READ ONLY)
DEFAULT_TRADING_SUMMARY_JSON = Path("report_html/trading_summary.json")
PAPER_TRADING_STATE = Path("paper_trading_state.json")

# Default output directory for dashboards
DEFAULT_OUTPUT_DIR = Path("report_html")


def format_number(value: float, decimals: int = 2, lang: str = "en") -> str:
    """Format number according to language conventions.

    English: 16,500.00
    German: 16.500,00
    """
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)

    if lang == "de":
        formatted = f"{value:,.{decimals}f}"
        formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
        return formatted
    else:
        return f"{value:,.{decimals}f}"


def get_current_price(symbol: str) -> float:
    """Get current price from Binance for a symbol.

    Args:
        symbol: Trading pair like "BTC/USDC" or "BTCUSDT"

    Returns:
        Current price as float, or 0.0 if failed
    """
    binance_symbol = symbol.replace("/", "")

    symbols_to_try = [binance_symbol]
    if "USDC" in binance_symbol:
        symbols_to_try.append(binance_symbol.replace("USDC", "USDT"))
    elif "EUR" in binance_symbol:
        base = binance_symbol.replace("EUR", "")
        symbols_to_try.append(f"{base}USDT")

    for sym in symbols_to_try:
        try:
            url = f"{BINANCE_PUBLIC_URL}/api/v3/ticker/price?symbol={sym}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data.get("price", 0))
        except Exception:
            continue

    return 0.0


def get_all_current_prices(symbols: list) -> dict:
    """Get current prices for multiple symbols at once."""
    prices = {}
    for symbol in symbols:
        price = get_current_price(symbol)
        if price > 0:
            prices[symbol] = price
    return prices


def load_trading_summary(summary_path: Path = None) -> dict:
    """Load trading summary from JSON file."""
    if summary_path is None:
        summary_path = DEFAULT_TRADING_SUMMARY_JSON

    if not summary_path.exists():
        print(f"[Warning] {summary_path} not found")
        return {}

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load {summary_path}: {e}")
        return {}


def load_paper_trading_state(state_path: Path = None) -> dict:
    """Load paper trading state from JSON file.

    Args:
        state_path: Path to state file. If None, uses default PAPER_TRADING_STATE.
    """
    if state_path is None:
        state_path = PAPER_TRADING_STATE
    else:
        state_path = Path(state_path)

    if not state_path.exists():
        print(f"[Warning] {state_path} not found")
        return {}

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load {state_path}: {e}")
        return {}


def generate_dashboard(
    trades_since: datetime = None,
    start_capital: float = 16500.0,
    max_positions: int = 10,
    lang: str = "en",
    output_dir: Path = None
) -> Path:
    """Generate HTML dashboard from trading_summary.json.

    Args:
        trades_since: Only show trades from this date onwards
        start_capital: Starting capital for display
        max_positions: Maximum positions for stake calculation
        lang: Language for dashboard ("en" or "de")
        output_dir: Output directory for dashboard files (default: report_html)
    """
    # Use defaults if not specified
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)

    # Input files are in the same directory as output
    trading_summary_json = output_dir / "trading_summary.json"
    state_json = output_dir / "state.json"

    print(f"[Dashboard] Loading data (lang={lang}, dir={output_dir})...")

    # Load trading summary
    summary = load_trading_summary(trading_summary_json)
    if not summary:
        print("[Dashboard] No trading summary data available")
        return None

    # Get trades from summary (only from trading_summary.json, not simulation files)
    all_trades = summary.get("trades", [])

    # Filter trades by date if specified
    closed_trades = []
    for t in all_trades:
        if trades_since:
            entry_time_str = t.get("entry_time", "")
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                if entry_time < trades_since:
                    continue
            except:
                pass
        closed_trades.append(t)

    # Get open positions from summary (use actual data from open_positions_data)
    summary_positions = summary.get("open_positions_data", [])

    # Fetch live prices for open positions
    if summary_positions:
        symbols = [p.get("symbol", "") for p in summary_positions]
        live_prices = get_all_current_prices(symbols)
        print(f"[Dashboard] Fetched {len(live_prices)} live prices")
    else:
        live_prices = {}

    # === RECALCULATE AS IF STARTING WITH INITIAL CAPITAL ===
    # Sort filtered trades chronologically by entry_time
    sorted_trades = sorted(closed_trades, key=lambda t: t.get("entry_time", "") or "")

    # Recalculate stakes and PnL starting from initial capital
    # Keep actual dates and prices, but recalculate stake = capital / max_positions
    capital = start_capital
    processed_trades = []
    total_realized_pnl = 0

    for t in sorted_trades:
        # Keep actual dates and prices
        entry_price = float(t.get("entry_price", 0) or 0)
        exit_price = float(t.get("exit_price", 0) or 0)
        direction = t.get("direction", "long").lower()

        # Recalculate stake based on current capital
        stake = capital / max_positions

        # Recalculate PnL based on recalculated stake and actual prices
        if entry_price > 0:
            if direction == "short":
                pnl = (entry_price - exit_price) / entry_price * stake
            else:  # long
                pnl = (exit_price - entry_price) / entry_price * stake
        else:
            pnl = 0

        # Update capital for next trade
        capital += pnl
        total_realized_pnl += pnl

        # Store with recalculated values but original dates/prices
        processed_trades.append({
            **t,
            "stake": stake,
            "pnl": pnl,
            "amount": stake / entry_price if entry_price > 0 else 0,
        })

    # Current capital after all closed trades
    current_capital = start_capital + total_realized_pnl

    # Recalculate open positions with current capital
    open_positions = []

    for p in summary_positions:
        symbol = p.get("symbol", "?")
        entry_price = float(p.get("entry_price", 0) or 0)
        live_price = live_prices.get(symbol, entry_price)

        # Recalculate stake based on current capital (after closed trades)
        stake = current_capital / max_positions
        if entry_price > 0:
            amount = stake / entry_price
            current_value = amount * live_price
            unrealized_pnl = current_value - stake
            unrealized_pct = (unrealized_pnl / stake * 100) if stake > 0 else 0
        else:
            amount = 0
            unrealized_pnl = 0
            unrealized_pct = 0

        open_positions.append({
            "symbol": symbol,
            "direction": p.get("direction", "long"),
            "entry_time": p.get("entry_time", "N/A"),
            "entry_price": entry_price,
            "last_price": live_price,
            "stake": stake,
            "amount": amount,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pct": unrealized_pct,
        })

    # Calculate totals (total_realized_pnl already calculated above)
    total_closed_trades = len(processed_trades)
    total_unrealized_pnl = sum(p["unrealized_pnl"] for p in open_positions)
    wins = sum(1 for t in processed_trades if float(t.get("pnl", 0) or 0) > 0)
    win_rate = wins / total_closed_trades * 100 if total_closed_trades > 0 else 0
    final_capital = current_capital + total_unrealized_pnl

    # Helper function
    def fmt(value: float, decimals: int = 2) -> str:
        return format_number(value, decimals, lang)

    # Labels
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
            "pnl": "PnL",
            "pnl_pct": "PnL %",
            "reason": "Reason",
            "no_positions": "No open positions",
            "no_trades": "No closed trades",
            "generated": "Generated",
            "data_source": "Data from trading_summary.json + paper_trading_state.json",
        },
        "de": {
            "title": "Paper Trading Dashboard",
            "subtitle": f"Nur SPOT - Nur Long | Start: {format_number(start_capital, 2, 'de')}€ | Max Positionen: {max_positions}",
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
            "pnl": "PnL",
            "pnl_pct": "PnL %",
            "reason": "Grund",
            "no_positions": "Keine offenen Positionen",
            "no_trades": "Keine geschlossenen Trades",
            "generated": "Erstellt",
            "data_source": "Daten aus trading_summary.json + paper_trading_state.json",
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
    </style>
</head>
<body>
<div class="container">
    <h1>{L['title']}</h1>
    <p class="subtitle">{L['subtitle']}</p>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>{L['start_capital']}</h3>
            <div class="value">{fmt(start_capital)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['current_capital']}</h3>
            <div class="value {'positive' if final_capital >= start_capital else 'negative'}">{fmt(final_capital)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['closed_trades']}</h3>
            <div class="value">{total_closed_trades}</div>
        </div>
        <div class="summary-box">
            <h3>{L['realized_pnl']}</h3>
            <div class="value {'positive' if total_realized_pnl >= 0 else 'negative'}">{fmt(total_realized_pnl)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['win_rate']}</h3>
            <div class="value">{fmt(win_rate, 1)}%</div>
        </div>
        <div class="summary-box">
            <h3>{L['open_positions']}</h3>
            <div class="value">{len(open_positions)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['unrealized_pnl']}</h3>
            <div class="value {'positive' if total_unrealized_pnl >= 0 else 'negative'}">{fmt(total_unrealized_pnl)}</div>
        </div>
    </div>

    <h2>{L['open_positions']} ({len(open_positions)})</h2>
    <table>
        <tr><th>{L['symbol']}</th><th>{L['entry_time']}</th><th>{L['entry_price']}</th><th>{L['last_price']}</th><th>{L['stake']}</th><th>{L['amount']}</th><th>{L['unrealized_pnl']}</th><th>{L['pnl_pct']}</th></tr>
"""
    if open_positions:
        for pos in open_positions:
            pnl_class = "positive" if pos["unrealized_pnl"] >= 0 else "negative"
            pct_sign = "+" if pos["unrealized_pct"] >= 0 else ""
            html += f"""        <tr>
            <td>{pos['symbol']}</td>
            <td>{pos['entry_time']}</td>
            <td>{fmt(pos['entry_price'], 8)}</td>
            <td>{fmt(pos['last_price'], 8)}</td>
            <td>{fmt(pos['stake'])}</td>
            <td>{fmt(pos['amount'], 6)}</td>
            <td class="{pnl_class}">{fmt(pos['unrealized_pnl'])}</td>
            <td class="{pnl_class}">{pct_sign}{fmt(pos['unrealized_pct'])}%</td>
        </tr>\n"""
    else:
        html += f"        <tr><td colspan='8'>{L['no_positions']}</td></tr>\n"

    # Closed Trades section - newest first
    html += f"""    </table>

    <h2>{L['closed_trades']} ({total_closed_trades}, PnL: <span class="{'positive' if total_realized_pnl >= 0 else 'negative'}">{fmt(total_realized_pnl)}</span>)</h2>
    <table>
        <tr class="long-header"><th>{L['symbol']}</th><th>{L['entry_time']}</th><th>{L['exit_time']}</th><th>{L['entry_price']}</th><th>{L['exit_price']}</th><th>{L['stake']}</th><th>{L['pnl']}</th><th>{L['pnl_pct']}</th><th>{L['reason']}</th></tr>
"""
    if processed_trades:
        # Sort by entry_time descending (newest first) for display
        display_trades = sorted(processed_trades, key=lambda t: t.get("entry_time", ""), reverse=True)
        for t in display_trades[:100]:  # Show max 100 trades
            pnl = float(t.get("pnl", 0) or 0)
            stake = float(t.get("stake", 0) or 0)
            pnl_pct = (pnl / stake * 100) if stake > 0 else 0
            pnl_class = "positive" if pnl >= 0 else "negative"
            pct_sign = "+" if pnl_pct >= 0 else ""
            html += f"""        <tr>
            <td>{t.get('symbol', '?')}</td>
            <td>{t.get('entry_time', 'N/A')}</td>
            <td>{t.get('exit_time', 'N/A')}</td>
            <td>{fmt(float(t.get('entry_price', 0) or 0), 8)}</td>
            <td>{fmt(float(t.get('exit_price', 0) or 0), 8)}</td>
            <td>{fmt(stake)}</td>
            <td class="{pnl_class}">{fmt(pnl)}</td>
            <td class="{pnl_class}">{pct_sign}{fmt(pnl_pct)}%</td>
            <td>{t.get('reason', '')}</td>
        </tr>\n"""
    else:
        html += f"        <tr><td colspan='9'>{L['no_trades']}</td></tr>\n"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>

    <p class="timestamp">{L['generated']}: {timestamp} | {L['data_source']}</p>
</div>
</body>
</html>"""

    # Write to file
    output_dir.mkdir(exist_ok=True)
    suffix = "_de" if lang == "de" else ""
    output_path = output_dir / f"dashboard{suffix}.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"[Dashboard] Saved to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="TestnetDashboard - READ-ONLY Dashboard für Paper Trading"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Only show trades from this date onwards (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run continuously in a loop"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Refresh interval in seconds when using --loop (default: 60)"
    )
    parser.add_argument(
        "--start-capital", type=float, default=16500.0,
        help="Starting capital for display (default: 16500)"
    )
    parser.add_argument(
        "--max-positions", type=int, default=10,
        help="Maximum open positions for stake calculation (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="report_html",
        help="Output directory for dashboard (default: report_html). Use report_testnet for testnet data."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse --start date if provided
    trades_since = None
    if args.start:
        try:
            trades_since = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            print(f"[Config] Filtering trades since: {trades_since.strftime('%Y-%m-%d')}")
        except ValueError:
            print(f"[Warning] Invalid date format '{args.start}', showing all trades. Use YYYY-MM-DD.")

    print(f"[Config] Start capital: {args.start_capital}, Max positions: {args.max_positions}")
    print(f"[Config] Output directory: {args.output_dir}")

    if args.loop:
        print(f"[Config] Running in loop mode, refreshing every {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                path_en = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="en", output_dir=args.output_dir)
                path_de = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="de", output_dir=args.output_dir)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboards updated")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        path_en = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="en", output_dir=args.output_dir)
        path_de = generate_dashboard(trades_since, args.start_capital, args.max_positions, lang="de", output_dir=args.output_dir)
        if path_en:
            print(f"\nOpen with: start {path_en}")
            print(f"           start {path_de}")
