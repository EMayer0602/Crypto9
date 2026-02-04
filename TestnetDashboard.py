#!/usr/bin/env python3
"""Generate HTML dashboard from simulation trades.

Reads directly from simulation JSON files:
- paper_trading_simulation_log.json (closed trades)
- paper_trading_actual_trades.json (open positions)

Stakes are recalculated with compound growth from 16500 initial capital.
Generates both English and German dashboards.
"""

import json
from datetime import datetime
from pathlib import Path


# Capital settings - same as paper_trader.py
START_CAPITAL = 16500.0
MAX_POSITIONS = 10

# Default paths
CLOSED_TRADES_JSON = "paper_trading_simulation_log.json"
OPEN_POSITIONS_JSON = "paper_trading_actual_trades.json"


def load_json(path: Path) -> list | dict:
    """Load JSON file, return empty list/dict if not found."""
    if not path.exists():
        print(f"Warning: {path} not found")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def fmt_num(value: float, lang: str = "en") -> str:
    """Format number according to language."""
    if lang == "de":
        if abs(value) >= 1000:
            return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{value:.2f}".replace(".", ",")
    return f"{value:,.2f}"


def generate_dashboard(output_dir: Path, lang: str = "en", start_date: datetime = None):
    """Generate HTML dashboard from simulation JSON files.

    Args:
        output_dir: Directory containing JSON files and for output
        lang: Language for labels ("en" or "de")
        start_date: Only include trades from this date onwards
    """
    closed_path = output_dir / CLOSED_TRADES_JSON
    open_path = output_dir / OPEN_POSITIONS_JSON

    # Load data directly from simulation JSONs
    closed_trades = load_json(closed_path)
    open_positions = load_json(open_path)

    if not isinstance(closed_trades, list):
        closed_trades = []
    if not isinstance(open_positions, list):
        open_positions = []

    # Filter by start_date if provided
    if start_date:
        filtered_trades = []
        for t in closed_trades:
            et = t.get("entry_time") or t.get("Zeit") or ""
            try:
                entry_dt = datetime.fromisoformat(et.replace("Z", "+00:00"))
                if entry_dt.replace(tzinfo=None) >= start_date:
                    filtered_trades.append(t)
            except:
                pass
        closed_trades = filtered_trades

        filtered_open = []
        for p in open_positions:
            et = p.get("entry_time", "")
            try:
                entry_dt = datetime.fromisoformat(et.replace("Z", "+00:00"))
                if entry_dt.replace(tzinfo=None) >= start_date:
                    filtered_open.append(p)
            except:
                pass
        open_positions = filtered_open

    print(f"Loaded {len(closed_trades)} closed trades, {len(open_positions)} open positions")

    # Sort helper
    def get_entry_time(t):
        et = t.get("entry_time") or t.get("Zeit") or ""
        try:
            return datetime.fromisoformat(et.replace("Z", "+00:00"))
        except:
            return datetime.min

    # === RECALCULATE CLOSED TRADES WITH COMPOUND GROWTH ===
    # Sort chronologically (oldest first) for compound calculation
    trades_chrono = sorted(closed_trades, key=get_entry_time, reverse=False)

    capital = START_CAPITAL
    recalculated_trades = []

    for t in trades_chrono:
        stake = capital / MAX_POSITIONS
        entry_price = t.get("entry_price") or t.get("Entry") or 0
        exit_price = t.get("exit_price") or t.get("ExitPreis") or 0
        direction = t.get("direction", "").lower()

        # Calculate PnL based on direction
        if entry_price > 0:
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - exit_price) / entry_price
            pnl = pnl_pct * stake
        else:
            pnl = 0
            pnl_pct = 0

        capital += pnl

        # Create recalculated trade entry
        recalc_trade = dict(t)
        recalc_trade["stake"] = stake
        recalc_trade["pnl"] = pnl
        recalc_trade["pnl_pct"] = pnl_pct * 100
        recalc_trade["equity_after"] = capital
        recalculated_trades.append(recalc_trade)

    final_capital = capital

    # === RECALCULATE OPEN POSITIONS WITH FINAL CAPITAL ===
    open_stake = final_capital / MAX_POSITIONS
    recalculated_open = []

    for p in open_positions:
        entry_price = p.get("entry_price", 0)
        last_price = p.get("last_price", 0)
        direction = p.get("direction", "").lower()

        if entry_price > 0:
            if direction == "long":
                pnl_pct = (last_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - last_price) / entry_price
            unrealized_pnl = pnl_pct * open_stake
        else:
            pnl_pct = 0
            unrealized_pnl = 0

        recalc_pos = dict(p)
        recalc_pos["stake"] = open_stake
        recalc_pos["unrealized_pnl"] = unrealized_pnl
        recalc_pos["unrealized_pct"] = pnl_pct * 100
        recalc_pos["status"] = "Gewinn" if unrealized_pnl > 0 else "Verlust" if unrealized_pnl < 0 else "Flat"
        recalculated_open.append(recalc_pos)

    # Sort by entry time (newest first) for display
    recalculated_trades.sort(key=get_entry_time, reverse=True)
    recalculated_open.sort(key=get_entry_time, reverse=True)

    # Separate long and short trades
    long_trades = [t for t in recalculated_trades if t.get("direction", "").lower() == "long"]
    short_trades = [t for t in recalculated_trades if t.get("direction", "").lower() == "short"]

    # Calculate statistics from recalculated values
    total_pnl = sum(t.get("pnl", 0) for t in recalculated_trades)
    long_pnl = sum(t.get("pnl", 0) for t in long_trades)
    short_pnl = sum(t.get("pnl", 0) for t in short_trades)
    long_wins = sum(1 for t in long_trades if t.get("pnl", 0) > 0)
    short_wins = sum(1 for t in short_trades if t.get("pnl", 0) > 0)

    # Open positions stats from recalculated values
    open_equity = sum(p.get("unrealized_pnl", 0) for p in recalculated_open)

    # Use recalculated data for display
    open_positions = recalculated_open

    # Language labels
    L = {
        "en": {
            "title": "Trading Dashboard",
            "source": f"Simulation trades, recalculated with {fmt_num(START_CAPITAL, 'en')} initial capital",
            "start_capital": "Start Capital", "final_capital": "Final Capital",
            "closed_trades": "Closed Trades", "realized_pnl": "Realized PnL",
            "long_trades": "Long Trades", "short_trades": "Short Trades",
            "open_positions": "Open Positions", "open_equity": "Open Equity",
            "symbol": "Symbol", "direction": "Direction", "indicator": "Indicator",
            "entry_time": "Entry Time", "exit_time": "Exit Time",
            "entry_price": "Entry Price", "last_price": "Last Price",
            "stake": "Stake", "bars": "Bars", "pnl": "PnL", "status": "Status", "reason": "Reason",
        },
        "de": {
            "title": "Trading Dashboard",
            "source": f"Simulation Trades, neu berechnet mit {fmt_num(START_CAPITAL, 'de')} Startkapital",
            "start_capital": "Startkapital", "final_capital": "Endkapital",
            "closed_trades": "Geschlossene Trades", "realized_pnl": "Realisierter PnL",
            "long_trades": "Long Trades", "short_trades": "Short Trades",
            "open_positions": "Offene Positionen", "open_equity": "Offenes Equity",
            "symbol": "Symbol", "direction": "Richtung", "indicator": "Indikator",
            "entry_time": "Einstieg", "exit_time": "Ausstieg",
            "entry_price": "Einstiegspreis", "last_price": "Aktueller Preis",
            "stake": "Einsatz", "bars": "Bars", "pnl": "PnL", "status": "Status", "reason": "Grund",
        },
    }[lang]

    def fmt(v): return fmt_num(v, lang)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{L['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
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
        .open-header {{ background: #17a2b8 !important; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .source-note {{ color: #666; font-style: italic; margin-bottom: 20px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>{L['title']}</h1>
    <p class="source-note">{L['source']}</p>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>{L['start_capital']}</h3>
            <div class="value">{fmt(START_CAPITAL)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['final_capital']}</h3>
            <div class="value {'positive' if final_capital >= START_CAPITAL else 'negative'}">{fmt(final_capital)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['closed_trades']}</h3>
            <div class="value">{len(recalculated_trades)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['realized_pnl']}</h3>
            <div class="value {'positive' if total_pnl >= 0 else 'negative'}">{fmt(total_pnl)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['long_trades']}</h3>
            <div class="value">{len(long_trades)} ({long_wins}W)</div>
        </div>
        <div class="summary-box">
            <h3>{L['short_trades']}</h3>
            <div class="value">{len(short_trades)} ({short_wins}W)</div>
        </div>
        <div class="summary-box">
            <h3>{L['open_positions']}</h3>
            <div class="value">{len(open_positions)}</div>
        </div>
        <div class="summary-box">
            <h3>{L['open_equity']}</h3>
            <div class="value {'positive' if open_equity >= 0 else 'negative'}">{fmt(open_equity)}</div>
        </div>
    </div>
"""

    # Open Positions section
    html += f"""
    <div class="section">
    <h2>{L['open_positions']} ({len(open_positions)}, Equity: <span class="{'positive' if open_equity >= 0 else 'negative'}">{fmt(open_equity)}</span>)</h2>
    <table>
        <tr class="open-header"><th>{L['symbol']}</th><th>{L['direction']}</th><th>{L['indicator']}</th><th>HTF</th><th>{L['entry_time']}</th><th>{L['entry_price']}</th><th>{L['last_price']}</th><th>{L['stake']}</th><th>{L['bars']}</th><th>{L['pnl']}</th><th>{L['status']}</th></tr>
"""
    if open_positions:
        for p in open_positions:
            entry_time = p.get("entry_time", "")
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                entry_str = entry_dt.strftime("%Y-%m-%d %H:%M")
            except:
                entry_str = entry_time[:16] if entry_time else "N/A"

            pnl = p.get("unrealized_pnl", 0)
            pnl_class = "positive" if pnl >= 0 else "negative"
            direction = p.get("direction", "").upper()

            html += f"""        <tr>
            <td>{p.get('symbol', 'N/A')}</td>
            <td>{direction}</td>
            <td>{p.get('indicator', 'N/A')}</td>
            <td>{p.get('htf', 'N/A')}</td>
            <td>{entry_str}</td>
            <td>{p.get('entry_price', 0):.6g}</td>
            <td>{p.get('last_price', 0):.6g}</td>
            <td>{fmt(p.get('stake', 0))}</td>
            <td>{p.get('bars_held', 0)}</td>
            <td class="{pnl_class}">{fmt(pnl)}</td>
            <td>{p.get('status', 'N/A')}</td>
        </tr>\n"""
    else:
        html += "        <tr><td colspan='11'>-</td></tr>\n"

    html += "    </table>\n    </div>\n"

    # Helper function to generate closed trade table rows
    def trade_rows(trades):
        if not trades:
            return "        <tr><td colspan='9'>-</td></tr>\n"
        rows = ""
        for t in trades:
            entry_time = t.get("entry_time") or t.get("Zeit") or ""
            exit_time = t.get("exit_time") or t.get("ExitZeit") or ""
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                entry_str = entry_dt.strftime("%Y-%m-%d %H:%M")
            except:
                entry_str = entry_time[:16] if entry_time else "N/A"
            try:
                exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                exit_str = exit_dt.strftime("%Y-%m-%d %H:%M")
            except:
                exit_str = exit_time[:16] if exit_time else "N/A"

            pnl = t.get("pnl", 0)
            stake = t.get("stake", 0)
            pnl_pct = t.get("pnl_pct", 0)
            pnl_class = "positive" if pnl >= 0 else "negative"

            rows += f"""        <tr>
            <td>{t.get('symbol', 'N/A')}</td>
            <td>{t.get('indicator', 'N/A')}</td>
            <td>{t.get('htf', 'N/A')}</td>
            <td>{entry_str}</td>
            <td>{exit_str}</td>
            <td>{fmt(stake)}</td>
            <td class="{pnl_class}">{fmt(pnl)}</td>
            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            <td>{t.get('reason', 'N/A')}</td>
        </tr>\n"""
        return rows

    # Long Trades section
    html += f"""
    <div class="section">
    <h2>{L['long_trades']} ({len(long_trades)}, PnL: <span class="{'positive' if long_pnl >= 0 else 'negative'}">{fmt(long_pnl)}</span>)</h2>
    <table>
        <tr class="long-header"><th>{L['symbol']}</th><th>{L['indicator']}</th><th>HTF</th><th>{L['entry_time']}</th><th>{L['exit_time']}</th><th>{L['stake']}</th><th>{L['pnl']}</th><th>%</th><th>{L['reason']}</th></tr>
"""
    html += trade_rows(long_trades)

    # Short Trades section
    html += f"""    </table>
    </div>

    <div class="section">
    <h2>{L['short_trades']} ({len(short_trades)}, PnL: <span class="{'positive' if short_pnl >= 0 else 'negative'}">{fmt(short_pnl)}</span>)</h2>
    <table>
        <tr class="short-header"><th>{L['symbol']}</th><th>{L['indicator']}</th><th>HTF</th><th>{L['entry_time']}</th><th>{L['exit_time']}</th><th>{L['stake']}</th><th>{L['pnl']}</th><th>%</th><th>{L['reason']}</th></tr>
"""
    html += trade_rows(short_trades)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">Generated: {timestamp}</p>
</div>
</body>
</html>"""

    # Write to file
    output_dir.mkdir(exist_ok=True)
    suffix = "_de" if lang == "de" else ""
    output_path = output_dir / f"dashboard{suffix}.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Generate trading dashboard from simulation data")
    parser.add_argument("--dir", "--output-dir", type=str, default="report_testnet", help="Directory with JSON files")
    parser.add_argument("--start", type=str, help="Only show trades from this date (YYYY-MM-DD)")
    parser.add_argument("--loop", action="store_true", help="Run continuously, refresh every 60 seconds")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval in seconds (default: 60)")
    args = parser.parse_args()

    output_dir = Path(args.dir)

    # Parse start date
    start_date = None
    if args.start:
        try:
            start_date = datetime.strptime(args.start, "%Y-%m-%d")
            print(f"Filtering trades from: {args.start}")
        except ValueError:
            print(f"Invalid date format: {args.start} (use YYYY-MM-DD)")

    print(f"Reading from: {output_dir}")

    if args.loop:
        print(f"Loop mode: refreshing every {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                path_en = generate_dashboard(output_dir, lang="en", start_date=start_date)
                path_de = generate_dashboard(output_dir, lang="de", start_date=start_date)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        path_en = generate_dashboard(output_dir, lang="en", start_date=start_date)
        path_de = generate_dashboard(output_dir, lang="de", start_date=start_date)
        print(f"\nOpen:\n  {path_en}\n  {path_de}")
