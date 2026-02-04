#!/usr/bin/env python3
"""Generate HTML dashboard from simulation trades.

This dashboard reads from the simulation's JSON output files to show
the exact same trades as trading_summary.html, just in dashboard format.
Stakes are recalculated with compound growth from initial capital.
"""

import json
import os
from datetime import datetime
from pathlib import Path


# Capital settings - same as paper_trader.py
START_CAPITAL = 16500.0
MAX_POSITIONS = 10

# Default paths - same as paper_trader.py
REPORT_DIR = Path("report_testnet")
CLOSED_TRADES_JSON = "paper_trading_simulation_log.json"
OPEN_POSITIONS_JSON = "paper_trading_actual_trades.json"
SUMMARY_JSON = "trading_summary.json"

# For testnet, files are in report_testnet/
TESTNET_CLOSED_TRADES_JSON = REPORT_DIR / "paper_trading_simulation_log.json"
TESTNET_OPEN_POSITIONS_JSON = REPORT_DIR / "paper_trading_actual_trades.json"
TESTNET_SUMMARY_JSON = REPORT_DIR / "trading_summary.json"


def load_json(path: Path) -> list | dict:
    """Load JSON file, return empty list/dict if not found."""
    if not path.exists():
        # Try without report_testnet prefix
        alt_path = Path(path.name)
        if alt_path.exists():
            path = alt_path
        else:
            print(f"Warning: {path} not found")
            return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def fmt_de(value: float) -> str:
    """Format number in German style."""
    if abs(value) >= 1000:
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:.2f}".replace(".", ",")


def generate_dashboard(use_testnet: bool = True):
    """Generate HTML dashboard from simulation JSON files."""
    print("Loading simulation data...")

    # Determine paths based on testnet mode
    if use_testnet:
        closed_path = TESTNET_CLOSED_TRADES_JSON
        open_path = TESTNET_OPEN_POSITIONS_JSON
        summary_path = TESTNET_SUMMARY_JSON
        output_dir = REPORT_DIR
    else:
        closed_path = Path(CLOSED_TRADES_JSON)
        open_path = Path(OPEN_POSITIONS_JSON)
        summary_path = Path(SUMMARY_JSON)
        output_dir = Path("report_html")

    # Load data
    closed_trades = load_json(closed_path)
    open_positions = load_json(open_path)
    summary = load_json(summary_path)

    if not isinstance(closed_trades, list):
        closed_trades = []
    if not isinstance(open_positions, list):
        open_positions = []
    if not isinstance(summary, dict):
        summary = {}

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

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Trading Dashboard (from Simulation)</title>
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
    <h1>Trading Dashboard</h1>
    <p class="source-note">Data source: Simulation trades, recalculated with {fmt_de(START_CAPITAL)} initial capital</p>

    <div class="summary-boxes">
        <div class="summary-box">
            <h3>Start Capital</h3>
            <div class="value">{fmt_de(START_CAPITAL)}</div>
        </div>
        <div class="summary-box">
            <h3>Final Capital</h3>
            <div class="value {'positive' if final_capital >= START_CAPITAL else 'negative'}">{fmt_de(final_capital)}</div>
        </div>
        <div class="summary-box">
            <h3>Closed Trades</h3>
            <div class="value">{len(recalculated_trades)}</div>
        </div>
        <div class="summary-box">
            <h3>Realized PnL</h3>
            <div class="value {'positive' if total_pnl >= 0 else 'negative'}">{fmt_de(total_pnl)}</div>
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
        <div class="summary-box">
            <h3>Open Equity</h3>
            <div class="value {'positive' if open_equity >= 0 else 'negative'}">{fmt_de(open_equity)}</div>
        </div>
    </div>
"""

    # Open Positions section
    html += f"""
    <div class="section">
    <h2>Open Positions ({len(open_positions)}, Equity: <span class="{'positive' if open_equity >= 0 else 'negative'}">{fmt_de(open_equity)}</span>)</h2>
    <table>
        <tr class="open-header"><th>Symbol</th><th>Direction</th><th>Indicator</th><th>HTF</th><th>Entry Time</th><th>Entry Price</th><th>Last Price</th><th>Stake</th><th>Bars</th><th>Unrealized PnL</th><th>Status</th></tr>
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
            <td>{fmt_de(p.get('stake', 0))}</td>
            <td>{p.get('bars_held', 0)}</td>
            <td class="{pnl_class}">{fmt_de(pnl)}</td>
            <td>{p.get('status', 'N/A')}</td>
        </tr>\n"""
    else:
        html += "        <tr><td colspan='11'>No open positions</td></tr>\n"

    html += "    </table>\n    </div>\n"

    # Helper function to generate closed trade table rows
    def trade_rows(trades):
        if not trades:
            return "        <tr><td colspan='10'>No trades</td></tr>\n"
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
            pnl_pct = t.get("pnl_pct", 0)  # Already calculated as percentage
            pnl_class = "positive" if pnl >= 0 else "negative"

            rows += f"""        <tr>
            <td>{t.get('symbol', 'N/A')}</td>
            <td>{t.get('indicator', 'N/A')}</td>
            <td>{t.get('htf', 'N/A')}</td>
            <td>{entry_str}</td>
            <td>{exit_str}</td>
            <td>{fmt_de(stake)}</td>
            <td class="{pnl_class}">{fmt_de(pnl)}</td>
            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            <td>{t.get('reason', 'N/A')}</td>
        </tr>\n"""
        return rows

    # Long Trades section
    html += f"""
    <div class="section">
    <h2>Long Trades ({len(long_trades)} closed, PnL: <span class="{'positive' if long_pnl >= 0 else 'negative'}">{fmt_de(long_pnl)}</span>)</h2>
    <table>
        <tr class="long-header"><th>Symbol</th><th>Indicator</th><th>HTF</th><th>Entry</th><th>Exit</th><th>Stake</th><th>PnL</th><th>PnL %</th><th>Reason</th></tr>
"""
    html += trade_rows(long_trades)

    # Short Trades section
    html += f"""    </table>
    </div>

    <div class="section">
    <h2>Short Trades ({len(short_trades)} closed, PnL: <span class="{'positive' if short_pnl >= 0 else 'negative'}">{fmt_de(short_pnl)}</span>)</h2>
    <table>
        <tr class="short-header"><th>Symbol</th><th>Indicator</th><th>HTF</th><th>Entry</th><th>Exit</th><th>Stake</th><th>PnL</th><th>PnL %</th><th>Reason</th></tr>
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
    output_path = output_dir / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate trading dashboard from simulation data")
    parser.add_argument("--testnet", action="store_true", default=True, help="Use testnet report directory")
    parser.add_argument("--live", action="store_true", help="Use live report directory")
    args = parser.parse_args()

    use_testnet = not args.live
    path = generate_dashboard(use_testnet=use_testnet)
    print(f"\nOpen with: xdg-open {path}")
