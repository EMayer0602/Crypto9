#!/usr/bin/env python3
"""Generate HTML dashboard from simulation trades.

This dashboard reads from report_html/trading_summary.json to show
all trades with recalculated stakes from initial capital.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from html.parser import HTMLParser


# Capital settings - same as paper_trader.py
START_CAPITAL = 16500.0
MAX_POSITIONS = 10
START_DATE = "2025-12-01"  # Trades before this date are ignored

# Fixed paths
SOURCE_JSON = Path("report_html/trading_summary.json")
SOURCE_HTML = Path("report_html/trading_summary.html")  # Fallback only
OPEN_POSITIONS_JSON = Path("paper_trading_open_positions.json")
OUTPUT_DIR = Path("report_testnet")


class PandasTableParser(HTMLParser):
    """Parse pandas-generated HTML tables from trading_summary.html."""

    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.is_header_cell = False
        self.current_row = []
        self.headers = []
        self.rows = []
        self.tables = []
        self.current_table_name = ""
        self.in_h2 = False
        self.h2_text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.rows = []
            self.headers = []
        elif tag == "thead":
            self.in_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag == "th" and self.in_row:
            self.in_cell = True
            self.is_header_cell = True
        elif tag == "td" and self.in_row:
            self.in_cell = True
            self.is_header_cell = False
        elif tag == "h2":
            self.in_h2 = True
            self.h2_text = ""

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
            if self.headers or self.rows:
                self.tables.append({
                    "name": self.current_table_name,
                    "headers": self.headers,
                    "rows": self.rows
                })
            self.rows = []
            self.headers = []
        elif tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr":
            self.in_row = False
            if self.current_row:
                if self.in_thead:
                    self.headers = self.current_row
                else:
                    self.rows.append(self.current_row)
            self.current_row = []
        elif tag in ("th", "td"):
            self.in_cell = False
            self.is_header_cell = False
        elif tag == "h2":
            self.in_h2 = False
            self.current_table_name = self.h2_text.strip()

    def handle_data(self, data):
        if self.in_cell:
            self.current_row.append(data.strip())
        elif self.in_h2:
            self.h2_text += data


def parse_trades_from_html(html_path: Path) -> tuple[list, list]:
    """Parse closed trades and open positions from trading_summary.html."""
    if not html_path.exists():
        print(f"Warning: {html_path} not found")
        return [], []

    html_content = html_path.read_text(encoding="utf-8")

    parser = PandasTableParser()
    parser.feed(html_content)

    long_trades = []
    short_trades = []
    open_positions = []

    for table in parser.tables:
        name = table["name"].lower()
        headers = [h.lower() for h in table.get("headers", [])]
        rows = table["rows"]

        if "long trades" in name and "open" not in name:
            # Parse Long Trades table (pandas format)
            for row in rows:
                trade = parse_pandas_trade_row(headers, row, "long")
                if trade:
                    long_trades.append(trade)

        elif "short trades" in name and "open" not in name:
            # Parse Short Trades table (pandas format)
            for row in rows:
                trade = parse_pandas_trade_row(headers, row, "short")
                if trade:
                    short_trades.append(trade)

        elif "open" in name.lower():
            # Parse Open Positions table
            for row in rows:
                pos = parse_pandas_open_position_row(headers, row)
                if pos:
                    open_positions.append(pos)

    # Only Long trades - we don't trade Short
    closed_trades = long_trades
    return closed_trades, open_positions


def parse_pandas_trade_row(headers: list, row: list, default_direction: str) -> dict | None:
    """Parse a single trade row from pandas-generated HTML table."""
    if not row or len(row) < 5:
        return None

    # Create a dict from headers and row values
    data = {}
    for i, h in enumerate(headers):
        if i < len(row):
            data[h] = row[i]

    symbol = data.get("symbol", "")
    if not symbol or "symbol" in symbol.lower():
        return None

    # Get entry and exit prices
    entry_price = parse_number(data.get("entry_price", "0"))
    exit_price = parse_number(data.get("exit_price", "0"))

    # Calculate PnL percentage from prices
    direction = data.get("direction", default_direction).lower()
    if entry_price > 0:
        if direction == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
    else:
        pnl_pct = 0

    return {
        "symbol": symbol,
        "indicator": data.get("indicator", ""),
        "htf": data.get("htf", ""),
        "entry_time": data.get("entry_time", ""),
        "exit_time": data.get("exit_time", ""),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "original_stake": parse_number(data.get("stake", "0")),
        "original_pnl": parse_number(data.get("pnl", "0")),
        "original_pnl_pct": pnl_pct,
        "reason": data.get("reason", ""),
        "direction": direction,
    }


def parse_pandas_open_position_row(headers: list, row: list) -> dict | None:
    """Parse a single open position row from pandas-generated HTML table."""
    if not row or len(row) < 5:
        return None

    # Create a dict from headers and row values
    data = {}
    for i, h in enumerate(headers):
        if i < len(row):
            data[h] = row[i]

    symbol = data.get("symbol", "")
    if not symbol or "symbol" in symbol.lower():
        return None

    return {
        "symbol": symbol,
        "direction": data.get("direction", "long").lower(),
        "indicator": data.get("indicator", ""),
        "htf": data.get("htf", ""),
        "entry_time": data.get("entry_time", ""),
        "entry_price": parse_number(data.get("entry_price", "0")),
        "last_price": parse_number(data.get("last_price", data.get("exit_price", "0"))),
        "bars_held": int(parse_number(data.get("bars_held", data.get("bars", "0")))),
    }


def parse_number(s: str) -> float:
    """Parse number from string, handling various formats."""
    if not s or s == "-":
        return 0.0
    # Remove any HTML tags
    s = re.sub(r"<[^>]+>", "", s)
    # Remove currency symbols and whitespace
    s = s.replace("USDT", "").replace("$", "").strip()
    # Handle German format: 1.234,56 -> 1234.56
    if "," in s and "." in s:
        # Determine which is decimal separator
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            # German format: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:
            # US format: 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0


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


def parse_entry_time(entry_time: str) -> datetime:
    """Parse entry_time string to datetime."""
    try:
        return datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
    except:
        try:
            return datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.min


def parse_trades_from_json(json_path: Path, start_date: str = None) -> tuple[list, list]:
    """Parse closed trades and open positions from trading_summary.json.

    Args:
        json_path: Path to trading_summary.json
        start_date: Optional start date filter (YYYY-MM-DD). Trades before this are skipped.
    """
    if not json_path.exists():
        print(f"Warning: {json_path} not found, falling back to HTML")
        return [], []

    data = load_json(json_path)
    if not isinstance(data, dict):
        print(f"Warning: Invalid JSON format in {json_path}")
        return [], []

    # Parse start_date filter
    start_dt = None
    if start_date:
        start_dt = datetime.fromisoformat(start_date + "T00:00:00+01:00")

    # Get trades from JSON
    trades = data.get("trades", [])
    open_positions = data.get("open_positions_data", [])

    # Filter to Long only + start_date filter
    long_trades = []
    for t in trades:
        direction = str(t.get("direction", "long")).lower()
        if direction != "long":
            continue

        entry_time = t.get("entry_time", "")

        # Apply start_date filter early
        if start_dt:
            entry_dt = parse_entry_time(entry_time)
            if entry_dt < start_dt:
                continue

        # Calculate PnL percentage from prices
        entry_price = float(t.get("entry_price", 0) or 0)
        exit_price = float(t.get("exit_price", 0) or 0)
        if entry_price > 0:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = 0

        long_trades.append({
            "symbol": t.get("symbol", ""),
            "indicator": t.get("indicator", ""),
            "htf": t.get("htf", ""),
            "entry_time": entry_time,
            "exit_time": t.get("exit_time", ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "original_stake": float(t.get("stake", 0) or 0),
            "original_pnl": float(t.get("pnl", 0) or 0),
            "original_pnl_pct": pnl_pct,
            "reason": t.get("reason", ""),
            "direction": "long",
        })

    # Parse open positions with start_date filter
    parsed_open = []
    for p in open_positions:
        direction = str(p.get("direction", "long")).lower()
        if direction != "long":
            continue

        entry_time = p.get("entry_time", "")

        # Apply start_date filter early
        if start_dt:
            entry_dt = parse_entry_time(entry_time)
            if entry_dt < start_dt:
                continue

        parsed_open.append({
            "symbol": p.get("symbol", ""),
            "direction": direction,
            "indicator": p.get("indicator", ""),
            "htf": p.get("htf", ""),
            "entry_time": entry_time,
            "entry_price": float(p.get("entry_price", 0) or 0),
            "last_price": float(p.get("last_price", 0) or 0),
            "bars_held": int(float(p.get("bars_held", 0) or 0)),
        })

    return long_trades, parsed_open


def fmt_de(value: float) -> str:
    """Format number in German style."""
    if abs(value) >= 1000:
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:.2f}".replace(".", ",")


def generate_dashboard(start_date: str = None, output_dir: Path = None, german: bool = False):
    """Generate HTML dashboard from simulation JSON file."""
    print(f"Loading simulation data from JSON (start_date={start_date})...")

    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Try JSON first (primary source) - filter is applied during parsing
    closed_trades, json_open_positions = parse_trades_from_json(SOURCE_JSON, start_date=start_date)
    if not closed_trades:
        print("Falling back to HTML parsing...")
        closed_trades, json_open_positions = parse_trades_from_html(SOURCE_HTML)
        # Apply filter for HTML fallback
        if start_date:
            start_dt = datetime.fromisoformat(start_date + "T00:00:00+01:00")
            closed_trades = [t for t in closed_trades if parse_entry_time(t.get("entry_time", "")) >= start_dt]
            json_open_positions = [p for p in json_open_positions if parse_entry_time(p.get("entry_time", "")) >= start_dt]

    # Load open positions from separate JSON file (paper_trading_open_positions.json)
    open_positions_file = load_json(OPEN_POSITIONS_JSON)
    if isinstance(open_positions_file, list) and open_positions_file:
        # Filter to Long only + start_date
        open_positions = []
        start_dt = datetime.fromisoformat(start_date + "T00:00:00+01:00") if start_date else None
        for p in open_positions_file:
            if str(p.get("direction", "long")).lower() != "long":
                continue
            if start_dt:
                entry_dt = parse_entry_time(p.get("entry_time", ""))
                if entry_dt < start_dt:
                    continue
            open_positions.append(p)
    else:
        open_positions = json_open_positions

    print(f"Loaded {len(closed_trades)} closed trades, {len(open_positions)} open positions")

    if not closed_trades:
        print("Warning: No trades found. Check if the file format is correct.")

    # Sort helper (reuse parse_entry_time)
    def get_entry_time(t):
        return parse_entry_time(t.get("entry_time", ""))

    # === RECALCULATE CLOSED TRADES WITH COMPOUND GROWTH ===
    # Sort chronologically (oldest first) for compound calculation
    trades_chrono = sorted(closed_trades, key=get_entry_time, reverse=False)

    capital = START_CAPITAL
    recalculated_trades = []

    for t in trades_chrono:
        stake = capital / MAX_POSITIONS
        original_pnl_pct = t.get("original_pnl_pct", 0) / 100  # Convert from percentage

        # Recalculate PnL with new stake
        pnl = original_pnl_pct * stake
        capital += pnl

        # Create recalculated trade entry
        recalc_trade = dict(t)
        recalc_trade["stake"] = stake
        recalc_trade["pnl"] = pnl
        recalc_trade["pnl_pct"] = original_pnl_pct * 100
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
    <title>Trading Dashboard</title>
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
    <p class="source-note">Simulation trades, recalculated with {fmt_de(START_CAPITAL)} initial capital</p>

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
            <h3>Open Positions</h3>
            <div class="value">{len(open_positions)}</div>
        </div>
        <div class="summary-box">
            <h3>Open Equity</h3>
            <div class="value {'positive' if open_equity >= 0 else 'negative'}">{fmt_de(open_equity)}</div>
        </div>
    </div>
"""

    # Open Positions section (FIRST - before closed trades)
    html += f"""
    <div class="section">
    <h2>Open Positions ({len(open_positions)}, Equity: <span class="{'positive' if open_equity >= 0 else 'negative'}">{fmt_de(open_equity)}</span>)</h2>
    <table>
        <tr class="open-header"><th>Symbol</th><th>Direction</th><th>Indicator</th><th>HTF</th><th>Entry Time</th><th>Entry Price</th><th>Last Price</th><th>Stake</th><th>Bars</th><th>PnL</th><th>Status</th></tr>
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
            return "        <tr><td colspan='9'>-</td></tr>\n"
        rows = ""
        for t in trades:
            entry_time = t.get("entry_time", "")
            exit_time = t.get("exit_time", "")
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
            <td>{fmt_de(stake)}</td>
            <td class="{pnl_class}">{fmt_de(pnl)}</td>
            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            <td>{t.get('reason', 'N/A')}</td>
        </tr>\n"""
        return rows

    # Long Trades section
    html += f"""
    <div class="section">
    <h2>Closed Trades ({len(long_trades)}, PnL: <span class="{'positive' if long_pnl >= 0 else 'negative'}">{fmt_de(long_pnl)}</span>)</h2>
    <table>
        <tr class="long-header"><th>Symbol</th><th>Indicator</th><th>HTF</th><th>Entry Time</th><th>Exit Time</th><th>Stake</th><th>PnL</th><th>%</th><th>Reason</th></tr>
"""
    html += trade_rows(long_trades)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>
    </div>

    <p class="timestamp">Generated: {timestamp}</p>
</div>
</body>
</html>"""

    # Write to file
    output_dir.mkdir(exist_ok=True)
    filename = "dashboard_de.html" if german else "dashboard.html"
    output_path = output_dir / filename
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Generate trading dashboard")
    parser.add_argument("--start", type=str, default=START_DATE, help="Start date YYYY-MM-DD (default: 2025-12-01)")
    parser.add_argument("--output-dir", type=str, default="report_testnet", help="Output directory")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds (default: 300)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    while True:
        # Generate both English and German dashboards
        path_en = generate_dashboard(start_date=args.start, output_dir=output_dir, german=False)
        path_de = generate_dashboard(start_date=args.start, output_dir=output_dir, german=True)
        print(f"\nOpen with: xdg-open {path_en}")
        print(f"German:    xdg-open {path_de}")

        if not args.loop:
            break

        print(f"\nWaiting {args.interval} seconds...")
        time.sleep(args.interval)
