#!/usr/bin/env python3
"""Regenerate trading_summary.html with filtered start date and compound growth stakes."""

import json
import re
from datetime import datetime
from pathlib import Path
from html.parser import HTMLParser


# Configuration
START_DATE = "2024-01-31"  # Filter trades from this date
START_CAPITAL = 16500.0
MAX_POSITIONS = 10
INPUT_HTML = Path("report_html/trading_summary.html")
OUTPUT_HTML = Path("report_html/trading_summary.html")
OUTPUT_JSON = Path("report_html/trading_summary.json")


class PandasTableParser(HTMLParser):
    """Parse pandas-generated HTML tables."""

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


def parse_number(s: str) -> float:
    """Parse number from string."""
    if not s or s == "-":
        return 0.0
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("USDT", "").replace("$", "").strip()
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0


def fmt_de(value, decimals=2):
    """Format number in German style."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def parse_datetime(s):
    """Parse datetime string."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.min


def main():
    print(f"Reading {INPUT_HTML}...")
    html_content = INPUT_HTML.read_text(encoding="utf-8")

    parser = PandasTableParser()
    parser.feed(html_content)

    # Extract Long Trades
    trades = []
    open_positions = []

    for table in parser.tables:
        name = table["name"].lower()
        headers = [h.lower() for h in table.get("headers", [])]
        rows = table["rows"]

        if "long trades" in name and "open" not in name:
            for row in rows:
                if len(row) < 5:
                    continue
                data = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                symbol = data.get("symbol", "")
                if not symbol or "symbol" in symbol.lower():
                    continue

                entry_time = data.get("entry_time", "")
                entry_dt = parse_datetime(entry_time)

                trades.append({
                    "symbol": symbol,
                    "direction": data.get("direction", "Long"),
                    "indicator": data.get("indicator", ""),
                    "htf": data.get("htf", ""),
                    "entry_time": entry_time,
                    "entry_price": parse_number(data.get("entry_price", "0")),
                    "exit_time": data.get("exit_time", ""),
                    "exit_price": parse_number(data.get("exit_price", "0")),
                    "reason": data.get("reason", ""),
                    "entry_dt": entry_dt,
                })

        elif "open" in name.lower():
            for row in rows:
                if len(row) < 5:
                    continue
                data = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                symbol = data.get("symbol", "")
                if not symbol or "symbol" in symbol.lower():
                    continue

                entry_time = data.get("entry_time", "")
                entry_dt = parse_datetime(entry_time)

                open_positions.append({
                    "symbol": symbol,
                    "direction": data.get("direction", "Long"),
                    "indicator": data.get("indicator", ""),
                    "htf": data.get("htf", ""),
                    "entry_time": entry_time,
                    "entry_price": parse_number(data.get("entry_price", "0")),
                    "last_price": parse_number(data.get("last_price", data.get("exit_price", "0"))),
                    "bars_held": int(parse_number(data.get("bars_held", data.get("bars", "0")))),
                    "entry_dt": entry_dt,
                })

    print(f"Found {len(trades)} Long trades, {len(open_positions)} open positions")

    # Filter by start date
    start_dt = datetime.fromisoformat(f"{START_DATE}T00:00:00+01:00")
    trades = [t for t in trades if t["entry_dt"] >= start_dt]
    open_positions = [p for p in open_positions if p["entry_dt"] >= start_dt]

    print(f"After filtering from {START_DATE}: {len(trades)} trades, {len(open_positions)} open positions")

    # Sort chronologically for compound growth
    trades.sort(key=lambda t: t["entry_dt"])

    # Recalculate stakes with compound growth
    capital = START_CAPITAL
    for t in trades:
        stake = capital / MAX_POSITIONS
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]

        if entry_price > 0:
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl = pnl_pct * stake
        else:
            pnl_pct = 0
            pnl = 0

        t["stake"] = stake
        t["pnl"] = pnl
        t["pnl_pct"] = pnl_pct * 100
        capital += pnl

    final_capital = capital

    # Recalculate open positions
    open_stake = final_capital / MAX_POSITIONS
    for p in open_positions:
        entry_price = p["entry_price"]
        last_price = p["last_price"]

        if entry_price > 0:
            pnl_pct = (last_price - entry_price) / entry_price
            unrealized_pnl = pnl_pct * open_stake
        else:
            pnl_pct = 0
            unrealized_pnl = 0

        p["stake"] = open_stake
        p["unrealized_pnl"] = unrealized_pnl
        p["unrealized_pct"] = pnl_pct * 100
        p["status"] = "Gewinn" if unrealized_pnl > 0 else "Verlust" if unrealized_pnl < 0 else "Flat"

    # Sort trades by entry time descending for display
    trades.sort(key=lambda t: t["entry_dt"], reverse=True)
    open_positions.sort(key=lambda t: t["entry_dt"], reverse=True)

    # Calculate statistics
    total_pnl = sum(t["pnl"] for t in trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] < 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    avg_pnl = total_pnl / len(trades) if trades else 0
    open_equity = sum(p["unrealized_pnl"] for p in open_positions)

    # Calculate by symbol
    symbol_stats = {}
    for t in trades:
        sym = t["symbol"]
        if sym not in symbol_stats:
            symbol_stats[sym] = {"trades": [], "pnl": 0}
        symbol_stats[sym]["trades"].append(t)
        symbol_stats[sym]["pnl"] += t["pnl"]

    # Build symbol stats list
    symbol_list = []
    for sym, data in sorted(symbol_stats.items(), key=lambda x: -x[1]["pnl"]):
        sym_trades = data["trades"]
        sym_wins = [t for t in sym_trades if t["pnl"] > 0]
        sym_losses = [t for t in sym_trades if t["pnl"] < 0]
        sym_pnl = data["pnl"]
        best = max(t["pnl"] for t in sym_trades) if sym_trades else 0
        worst = min(t["pnl"] for t in sym_trades) if sym_trades else 0

        # Max drawdown calculation
        running = 0
        peak = 0
        max_dd = 0
        for t in sorted(sym_trades, key=lambda x: x["entry_dt"]):
            running += t["pnl"]
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        gross_profit = sum(t["pnl"] for t in sym_wins)
        gross_loss = abs(sum(t["pnl"] for t in sym_losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        symbol_list.append({
            "symbol": sym,
            "trades": len(sym_trades),
            "wins": len(sym_wins),
            "losses": len(sym_losses),
            "win_rate": len(sym_wins) / len(sym_trades) * 100 if sym_trades else 0,
            "pnl": sym_pnl,
            "avg_pnl": sym_pnl / len(sym_trades) if sym_trades else 0,
            "best": best,
            "worst": worst,
            "max_dd": max_dd,
            "pf": pf,
        })

    # Get time range
    if trades:
        start_ts = min(t["entry_dt"] for t in trades)
        end_ts = max(parse_datetime(t["exit_time"]) for t in trades)
    else:
        start_ts = start_dt
        end_ts = datetime.now()

    # Generate HTML
    html = f"""<html><head><meta charset='utf-8'><title>Paper Trading Simulation Summary</title><style>body{{font-family:Arial,sans-serif;margin:20px;}}table{{border-collapse:collapse;margin-top:12px;width:auto;}}th,td{{border:1px solid #ccc;padding:6px 10px;text-align:right;}}th{{text-align:center;background:#f0f0f0;font-weight:bold;}}td:first-child{{text-align:left;}}h1{{margin-bottom:10px;}}h2{{margin-top:30px;margin-bottom:10px;}}.stats-container{{display:flex;gap:20px;flex-wrap:wrap;}}</style></head><body><h1>Simulation Summary {start_ts.isoformat()} → {end_ts.isoformat()}</h1><h2>Statistics</h2><table><tr><th>Metric</th><th>Overall</th><th>Long</th><th>Short</th></tr><tr><td>Closed trades</td><td>{len(trades)}</td><td>{len(trades)}</td><td>0</td></tr><tr><td>Open positions</td><td>{len(open_positions)}</td><td>{len(open_positions)}</td><td>0</td></tr><tr><td>PnL (USDT)</td><td>{fmt_de(total_pnl)}</td><td>{fmt_de(total_pnl)}</td><td>0,00</td></tr><tr><td>Avg PnL (USDT)</td><td>{fmt_de(avg_pnl)}</td><td>{fmt_de(avg_pnl)}</td><td>0,00</td></tr><tr><td>Win rate (%)</td><td>{fmt_de(win_rate)}</td><td>{fmt_de(win_rate)}</td><td>0,00</td></tr><tr><td>Winners</td><td>{len(winners)}</td><td>{len(winners)}</td><td>0</td></tr><tr><td>Losers</td><td>{len(losers)}</td><td>{len(losers)}</td><td>0</td></tr><tr><td>Open equity (USDT)</td><td>{fmt_de(open_equity)}</td><td>{fmt_de(open_equity)}</td><td>0,00</td></tr><tr style='font-weight:bold;'><td>Final capital (USDT)</td><td>{fmt_de(final_capital + open_equity)}</td><td>-</td><td>-</td></tr></table>"""

    # Symbol stats table
    html += """<h2>Statistics by Symbol</h2><table><tr><th>Symbol</th><th>Trades</th><th>Win</th><th>Loss</th><th>Win%</th><th>Total PnL</th><th>Avg PnL</th><th>Best</th><th>Worst</th><th>Max DD</th><th>PF</th><th>Long</th><th>Short</th><th>Long PnL</th><th>Short PnL</th></tr>"""
    for s in symbol_list:
        html += f"""<tr><td>{s['symbol']}</td><td>{s['trades']}</td><td>{s['wins']}</td><td>{s['losses']}</td><td>{s['win_rate']:.1f}%</td><td style='color:{'green' if s['pnl'] >= 0 else 'red'}'>{fmt_de(s['pnl'])}</td><td>{fmt_de(s['avg_pnl'])}</td><td style='color:green'>{fmt_de(s['best'])}</td><td style='color:red'>{fmt_de(s['worst'])}</td><td style='color:orange'>{fmt_de(s['max_dd'])}</td><td>{s['pf']:.2f}</td><td>{s['trades']}</td><td>0</td><td>{fmt_de(s['pnl'])}</td><td>0,00</td></tr>"""
    html += "</table>"

    # Open positions table
    if open_positions:
        html += f"""<h2>Long Open Positions ({len(open_positions)} positions, Equity: {fmt_de(open_equity)} USDT)</h2><table border="1" class="dataframe"><thead><tr style="text-align: right;"><th>symbol</th><th>direction</th><th>indicator</th><th>htf</th><th>entry_time</th><th>entry_price</th><th>last_price</th><th>stake</th><th>bars_held</th><th>unrealized_pnl</th><th>status</th></tr></thead><tbody>"""
        for p in open_positions:
            html += f"""<tr><td>{p['symbol']}</td><td>{p['direction']}</td><td>{p['indicator']}</td><td>{p['htf']}</td><td>{p['entry_time']}</td><td>{p['entry_price']:.8f}</td><td>{p['last_price']:.8f}</td><td>{fmt_de(p['stake'], 8)}</td><td>{p['bars_held']}</td><td>{fmt_de(p['unrealized_pnl'], 8)}</td><td>{p['status']}</td></tr>"""
        html += "</tbody></table>"

    # Trades table
    html += f"""<h2>Long Trades ({len(trades)} trades, PnL: {fmt_de(total_pnl)} USDT)</h2><table border="1" class="dataframe"><thead><tr style="text-align: right;"><th>symbol</th><th>direction</th><th>indicator</th><th>htf</th><th>entry_time</th><th>entry_price</th><th>exit_time</th><th>exit_price</th><th>stake</th><th>pnl</th><th>reason</th></tr></thead><tbody>"""
    for t in trades:
        html += f"""<tr><td>{t['symbol']}</td><td>{t['direction']}</td><td>{t['indicator']}</td><td>{t['htf']}</td><td>{t['entry_time']}</td><td>{t['entry_price']:.8f}</td><td>{t['exit_time']}</td><td>{t['exit_price']:.8f}</td><td>{fmt_de(t['stake'], 8)}</td><td>{fmt_de(t['pnl'], 8)}</td><td>{t['reason']}</td></tr>"""
    html += "</tbody></table></body></html>"

    # Write HTML
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved {OUTPUT_HTML}")

    # Build JSON
    trades_json = []
    for t in trades:
        trades_json.append({
            "symbol": t["symbol"],
            "direction": t["direction"].lower(),
            "indicator": t["indicator"],
            "htf": t["htf"],
            "entry_time": t["entry_time"],
            "entry_price": t["entry_price"],
            "exit_time": t["exit_time"],
            "exit_price": t["exit_price"],
            "stake": t["stake"],
            "pnl": t["pnl"],
            "reason": t["reason"],
        })

    open_json = []
    for p in open_positions:
        open_json.append({
            "symbol": p["symbol"],
            "direction": p["direction"].lower(),
            "indicator": p["indicator"],
            "htf": p["htf"],
            "entry_time": p["entry_time"],
            "entry_price": p["entry_price"],
            "last_price": p["last_price"],
            "stake": p["stake"],
            "bars_held": p["bars_held"],
            "unrealized_pnl": p["unrealized_pnl"],
            "status": p["status"],
        })

    summary_json = {
        "generated_at": datetime.now().isoformat(),
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "closed_trades": len(trades),
        "open_positions": len(open_positions),
        "closed_pnl": round(total_pnl, 6),
        "avg_trade_pnl": round(avg_pnl, 6),
        "win_rate_pct": round(win_rate, 4),
        "winners": len(winners),
        "losers": len(losers),
        "open_equity": round(open_equity, 6),
        "final_capital": round(final_capital + open_equity, 6),
        "long_trades": len(trades),
        "long_pnl": round(total_pnl, 6),
        "long_avg_pnl": round(avg_pnl, 6),
        "long_win_rate": round(win_rate, 4),
        "long_winners": len(winners),
        "long_losers": len(losers),
        "short_trades": 0,
        "short_pnl": 0.0,
        "short_avg_pnl": 0.0,
        "short_win_rate": 0.0,
        "short_winners": 0,
        "short_losers": 0,
        "long_open": len(open_positions),
        "long_open_equity": round(open_equity, 6),
        "short_open": 0,
        "short_open_equity": 0.0,
        "symbol_stats": symbol_list,
        "open_positions_data": open_json,
        "trades": trades_json,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"Saved {OUTPUT_JSON}")

    print(f"\nSummary:")
    print(f"  Trades: {len(trades)}")
    print(f"  PnL: {fmt_de(total_pnl)} USDT")
    print(f"  Winners: {len(winners)}")
    print(f"  Losers: {len(losers)}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Final Capital: {fmt_de(final_capital + open_equity)} USDT")


if __name__ == "__main__":
    main()
