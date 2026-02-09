#!/usr/bin/env python3
"""
Exact Hourly Delay Backtest - Uses minute data for precise prices.

Problem mit dem alten Skript:
- Bei 1h OHLCV-Daten wird immer die naechste volle Stundenkerze genommen
- Signal um 10:30 + 1h Delay = Preis von 12:00 statt 11:30

Loesung:
- Minutendaten (1m) von Binance laden
- Exakten Preis bei T+Xh zurueckgeben

Usage:
    python test_hourly_delay_exact.py                    # Test 1h to 8h
    python test_hourly_delay_exact.py --hours 1,2,4,8   # Specific hours
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import time

import pandas as pd

import Supertrend_5Min as st

# Settings
SOURCE_JSON = Path("report_html/trading_summary.json")
START_CAPITAL = 16500.0
MAX_POSITIONS = 10

# Cache for minute data per symbol
_minute_cache: Dict[str, pd.DataFrame] = {}


def load_trades(start_date: str = "2024-01-31") -> list:
    """Load long trades from JSON."""
    with open(SOURCE_JSON, "r") as f:
        data = json.load(f)

    start_dt = datetime.fromisoformat(start_date + "T00:00:00")
    trades = []

    for t in data.get("trades", []):
        if str(t.get("direction", "")).lower() != "long":
            continue

        entry_time = t.get("entry_time", "")
        try:
            entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            entry_dt = entry_dt.replace(tzinfo=None)
        except:
            try:
                entry_dt = datetime.strptime(entry_time[:19], "%Y-%m-%d %H:%M:%S")
            except:
                continue

        if entry_dt < start_dt:
            continue

        entry_price = float(t.get("entry_price", 0) or 0)
        exit_price = float(t.get("exit_price", 0) or 0)

        if entry_price <= 0:
            continue

        trades.append({
            "symbol": t.get("symbol", ""),
            "entry_time": entry_dt,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": (exit_price - entry_price) / entry_price * 100
        })

    return trades


def fetch_minute_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch 1-minute OHLCV data from Binance.

    Caches data to avoid repeated API calls.
    """
    global _minute_cache

    if symbol in _minute_cache:
        return _minute_cache[symbol]

    print(f"  Lade 1m Daten fuer {symbol}...")

    try:
        exchange = st.get_data_exchange()

        # Calculate how many minutes we need
        total_minutes = int((end_date - start_date).total_seconds() / 60) + 60 * 24  # +1 day buffer

        all_data = []
        current_start = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        while current_start < end_ms:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe="1m",
                    since=current_start,
                    limit=1000
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                # Move to next batch
                last_ts = ohlcv[-1][0]
                if last_ts <= current_start:
                    break
                current_start = last_ts + 60000  # +1 minute

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"    Warnung: {e}")
                time.sleep(1)
                break

        if not all_data:
            return None

        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Make timezone naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        _minute_cache[symbol] = df
        print(f"    {len(df)} Minuten-Kerzen geladen ({df.index[0]} bis {df.index[-1]})")

        return df

    except Exception as e:
        print(f"    Fehler beim Laden von {symbol}: {e}")
        return None


def get_exact_price_at_delay(df: pd.DataFrame, entry_time: datetime, hours: int) -> Optional[float]:
    """Get the exact close price at entry_time + hours.

    Unlike the old function that rounds to the next hourly bar,
    this returns the actual price at the exact minute.
    """
    if df is None or df.empty:
        return None

    target_time = entry_time + timedelta(hours=hours)

    # Find the exact minute or closest before
    mask = df.index <= target_time
    if mask.any():
        idx = df.index[mask][-1]  # Last bar at or before target
        return float(df.loc[idx, "close"])

    # If no data before, try first bar after
    mask_after = df.index >= target_time
    if mask_after.any():
        idx = df.index[mask_after][0]
        # Only use if within 5 minutes
        if (idx - target_time).total_seconds() <= 300:
            return float(df.loc[idx, "close"])

    return None


def backtest_delay_exact(trades: list, delay_hours: int, minute_cache: dict) -> list:
    """Backtest with exact price at T+delay_hours using minute data."""

    passed = []
    skipped_no_data = 0

    for t in trades:
        symbol = t["symbol"]
        df = minute_cache.get(symbol)

        if df is None or df.empty:
            skipped_no_data += 1
            continue

        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        entry_time = t["entry_time"]

        # Exakter Preis nach delay_hours Stunden
        price_at_delay = get_exact_price_at_delay(df, entry_time, delay_hours)

        if price_at_delay is None:
            skipped_no_data += 1
            continue

        # Momentum-Check: Preis muss gestiegen sein
        if price_at_delay > entry_price:
            # Entry zum exakten verzögerten Preis, Exit zum Original-Exit
            new_pnl_pct = (exit_price - price_at_delay) / price_at_delay * 100
            passed.append({
                "symbol": symbol,
                "entry_time": entry_time,
                "pnl_pct": new_pnl_pct,
                "price_at_delay": price_at_delay,
                "original_entry": entry_price,
                "original_pnl_pct": t["pnl_pct"]
            })

    return passed


def calculate_stats(trades: list, passed: list) -> dict:
    """Calculate statistics."""
    total = len(trades)
    filtered = len(passed)

    if filtered == 0:
        return {
            "total": total,
            "filtered": filtered,
            "pass_rate": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "total_return": 0
        }

    wins = sum(1 for t in passed if t["pnl_pct"] > 0)
    win_rate = wins / filtered * 100
    avg_pnl = sum(t["pnl_pct"] for t in passed) / filtered

    # Compound growth
    capital = START_CAPITAL
    for t in passed:
        stake = capital / MAX_POSITIONS
        pnl = stake * (t["pnl_pct"] / 100)
        capital += pnl

    total_return = (capital - START_CAPITAL) / START_CAPITAL * 100

    return {
        "total": total,
        "filtered": filtered,
        "pass_rate": filtered / total * 100,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_return": total_return
    }


def generate_html_report(results: list, orig_win_rate: float, output_path: Path):
    """Generate HTML comparison report."""

    best_wr = max(results, key=lambda x: x["win_rate"])
    best_ret = max(results, key=lambda x: x["total_return"])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Exakter Stunden-Delay Backtest</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        .info {{ background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .info h3 {{ color: #00d4ff; margin-top: 0; }}
        table {{ border-collapse: collapse; width: 100%; background: #16213e; border-radius: 8px; overflow: hidden; }}
        th {{ background: #0f3460; color: #00d4ff; padding: 12px; text-align: center; }}
        td {{ padding: 10px; text-align: right; border-bottom: 1px solid #0f3460; }}
        td:first-child {{ text-align: center; font-weight: bold; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4757; }}
        .best {{ background: #0f3460; }}
        .summary {{ margin-top: 20px; background: #16213e; padding: 15px; border-radius: 8px; }}
        .timestamp {{ color: #666; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Exakter Stunden-Delay Backtest (mit Minutendaten)</h1>

    <div class="info">
        <h3>Unterschied zum alten Test</h3>
        <p><strong>Alt:</strong> Signal um 10:30 + 1h Delay → Preis von 12:00 (naechste volle Stundenkerze)</p>
        <p><strong>Neu:</strong> Signal um 10:30 + 1h Delay → Preis von 11:30 (exakter Zeitpunkt)</p>
        <p>Original Win Rate: <strong>{orig_win_rate:.1f}%</strong></p>
    </div>

    <table>
        <tr>
            <th>Delay</th>
            <th>Trades</th>
            <th>Pass Rate</th>
            <th>Win Rate</th>
            <th>vs Original</th>
            <th>Avg PnL</th>
            <th>Total Return</th>
        </tr>
"""

    for r in results:
        improvement = r["win_rate"] - orig_win_rate
        imp_class = "positive" if improvement > 0 else "negative"
        ret_class = "positive" if r["total_return"] > 0 else "negative"
        row_class = "best" if r == best_wr or r == best_ret else ""

        html += f"""        <tr class="{row_class}">
            <td>{r['hours']}h</td>
            <td>{r['filtered']}</td>
            <td>{r['pass_rate']:.1f}%</td>
            <td>{r['win_rate']:.1f}%</td>
            <td class="{imp_class}">{improvement:+.1f}%</td>
            <td>{r['avg_pnl']:.2f}%</td>
            <td class="{ret_class}">{r['total_return']:+.1f}%</td>
        </tr>
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""    </table>

    <div class="summary">
        <h3>Beste Ergebnisse</h3>
        <p>Beste Win Rate: <strong>{best_wr['hours']}h</strong> mit {best_wr['win_rate']:.1f}%</p>
        <p>Bester Return: <strong>{best_ret['hours']}h</strong> mit {best_ret['total_return']:+.1f}%</p>
    </div>

    <p class="timestamp">Generiert: {timestamp}</p>
</div>
</body>
</html>"""

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML Report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Exakter Stunden-Delay Backtest mit Minutendaten")
    parser.add_argument("--start", default="2024-01-31", help="Start date YYYY-MM-DD")
    parser.add_argument("--hours", default="1,2,3,4,5,6,7,8",
                        help="Comma-separated hours to test")
    args = parser.parse_args()

    hours_list = [int(h) for h in args.hours.split(",")]

    print(f"\n{'='*70}")
    print("EXAKTER STUNDEN-DELAY BACKTEST (mit Minutendaten)")
    print(f"{'='*70}")

    print(f"\nLade Trades seit {args.start}...")
    trades = load_trades(args.start)
    print(f"  {len(trades)} Long-Trades geladen")

    if not trades:
        print("Keine Trades gefunden!")
        return

    # Original stats
    orig_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    orig_win_rate = orig_wins / len(trades) * 100
    print(f"  Original Win Rate: {orig_win_rate:.1f}%")

    # Find date range
    min_date = min(t["entry_time"] for t in trades)
    max_date = max(t["entry_time"] for t in trades) + timedelta(hours=max(hours_list) + 24)

    print(f"\nLade Minutendaten von Binance...")
    print(f"  Zeitraum: {min_date.strftime('%Y-%m-%d')} bis {max_date.strftime('%Y-%m-%d')}")

    # Load minute data for all symbols
    symbols = list(set(t["symbol"] for t in trades))
    print(f"  {len(symbols)} Symbole zu laden...")

    minute_cache = {}
    for symbol in symbols:
        df = fetch_minute_data(symbol, min_date, max_date)
        if df is not None:
            minute_cache[symbol] = df

    print(f"\n  {len(minute_cache)} Symbole mit Minutendaten geladen")

    # Run backtests
    print(f"\n{'='*70}")
    print(f"{'Delay':>6} | {'Trades':>7} | {'Pass%':>7} | {'WinRate':>8} | "
          f"{'vs Orig':>8} | {'Avg PnL':>8} | {'Return':>8}")
    print("-" * 70)

    results = []

    for hours in hours_list:
        passed = backtest_delay_exact(trades, hours, minute_cache)
        stats = calculate_stats(trades, passed)

        improvement = stats["win_rate"] - orig_win_rate

        print(f"{hours:>5}h | {stats['filtered']:>7} | {stats['pass_rate']:>6.1f}% | "
              f"{stats['win_rate']:>7.1f}% | {improvement:>+7.1f}% | "
              f"{stats['avg_pnl']:>+7.2f}% | {stats['total_return']:>+7.1f}%")

        results.append({
            "hours": hours,
            **stats,
            "improvement": improvement
        })

    print("-" * 70)

    # Best results
    if results:
        best_wr = max(results, key=lambda x: x["win_rate"])
        best_ret = max(results, key=lambda x: x["total_return"])

        print(f"\nBeste Win Rate:  {best_wr['hours']}h mit {best_wr['win_rate']:.1f}%")
        print(f"Bester Return:   {best_ret['hours']}h mit {best_ret['total_return']:+.1f}%")

        # Generate HTML report
        html_path = Path("report_html/backtest_hourly_exact.html")
        generate_html_report(results, orig_win_rate, html_path)


if __name__ == "__main__":
    main()
