#!/usr/bin/env python3
"""
Delay Backtest - Synchroner Exit mit verzögertem Entry.

Konzept:
- Entry wird um X Minuten verzögert
- Exit erfolgt zur GLEICHEN Zeit wie das Original
- Preis bei T+Delay muss über dem Signal-Preis liegen (Momentum-Check)
- Entry-Preis = Preis bei T+Delay (kein Look-Ahead Bias)

Verwendung:
    python delay_backtest_sync.py --delays 5,10,15,30,45
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Settings
REPORT_DIR = Path("report_html")
SOURCE_JSON = REPORT_DIR / "trading_summary.json"
OHLCV_5M_CACHE = Path("ohlcv_cache_5m")
START_CAPITAL = 16500.0
MAX_POSITIONS = 10


def load_trades(json_path: Path, start_date: str = None) -> list:
    """Lade Trades aus JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trades = []
    start_dt = datetime.fromisoformat(start_date + "T00:00:00") if start_date else None

    for t in data.get("trades", []):
        if t.get("direction", "").lower() != "long":
            continue

        entry_time_str = t.get("entry_time", "")
        exit_time_str = t.get("exit_time", "")

        try:
            entry_dt = datetime.fromisoformat(entry_time_str).replace(tzinfo=None)
            exit_dt = datetime.fromisoformat(exit_time_str).replace(tzinfo=None)
        except:
            continue

        if start_dt and entry_dt < start_dt:
            continue

        trades.append({
            "symbol": t.get("symbol"),
            "entry_time": entry_dt,
            "exit_time": exit_dt,
            "entry_price": float(t.get("entry_price", 0)),
            "exit_price": float(t.get("exit_price", 0)),
        })

    return trades


def fetch_5m_ohlcv(exchange, symbol: str, since: datetime) -> pd.DataFrame:
    """Hole 5m OHLCV-Daten von der API."""
    OHLCV_5M_CACHE.mkdir(exist_ok=True)
    cache_file = OHLCV_5M_CACHE / f"{symbol.replace('/', '_')}_5m.csv"

    # Cache prüfen
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    print(f"  Fetching 5m data: {symbol}...")
    all_data = []
    current_since = int(since.timestamp() * 1000)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '5m', since=current_since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            if ohlcv[-1][0] >= datetime.now().timestamp() * 1000 - 300000:
                break
            current_since = ohlcv[-1][0] + 1
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df.to_csv(cache_file)
    print(f"    Cached: {len(df)} bars")
    return df


def get_price_at_delay(df: pd.DataFrame, entry_time: datetime, delay_min: int) -> float | None:
    """Hole Preis bei Entry + Delay Minuten."""
    if df is None or df.empty:
        return None

    target = entry_time + timedelta(minutes=delay_min)

    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    mask = df.index >= target
    if mask.any():
        return float(df.loc[df.index[mask][0], "close"])
    return None


def run_backtest(trades: list, delay_min: int, ohlcv_cache: dict) -> dict:
    """
    Backtest mit verzögertem Entry und synchronem Exit.

    Für jeden Trade:
    1. Prüfe ob Preis bei T+Delay > Entry-Preis (Momentum-Check)
    2. Wenn ja: Entry zum Preis bei T+Delay
    3. Exit zur GLEICHEN Zeit wie Original
    """
    passed_trades = []

    for t in trades:
        symbol = t["symbol"]
        entry_time = t["entry_time"]
        exit_time = t["exit_time"]
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]

        df = ohlcv_cache.get(symbol)
        if df is None:
            continue

        # Preis bei T+Delay holen
        price_at_delay = get_price_at_delay(df, entry_time, delay_min)
        if price_at_delay is None:
            continue

        # Momentum-Check: Preis muss gestiegen sein
        if price_at_delay <= entry_price:
            continue

        # Trade besteht den Filter
        # PnL berechnen: Entry bei price_at_delay, Exit bei exit_price
        pnl_pct = (exit_price - price_at_delay) / price_at_delay * 100

        passed_trades.append({
            "symbol": symbol,
            "entry_time": entry_time + timedelta(minutes=delay_min),
            "exit_time": exit_time,
            "signal_price": entry_price,
            "entry_price": price_at_delay,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "price_change_pct": (price_at_delay - entry_price) / entry_price * 100,
        })

    # Statistiken
    total = len(trades)
    passed = len(passed_trades)
    wins = sum(1 for t in passed_trades if t["pnl_pct"] > 0)
    win_rate = wins / passed * 100 if passed > 0 else 0

    # Equity-Kurve berechnen
    passed_trades.sort(key=lambda x: x["exit_time"])
    capital = START_CAPITAL
    equity_curve = []

    for t in passed_trades:
        stake = capital / MAX_POSITIONS
        pnl = stake * (t["pnl_pct"] / 100)
        capital += pnl
        equity_curve.append({
            "time": t["exit_time"],
            "equity": capital,
        })

    avg_price_delta = sum(t["price_change_pct"] for t in passed_trades) / passed if passed > 0 else 0
    avg_pnl = sum(t["pnl_pct"] for t in passed_trades) / passed if passed > 0 else 0
    total_return = (capital - START_CAPITAL) / START_CAPITAL * 100

    return {
        "delay_min": delay_min,
        "total_trades": total,
        "passed_trades": passed,
        "pass_rate": passed / total * 100 if total > 0 else 0,
        "wins": wins,
        "win_rate": win_rate,
        "avg_price_delta": avg_price_delta,
        "avg_pnl": avg_pnl,
        "final_capital": capital,
        "total_return": total_return,
        "equity_curve": equity_curve,
        "trades": passed_trades,
    }


def generate_dashboard(results: list, original_win_rate: float, output_path: Path):
    """Generiere HTML Dashboard mit Equity-Kurve."""

    # Plotly Chart
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve - Delay Strategien", "Win Rate Vergleich"),
        vertical_spacing=0.12
    )

    colors = ["#2196f3", "#4caf50", "#ff9800", "#9c27b0", "#f44336", "#00bcd4", "#795548"]

    for i, r in enumerate(results):
        delay = r["delay_min"]
        eq = r["equity_curve"]
        if not eq:
            continue

        times = [e["time"] for e in eq]
        equities = [e["equity"] for e in eq]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=equities,
                mode="lines",
                name=f"{delay}m",
                line=dict(color=colors[i % len(colors)], width=1.5),
            ),
            row=1, col=1
        )

    # Win Rate Bar
    labels = [f"{r['delay_min']}m" for r in results]
    win_rates = [r["win_rate"] for r in results]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=win_rates,
            marker_color=colors[:len(results)],
            text=[f"{wr:.1f}%" for wr in win_rates],
            textposition="outside",
            showlegend=False,
        ),
        row=2, col=1
    )

    # Original Win Rate Line
    fig.add_hline(y=original_win_rate, line_dash="dash", line_color="gray",
                  annotation_text=f"Original: {original_win_rate:.1f}%", row=2, col=1)

    fig.update_layout(height=900, title="Momentum Delay Dashboard", hovermode="x unified")
    fig.update_xaxes(title_text="Zeit", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # HTML generieren
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Delay Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #2196f3; padding-bottom: 10px; }}
        .note {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #2196f3; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
        th {{ background: #2196f3; color: white; }}
        td:first-child {{ text-align: center; font-weight: bold; }}
        .pos {{ color: green; font-weight: bold; }}
        .neg {{ color: red; font-weight: bold; }}
        .best {{ background: #e8f5e9; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Delay Dashboard - Momentum Filter</h1>

    <div class="note">
        <strong>Konzept:</strong> Wenn der Preis nach X Minuten gestiegen ist, wird zum verzögerten Preis eingestiegen.<br>
        <strong>Kein Look-Ahead Bias:</strong> Entry-Preis = Preis bei T+Delay (nicht Signal-Preis)<br>
        <strong>Original Win Rate:</strong> {original_win_rate:.1f}%
    </div>

    <table>
        <tr>
            <th>Delay</th>
            <th>Trades</th>
            <th>Pass Rate</th>
            <th>Win Rate</th>
            <th>vs Original</th>
            <th>Avg Price Δ</th>
            <th>Avg PnL</th>
            <th>Total Return</th>
        </tr>
"""

    best_wr = max(results, key=lambda x: x["win_rate"])

    for r in results:
        diff = r["win_rate"] - original_win_rate
        diff_class = "pos" if diff > 0 else ("neg" if diff < 0 else "")
        ret_class = "pos" if r["total_return"] > 0 else "neg"
        row_class = "best" if r == best_wr else ""

        html += f"""        <tr class="{row_class}">
            <td>{r['delay_min']}m</td>
            <td>{r['passed_trades']}</td>
            <td>{r['pass_rate']:.1f}%</td>
            <td>{r['win_rate']:.1f}%</td>
            <td class="{diff_class}">{diff:+.1f}%</td>
            <td>+{r['avg_price_delta']:.2f}%</td>
            <td>{r['avg_pnl']:.2f}%</td>
            <td class="{ret_class}">{r['total_return']:+.1f}%</td>
        </tr>
"""

    html += f"""    </table>

    <div style="background: white; padding: 20px; border-radius: 5px; margin: 20px 0;">
        {chart_html}
    </div>

    <p style="color: #666; font-size: 12px;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Delay Backtest mit synchronem Exit")
    parser.add_argument("--start", type=str, default="2024-01-31")
    parser.add_argument("--delays", type=str, default="5,10,15,30,45")
    parser.add_argument("--exchange", type=str, default="hyperliquid")
    parser.add_argument("--output", type=str, default="report_html/delay_sync_dashboard.html")
    args = parser.parse_args()

    delays = [int(x.strip()) for x in args.delays.split(",")]

    # Trades laden
    print(f"Loading trades from {SOURCE_JSON}...")
    trades = load_trades(SOURCE_JSON, args.start)
    print(f"Loaded: {len(trades)} trades")

    # Original Win Rate
    original_wins = sum(1 for t in trades if t["exit_price"] > t["entry_price"])
    original_win_rate = original_wins / len(trades) * 100

    # Symbole
    symbols = list(set(t["symbol"] for t in trades))
    earliest = min(t["entry_time"] for t in trades) - timedelta(days=1)

    # OHLCV laden
    print(f"\nFetching 5m OHLCV from {args.exchange}...")
    if args.exchange == "hyperliquid":
        exchange = ccxt.hyperliquid({'enableRateLimit': True})
    else:
        exchange = ccxt.binance({'enableRateLimit': True})

    ohlcv_cache = {}
    for sym in symbols:
        try:
            df = fetch_5m_ohlcv(exchange, sym, earliest)
            if not df.empty:
                ohlcv_cache[sym] = df
        except Exception as e:
            print(f"  Error {sym}: {e}")

    print(f"Loaded: {len(ohlcv_cache)} symbols")

    # Backtests
    print("\nRunning backtests...")
    results = []
    for delay in delays:
        print(f"  {delay}m...")
        r = run_backtest(trades, delay, ohlcv_cache)
        results.append(r)
        print(f"    Trades: {r['passed_trades']} | Win Rate: {r['win_rate']:.1f}% | Return: {r['total_return']:+.1f}%")

    # Dashboard
    generate_dashboard(results, original_win_rate, Path(args.output))
    print(f"\nOpen: xdg-open {args.output}")


if __name__ == "__main__":
    main()
