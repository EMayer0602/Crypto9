#!/usr/bin/env python3
"""
Phase Baseline: Run paper_trader simulation per indicator, tag with 4 phase classifiers.

Uses paper_trader.run_simulation() directly to get IDENTICAL trades as the BCK.
Then tags each trade with 4 different phase classifiers.
Uses paper_trader.build_summary_payload() for correct compound PnL,
then renders dark-blue themed HTML with phase column.

Generates per (trading indicator, phase classifier):
  - trading_summary_ph1_{indicator}_{classifier}.json
  - trading_summary_ph1_{indicator}_{classifier}.html

Usage:
    python test_baseline_phases.py
    python test_baseline_phases.py --sim-start 2024-01-31
"""
import argparse
import os
from collections import Counter
from datetime import datetime

import pandas as pd

import Supertrend_5Min as st
import paper_trader as pt

# Phase classification: FIXED params (independent of trading params)
PHASE_HTF_LENGTH = 10
PHASE_HTF_FACTOR = 3.0
PHASE_HTF = "6h"

TRADING_INDICATORS = ["jma"]

CLASSIFIER_DISPLAY = {
    "supertrend": "ST-Phasen",
    "htf_crossover": "HTF-Phasen",
    "jma": "JMA-Phasen",
    "kama": "KAMA-Phasen",
}

PHASE_COLORS = {"Up": "#27ae60", "Down": "#e74c3c", "Flat": "#f39c12"}


def get_phase_at_entry(phases, entry_ts):
    """Look up market phase at entry timestamp using ffill."""
    if phases is None or phases.empty:
        return "Flat"
    try:
        if entry_ts in phases.index:
            return phases.loc[entry_ts]
        idx = phases.index.get_indexer([entry_ts], method="ffill")[0]
        if idx >= 0:
            return phases.iloc[idx]
    except Exception:
        pass
    return "Flat"


def _fmt_de(val):
    s = f"{val:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_price(val):
    v = abs(val)
    if v < 0.0001:
        return f"{val:.8f}"
    elif v < 1:
        return f"{val:.6f}"
    elif v < 100:
        return f"{val:.4f}"
    return _fmt_de(val)


def _apply_compound(trades, max_positions):
    """Apply compound growth from START_EQUITY, stake = capital/max_positions."""
    capital = st.START_EQUITY
    result = []
    for t in trades:
        stake = capital / max_positions
        ep = float(t.get("entry_price", 0) or 0)
        xp = float(t.get("exit_price", 0) or 0)
        direction = str(t.get("direction", "long")).lower()
        if direction == "long":
            pnl_pct = (xp - ep) / ep if ep else 0
        else:
            pnl_pct = (ep - xp) / ep if ep else 0
        pnl = pnl_pct * stake
        capital += pnl
        tc = dict(t)
        tc["stake"] = stake
        tc["pnl"] = pnl
        tc["pnl_pct"] = pnl_pct
        tc["equity_after"] = capital
        result.append(tc)
    return result


def write_dark_phase_html(summary, title, html_path):
    """Write dark-blue themed HTML from a build_summary_payload() result."""
    now = datetime.now(st.BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    trades = summary.get("trades", [])
    open_positions = summary.get("open_positions_data", [])

    start_cap = st.START_EQUITY
    n_closed = summary.get("closed_trades", len(trades))
    win_rate = summary.get("win_rate_pct", 0)
    max_pos = pt.MAX_OPEN_POSITIONS

    # Compound-recalculate: final closed capital from compound trades
    compound_closed_pnl = sum(float(t.get("pnl", 0) or 0) for t in trades)
    final_closed_capital = start_cap + compound_closed_pnl
    closed_pnl = compound_closed_pnl

    # Recalculate open position stakes: uniform stake = final_closed_capital / max_pos
    uniform_stake = final_closed_capital / max_pos
    for op in open_positions:
        ep = float(op.get("entry_price", 0) or 0)
        lp = float(op.get("last_price", 0) or op.get("exit_price", 0) or 0)
        direction = str(op.get("direction", "long")).lower()
        if direction == "long":
            pnl_pct = (lp - ep) / ep if ep else 0
        else:
            pnl_pct = (ep - lp) / ep if ep else 0
        op["stake"] = uniform_stake
        op["unrealized_pnl"] = pnl_pct * uniform_stake
        op["unrealized_pct"] = pnl_pct * 100

    n_open = len(open_positions)

    # Open PnL (recalculated)
    open_pnl = sum(float(op.get("unrealized_pnl", 0) or 0) for op in open_positions)
    final_cap = final_closed_capital + open_pnl

    cap_color = "#27ae60" if final_cap >= start_cap else "#e74c3c"
    pnl_color = "#27ae60" if closed_pnl >= 0 else "#e74c3c"
    open_pnl_color = "#27ae60" if open_pnl >= 0 else "#e74c3c"

    # Trade rows
    def _trade_row(t, show_exit=True):
        entry_p = float(t.get("entry_price", 0) or 0)
        phase = t.get("phase", "")
        phase_color = PHASE_COLORS.get(phase, "#888")
        stake = float(t.get("stake", 0) or 0)

        if show_exit:
            # Closed trade: calculate from exit_price
            exit_p = float(t.get("exit_price", 0) or 0)
            pnl = float(t.get("pnl", 0) or 0)
            pnl_pct = (exit_p / entry_p - 1) * 100 if entry_p else 0
        else:
            # Open position: use unrealized fields
            pnl = float(t.get("unrealized_pnl", 0) or 0)
            pnl_pct = float(t.get("unrealized_pct", 0) or 0)

        pnl_cls = "pos" if pnl >= 0 else "neg"

        cols = [
            f'<td>{t.get("symbol", "")}</td>',
            f'<td style="color:{phase_color};font-weight:bold">{phase}</td>',
            f'<td>{t.get("indicator", "")}</td>',
            f'<td>{t.get("htf", "")}</td>',
            f'<td>{str(t.get("entry_time", ""))[:16]}</td>',
            f'<td style="text-align:right">{_fmt_price(entry_p)}</td>',
        ]
        if show_exit:
            exit_p = float(t.get("exit_price", 0) or 0)
            cols.append(f'<td>{str(t.get("exit_time", ""))[:16]}</td>')
            cols.append(f'<td style="text-align:right">{_fmt_price(exit_p)}</td>')
        else:
            last_p = float(t.get("last_price", 0) or t.get("exit_price", 0) or 0)
            cols.append(f'<td style="text-align:right">{_fmt_price(last_p)}</td>')
        cols += [
            f'<td style="text-align:right">{_fmt_de(stake)}</td>',
            f'<td class="{pnl_cls}" style="text-align:right">{_fmt_de(pnl)}</td>',
            f'<td class="{pnl_cls}" style="text-align:right">{pnl_pct:+.2f}%</td>',
        ]
        if show_exit:
            reason = t.get("exit_reason", "") or t.get("reason", "")
            cols.append(f'<td>{reason}</td>')
        else:
            cols.append('<td>Open</td>')
        return "<tr>" + "".join(cols) + "</tr>"

    # Open position rows
    if open_positions:
        open_rows = "\n".join(_trade_row(p, show_exit=False) for p in open_positions)
    else:
        open_rows = '<tr><td colspan="11" style="text-align:center;color:#888">Keine offenen Positionen</td></tr>'

    # Closed trade rows (newest first)
    sorted_trades = sorted(trades, key=lambda t: t.get("entry_time", ""), reverse=True)
    if sorted_trades:
        closed_rows = "\n".join(_trade_row(t, show_exit=True) for t in sorted_trades)
    else:
        closed_rows = '<tr><td colspan="12" style="text-align:center;color:#888">Keine Trades</td></tr>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #00d2ff; margin-bottom: 5px; }}
h2 {{ color: #f39c12; margin-top: 30px; }}
p.sub {{ color: #888; margin-top: 0; }}
.cards {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }}
.card {{ background: #16213e; border-radius: 8px; padding: 15px 20px; min-width: 150px; }}
.card .label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
.card .value {{ font-size: 22px; font-weight: bold; margin-top: 5px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }}
th {{ background: #16213e; color: #00d2ff; padding: 8px 10px; text-align: left; border-bottom: 2px solid #0f3460; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #333; }}
tr:hover {{ background: #16213e; }}
.pos {{ color: #27ae60; }}
.neg {{ color: #e74c3c; }}
</style>
</head><body>
<h1>{title}</h1>
<p class="sub">Stand: {now} | Compound Growth (stake = kapital/{max_pos})</p>

<div class="cards">
<div class="card"><div class="label">Start Kapital</div><div class="value">{_fmt_de(start_cap)} USDT</div></div>
<div class="card"><div class="label">Final Kapital</div><div class="value" style="color:{cap_color}">{_fmt_de(final_cap)} USDT</div></div>
<div class="card"><div class="label">Closed Trades</div><div class="value">{n_closed}</div></div>
<div class="card"><div class="label">Realized PnL</div><div class="value" style="color:{pnl_color}">{_fmt_de(closed_pnl)} USDT</div></div>
<div class="card"><div class="label">Open Positionen (max {max_pos})</div><div class="value">{n_open}</div></div>
<div class="card"><div class="label">Open PnL</div><div class="value" style="color:{open_pnl_color}">{_fmt_de(open_pnl)} USDT</div></div>
<div class="card"><div class="label">Win Rate</div><div class="value">{win_rate:.1f}%</div></div>
</div>

<h2>Open Positionen</h2>
<table>
<tr><th>Symbol</th><th>Phase</th><th>Indikator</th><th>HTF</th><th>Entry</th><th>Entry Preis</th><th>Aktuell</th><th>Stake</th><th>PnL</th><th>PnL%</th><th>Status</th></tr>
{open_rows}
</table>

<h2>Closed Trades ({n_closed})</h2>
<table>
<tr><th>Symbol</th><th>Phase</th><th>Indikator</th><th>HTF</th><th>Entry</th><th>Entry Preis</th><th>Exit</th><th>Exit Preis</th><th>Stake</th><th>PnL</th><th>PnL%</th><th>Reason</th></tr>
{closed_rows}
</table>

</body></html>"""

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Phase HTML] {html_path}")


def write_dark_dashboard_html(all_trades, date_from, title, html_path, max_positions,
                              open_positions_data=None):
    """Filter trades from date_from, apply fresh compound from 16500, write dark-blue HTML."""
    filtered = sorted(
        [t for t in all_trades if str(t.get("entry_time", ""))[:10] >= date_from],
        key=lambda t: t.get("entry_time", ""),
    )
    processed = _apply_compound(filtered, max_positions)

    now = datetime.now(st.BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    start_cap = st.START_EQUITY
    n_closed = len(processed)
    closed_pnl = sum(t["pnl"] for t in processed) if processed else 0
    final_closed_cap = processed[-1]["equity_after"] if processed else start_cap
    winners = sum(1 for t in processed if t["pnl"] > 0)
    win_rate = (winners / n_closed * 100) if n_closed else 0

    # Open positions: recalculate stakes from dashboard compound equity
    open_t = []
    for op in (open_positions_data or []):
        op = dict(op)  # copy
        uniform_stake = final_closed_cap / max_positions
        ep = float(op.get("entry_price", 0) or 0)
        lp = float(op.get("last_price", 0) or op.get("exit_price", 0) or 0)
        direction = str(op.get("direction", "long")).lower()
        if direction == "long":
            pnl_pct = (lp - ep) / ep if ep else 0
        else:
            pnl_pct = (ep - lp) / ep if ep else 0
        op["stake"] = uniform_stake
        op["unrealized_pnl"] = pnl_pct * uniform_stake
        op["unrealized_pct"] = pnl_pct * 100
        open_t.append(op)

    open_pnl = sum(float(t.get("unrealized_pnl", 0) or 0) for t in open_t)
    final_cap = final_closed_cap + open_pnl

    cap_color = "#27ae60" if final_cap >= start_cap else "#e74c3c"
    pnl_color = "#27ae60" if closed_pnl >= 0 else "#e74c3c"
    open_pnl_color = "#27ae60" if open_pnl >= 0 else "#e74c3c"

    def _closed_row(t):
        pnl = t["pnl"]
        pnl_pct = t["pnl_pct"] * 100
        pnl_cls = "pos" if pnl >= 0 else "neg"
        phase = t.get("phase", "")
        phase_color = PHASE_COLORS.get(phase, "#888")
        cols = [
            f'<td>{t.get("symbol", "")}</td>',
            f'<td style="color:{phase_color};font-weight:bold">{phase}</td>',
            f'<td>{t.get("indicator", "")}</td>',
            f'<td>{t.get("htf", "")}</td>',
            f'<td>{str(t.get("entry_time", ""))[:16]}</td>',
            f'<td style="text-align:right">{_fmt_price(t["entry_price"])}</td>',
            f'<td>{str(t.get("exit_time", ""))[:16]}</td>',
            f'<td style="text-align:right">{_fmt_price(t["exit_price"])}</td>',
            f'<td style="text-align:right">{_fmt_de(t["stake"])}</td>',
            f'<td class="{pnl_cls}" style="text-align:right">{_fmt_de(pnl)}</td>',
            f'<td class="{pnl_cls}" style="text-align:right">{pnl_pct:+.2f}%</td>',
            f'<td>{t.get("reason", "")}</td>',
        ]
        return "<tr>" + "".join(cols) + "</tr>"

    def _open_row(t):
        pnl = float(t.get("unrealized_pnl", 0) or 0)
        pnl_pct = float(t.get("unrealized_pct", 0) or 0)
        pnl_cls = "pos" if pnl >= 0 else "neg"
        phase = t.get("phase", "")
        phase_color = PHASE_COLORS.get(phase, "#888")
        entry_p = float(t.get("entry_price", 0) or 0)
        last_p = float(t.get("last_price", entry_p) or entry_p)
        stake = float(t.get("stake", 0) or 0)
        status = t.get("status", "Gewinn" if pnl >= 0 else "Verlust")
        cols = [
            f'<td>{t.get("symbol", "")}</td>',
            f'<td style="color:{phase_color};font-weight:bold">{phase}</td>',
            f'<td>{t.get("indicator", "")}</td>',
            f'<td>{t.get("htf", "")}</td>',
            f'<td>{str(t.get("entry_time", ""))[:16]}</td>',
            f'<td style="text-align:right">{_fmt_price(entry_p)}</td>',
            f'<td style="text-align:right">{_fmt_price(last_p)}</td>',
            f'<td style="text-align:right">{_fmt_de(stake)}</td>',
            f'<td class="{pnl_cls}" style="text-align:right">{_fmt_de(pnl)}</td>',
            f'<td class="{pnl_cls}" style="text-align:right">{pnl_pct:+.2f}%</td>',
            f'<td>{status}</td>',
        ]
        return "<tr>" + "".join(cols) + "</tr>"

    open_rows = "\n".join(_open_row(t) for t in open_t) if open_t else \
        '<tr><td colspan="11" style="text-align:center;color:#888">Keine offenen Positionen</td></tr>'
    closed_rows = "\n".join(_closed_row(t) for t in reversed(processed)) if processed else \
        '<tr><td colspan="12" style="text-align:center;color:#888">Keine Trades</td></tr>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>{title} (ab {date_from})</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #00d2ff; margin-bottom: 5px; }}
h2 {{ color: #f39c12; margin-top: 30px; }}
p.sub {{ color: #888; margin-top: 0; }}
.cards {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }}
.card {{ background: #16213e; border-radius: 8px; padding: 15px 20px; min-width: 150px; }}
.card .label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
.card .value {{ font-size: 22px; font-weight: bold; margin-top: 5px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }}
th {{ background: #16213e; color: #00d2ff; padding: 8px 10px; text-align: left; border-bottom: 2px solid #0f3460; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #333; }}
tr:hover {{ background: #16213e; }}
.pos {{ color: #27ae60; }}
.neg {{ color: #e74c3c; }}
</style>
</head><body>
<h1>{title}</h1>
<p class="sub">Ab {date_from} | Stand: {now} | Compound Growth (stake = kapital/{max_positions})</p>

<div class="cards">
<div class="card"><div class="label">Start Kapital</div><div class="value">{_fmt_de(start_cap)} USDT</div></div>
<div class="card"><div class="label">Final Kapital</div><div class="value" style="color:{cap_color}">{_fmt_de(final_cap)} USDT</div></div>
<div class="card"><div class="label">Closed Trades</div><div class="value">{n_closed}</div></div>
<div class="card"><div class="label">Realized PnL</div><div class="value" style="color:{pnl_color}">{_fmt_de(closed_pnl)} USDT</div></div>
<div class="card"><div class="label">Open Positionen (max {max_positions})</div><div class="value">{len(open_t)}</div></div>
<div class="card"><div class="label">Open PnL</div><div class="value" style="color:{open_pnl_color}">{_fmt_de(open_pnl)} USDT</div></div>
<div class="card"><div class="label">Win Rate</div><div class="value">{win_rate:.1f}%</div></div>
</div>

<h2>Open Positionen</h2>
<table>
<tr><th>Symbol</th><th>Phase</th><th>Indikator</th><th>HTF</th><th>Entry</th><th>Entry Preis</th><th>Aktuell</th><th>Stake</th><th>PnL</th><th>PnL%</th><th>Status</th></tr>
{open_rows}
</table>

<h2>Closed Trades ({n_closed})</h2>
<table>
<tr><th>Symbol</th><th>Phase</th><th>Indikator</th><th>HTF</th><th>Entry</th><th>Entry Preis</th><th>Exit</th><th>Exit Preis</th><th>Stake</th><th>PnL</th><th>PnL%</th><th>Reason</th></tr>
{closed_rows}
</table>

</body></html>"""

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Dashboard] {html_path} ({n_closed} trades ab {date_from})")


def main():
    parser = argparse.ArgumentParser(description="Phase baseline using paper_trader simulation")
    parser.add_argument("--start", type=str, default="2024-01-31",
                        help="Simulation start date (default: 2024-01-31)")
    parser.add_argument("--dashboard-start", type=str, default="2025-01-01",
                        help="Dashboard start date (default: 2025-01-01)")
    args = parser.parse_args()

    dashboard_start = args.dashboard_start
    sim_start = pd.Timestamp(args.start, tz=st.BERLIN_TZ)
    sim_end = pd.Timestamp.now(tz=st.BERLIN_TZ)
    symbols = st.SYMBOLS

    print(f"=== Phase Baseline (via paper_trader) ===")
    print(f"Indicators: {TRADING_INDICATORS}")
    print(f"MAX_OPEN_POSITIONS: {pt.MAX_OPEN_POSITIONS} | STAKE_DIVISOR: {pt.STAKE_DIVISOR}")
    print(f"Simulation: {args.start} to now | Dashboard ab {dashboard_start}")

    # Save original globals
    orig_htf_length = st.HTF_LENGTH
    orig_htf_factor = st.HTF_FACTOR

    for indicator in TRADING_INDICATORS:
        print(f"\n{'='*60}")
        print(f"Trading indicator: {indicator}")
        print(f"{'='*60}")

        # ── Step 1: Run simulation (same engine as BCK) ──
        trade_results, sim_state = pt.run_simulation(
            start_ts=sim_start,
            end_ts=sim_end,
            use_saved_state=False,
            allowed_indicators=[indicator],
            use_testnet=False,
            reset_state=True,
        )

        # Convert to DataFrame using paper_trader's own function
        trades_df = pt.trades_to_dataframe(trade_results)

        # Open positions from sim_state
        open_positions = sim_state.get("positions", [])
        open_df = pt.open_positions_to_dataframe(open_positions)

        n_closed = len(trades_df)
        n_open = len(open_df)
        print(f"  Closed: {n_closed} | Open: {n_open}")
        if not trades_df.empty:
            sym_counts = Counter(trades_df["symbol"].values)
            pnl_sum = float(trades_df["pnl"].sum())
            print(f"  PnL: {pnl_sum:+,.2f} | Symbols: {len(sym_counts)}")

        # ── Step 2: Tag with 4 phase classifiers ──
        for classifier in sorted(CLASSIFIER_DISPLAY.keys()):
            cls_display = CLASSIFIER_DISPLAY[classifier]
            print(f"\n  Phase: {cls_display}")

            st.apply_indicator_type(classifier)
            st.apply_higher_timeframe(PHASE_HTF)
            st.HTF_LENGTH = PHASE_HTF_LENGTH
            st.HTF_FACTOR = PHASE_HTF_FACTOR
            st.clear_data_cache()

            # Classify phases per symbol
            phase_cache = {}
            for symbol in symbols:
                df_for_phases = st.fetch_data(symbol, st.TIMEFRAME, st.LOOKBACK)
                if df_for_phases.empty:
                    phase_cache[symbol] = pd.Series(dtype=str)
                    continue
                phases = st.classify_market_phases(df_for_phases, symbol)
                phase_cache[symbol] = phases

            # Tag trades with phase
            tagged_df = trades_df.copy()
            phase_tags = []
            for _, row in tagged_df.iterrows():
                phases_s = phase_cache.get(row["symbol"], pd.Series(dtype=str))
                entry_ts = pd.Timestamp(row["entry_time"])
                phase = get_phase_at_entry(phases_s, entry_ts)
                phase_tags.append(phase)
            tagged_df["phase"] = phase_tags

            phase_dist = Counter(phase_tags)
            print(f"    Distribution: {dict(phase_dist)}")

            # Build summary using paper_trader's own function (same as BCK)
            summary = pt.build_summary_payload(
                tagged_df, open_df, sim_state, sim_start, sim_end
            )

            # Recalculate summary stats from compound-recalculated trade PnLs
            # (build_summary_payload recalculates stake/pnl in the export,
            #  but uses original simulation PnLs for summary stats - fix that)
            exported_trades = summary.get("trades", [])
            if exported_trades:
                compound_pnl = sum(float(t.get("pnl", 0) or 0) for t in exported_trades)
                compound_winners = sum(1 for t in exported_trades if float(t.get("pnl", 0) or 0) > 0)
                compound_losers = sum(1 for t in exported_trades if float(t.get("pnl", 0) or 0) < 0)
                n_trades = len(exported_trades)
                compound_wr = (compound_winners / n_trades * 100) if n_trades else 0
                open_eq = summary.get("open_equity", 0)
                summary["closed_pnl"] = round(compound_pnl, 6)
                summary["avg_trade_pnl"] = round(compound_pnl / n_trades, 6) if n_trades else 0
                summary["win_rate_pct"] = round(compound_wr, 4)
                summary["winners"] = compound_winners
                summary["losers"] = compound_losers
                summary["final_capital"] = round(st.START_EQUITY + compound_pnl + open_eq, 6)
                # Long stats (all trades are long)
                summary["long_pnl"] = summary["closed_pnl"]
                summary["long_avg_pnl"] = summary["avg_trade_pnl"]
                summary["long_win_rate"] = summary["win_rate_pct"]
                summary["long_winners"] = summary["winners"]
                summary["long_losers"] = summary["losers"]

            # Write JSON
            prefix = f"ph1_{indicator}_{classifier}"
            json_path = os.path.join(st.BASE_OUT_DIR, f"trading_summary_{prefix}.json")
            pt.write_summary_json(summary, json_path)

            # Write dark-blue themed Summary HTML with phase column
            ind_display = indicator.upper()
            title = f"Phase Trading Summary — {ind_display} ({cls_display})"
            html_path = os.path.join(st.BASE_OUT_DIR, f"trading_summary_{prefix}.html")
            write_dark_phase_html(summary, title, html_path)

            # Write dark-blue themed Dashboard HTML (trades ab dashboard_start, fresh compound)
            dash_title = f"Phase Dashboard — {ind_display} ({cls_display})"
            dash_path = os.path.join(st.BASE_OUT_DIR, f"dashboard_{prefix}.html")
            dash_trades = summary.get("trades", [])
            open_pos_data = summary.get("open_positions_data", [])
            write_dark_dashboard_html(
                dash_trades, dashboard_start, dash_title, dash_path, pt.MAX_OPEN_POSITIONS,
                open_positions_data=open_pos_data,
            )

            print(f"    -> {json_path} ({summary['closed_trades']} closed, PnL: {summary['closed_pnl']:+,.2f})")

    # Restore globals
    st.HTF_LENGTH = orig_htf_length
    st.HTF_FACTOR = orig_htf_factor

    print(f"\n{'='*60}")
    print(f"Done! {len(TRADING_INDICATORS)} indicators × 4 classifiers = {len(TRADING_INDICATORS) * 4} output sets")


if __name__ == "__main__":
    main()
