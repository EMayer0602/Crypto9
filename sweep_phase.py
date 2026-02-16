#!/usr/bin/env python3
"""
Phase-filtered parameter sweep.

Runs per-symbol backtests for all JMA param combinations,
filters trades by market phase (Up/Down/Flat), and finds
the best parameters for each phase.

Tracks both full-period and dashboard-period (ab --dash-start) metrics.

Usage:
    python sweep_phase.py --phase Down
    python sweep_phase.py --phase Down --phase-length 20 --phase-factor 4.0
"""
import argparse
import os
import time
from collections import defaultdict

import pandas as pd

import Supertrend_5Min as st

# Sweep dimensions (same as run_parameter_sweep)
JMA_PARAM_A = [20, 30, 50]       # Length
JMA_PARAM_B = [-50, 0, 50]       # Phase
ATR_MULTS = [None, 1.0, 1.5, 2.0]
MIN_HOLD_VALUES = [0, 12, 24]
MAX_HOLD_VALUES = [0, 24, 48, 60]

DASHBOARD_START = "2025-12-01"


def get_phase_at(phases_series, ts):
    """Look up market phase at timestamp using ffill."""
    if phases_series is None or phases_series.empty:
        return "Flat"
    try:
        if ts in phases_series.index:
            return phases_series.loc[ts]
        idx = phases_series.index.get_indexer([ts], method="ffill")[0]
        if idx >= 0:
            return phases_series.iloc[idx]
    except Exception:
        pass
    return "Flat"


def fmt_time(seconds):
    """Format seconds as Xm Ys."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s"


def sweep_phase(target_phase, phase_htf_length=10, phase_htf_factor=3.0,
                phase_htf="6h", phase_classifier="jma", dash_start=DASHBOARD_START):
    """
    Sweep JMA trading params, evaluate only trades in target_phase.

    Returns dict keyed by (param_a, param_b, atr, min_hold, max_hold)
    with full-period AND dashboard-period stats.
    """
    symbols = st.SYMBOLS

    # Save original settings
    orig_indicator = st.INDICATOR_TYPE
    orig_htf = st.HIGHER_TIMEFRAME
    orig_htf_len = st.HTF_LENGTH
    orig_htf_fac = st.HTF_FACTOR

    # Pre-compute phases per symbol using phase classifier
    print(f"\n=== Phase Sweep: {target_phase} ===")
    print(f"Phase classifier: {phase_classifier} (HTF={phase_htf}, L={phase_htf_length}, F={phase_htf_factor})")
    print(f"Trading indicator: JMA")
    print(f"Dashboard ab: {dash_start}")
    print(f"Symbols: {len(symbols)}")

    st.apply_indicator_type(phase_classifier)
    st.apply_higher_timeframe(phase_htf)
    st.HTF_LENGTH = phase_htf_length
    st.HTF_FACTOR = phase_htf_factor
    st.clear_data_cache()

    phase_cache = {}
    for symbol in symbols:
        df_phases = st.fetch_data(symbol, st.TIMEFRAME, st.LOOKBACK)
        if df_phases.empty:
            phase_cache[symbol] = pd.Series(dtype=str)
            continue
        phases = st.classify_market_phases(df_phases, symbol)
        phase_cache[symbol] = phases
        n_up = (phases == "Up").sum()
        n_down = (phases == "Down").sum()
        n_flat = (phases == "Flat").sum()
        print(f"  {symbol}: Up={n_up}, Down={n_down}, Flat={n_flat}")

    # Switch to JMA trading indicator
    st.apply_indicator_type("jma")
    st.clear_data_cache()

    # Ensure cache populated
    st.ensure_cache_populated(symbols, st.TIMEFRAME, st.LOOKBACK)

    combos_per_sym = len(JMA_PARAM_A) * len(JMA_PARAM_B) * len(ATR_MULTS) * len(MIN_HOLD_VALUES) * len(MAX_HOLD_VALUES)
    total_backtests = combos_per_sym * len(symbols)
    print(f"\nSweep: {combos_per_sym} combos/symbol x {len(symbols)} symbols = {total_backtests} backtests")

    dash_ts = pd.Timestamp(dash_start, tz=st.BERLIN_TZ)

    # Aggregated results per combo
    agg_results = {}

    t0 = time.time()
    done_backtests = 0

    for si, symbol in enumerate(symbols):
        df_raw = st.prepare_symbol_dataframe(symbol)
        phases_s = phase_cache.get(symbol, pd.Series(dtype=str))
        df_cache = {}

        for param_a in JMA_PARAM_A:
            for param_b in JMA_PARAM_B:
                cache_key = (param_a, param_b)
                if cache_key not in df_cache:
                    df_tmp = st.compute_indicator(df_raw, param_a, param_b)
                    for col in ("htf_trend", "htf_indicator", "momentum"):
                        if col in df_raw.columns:
                            df_tmp[col] = df_raw[col]
                    df_cache[cache_key] = df_tmp
                df_st = df_cache[cache_key]

                for atr_mult in ATR_MULTS:
                    for min_hold in MIN_HOLD_VALUES:
                        for max_hold in MAX_HOLD_VALUES:
                            df_bt = df_st.copy()
                            for col in ("htf_trend", "htf_indicator", "momentum"):
                                if col in df_raw.columns:
                                    df_bt[col] = df_raw[col]

                            trades_df = st.backtest_supertrend(
                                df_bt,
                                atr_stop_mult=atr_mult,
                                direction="long",
                                min_hold_bars=min_hold,
                                max_hold_bars=max_hold,
                            )

                            # Filter trades by phase
                            if not trades_df.empty:
                                phase_tags = trades_df["Zeit"].apply(
                                    lambda ts: get_phase_at(phases_s, ts)
                                )
                                filtered = trades_df[phase_tags == target_phase]
                            else:
                                filtered = trades_df

                            # Full-period stats
                            n_all = len(filtered)
                            pnl_all = float(filtered["PnL (USD)"].sum()) if n_all > 0 else 0
                            win_all = int((filtered["PnL (USD)"] > 0).sum()) if n_all > 0 else 0

                            # Dashboard-period stats (ab dash_start)
                            if n_all > 0:
                                dash_mask = filtered["Zeit"] >= dash_ts
                                dash_trades = filtered[dash_mask]
                                n_dash = len(dash_trades)
                                pnl_dash = float(dash_trades["PnL (USD)"].sum()) if n_dash > 0 else 0
                                win_dash = int((dash_trades["PnL (USD)"] > 0).sum()) if n_dash > 0 else 0
                            else:
                                n_dash = 0
                                pnl_dash = 0
                                win_dash = 0

                            combo_key = (param_a, param_b, atr_mult, min_hold, max_hold)
                            if combo_key not in agg_results:
                                agg_results[combo_key] = {
                                    'total_pnl': 0, 'total_trades': 0, 'winners': 0,
                                    'dash_pnl': 0, 'dash_trades': 0, 'dash_winners': 0,
                                    'per_symbol': {},
                                }
                            r = agg_results[combo_key]
                            r['total_pnl'] += pnl_all
                            r['total_trades'] += n_all
                            r['winners'] += win_all
                            r['dash_pnl'] += pnl_dash
                            r['dash_trades'] += n_dash
                            r['dash_winners'] += win_dash
                            r['per_symbol'][symbol] = {
                                'trades': n_all, 'pnl': pnl_all, 'winners': win_all,
                                'dash_trades': n_dash, 'dash_pnl': pnl_dash, 'dash_winners': win_dash,
                            }

                            done_backtests += 1

        # Progress
        elapsed = time.time() - t0
        pct = done_backtests / total_backtests * 100
        rate = done_backtests / elapsed if elapsed > 0 else 0
        remaining = (total_backtests - done_backtests) / rate if rate > 0 else 0
        print(f"  [{si+1}/{len(symbols)}] {symbol} fertig | "
              f"{done_backtests}/{total_backtests} ({pct:.0f}%) | "
              f"Dauer: {fmt_time(elapsed)} | Rest: ~{fmt_time(remaining)}")

    total_elapsed = time.time() - t0
    print(f"\nSweep fertig in {fmt_time(total_elapsed)}")

    # Restore original settings
    st.apply_indicator_type(orig_indicator)
    st.apply_higher_timeframe(orig_htf)
    st.HTF_LENGTH = orig_htf_len
    st.HTF_FACTOR = orig_htf_fac

    return agg_results


def print_results(agg_results, target_phase, dash_start=DASHBOARD_START, top_n=20):
    """Print top N parameter combinations ranked by total PnL."""
    ranked = sorted(agg_results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

    print(f"\n{'='*110}")
    print(f"TOP {top_n} fuer {target_phase}-Phase (nach Gesamt-PnL)")
    print(f"{'='*110}")
    print(f"{'#':>3} {'A':>4} {'B':>4} {'ATR':>5} {'MinH':>5} {'MaxH':>5} | "
          f"{'Trades':>7} {'PnL':>12} {'WR%':>6} {'AvgPnL':>9} | "
          f"{'D-Trd':>6} {'D-PnL':>12} {'D-WR%':>6}")
    print("-" * 110)

    for i, (combo, s) in enumerate(ranked[:top_n]):
        pa, pb, atr, minh, maxh = combo
        n = s['total_trades']
        pnl = s['total_pnl']
        wr = (s['winners'] / n * 100) if n > 0 else 0
        avg = pnl / n if n > 0 else 0
        dn = s['dash_trades']
        dpnl = s['dash_pnl']
        dwr = (s['dash_winners'] / dn * 100) if dn > 0 else 0
        atr_s = f"{atr}" if atr is not None else "None"
        print(f"{i+1:>3} {pa:>4} {pb:>4} {atr_s:>5} {minh:>5} {maxh:>5} | "
              f"{n:>7} {pnl:>+12,.0f} {wr:>5.1f}% {avg:>+9,.0f} | "
              f"{dn:>6} {dpnl:>+12,.0f} {dwr:>5.1f}%")

    # Baseline comparison
    print(f"\n--- Baseline (Default-Params) ---")
    for combo, s in ranked:
        if combo == (30, 0, None, 0, 0):
            n = s['total_trades']
            pnl = s['total_pnl']
            wr = (s['winners'] / n * 100) if n > 0 else 0
            dn = s['dash_trades']
            dpnl = s['dash_pnl']
            dwr = (s['dash_winners'] / dn * 100) if dn > 0 else 0
            print(f"    (30, 0, None, 0, 0): {n} trades, PnL={pnl:+,.0f}, WR={wr:.1f}% | "
                  f"Dash: {dn} trades, PnL={dpnl:+,.0f}, WR={dwr:.1f}%")
            break

    # Bottom 5
    print(f"\n--- Schlechteste 5 ---")
    for combo, s in ranked[-5:]:
        pa, pb, atr, minh, maxh = combo
        n = s['total_trades']
        pnl = s['total_pnl']
        wr = (s['winners'] / n * 100) if n > 0 else 0
        atr_s = f"{atr}" if atr is not None else "None"
        print(f"    ({pa}, {pb}, {atr_s}, {minh}, {maxh}): "
              f"{n} trades, PnL={pnl:+,.0f}, WR={wr:.1f}%")

    return ranked


def save_results_csv(ranked, target_phase, phase_params, out_dir="report_html"):
    """Save sweep results to CSV."""
    rows = []
    for combo, stats in ranked:
        param_a, param_b, atr_mult, min_hold, max_hold = combo
        n = stats['total_trades']
        pnl = stats['total_pnl']
        wr = (stats['winners'] / n * 100) if n > 0 else 0
        dn = stats['dash_trades']
        dpnl = stats['dash_pnl']
        dwr = (stats['dash_winners'] / dn * 100) if dn > 0 else 0
        rows.append({
            'Phase': target_phase,
            'PhaseClassifier': phase_params['classifier'],
            'PhaseHTF': phase_params['htf'],
            'PhaseLength': phase_params['length'],
            'PhaseFactor': phase_params['factor'],
            'ParamA': param_a,
            'ParamB': param_b,
            'ATRStopMult': atr_mult if atr_mult is not None else 'None',
            'MinHoldBars': min_hold,
            'MaxHoldBars': max_hold,
            'Trades': n,
            'PnL': round(pnl, 2),
            'WinRate': round(wr, 2),
            'AvgPnL': round(pnl / n, 2) if n > 0 else 0,
            'Winners': stats['winners'],
            'DashTrades': dn,
            'DashPnL': round(dpnl, 2),
            'DashWinRate': round(dwr, 2),
            'DashWinners': stats['dash_winners'],
        })

    df = pd.DataFrame(rows)
    fname = f"sweep_{target_phase.lower()}_{phase_params['classifier']}_L{phase_params['length']}_F{phase_params['factor']}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    print(f"\nCSV gespeichert: {path} ({len(df)} Zeilen)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Phase-filtered parameter sweep")
    parser.add_argument("--phase", type=str, default="Down",
                        help="Target phase: Up, Down, Flat (default: Down)")
    parser.add_argument("--phase-classifier", type=str, default="jma",
                        help="Phase classifier: jma, kama, supertrend, htf_crossover")
    parser.add_argument("--phase-htf", type=str, default="6h",
                        help="Phase HTF timeframe (default: 6h)")
    parser.add_argument("--phase-length", type=int, default=10,
                        help="Phase HTF_LENGTH (default: 10)")
    parser.add_argument("--phase-factor", type=float, default=3.0,
                        help="Phase HTF_FACTOR (default: 3.0)")
    parser.add_argument("--dash-start", type=str, default=DASHBOARD_START,
                        help="Dashboard start date (default: 2025-12-01)")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N results (default: 20)")
    args = parser.parse_args()

    agg_results = sweep_phase(
        target_phase=args.phase,
        phase_htf_length=args.phase_length,
        phase_htf_factor=args.phase_factor,
        phase_htf=args.phase_htf,
        phase_classifier=args.phase_classifier,
        dash_start=args.dash_start,
    )

    ranked = print_results(agg_results, args.phase, dash_start=args.dash_start, top_n=args.top)

    phase_params = {
        'classifier': args.phase_classifier,
        'htf': args.phase_htf,
        'length': args.phase_length,
        'factor': args.phase_factor,
    }
    save_results_csv(ranked, args.phase, phase_params)


if __name__ == "__main__":
    main()
