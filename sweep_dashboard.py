#!/usr/bin/env python3
"""Sweep Dashboard - Zeigt optimale Trades mit Marktphase, Indikator und Parametern.

Features:
- Lädt best_params_overall.csv für optimale Parameter pro Symbol/Direction/Indikator
- Ermittelt aktuelle Marktphasen (Up/Down/Flat) pro Symbol
- Zeigt Trades mit dynamischem Stake (wie trading_summary)
- Sortiert nach Entry-Zeit (neueste oben)
- Spalte Marktphase mit Indikator und optimalen Parametern
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Import from project
import Supertrend_5Min as st
from Supertrend_5Min import (
    SYMBOLS,
    TIMEFRAME,
    LOOKBACK,
    BERLIN_TZ,
    prepare_symbol_dataframe,
    compute_indicator,
    backtest_supertrend,
    backtest_htf_crossover,
    determine_market_phase_from_df,
    determine_indicator_phase,
    PHASE_UP,
    PHASE_DOWN,
    PHASE_FLAT,
)

# Paths
REPORT_DIR = Path("report_html")
BEST_PARAMS_CSV = REPORT_DIR / "best_params_overall.csv"
OUTPUT_HTML = REPORT_DIR / "sweep_dashboard.html"
OUTPUT_JSON = REPORT_DIR / "sweep_dashboard.json"
OUTPUT_PHASE_PARAMS_CSV = REPORT_DIR / "best_params_per_phase.csv"

# Capital settings - same as paper_trader.py
START_CAPITAL = 16500.0
MAX_POSITIONS = 10


@dataclass
class OptimalParams:
    """Optimale Parameter für ein Symbol/Direction/Indikator."""
    symbol: str
    direction: str
    indicator: str
    indicator_display: str
    param_a: float
    param_b: float
    atr_stop_mult: Optional[float]
    min_hold_bars: int
    htf: str
    final_equity: float
    trades: int
    win_rate: float
    max_drawdown: float
    phase: str


def load_best_params() -> Dict[Tuple[str, str, str], OptimalParams]:
    """Lade best_params_overall.csv und parse zu Dict."""
    if not BEST_PARAMS_CSV.exists():
        print(f"[WARN] {BEST_PARAMS_CSV} nicht gefunden!")
        return {}

    # CSV mit Semikolon-Separator, Dezimal-Komma
    df = pd.read_csv(BEST_PARAMS_CSV, sep=";", decimal=",")

    params_dict = {}
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        direction = str(row.get("Direction", "")).strip().lower()
        indicator = str(row.get("Indicator", "")).strip().lower()

        if not symbol or not direction or not indicator:
            continue

        # Parse ATRStopMult
        atr_raw = row.get("ATRStopMult", "None")
        atr_stop_mult = None
        if atr_raw and str(atr_raw).lower() not in ("none", "nan", ""):
            try:
                atr_stop_mult = float(str(atr_raw).replace(",", "."))
            except (ValueError, TypeError):
                pass

        # Parse other values
        try:
            param_a = float(str(row.get("ParamA", 7)).replace(",", "."))
            param_b = float(str(row.get("ParamB", 2.0)).replace(",", "."))
            min_hold_bars = int(float(str(row.get("MinHoldBars", 0)).replace(",", ".")))
            final_equity = float(str(row.get("FinalEquity", 10000)).replace(",", "."))
            trades = int(float(str(row.get("Trades", 0)).replace(",", ".")))
            win_rate = float(str(row.get("WinRate", 0)).replace(",", "."))
            max_drawdown = float(str(row.get("MaxDrawdown", 0)).replace(",", "."))
        except (ValueError, TypeError):
            continue

        key = (symbol, direction, indicator)
        params_dict[key] = OptimalParams(
            symbol=symbol,
            direction=direction,
            indicator=indicator,
            indicator_display=str(row.get("IndicatorDisplay", indicator)),
            param_a=param_a,
            param_b=param_b,
            atr_stop_mult=atr_stop_mult,
            min_hold_bars=min_hold_bars,
            htf=str(row.get("HTF", "6h")),
            final_equity=final_equity,
            trades=trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            phase=str(row.get("Phase", "")),
        )

    print(f"[INFO] {len(params_dict)} Parameter-Sets geladen")
    return params_dict


def get_daily_market_phases(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Ermittle Marktphasen auf DAILY Basis für ein Symbol.

    Verwendet alle drei Indikatoren (Supertrend, JMA, KAMA) und bestimmt
    die Phase durch Mehrheitsentscheidung. Dadurch dauern Phasen
    typischerweise WOCHEN oder MONATE.

    Returns:
        DataFrame mit Datum und Phase für jeden Tag
    """
    from Supertrend_5Min import (
        download_historical_ohlcv,
        compute_supertrend,
        compute_jma,
        compute_kama,
        calculate_indicator_slope,
        determine_phase_from_slope,
        load_ohlcv_from_cache,
        save_ohlcv_to_cache,
    )

    try:
        # Versuche aus Cache zu laden
        df_daily = load_ohlcv_from_cache(symbol, "1d")

        if df_daily is None or df_daily.empty:
            # Download Daily Daten
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
            df_daily = download_historical_ohlcv(symbol, "1d", start_date)
            if df_daily is not None and not df_daily.empty:
                save_ohlcv_to_cache(symbol, "1d", df_daily)

        if df_daily is None or df_daily.empty:
            return pd.DataFrame()

        # Alle drei Indikatoren auf Daily berechnen
        df_daily = compute_supertrend(df_daily, length=10, factor=3.0)

        try:
            df_jma = compute_jma(df_daily.copy(), length=30, phase=0)
            df_daily["jma"] = df_jma["jma"]
        except Exception:
            df_daily["jma"] = df_daily["supertrend"]  # Fallback

        try:
            df_kama = compute_kama(df_daily.copy(), length=20, slow_length=40)
            df_daily["kama"] = df_kama["kama"]
        except Exception:
            df_daily["kama"] = df_daily["supertrend"]  # Fallback

        # Slope-Threshold für Daily (höher als bei Stundendaten)
        threshold = 0.0002

        # Phase für jeden Tag berechnen (Mehrheitsentscheidung)
        phases = []
        lookback = 10  # 10 Tage für Slope-Berechnung

        for i in range(lookback, len(df_daily)):
            date = df_daily.index[i]

            # Berechne Phase für jeden Indikator
            indicator_phases = []

            for indicator_col in ["supertrend", "jma", "kama"]:
                if indicator_col in df_daily.columns:
                    values = df_daily[indicator_col].iloc[i-lookback:i+1]
                    if len(values) >= 2 and not values.isna().all():
                        slope = calculate_indicator_slope(values, lookback=lookback)
                        phase = determine_phase_from_slope(slope, threshold)
                        indicator_phases.append(phase)

            # Mehrheitsentscheidung
            if indicator_phases:
                up_count = indicator_phases.count(PHASE_UP)
                down_count = indicator_phases.count(PHASE_DOWN)

                if up_count >= 2:
                    final_phase = PHASE_UP
                elif down_count >= 2:
                    final_phase = PHASE_DOWN
                else:
                    final_phase = PHASE_FLAT
            else:
                final_phase = PHASE_FLAT

            phases.append({
                "date": date.date() if hasattr(date, 'date') else date,
                "phase": final_phase
            })

        return pd.DataFrame(phases)
    except Exception as e:
        print(f"[WARN] Daily Phase für {symbol} nicht ermittelbar: {e}")
        return pd.DataFrame()


def get_phase_for_date(phase_df: pd.DataFrame, entry_date) -> str:
    """Finde die Marktphase für ein bestimmtes Datum."""
    if phase_df.empty:
        return PHASE_FLAT

    try:
        # Konvertiere entry_date zu date
        if isinstance(entry_date, str):
            entry_date = pd.to_datetime(entry_date).date()
        elif hasattr(entry_date, 'date'):
            entry_date = entry_date.date()

        # Finde die passende Phase
        matching = phase_df[phase_df["date"] <= entry_date]
        if not matching.empty:
            return matching.iloc[-1]["phase"]
    except Exception:
        pass

    return PHASE_FLAT


def get_current_market_phase(symbol: str, indicator: str = "supertrend") -> str:
    """Ermittle aktuelle Marktphase für ein Symbol (auf Daily Basis)."""
    phase_df = get_daily_market_phases(symbol, lookback_days=30)
    if phase_df.empty:
        return PHASE_FLAT
    return phase_df.iloc[-1]["phase"] if not phase_df.empty else PHASE_FLAT


def run_backtest_for_params(
    symbol: str,
    params: OptimalParams,
    lookback: int = 500
) -> List[Dict]:
    """Führe Backtest mit optimalen Parametern aus und gebe Trades zurück."""
    try:
        # Daten laden
        df_raw = prepare_symbol_dataframe(symbol, use_all_cached_data=False, lookback=lookback)
        if df_raw.empty:
            return []

        # Indikator berechnen
        df = compute_indicator(df_raw, params.param_a, params.param_b)

        # HTF Daten hinzufügen falls vorhanden
        for col in ("htf_trend", "htf_indicator", "momentum"):
            if col in df_raw.columns:
                df[col] = df_raw[col]

        # Backtest ausführen - gibt DataFrame zurück
        if params.indicator == "htf_crossover":
            trades_df = backtest_htf_crossover(
                df,
                atr_stop_mult=params.atr_stop_mult,
                direction=params.direction,
                min_hold_bars=params.min_hold_bars,
            )
        else:
            trades_df = backtest_supertrend(
                df,
                atr_stop_mult=params.atr_stop_mult,
                direction=params.direction,
                min_hold_bars=params.min_hold_bars,
            )

        # DataFrame zu Liste von Dicts konvertieren mit einheitlichen Spaltennamen
        if trades_df.empty:
            return []

        trades = []
        for _, row in trades_df.iterrows():
            trade = {
                "entry_time": str(row.get("Zeit", "")),
                "entry_price": float(row.get("Entry", 0) or 0),
                "exit_time": str(row.get("ExitZeit", "")),
                "exit_price": float(row.get("ExitPreis", 0) or 0),
                "stake": float(row.get("Stake", 0) or 0),
                "fees": float(row.get("Fees", 0) or 0),
                "pnl": float(row.get("PnL (USD)", 0) or 0),
                "equity_after": float(row.get("Equity", 0) or 0),
                "reason": str(row.get("ExitReason", "")),
                "direction": str(row.get("Direction", "")).lower(),
                "min_hold_bars": int(row.get("MinHoldBars", 0) or 0),
                "phase": str(row.get("Phase", "")),
            }
            trades.append(trade)

        return trades
    except Exception as e:
        print(f"[ERROR] Backtest für {symbol} fehlgeschlagen: {e}")
        return []


def calculate_dynamic_stake_trades(
    all_trades: List[Dict],
    start_capital: float = START_CAPITAL,
    max_positions: int = MAX_POSITIONS
) -> Tuple[List[Dict], float]:
    """Berechne dynamischen Stake für alle Trades (Compound Growth)."""
    if not all_trades:
        return [], start_capital

    # Nach Entry-Zeit sortieren (älteste zuerst für Compound)
    sorted_trades = sorted(all_trades, key=lambda t: t.get("entry_time", "") or "")

    capital = start_capital
    enriched_trades = []

    for trade in sorted_trades:
        stake = capital / max_positions
        entry_price = float(trade.get("entry_price", 0) or 0)
        exit_price = float(trade.get("exit_price", 0) or 0)
        direction = str(trade.get("direction", "long")).lower()

        if entry_price > 0:
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            pnl = pnl_pct * stake
        else:
            pnl_pct = 0
            pnl = 0

        enriched = dict(trade)
        enriched["stake"] = stake
        enriched["pnl"] = pnl
        enriched["pnl_pct"] = pnl_pct * 100
        enriched["equity_after"] = capital + pnl

        enriched_trades.append(enriched)
        capital += pnl

    return enriched_trades, capital


def fetch_live_price(symbol: str) -> Optional[float]:
    """Hole aktuellen Preis von Binance."""
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 30000})
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get('last')
    except Exception as e:
        print(f"[WARN] Preis für {symbol} nicht abrufbar: {e}")
        return None


def run_sweep_dashboard():
    """Hauptfunktion: Führe Sweep aus und generiere Dashboard."""
    print("=" * 60)
    print("SWEEP DASHBOARD - Optimale Trades mit Marktphasen")
    print("=" * 60)
    print("  Marktphase: DAILY Charts")
    print("  Trades: Stundenbasis (1h)")
    print("=" * 60)

    # 1. Lade optimale Parameter
    params_dict = load_best_params()
    if not params_dict:
        print("[ERROR] Keine Parameter gefunden. Führe zuerst den Parameter-Sweep aus.")
        return

    # 2. Ermittle aktuelle Marktphasen und führe Backtests aus
    all_trades = []
    phase_info = {}  # Symbol -> aktuelle Phase
    daily_phases = {}  # Symbol -> DataFrame mit täglichen Phasen
    indicator_info = {}  # Key -> Params

    indicators = ["supertrend", "htf_crossover", "jma", "kama"]
    directions = ["long", "short"]

    # Lookback für 2 Jahre Stundendaten
    LOOKBACK_HOURS = 17520  # 730 Tage * 24 Stunden (2 Jahre)

    print("\n[1/4] Lade Daily-Phasen für alle Symbole...")

    for symbol in SYMBOLS:
        print(f"  {symbol}: Lade Daily-Daten...")
        phase_df = get_daily_market_phases(symbol, lookback_days=800)  # 2+ Jahre
        daily_phases[symbol] = phase_df
        if not phase_df.empty:
            current_phase = phase_df.iloc[-1]["phase"]
            phase_info[symbol] = current_phase
            # Zähle Phasen
            up_count = (phase_df["phase"] == PHASE_UP).sum()
            down_count = (phase_df["phase"] == PHASE_DOWN).sum()
            flat_count = (phase_df["phase"] == PHASE_FLAT).sum()
            print(f"    {len(phase_df)} Tage: UP={up_count}, DOWN={down_count}, FLAT={flat_count}")
            print(f"    Aktuelle Phase: {current_phase}")
        else:
            phase_info[symbol] = PHASE_FLAT
            print(f"    Keine Daily-Daten verfügbar")

    print("\n[2/4] Führe Backtests aus (2 Jahre Stundendaten)...")

    for symbol in SYMBOLS:
        print(f"\n  Processing {symbol}...")
        phase_df = daily_phases.get(symbol, pd.DataFrame())

        for indicator in indicators:
            for direction in directions:
                key = (symbol, direction, indicator)
                if key not in params_dict:
                    continue

                params = params_dict[key]
                indicator_info[key] = params

                # Backtest mit 1 Jahr Daten
                trades = run_backtest_for_params(symbol, params, lookback=LOOKBACK_HOURS)

                # Trades anreichern mit Metadaten und DAILY Phase
                for trade in trades:
                    entry_time = trade.get("entry_time", "")
                    # Phase basierend auf Entry-Datum (DAILY)
                    trade_phase = get_phase_for_date(phase_df, entry_time)

                    trade["symbol"] = symbol
                    trade["direction"] = direction
                    trade["indicator"] = params.indicator_display
                    trade["market_phase"] = trade_phase  # Daily Phase zum Entry-Zeitpunkt
                    trade["param_a"] = params.param_a
                    trade["param_b"] = params.param_b
                    trade["atr_stop_mult"] = params.atr_stop_mult
                    trade["min_hold_bars"] = params.min_hold_bars
                    trade["htf"] = params.htf
                    trade["optimal_params"] = f"{params.param_a}/{params.param_b}"
                    if params.atr_stop_mult:
                        trade["optimal_params"] += f"/ATR{params.atr_stop_mult}"

                all_trades.extend(trades)
                if trades:
                    # Zähle Trades pro Phase
                    up_trades = sum(1 for t in trades if get_phase_for_date(phase_df, t.get("entry_time", "")) == PHASE_UP)
                    down_trades = sum(1 for t in trades if get_phase_for_date(phase_df, t.get("entry_time", "")) == PHASE_DOWN)
                    flat_trades = sum(1 for t in trades if get_phase_for_date(phase_df, t.get("entry_time", "")) == PHASE_FLAT)
                    print(f"    {indicator} {direction}: {len(trades)} Trades (UP={up_trades}, DOWN={down_trades}, FLAT={flat_trades})")

    print(f"\n[3/4] Berechne dynamischen Stake für {len(all_trades)} Trades...")

    # 3. Dynamischen Stake berechnen
    enriched_trades, final_capital = calculate_dynamic_stake_trades(all_trades)

    # Nach Entry-Zeit sortieren (neueste zuerst für Anzeige)
    enriched_trades.sort(key=lambda t: t.get("entry_time", "") or "", reverse=True)

    print(f"    Final Capital: {final_capital:,.2f} USDT")

    # 4. HTML Dashboard und CSV generieren
    print("\n[4/4] Generiere Dashboard und speichere Parameter...")
    generate_html_dashboard(enriched_trades, phase_info, final_capital)

    # 5. JSON speichern
    save_json_summary(enriched_trades, phase_info, final_capital)

    # 6. Phase-Parameter CSV speichern
    save_phase_params_csv(enriched_trades)

    print(f"\n{'=' * 60}")
    print(f"Dashboard generiert: {OUTPUT_HTML}")
    print(f"JSON gespeichert: {OUTPUT_JSON}")
    print(f"Phase-Parameter: {OUTPUT_PHASE_PARAMS_CSV}")
    print(f"{'=' * 60}")


def calculate_best_params_per_phase(trades: List[Dict]) -> Dict[str, List[Dict]]:
    """Berechne die besten Parameter pro Marktphase basierend auf Trade-Performance."""
    # Gruppiere Trades nach Symbol, Direction, Indikator und Phase
    from collections import defaultdict

    phase_performance = defaultdict(lambda: defaultdict(list))

    for t in trades:
        symbol = t.get("symbol", "")
        direction = t.get("direction", "")
        indicator = t.get("indicator", "")
        phase = t.get("phase", "") or t.get("market_phase", PHASE_FLAT)
        optimal_params = t.get("optimal_params", "")
        pnl = float(t.get("pnl", 0) or 0)

        key = (symbol, direction, indicator)
        phase_performance[key][phase].append({
            "pnl": pnl,
            "params": optimal_params,
            "htf": t.get("htf", ""),
            "param_a": t.get("param_a", 0),
            "param_b": t.get("param_b", 0),
            "atr_stop_mult": t.get("atr_stop_mult"),
            "min_hold_bars": t.get("min_hold_bars", 0),
        })

    # Berechne Statistiken pro Phase
    results = []
    for key, phases in phase_performance.items():
        symbol, direction, indicator = key
        row = {
            "symbol": symbol,
            "direction": direction,
            "indicator": indicator,
        }

        for phase in [PHASE_UP, PHASE_DOWN, PHASE_FLAT]:
            phase_trades = phases.get(phase, [])
            if phase_trades:
                total_pnl = sum(t["pnl"] for t in phase_trades)
                winners = sum(1 for t in phase_trades if t["pnl"] > 0)
                win_rate = (winners / len(phase_trades) * 100) if phase_trades else 0
                # Nehme die Parameter des ersten Trades (sollten alle gleich sein)
                params = phase_trades[0]["params"]
                htf = phase_trades[0]["htf"]

                row[f"{phase.lower()}_trades"] = len(phase_trades)
                row[f"{phase.lower()}_pnl"] = total_pnl
                row[f"{phase.lower()}_win_rate"] = win_rate
                row[f"{phase.lower()}_params"] = params
                row[f"{phase.lower()}_htf"] = htf
                row[f"{phase.lower()}_param_a"] = phase_trades[0]["param_a"]
                row[f"{phase.lower()}_param_b"] = phase_trades[0]["param_b"]
                row[f"{phase.lower()}_atr_stop_mult"] = phase_trades[0]["atr_stop_mult"]
                row[f"{phase.lower()}_min_hold_bars"] = phase_trades[0]["min_hold_bars"]
            else:
                row[f"{phase.lower()}_trades"] = 0
                row[f"{phase.lower()}_pnl"] = 0
                row[f"{phase.lower()}_win_rate"] = 0
                row[f"{phase.lower()}_params"] = "-"
                row[f"{phase.lower()}_htf"] = "-"
                row[f"{phase.lower()}_param_a"] = None
                row[f"{phase.lower()}_param_b"] = None
                row[f"{phase.lower()}_atr_stop_mult"] = None
                row[f"{phase.lower()}_min_hold_bars"] = None

        results.append(row)

    return results


def save_phase_params_csv(trades: List[Dict]) -> None:
    """Speichere die besten Parameter pro Marktphase als CSV.

    Spalten: Symbol, Direction, Indicator, Phase, ParamA, ParamB, ATRStopMult,
             MinHoldBars, HTF, Trades, PnL, WinRate
    """
    from collections import defaultdict

    phase_performance = defaultdict(lambda: defaultdict(list))

    for t in trades:
        symbol = t.get("symbol", "")
        direction = t.get("direction", "")
        indicator = t.get("indicator", "")
        phase = t.get("phase", "") or t.get("market_phase", PHASE_FLAT)
        pnl = float(t.get("pnl", 0) or 0)

        key = (symbol, direction, indicator)
        phase_performance[key][phase].append({
            "pnl": pnl,
            "htf": t.get("htf", ""),
            "param_a": t.get("param_a", 0),
            "param_b": t.get("param_b", 0),
            "atr_stop_mult": t.get("atr_stop_mult"),
            "min_hold_bars": t.get("min_hold_bars", 0),
        })

    # Erstelle flache Liste für CSV
    rows = []
    for key, phases in phase_performance.items():
        symbol, direction, indicator = key

        for phase in [PHASE_UP, PHASE_DOWN, PHASE_FLAT]:
            phase_trades = phases.get(phase, [])
            if phase_trades:
                total_pnl = sum(t["pnl"] for t in phase_trades)
                winners = sum(1 for t in phase_trades if t["pnl"] > 0)
                win_rate = (winners / len(phase_trades) * 100) if phase_trades else 0
                # Parameter vom ersten Trade
                first = phase_trades[0]

                rows.append({
                    "Symbol": symbol,
                    "Direction": direction,
                    "Indicator": indicator,
                    "Phase": phase,
                    "ParamA": first["param_a"],
                    "ParamB": first["param_b"],
                    "ATRStopMult": first["atr_stop_mult"] if first["atr_stop_mult"] else "None",
                    "MinHoldBars": first["min_hold_bars"],
                    "HTF": first["htf"],
                    "Trades": len(phase_trades),
                    "PnL": round(total_pnl, 2),
                    "WinRate": round(win_rate, 2),
                })

    # Als DataFrame speichern
    df = pd.DataFrame(rows)

    # Sortieren
    df = df.sort_values(["Symbol", "Direction", "Indicator", "Phase"]).reset_index(drop=True)

    # CSV speichern (Semikolon-Separator, Dezimal-Komma für Deutschland)
    OUTPUT_PHASE_PARAMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PHASE_PARAMS_CSV, sep=";", decimal=",", index=False)
    print(f"    Phase-Parameter gespeichert: {OUTPUT_PHASE_PARAMS_CSV}")
    print(f"    {len(rows)} Einträge (Symbol/Direction/Indicator/Phase Kombinationen)")


def generate_html_dashboard(
    trades: List[Dict],
    phase_info: Dict[str, str],
    final_capital: float
):
    """Generiere HTML Dashboard mit allen Trades."""

    # Berechne Phase-spezifische Parameter
    phase_params = calculate_best_params_per_phase(trades)

    # Statistiken berechnen
    total_trades = len(trades)
    winners = sum(1 for t in trades if float(t.get("pnl", 0) or 0) > 0)
    losers = sum(1 for t in trades if float(t.get("pnl", 0) or 0) < 0)
    total_pnl = sum(float(t.get("pnl", 0) or 0) for t in trades)
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

    # Statistiken pro Marktphase
    phase_stats = {}
    for phase in [PHASE_UP, PHASE_DOWN, PHASE_FLAT]:
        phase_trades = [t for t in trades if t.get("market_phase") == phase]
        if phase_trades:
            phase_pnl = sum(float(t.get("pnl", 0) or 0) for t in phase_trades)
            phase_winners = sum(1 for t in phase_trades if float(t.get("pnl", 0) or 0) > 0)
            phase_stats[phase] = {
                "trades": len(phase_trades),
                "pnl": phase_pnl,
                "win_rate": (phase_winners / len(phase_trades) * 100) if phase_trades else 0
            }

    def fmt(val):
        """Format number with German locale."""
        if val is None or str(val) == "NaN":
            return "-"
        try:
            return f"{float(val):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (ValueError, TypeError):
            return str(val)

    def fmt_pct(val):
        """Format percentage."""
        if val is None or str(val) == "NaN":
            return "-"
        try:
            return f"{float(val):+.2f}%"
        except (ValueError, TypeError):
            return str(val)

    def fmt_price(price):
        """Format price with appropriate decimals."""
        if price is None or str(price) == "NaN":
            return "-"
        try:
            p = float(price)
            if p < 0.0001:
                return f"{p:.8f}"
            elif p < 1:
                return f"{p:.6f}"
            elif p < 100:
                return f"{p:.4f}"
            else:
                return fmt(p)
        except (ValueError, TypeError):
            return str(price)

    def pnl_class(val):
        try:
            return "pos" if float(val) >= 0 else "neg"
        except (ValueError, TypeError):
            return ""

    def phase_class(phase):
        """CSS class für Phase."""
        if phase == PHASE_UP:
            return "phase-up"
        elif phase == PHASE_DOWN:
            return "phase-down"
        return "phase-flat"

    def phase_icon(phase):
        """Icon für Phase."""
        if phase == PHASE_UP:
            return "▲"
        elif phase == PHASE_DOWN:
            return "▼"
        return "◆"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # HTML generieren
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<meta http-equiv='refresh' content='300'>",  # Auto-refresh alle 5 Min
        "<title>Sweep Dashboard - Optimale Trades</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        ".container { max-width: 1600px; margin: 0 auto; }",
        "h1 { color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }",
        "h2 { color: #555; margin-top: 30px; }",
        ".stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }",
        ".stat-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".stat-card h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }",
        ".stat-card .value { font-size: 24px; font-weight: bold; color: #333; }",
        ".stat-card .value.pos { color: #4CAF50; }",
        ".stat-card .value.neg { color: #f44336; }",
        "table { border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; font-size: 13px; }",
        "th { background: #2196F3; color: white; font-weight: 600; position: sticky; top: 0; }",
        "td:first-child, td:nth-child(2), td:nth-child(3), td:nth-child(4) { text-align: left; }",
        "tr:nth-child(even) { background: #f9f9f9; }",
        "tr:hover { background: #e3f2fd; }",
        ".pos { color: #4CAF50; font-weight: 600; }",
        ".neg { color: #f44336; font-weight: 600; }",
        ".phase-up { background: #e8f5e9; color: #2e7d32; }",
        ".phase-down { background: #ffebee; color: #c62828; }",
        ".phase-flat { background: #fff3e0; color: #ef6c00; }",
        ".phase-badge { padding: 3px 8px; border-radius: 4px; font-weight: 600; display: inline-block; }",
        ".params { font-family: monospace; font-size: 12px; color: #666; }",
        ".source-note { color: #888; font-style: italic; margin-bottom: 20px; }",
        ".phase-summary { display: flex; gap: 20px; margin: 15px 0; }",
        ".phase-box { padding: 15px 20px; border-radius: 8px; flex: 1; }",
        ".phase-box h4 { margin: 0 0 10px 0; }",
        "</style></head><body>",
        "<div class='container'>",
        "<h1>Sweep Dashboard - Optimale Trades mit Marktphasen</h1>",
        f"<p class='source-note'>Generiert: {timestamp} | Auto-Refresh: 5 Min</p>",
    ]

    # Statistik-Karten
    html_parts.append("<div class='stats-grid'>")
    html_parts.append(f"<div class='stat-card'><h3>Total Trades</h3><div class='value'>{total_trades}</div></div>")
    html_parts.append(f"<div class='stat-card'><h3>Final Capital</h3><div class='value {pnl_class(final_capital - START_CAPITAL)}'>{fmt(final_capital)} USDT</div></div>")
    html_parts.append(f"<div class='stat-card'><h3>Total PnL</h3><div class='value {pnl_class(total_pnl)}'>{fmt(total_pnl)} USDT</div></div>")
    html_parts.append(f"<div class='stat-card'><h3>Win Rate</h3><div class='value'>{win_rate:.1f}%</div></div>")
    html_parts.append(f"<div class='stat-card'><h3>Winners</h3><div class='value pos'>{winners}</div></div>")
    html_parts.append(f"<div class='stat-card'><h3>Losers</h3><div class='value neg'>{losers}</div></div>")
    html_parts.append("</div>")

    # Phase Summary
    html_parts.append("<h2>Performance nach Marktphase</h2>")
    html_parts.append("<div class='phase-summary'>")
    for phase, stats in phase_stats.items():
        css_class = phase_class(phase)
        html_parts.append(f"<div class='phase-box {css_class}'>")
        html_parts.append(f"<h4>{phase_icon(phase)} {phase}</h4>")
        html_parts.append(f"<div>Trades: {stats['trades']}</div>")
        html_parts.append(f"<div>PnL: <strong class='{pnl_class(stats['pnl'])}'>{fmt(stats['pnl'])} USDT</strong></div>")
        html_parts.append(f"<div>Win Rate: {stats['win_rate']:.1f}%</div>")
        html_parts.append("</div>")
    html_parts.append("</div>")

    # Aktuelle Marktphasen pro Symbol
    html_parts.append("<h2>Aktuelle Marktphasen</h2>")
    html_parts.append("<table><tr><th>Symbol</th><th>Phase</th></tr>")
    for symbol, phase in sorted(phase_info.items()):
        html_parts.append(f"<tr><td>{symbol}</td><td><span class='phase-badge {phase_class(phase)}'>{phase_icon(phase)} {phase}</span></td></tr>")
    html_parts.append("</table>")

    # Beste Parameter pro Marktphase Tabelle
    html_parts.append("<h2>Beste Parameter pro Marktphase</h2>")
    html_parts.append("<p class='source-note'>Zeigt die optimalen Parameter und Performance für jedes Symbol/Indikator in jeder Marktphase</p>")
    html_parts.append("<table><tr>")
    html_parts.append("<th>Symbol</th>")
    html_parts.append("<th>Direction</th>")
    html_parts.append("<th>Indikator</th>")
    html_parts.append(f"<th colspan='3' class='phase-up'>{phase_icon(PHASE_UP)} UP Phase</th>")
    html_parts.append(f"<th colspan='3' class='phase-down'>{phase_icon(PHASE_DOWN)} DOWN Phase</th>")
    html_parts.append(f"<th colspan='3' class='phase-flat'>{phase_icon(PHASE_FLAT)} FLAT Phase</th>")
    html_parts.append("</tr>")
    html_parts.append("<tr>")
    html_parts.append("<th></th><th></th><th></th>")
    html_parts.append("<th>Params/HTF</th><th>Trades</th><th>PnL</th>")
    html_parts.append("<th>Params/HTF</th><th>Trades</th><th>PnL</th>")
    html_parts.append("<th>Params/HTF</th><th>Trades</th><th>PnL</th>")
    html_parts.append("</tr>")

    for row in sorted(phase_params, key=lambda r: (r["symbol"], r["direction"], r["indicator"])):
        html_parts.append("<tr>")
        html_parts.append(f"<td>{row['symbol']}</td>")
        html_parts.append(f"<td>{row['direction']}</td>")
        html_parts.append(f"<td>{row['indicator']}</td>")

        # UP Phase
        up_params = row.get("up_params", "-")
        up_htf = row.get("up_htf", "-")
        up_trades = row.get("up_trades", 0)
        up_pnl = row.get("up_pnl", 0)
        html_parts.append(f"<td class='params'>{up_params}<br/><small>{up_htf}</small></td>")
        html_parts.append(f"<td>{up_trades}</td>")
        html_parts.append(f"<td class='{pnl_class(up_pnl)}'>{fmt(up_pnl)}</td>")

        # DOWN Phase
        down_params = row.get("down_params", "-")
        down_htf = row.get("down_htf", "-")
        down_trades = row.get("down_trades", 0)
        down_pnl = row.get("down_pnl", 0)
        html_parts.append(f"<td class='params'>{down_params}<br/><small>{down_htf}</small></td>")
        html_parts.append(f"<td>{down_trades}</td>")
        html_parts.append(f"<td class='{pnl_class(down_pnl)}'>{fmt(down_pnl)}</td>")

        # FLAT Phase
        flat_params = row.get("flat_params", "-")
        flat_htf = row.get("flat_htf", "-")
        flat_trades = row.get("flat_trades", 0)
        flat_pnl = row.get("flat_pnl", 0)
        html_parts.append(f"<td class='params'>{flat_params}<br/><small>{flat_htf}</small></td>")
        html_parts.append(f"<td>{flat_trades}</td>")
        html_parts.append(f"<td class='{pnl_class(flat_pnl)}'>{fmt(flat_pnl)}</td>")

        html_parts.append("</tr>")

    html_parts.append("</table>")

    # Trades Tabelle (sortiert nach Entry Zeit, neueste zuerst)
    html_parts.append(f"<h2>Trades ({total_trades}) - sortiert nach Entry Zeit (neueste zuerst)</h2>")
    html_parts.append("<table><tr>")
    html_parts.append("<th>Symbol</th>")
    html_parts.append("<th>Direction</th>")
    html_parts.append("<th>Marktphase</th>")
    html_parts.append("<th>Indikator</th>")
    html_parts.append("<th>Opt. Params</th>")
    html_parts.append("<th>HTF</th>")
    html_parts.append("<th>Entry Zeit</th>")
    html_parts.append("<th>Entry Preis</th>")
    html_parts.append("<th>Exit Zeit</th>")
    html_parts.append("<th>Exit Preis</th>")
    html_parts.append("<th>Stake</th>")
    html_parts.append("<th>PnL</th>")
    html_parts.append("<th>%</th>")
    html_parts.append("</tr>")

    for t in trades:
        symbol = t.get("symbol", "")
        direction = t.get("direction", "")
        phase = t.get("market_phase", PHASE_FLAT)
        indicator = t.get("indicator", "")
        optimal_params = t.get("optimal_params", "")
        htf = t.get("htf", "")
        entry_time = (t.get("entry_time", "") or "")[:16]
        entry_price = float(t.get("entry_price", 0) or 0)
        exit_time = (t.get("exit_time", "") or "")[:16]
        exit_price = float(t.get("exit_price", 0) or 0)
        stake = float(t.get("stake", 0) or 0)
        pnl = float(t.get("pnl", 0) or 0)
        pnl_pct = float(t.get("pnl_pct", 0) or 0)

        html_parts.append("<tr>")
        html_parts.append(f"<td>{symbol}</td>")
        html_parts.append(f"<td>{direction}</td>")
        html_parts.append(f"<td><span class='phase-badge {phase_class(phase)}'>{phase_icon(phase)} {phase}</span></td>")
        html_parts.append(f"<td>{indicator}</td>")
        html_parts.append(f"<td class='params'>{optimal_params}</td>")
        html_parts.append(f"<td>{htf}</td>")
        html_parts.append(f"<td>{entry_time}</td>")
        html_parts.append(f"<td>{fmt_price(entry_price)}</td>")
        html_parts.append(f"<td>{exit_time}</td>")
        html_parts.append(f"<td>{fmt_price(exit_price)}</td>")
        html_parts.append(f"<td>{fmt(stake)}</td>")
        html_parts.append(f"<td class='{pnl_class(pnl)}'>{fmt(pnl)}</td>")
        html_parts.append(f"<td class='{pnl_class(pnl_pct)}'>{fmt_pct(pnl_pct)}</td>")
        html_parts.append("</tr>")

    html_parts.append("</table>")
    html_parts.append("</div></body></html>")

    # HTML speichern
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text("\n".join(html_parts), encoding="utf-8")


def save_json_summary(
    trades: List[Dict],
    phase_info: Dict[str, str],
    final_capital: float
):
    """Speichere Zusammenfassung als JSON."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_trades": len(trades),
        "final_capital": final_capital,
        "start_capital": START_CAPITAL,
        "total_pnl": sum(float(t.get("pnl", 0) or 0) for t in trades),
        "winners": sum(1 for t in trades if float(t.get("pnl", 0) or 0) > 0),
        "losers": sum(1 for t in trades if float(t.get("pnl", 0) or 0) < 0),
        "current_phases": phase_info,
        "trades": trades,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    run_sweep_dashboard()
