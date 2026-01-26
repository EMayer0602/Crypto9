# Crypto6 - Supertrend Trading Bot

## Overview
Paper trading and backtesting system for Supertrend-based crypto strategies with multiple indicators (JMA, KAMA, MAMA, etc.) and higher timeframe filters.

## Prerequisites
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`
- `.env` file with API keys:
  ```
  BINANCE_API_KEY_TEST=your_testnet_key
  BINANCE_API_SECRET_TEST=your_testnet_secret
  BINANCE_API_KEY=your_live_key (optional)
  BINANCE_API_SECRET=your_live_secret (optional)
  ```

---

## Usage Scenarios

### 1) Single Live Tick (Default)
Run one trading cycle - check signals and update positions:
```bash
python paper_trader.py
```

With specific symbols:
```bash
python paper_trader.py --symbols "BTC/EUR,ETH/EUR"
```

### 2) Historical Simulation

**Last 24 hours (default):**
```bash
python paper_trader.py --simulate
```

**Last 48 hours:**
```bash
python paper_trader.py --simulate --start 2025-12-30T00:00:00
```

**Full year backtest:**
```bash
python paper_trader.py --simulate --start 2025-01-01
```

**Custom date range:**
```bash
python paper_trader.py --simulate --start 2025-06-01 --end 2025-12-31
```

**With custom output files:**
```bash
python paper_trader.py --simulate --start 2025-01-01 \
  --sim-log simulation_logs/my_trades.csv \
  --sim-json simulation_logs/my_trades.json \
  --summary-html report_html/my_summary.html
```

### 3) Continuous Monitor Mode
Run indefinitely with scheduled cycles:
```bash
python paper_trader.py --monitor
```

**Custom intervals:**
```bash
python paper_trader.py --monitor \
  --signal-interval 15 \
  --spike-interval 5 \
  --atr-mult 2.5 \
  --poll-seconds 30
```

### 4) Testnet Trading (Real Orders)
Place actual orders on Binance testnet:
```bash
python paper_trader.py --testnet --place-orders
```

**Force entry on specific symbol:**
```bash
python paper_trader.py --testnet --place-orders --force-entry "BTC/EUR:long"
```

**Force entry with lookback window:**
```bash
python paper_trader.py --testnet --place-orders \
  --force-entry "ETH/EUR:short" \
  --force-lookback-hours 48
```

### 5) Filter by Symbols/Indicators

**Specific symbols only:**
```bash
python paper_trader.py --symbols "BTC/EUR,ETH/EUR,SOL/EUR"
```

**Specific indicator only:**
```bash
python paper_trader.py --indicators "jma"
```

**Combined:**
```bash
python paper_trader.py --simulate --start 2025-01-01 \
  --symbols "BTC/EUR,ETH/EUR" \
  --indicators "jma,kama"
```

### 6) Position Sizing

**Fixed stake per trade:**
```bash
python paper_trader.py --stake 100
```

**Dynamic sizing (default):**
```bash
python paper_trader.py  # Uses total_capital / 7
```

**Max open positions:**
```bash
python paper_trader.py --max-open-positions 10
```

### 7) State Management

**Clear all positions:**
```bash
python paper_trader.py --clear-positions
```

**Reset state (fresh start):**
```bash
python paper_trader.py --reset-state
```

**Clear all outputs + state:**
```bash
python paper_trader.py --clear-all
```

**Clear outputs only:**
```bash
python paper_trader.py --clear-outputs
```

### 8) Parameter Refresh
Re-run parameter optimization before trading:
```bash
python paper_trader.py --refresh-params
```

### 9) Debug Mode
Verbose logging for signal filtering decisions:
```bash
python paper_trader.py --debug-signals
```

### 10) SMS Notifications
Send trade alerts via Twilio:
```bash
python paper_trader.py --notify-sms --sms-to "+491234567890"
```

### 11) Replay Precomputed Trades
Use existing CSV instead of simulating:
```bash
python paper_trader.py --replay-trades-csv report_html/last48_trades.csv
```

---

## Analysis & Visualization

### 12) Equity Curve (Kapitalkurve)
After running a simulation, an equity curve is automatically generated showing:
- Capital growth over time
- Peak equity line
- Drawdown visualization
- Cumulative PnL by symbol

Output: `report_html/charts/equity_curve.html`

The equity curve shows:
- **Start → Final Capital** with return percentage
- **Maximum Drawdown** in USDT and percentage
- **Trade count** and timing

### 13) Monthly PnL Bar Chart (Monatliche PnL)
Automatically generated alongside the equity curve:
- Bar chart with monthly returns
- Color-coded (green = profit, red = loss)
- Win rate per month
- Trade count per month

Output: `report_html/charts/monthly_returns.html`

Shows:
- **Total PnL** across all months
- **Average monthly return**
- **Best/Worst month** identification

### 14) Equity Curve Comparison (Kapitalkurvenvergleich)
Compare performance across different time periods:

**Using existing simulation logs:**
```bash
python compare_equity_curves.py
```

**Run new simulations and compare:**
```bash
python compare_equity_curves.py --run-simulations
```

**Custom periods:**
```bash
python compare_equity_curves.py --periods "2024-06-01,2024-12-31,H2_2024" "2025-01-01,2025-06-30,H1_2025"
```

**Adding a new period:**

Step 1 - Run simulation for the new period:
```bash
python paper_trader.py --simulate --start 2025-01-01 --end 2025-12-31 \
  --sim-log simulation_logs/full_2025_trades.csv \
  --sim-json simulation_logs/full_2025_trades.json
```

Step 2 - Include in comparison (label must match filename: `Full_2025` → `full_2025_trades.csv`):
```bash
python compare_equity_curves.py --periods "2024-06-01,2024-12-31,H2_2024" "2025-01-01,2025-06-30,H1_2025" "2025-01-01,2025-12-31,Full_2025"
```

Or run all simulations automatically:
```bash
python compare_equity_curves.py --run-simulations --periods "2024-06-01,2024-12-31,H2_2024" "2025-01-01,2025-06-30,H1_2025" "2025-01-01,2025-12-31,Full_2025"
```

Default periods compared:
- H2 2024 (Jun-Dec 2024)
- H1 2025 (Jan-Jun 2025)
- Full 2025 (Jan-Dec 2025)

Output: `report_html/charts/equity_comparison.html`

The comparison shows:
- **Overlaid equity curves** (normalized by days from start)
- **Drawdown comparison**
- **Summary bar chart** with return %, trade count, max drawdown

---

## Common Workflows

### Full Backtest with Reports
```bash
python paper_trader.py --simulate --start 2025-01-01 \
  --sim-log simulation_logs/full_2025_trades.csv \
  --sim-json simulation_logs/full_2025_trades.json \
  --summary-html report_html/trading_summary.html \
  --summary-json report_html/trading_summary.json
```

### Fresh Start + Simulation
```bash
python paper_trader.py --clear-all
python paper_trader.py --simulate --start 2025-01-01
```

### Live Paper Trading (Continuous)
```bash
python paper_trader.py --monitor --signal-interval 15
```

### Testnet Order Testing
```bash
python paper_trader.py --testnet --place-orders --stake 50 \
  --force-entry "LUNC/USDT:long" --max-open-positions 50
```

---

## Helper Scripts

### Add/Remove Single Symbol (without full sweep)
```bash
# Add a new symbol (runs sweep only for that symbol)
python sweep_single_symbol.py --add "DOGE/USDT"

# Remove a symbol from all files
python sweep_single_symbol.py --remove "DOGE/USDT"

# List all current symbols
python sweep_single_symbol.py --list
```

### Download OHLCV Data
```bash
python download_ohlcv.py
```

### Check Testnet Order History
```bash
python BinTestnetOrderHistory.py
```

### Clear Testnet Positions
```bash
python ClearPositions.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `paper_trading_state.json` | Current positions and capital |
| `paper_trading_simulation_log.csv/json` | Simulation trade history |
| `paper_trading_actual_trades.csv/json` | Live trade history |
| `report_html/best_params_overall.csv` | Optimized parameters per symbol |
| `report_html/charts/*.html` | Interactive price charts per symbol |
| `report_html/charts/equity_curve.html` | Equity curve with drawdown |
| `report_html/charts/monthly_returns.html` | Monthly PnL bar chart |
| `report_html/charts/equity_comparison.html` | Multi-period comparison |
| `simulation_logs/*_trades.csv/json` | Period-specific simulation logs |
| `ohlcv_cache/*.csv` | Cached OHLCV data |

---

## Configuration

Key settings in `Supertrend_5Min.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `HIGHER_TIMEFRAME` | "6h" | HTF filter timeframe |
| `HTF_LOOKBACK` | 1000 | Bars for HTF calculation |
| `LOOKBACK` | 8760 | 1 year of hourly bars |
| `SYMBOLS` | [...] | Trading pairs |

---

## Git Workflow
```bash
git status
git add .
git commit -m "Your message"
git push origin branch-name
```

---

## Dashboard (TestnetDashboard.py)

Das Dashboard zeigt alle offenen Positionen, geschlossene Trades und Performance-Statistiken in einer HTML-Übersicht.

### Dashboard starten

**Einmalig generieren:**
```bash
python TestnetDashboard.py
```

**Kontinuierlich aktualisieren (alle 30 Sekunden):**
```bash
python TestnetDashboard.py --loop
```

**Mit benutzerdefiniertem Intervall (z.B. 60 Sekunden):**
```bash
python TestnetDashboard.py --loop --interval 60
```

### Dashboard Optionen

| Option | Beschreibung | Standard |
|--------|--------------|----------|
| `--loop` | Kontinuierlicher Modus - aktualisiert automatisch | Aus |
| `--interval N` | Aktualisierungsintervall in Sekunden | 30 |

### Dashboard Ausgabe

Das Dashboard generiert `report_testnet/dashboard.html` mit:
- **Open Positions**: Aktuelle offene Trades mit Echtzeit-Preisen von Binance
- **Closed Trades**: Alle geschlossenen Trades mit PnL
- **Performance Summary**: Win Rate, Total PnL, Equity Curve
- **Symbol Statistics**: Performance pro Symbol

### Beispiele

```bash
# Dashboard einmal generieren und im Browser öffnen
python TestnetDashboard.py
start report_testnet\dashboard.html

# Dashboard alle 60 Sekunden aktualisieren
python TestnetDashboard.py --loop --interval 60

# Dashboard alle 5 Minuten aktualisieren
python TestnetDashboard.py --loop --interval 300
```

---

## Stable Versions (Tags)

### v1.0-long-only-optimal (2025-01-22)
**Commit:** `15c9747`

Beste Long-Only Konfiguration mit verdoppeltem Kapital.

| Einstellung | Wert |
|-------------|------|
| STAKE_DIVISOR | 10 |
| MAX_OPEN_POSITIONS | 10 |
| MAX_SHORT_POSITIONS | 0 (Long only) |
| Exit-Strategie | Forced nach optimal_hold_bars |
| Trend flip | aktiv |
| USE_TIME_BASED_EXIT | True |
| DISABLE_TREND_FLIP_EXIT | False |

**Wiederherstellen:**
```bash
git checkout v1.0-long-only-optimal
```

**Tag lokal erstellen (falls nicht vorhanden):**
```bash
git tag v1.0-long-only-optimal 15c9747
```

**Wichtige Dateien:**
- `paper_trader.py` - Hauptlogik mit evaluate_exit
- `optimal_hold_times_defaults.py` - USDC Symbole mit optimal bars
- `report_html/best_params_overall.csv` - Parameter für alle Indikatoren

**Exit-Logik:**
1. ATR Stop (falls konfiguriert)
2. **Forced Time-based Exit** nach optimal_hold_bars (auch bei Verlust!)
3. Trend Flip (nach min_bars_for_trend_flip)

---

### v1.1-optimized-bars (2025-01-22)
**Commit:** `4a4c969`

Optimierte optimal_hold_bars aus echten Simulationsdaten + TAO alle Indikatoren.

**Änderungen gegenüber v1.0:**

| Symbol | Alt | Neu | Grund |
|--------|-----|-----|-------|
| ETH | 5 | **2** | Größter Verlierer (-8573 PnL) |
| BTC | 5 | **3** | Verlierer (-257 PnL) |
| XRP | 5 | **3** | Verlierer (-208 PnL) |
| TNSR, LUNC, ZEC, SOL, LINK, SUI | 2-5 | **4** | Beste Win Rate (74.7%) |

**TAO erweitert:**
- supertrend (existierte bereits)
- htf_crossover (neu)
- jma (neu)
- kama (neu)

**Wiederherstellen:**
```bash
git checkout v1.1-optimized-bars
```

**Tag lokal erstellen (falls nicht vorhanden):**
```bash
git tag v1.1-optimized-bars 4a4c969
```

---

### v1.2-winners-only (2025-01-22) ⭐ BEST
**Commit:** `33b7bb1`

**BESTES ERGEBNIS!** Nur Gewinner optimiert, Verlierer auf Original belassen.

**Strategie:**
- Gewinner (TNSR, LUNC, ZEC, SOL, LINK, SUI) → 4 bars (optimiert)
- Verlierer (ETH, BTC, XRP) → 5 bars (original, nicht anfassen)

| Symbol | Bars | Status |
|--------|------|--------|
| TNSR, LUNC, ZEC, SOL, LINK, SUI | **4** | Optimiert (74.7% Win Rate) |
| ETH, BTC, XRP | **5** | Original (nicht reduziert) |

**Wiederherstellen:**
```bash
git checkout v1.2-winners-only
```

**Tag lokal erstellen (falls nicht vorhanden):**
```bash
git tag v1.2-winners-only 33b7bb1
```

---

### Rollback Übersicht

| Version | Commit | Beschreibung |
|---------|--------|--------------|
| v1.0-long-only-optimal | `15c9747` | Basis Long-Only, verdoppeltes Kapital |
| v1.1-optimized-bars | `4a4c969` | Alle Bars optimiert + TAO Indikatoren |
| **v1.2-winners-only** | `33b7bb1` | ⭐ **BEST** - Nur Gewinner optimiert |

**Schnell-Rollback:**
```bash
# Zurück zu v1.0
git checkout v1.0-long-only-optimal

# Zurück zu v1.1
git checkout v1.1-optimized-bars

# Zurück zu v1.2 (BEST)
git checkout v1.2-winners-only

# Zurück zum neuesten Stand
git checkout claude/review-project-0ktcy
```

