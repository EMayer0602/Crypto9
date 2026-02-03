# Crypto9 - Supertrend Trading Bot

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

## Trading Progression (Recommended Workflow)

The system supports a safe progression from simulation to live trading:

```
┌────────────┐    ┌──────────┐    ┌─────────┐    ┌────────────┐    ┌───────────┐
│ SIMULATION │ → │ TESTNET  │ → │ DRY-RUN │ → │ MICRO-LIVE │ → │ FULL-LIVE │
│  (Paper)   │    │  (Fake$) │    │(Real,no│    │ (10 USD)   │    │ (Full $)  │
└────────────┘    └──────────┘    │ orders)│    └────────────┘    └───────────┘
                                  └─────────┘
```

| Stage | Description | Risk |
|-------|-------------|------|
| **Simulation** | Historical backtesting with fake capital | None |
| **Testnet** | Real orders on Binance testnet (fake money) | None |
| **Dry-Run** | Live signals, no order execution | None |
| **Micro-Live** | Small stakes (10 USD) with limit orders | Minimal |
| **Full-Live** | Full capital trading | Full |

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

**Continuous simulation loop (auto-refresh):**
```bash
python paper_trader.py --simulate --loop --signal-interval 15
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

### 4) Testnet Trading (Real Orders on Testnet)
Place actual orders on Binance testnet (fake money, real order flow):
```bash
python paper_trader.py --testnet --place-orders
```

**Testnet uses:**
- Spot: `testnet.binance.vision`
- Futures: `testnet.binancefuture.com`
- HybridOrderExecutor: Long trades → Spot, Short trades → Futures

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

**Testnet simulation (backfill + forward):**
```bash
python paper_trader.py --testnet --simulate --start 2025-01-01
```

### 5) Dry-Run Mode (Live Signals, No Orders)
Monitor live signals without executing orders:
```bash
python paper_trader.py --monitor
# (without --place-orders flag)
```

### 6) Filter by Symbols/Indicators

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

### 7) Position Sizing

**Fixed stake per trade:**
```bash
python paper_trader.py --stake 100
```

**Dynamic sizing (default):**
```bash
python paper_trader.py  # Uses total_capital / 10 with compound growth
```

**Max open positions:**
```bash
python paper_trader.py --max-open-positions 10
```

### 8) State Management

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

### 9) Parameter Refresh
Re-run parameter optimization before trading:
```bash
python paper_trader.py --refresh-params
```

### 10) Debug Mode
Verbose logging for signal filtering decisions:
```bash
python paper_trader.py --debug-signals
```

### 11) SMS Notifications
Send trade alerts via Twilio:
```bash
python paper_trader.py --notify-sms --sms-to "+491234567890"
```

### 12) Replay Precomputed Trades
Use existing CSV instead of simulating:
```bash
python paper_trader.py --replay-trades-csv report_html/last48_trades.csv
```

### 13) Use Futures Signals
Generate signals from futures data (execute on spot):
```bash
python paper_trader.py --use-futures-signals
```

---

## Parameter Combinations and Priorities

### Mode Priority (paper_trader.py)

When multiple mode flags are set, they follow this priority:

| Priority | Flag | Behavior |
|----------|------|----------|
| 1 | `--force-entry` | Force single entry, then exit |
| 2 | `--monitor` | Continuous monitoring loop |
| 3 | `--simulate` | Historical simulation |
| 4 | (none) | Single tick (one-time check) |

**Important:** `--simulate` overrides `--monitor`! For continuous simulation use `--loop`:
```bash
# WRONG: --monitor is ignored
python paper_trader.py --simulate --testnet --monitor

# CORRECT: Use --loop for continuous simulation
python paper_trader.py --simulate --testnet --loop
```

### --monitor vs --signal-interval

| Parameter | Purpose |
|-----------|---------|
| `--monitor` | **Mode**: Enables continuous operation |
| `--signal-interval` | **Timing**: Minutes between signal checks (default: 15) |

```bash
# Monitor mode checking every 10 minutes
python paper_trader.py --monitor --signal-interval 10
```

### Dashboard: --loop vs --interval

| Parameter | Purpose |
|-----------|---------|
| `--loop` | **Mode**: Run dashboard continuously |
| `--interval` | **Timing**: Seconds between refreshes (default: 60) |

| Combination | Behavior |
|-------------|----------|
| Neither | Single dashboard generation |
| `--loop` only | Continuous, refresh every 60 sec |
| `--interval 30` only | Single run (interval ignored) |
| `--loop --interval 30` | Continuous, refresh every 30 sec |

```bash
# Dashboard updating every 30 seconds
python TestnetDashboard.py --loop --interval 30 --start 2026-01-01
```

### Dashboard Start Date

| Command | Trades shown from |
|---------|-------------------|
| `python TestnetDashboard.py` | 2025-12-01 (default) |
| `python TestnetDashboard.py --start 2026-01-15` | 2026-01-15 |
| `python paper_trader.py --testnet --dashboard-start 2026-01-01` | 2026-01-01 |

**Note:** When filtering by date, all displayed values (stakes, PnL) are recalculated dynamically starting from 16,500 initial capital.

---

## Output Directory Structure

Different modes write to separate directories to prevent data conflicts:

| Mode | Output Directory | Description |
|------|------------------|-------------|
| **Normal** | `report_html/` | Live trading reports |
| **Testnet** | `report_testnet/` | Testnet trading reports |
| **Simulation** | `report_simulation/` | Simulation-only reports |
| **Simulation + Testnet** | `report_simulation_testnet/` | Testnet simulation reports |

Each directory contains:
```
report_*/
├── trading_summary.html      # HTML summary report
├── trading_summary.json      # JSON data for dashboard
├── dashboard.html            # English dashboard
├── dashboard_de.html         # German dashboard (Deutsch)
└── charts/
    ├── equity_curve.html     # Capital growth visualization
    ├── monthly_returns.html  # Monthly PnL bar chart
    └── [SYMBOL]_chart.html   # Per-symbol price charts
```

---

## Dashboard Features

The trading dashboard includes:

- **Dark Theme** - Easy on the eyes
- **Dynamic Stake** - Stake = Capital / max_positions with compound growth
- **Amount Column** - Shows position size (Amount = Stake / Entry Price)
- **Live Prices** - Real-time price updates for open positions
- **Unrealized PnL** - Current profit/loss for open positions
- **German Support** - Full German translation (`dashboard_de.html`)
- **Auto-Refresh** - Updates every 60 seconds

### Standalone Dashboard Generation

**Single run:**
```bash
python TestnetDashboard.py --start 2025-12-31
```

**Continuous loop with auto-refresh:**
```bash
python TestnetDashboard.py --start 2025-12-31 --loop --interval 60
```

**With custom capital and max positions:**
```bash
python TestnetDashboard.py --start 2025-12-31 --loop --interval 60 \
  --start-capital 16500 --max-positions 10
```

---

## Analysis & Visualization

### Equity Curve (Kapitalkurve)
After running a simulation, an equity curve is automatically generated showing:
- Capital growth over time
- Peak equity line
- Drawdown visualization
- Cumulative PnL by symbol

Output: `report_*/charts/equity_curve.html`

The equity curve shows:
- **Start → Final Capital** with return percentage
- **Maximum Drawdown** in USDT and percentage
- **Trade count** and timing

### Monthly PnL Bar Chart (Monatliche PnL)
Automatically generated alongside the equity curve:
- Bar chart with monthly returns
- Color-coded (green = profit, red = loss)
- Win rate per month
- Trade count per month

Output: `report_*/charts/monthly_returns.html`

Shows:
- **Total PnL** across all months
- **Average monthly return**
- **Best/Worst month** identification

### Equity Curve Comparison (Kapitalkurvenvergleich)
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

### Testnet with Dashboard
```bash
# Terminal 1: Run testnet trading
python paper_trader.py --testnet --place-orders --monitor

# Terminal 2: Run dashboard (optional, auto-updates from testnet)
python TestnetDashboard.py --loop --interval 60
```

---

## Helper Scripts

| Script | Description |
|--------|-------------|
| `download_ohlcv.py` | Download OHLCV data for all symbols |
| `BinTestnetOrderHistory.py` | Check testnet order history |
| `BinTestnetFaucet.py` | Get testnet funds from faucet |
| `BinTestnetProbe.py` | Test testnet API connectivity |
| `ClearPositions.py` | Clear paper trading positions |
| `ClearTestnetPositions.py` | Clear testnet positions |
| `compare_equity_curves.py` | Compare equity curves across periods |
| `regenerate_summary.py` | Regenerate summary from trade logs |
| `merge_history.py` | Merge multiple trade history files |
| `extract_last24_from_detailed.py` | Extract last 24h from detailed logs |
| `analyze_futures_lead.py` | Analyze futures lead/lag vs spot |
| `lead_lag_analysis.py` | Lead/lag correlation analysis |
| `optimized_sweep.py` | Parameter optimization sweep |

---

## CLI Arguments Reference

### Core Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--simulate` | false | Run historical simulation |
| `--start` | 24h ago | Simulation start (ISO format) |
| `--end` | now | Simulation end (ISO format) |
| `--monitor` | false | Continuous monitor loop |
| `--loop` | false | Continuous simulation loop |

### Trading Mode
| Argument | Default | Description |
|----------|---------|-------------|
| `--testnet` | false | Use Binance testnet |
| `--place-orders` | false | Execute real orders |
| `--use-futures-signals` | false | Signals from futures data |

### Position Management
| Argument | Default | Description |
|----------|---------|-------------|
| `--stake` | dynamic | Fixed stake per trade |
| `--max-open-positions` | 10 | Max concurrent positions |
| `--force-entry` | none | Force entry "SYMBOL:direction" |
| `--force-lookback-hours` | 24 | Lookback for force entry signal |

### Filters
| Argument | Default | Description |
|----------|---------|-------------|
| `--symbols` | all | Comma-separated symbol list |
| `--indicators` | all | Comma-separated indicator list |

### Monitor Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--signal-interval` | 15 | Minutes between cycles |
| `--spike-interval` | 5 | Minutes between ATR scans |
| `--atr-mult` | 2.5 | ATR spike threshold |
| `--poll-seconds` | 30 | Sleep in monitor loop |

### State Management
| Argument | Default | Description |
|----------|---------|-------------|
| `--reset-state` | false | Delete saved state |
| `--clear-positions` | false | Clear open positions |
| `--clear-all` | false | Clear state + outputs |
| `--clear-outputs` | false | Clear outputs only |
| `--use-saved-state` | false | Seed sim with saved state |

### Output Files
| Argument | Default | Description |
|----------|---------|-------------|
| `--sim-log` | auto | CSV path for trades |
| `--sim-json` | auto | JSON path for trades |
| `--summary-html` | auto | HTML summary path |
| `--summary-json` | auto | JSON summary path |

### Other
| Argument | Default | Description |
|----------|---------|-------------|
| `--debug-signals` | false | Verbose signal logging |
| `--refresh-params` | false | Re-run optimization |
| `--notify-sms` | false | SMS alerts via Twilio |
| `--sms-to` | env | Phone numbers for SMS |
| `--dashboard-start` | 2025-12-01 | Dashboard start date |
| `--replay-trades-csv` | none | Replay from CSV |
| `--verbose-sim-entries` | false | Print entry messages |

---

## Output Files

| File | Description |
|------|-------------|
| `paper_trading_state.json` | Current positions and capital |
| `paper_trading_simulation_log.csv/json` | Simulation trade history |
| `paper_trading_actual_trades.csv/json` | Live trade history |
| `report_html/best_params_overall.csv` | Optimized parameters per symbol |
| `report_*/charts/*.html` | Interactive price charts per symbol |
| `report_*/charts/equity_curve.html` | Equity curve with drawdown |
| `report_*/charts/monthly_returns.html` | Monthly PnL bar chart |
| `report_*/charts/equity_comparison.html` | Multi-period comparison |
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

Key settings in `paper_trader.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `START_TOTAL_CAPITAL` | 16,500 | Starting capital |
| `MAX_OPEN_POSITIONS` | 10 | Max concurrent positions |
| `STAKE_DIVISOR` | 10 | Stake = capital / divisor |
| `MAX_LONG_POSITIONS` | 10 | Max long (spot) positions |
| `MAX_SHORT_POSITIONS` | 5 | Max short (futures) positions |

---

## Git Workflow
```bash
git status
git add .
git commit -m "Your message"
git push origin branch-name
```
