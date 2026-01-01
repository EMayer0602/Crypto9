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
| `report_html/charts/*.html` | Interactive price charts |
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
