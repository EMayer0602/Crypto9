"""
Optimal hold times based on peak profit analysis - REDUCED FOR LONGS.

Analysis results:
- Long trades: Avg optimal 5 bars (saves 59.82 USDT/trade, total 1,196 USDT)
- Short trades: Avg optimal 3 bars (saves 43.63 USDT/trade, total 4,145 USDT)
- Total potential savings: 5,340 USDT

Key insight: Peaks occur MUCH earlier than expected (2-10 bars vs 12-15)!

ADJUSTMENT: Long hold times reduced by 40-50% after poor performance.
Long trades showed -128 USDT loss vs Short +1,461 USDT profit.
Longs need to exit faster to capture peaks before reversals.
"""

# Optimal hold times from real data analysis (115 trades)
# Format: (symbol, direction) -> bars
OPTIMAL_HOLD_BARS = {
    # Real data from find_optimal_hold_times.py analysis

    # BTC/EUR
    ("BTC/EUR", "short"): 5,   # Peak at 73%, saves 22.94 USDT/trade

    # ETH/EUR - REDUCED LONG from 10 to 5
    ("ETH/EUR", "long"): 5,    # REDUCED: Was 10, cut 50% for faster exits
    ("ETH/EUR", "short"): 2,   # Peak at 51%, saves 27.75 USDT/trade

    # LINK/EUR
    ("LINK/EUR", "long"): 3,   # Peak at 53%, saves 35.94 USDT/trade (OK)
    ("LINK/EUR", "short"): 2,  # Peak at 53%, saves 24.93 USDT/trade

    # LUNC/USDT - REDUCED LONG from 7 to 4
    ("LUNC/USDT", "long"): 4,  # REDUCED: Was 7, cut 43% for faster exits
    ("LUNC/USDT", "short"): 2, # Peak at 41%, saves 70.75 USDT/trade

    # SOL/EUR
    ("SOL/EUR", "long"): 3,    # Peak at 51%, saves 39.67 USDT/trade (OK)
    ("SOL/EUR", "short"): 2,   # Peak at 37%, saves 44.10 USDT/trade

    # SUI/EUR - REDUCED LONG from 9 to 5
    ("SUI/EUR", "long"): 5,    # REDUCED: Was 9, cut 44% for faster exits
    ("SUI/EUR", "short"): 2,   # Peak at 61%, saves 26.04 USDT/trade

    # TNSR/USDC
    ("TNSR/USDC", "long"): 2,  # Peak at 46%, saves 98.40 USDT/trade (OK - already aggressive)
    ("TNSR/USDC", "short"): 2, # Peak at 36%, saves 51.72 USDT/trade

    # XRP/EUR
    ("XRP/EUR", "short"): 2,   # Peak at 53%, saves 27.10 USDT/trade

    # ZEC/USDC
    ("ZEC/USDC", "long"): 2,   # Peak at 29%, saves 53.83 USDT/trade (OK - already aggressive)
    ("ZEC/USDC", "short"): 4,  # Peak at 31%, saves 97.30 USDT/trade
}

def get_optimal_hold_bars(symbol: str, direction: str) -> int:
    """
    Get optimal hold time for symbol/direction based on real data.

    Returns conservative defaults if symbol/direction not found:
    - Long: 5 bars (from analysis average)
    - Short: 3 bars (from analysis average)
    """
    direction = direction.lower()
    key = (symbol, direction)

    if key in OPTIMAL_HOLD_BARS:
        return OPTIMAL_HOLD_BARS[key]

    # Fallback to analysis averages
    return 5 if direction == "long" else 3
