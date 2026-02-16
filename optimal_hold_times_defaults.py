"""
Optimal hold times based on peak profit analysis - LONG ONLY (USDC pairs).

Analysis results:
- Long trades: Avg optimal 5 bars (saves 59.82 USDT/trade, total 1,196 USDT)

Key insight: Peaks occur MUCH earlier than expected (2-10 bars vs 12-15)!

ADJUSTMENT: Long hold times reduced by 40-50% after poor performance.
Longs need to exit faster to capture peaks before reversals.
"""

# Optimal hold times from real data analysis
# Format: (symbol, direction) -> bars
# Updated to match USDC trading symbols (previously EUR)
OPTIMAL_HOLD_BARS = {
    # BTC/USDC
    ("BTC/USDC", "long"): 5,

    # ETH/USDC - REDUCED LONG from 10 to 5
    ("ETH/USDC", "long"): 5,    # REDUCED: Was 10, cut 50% for faster exits

    # LINK/USDC
    ("LINK/USDC", "long"): 3,   # Peak at 53%, saves 35.94 USDT/trade

    # LUNC/USDT - REDUCED LONG from 7 to 4
    ("LUNC/USDT", "long"): 4,   # REDUCED: Was 7, cut 43% for faster exits

    # SOL/USDC
    ("SOL/USDC", "long"): 3,    # Peak at 51%, saves 39.67 USDT/trade

    # SUI/USDC - REDUCED LONG from 9 to 5
    ("SUI/USDC", "long"): 5,    # REDUCED: Was 9, cut 44% for faster exits

    # TNSR/USDC
    ("TNSR/USDC", "long"): 2,   # Peak at 46%, saves 98.40 USDT/trade

    # XRP/USDC
    ("XRP/USDC", "long"): 5,

    # ZEC/USDC
    ("ZEC/USDC", "long"): 2,    # Peak at 29%, saves 53.83 USDT/trade

    # ADA/USDC
    ("ADA/USDC", "long"): 5,

    # ICP/USDC
    ("ICP/USDC", "long"): 5,

    # BNB/USDC
    ("BNB/USDC", "long"): 5,
}

def get_optimal_hold_bars(symbol: str, direction: str) -> int:
    """
    Get optimal hold time for symbol/direction based on real data.

    Returns conservative default of 5 bars if symbol/direction not found.
    """
    direction = direction.lower()
    key = (symbol, direction)

    if key in OPTIMAL_HOLD_BARS:
        return OPTIMAL_HOLD_BARS[key]

    # Fallback to analysis average
    return 5
