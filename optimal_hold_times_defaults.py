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

# Optimal hold times from real data analysis
# Format: (symbol, direction) -> bars
OPTIMAL_HOLD_BARS = {
    # USDC pairs (primary trading symbols) - all use optimal=4 for long
    ("BTC/USDC", "long"): 4,
    ("BTC/USDC", "short"): 5,
    ("ETH/USDC", "long"): 4,
    ("ETH/USDC", "short"): 3,
    ("ADA/USDC", "long"): 4,
    ("ADA/USDC", "short"): 3,
    ("BNB/USDC", "long"): 4,
    ("BNB/USDC", "short"): 3,
    ("SOL/USDC", "long"): 4,
    ("SOL/USDC", "short"): 3,
    ("XRP/USDC", "long"): 4,
    ("XRP/USDC", "short"): 3,
    ("LINK/USDC", "long"): 4,
    ("LINK/USDC", "short"): 3,
    ("SUI/USDC", "long"): 4,
    ("SUI/USDC", "short"): 3,
    ("ICP/USDC", "long"): 4,
    ("ICP/USDC", "short"): 3,
    ("TAO/USDC", "long"): 4,
    ("TAO/USDC", "short"): 3,

    # USDT pairs
    ("LUNC/USDT", "long"): 4,
    ("LUNC/USDT", "short"): 3,

    # Special cases with different optimal values
    ("TNSR/USDC", "long"): 2,  # Aggressive exit
    ("TNSR/USDC", "short"): 2,
    ("ZEC/USDC", "long"): 2,   # Aggressive exit
    ("ZEC/USDC", "short"): 4,

    # EUR pairs (legacy)
    ("BTC/EUR", "short"): 5,
    ("ETH/EUR", "long"): 4,
    ("ETH/EUR", "short"): 2,
    ("LINK/EUR", "long"): 3,
    ("LINK/EUR", "short"): 2,
    ("SOL/EUR", "long"): 3,
    ("SOL/EUR", "short"): 2,
    ("SUI/EUR", "long"): 4,
    ("SUI/EUR", "short"): 2,
    ("XRP/EUR", "short"): 2,
}

def get_optimal_hold_bars(symbol: str, direction: str) -> int:
    """
    Get optimal hold time for symbol/direction based on real data.

    Returns conservative defaults if symbol/direction not found:
    - Long: 4 bars (matches checked-in simulation results)
    - Short: 3 bars (from analysis average)
    """
    direction = direction.lower()
    key = (symbol, direction)

    if key in OPTIMAL_HOLD_BARS:
        return OPTIMAL_HOLD_BARS[key]

    # Fallback to defaults matching checked-in results
    return 4 if direction == "long" else 3
