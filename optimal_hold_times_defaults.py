"""
Optimal hold times based on simulation analysis (2025-01-22).

Analysis Results:
- 4 bars: Best win rate (74.7%)
- Winners: TNSR, LUNC, ZEC, SOL, LINK, SUI
- Losers: ETH (-8573), BTC (-257), XRP (-208) -> reduced bars to cut losses faster

Updated for USDC pairs with data-driven values.
"""

# Optimal hold times from real simulation data
# Format: (symbol, direction) -> bars
OPTIMAL_HOLD_BARS = {
    # BTC - LOSER: reduced from 5 to 3 to cut losses faster
    ("BTC/USDC", "long"): 3,
    ("BTC/USDC", "short"): 3,

    # ETH - BIGGEST LOSER: reduced from 5 to 2 to cut losses faster
    ("ETH/USDC", "long"): 2,
    ("ETH/USDC", "short"): 2,

    # LINK - small winner, keep at 4
    ("LINK/USDC", "long"): 4,
    ("LINK/USDC", "short"): 2,

    # LUNC - WINNER: keep at 4 (57% win rate, +4376 PnL)
    ("LUNC/USDT", "long"): 4,
    ("LUNC/USDT", "short"): 2,

    # SOL - winner, increase to 4
    ("SOL/USDC", "long"): 4,
    ("SOL/USDC", "short"): 2,

    # SUI - small winner, keep at 4
    ("SUI/USDC", "long"): 4,
    ("SUI/USDC", "short"): 2,

    # TNSR - BIGGEST WINNER: keep at 4 (59% win rate, +8552 PnL)
    ("TNSR/USDC", "long"): 4,
    ("TNSR/USDC", "short"): 2,

    # XRP - LOSER: reduced from 5 to 3
    ("XRP/USDC", "long"): 3,
    ("XRP/USDC", "short"): 2,

    # ZEC - WINNER: keep at 4 (56% win rate, +1648 PnL)
    ("ZEC/USDC", "long"): 4,
    ("ZEC/USDC", "short"): 4,

    # ADA - no data yet, use default 4
    ("ADA/USDC", "long"): 4,
    ("ADA/USDC", "short"): 3,

    # ICP - no data yet, use default 4
    ("ICP/USDC", "long"): 4,
    ("ICP/USDC", "short"): 3,

    # BNB - no data yet, use default 4
    ("BNB/USDC", "long"): 4,
    ("BNB/USDC", "short"): 3,

    # TAO - no data yet, use default 4
    ("TAO/USDC", "long"): 4,
    ("TAO/USDC", "short"): 3,
}

def get_optimal_hold_bars(symbol: str, direction: str) -> int:
    """
    Get optimal hold time for symbol/direction based on real data.

    Returns conservative defaults if symbol/direction not found:
    - Long: 4 bars (best win rate from analysis)
    - Short: 3 bars (from analysis average)
    """
    direction = direction.lower()
    key = (symbol, direction)

    if key in OPTIMAL_HOLD_BARS:
        return OPTIMAL_HOLD_BARS[key]

    # Fallback: 4 bars for long (best win rate), 3 for short
    return 4 if direction == "long" else 3
