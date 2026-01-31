"""
Optimal hold times based on simulation analysis (2025-01-22).

v1.2: Only optimize winners, keep losers unchanged.

Analysis Results:
- 4 bars: Best win rate (74.7%)
- Winners optimized: TNSR, LUNC, ZEC, SOL, LINK, SUI -> 4 bars
- Losers unchanged: ETH, BTC, XRP -> 5 bars (original)

Updated for USDC pairs with data-driven values.
"""

# Optimal hold times from real simulation data
# Format: (symbol, direction) -> bars
OPTIMAL_HOLD_BARS = {
    # BTC - LOSER but keep original 5 bars
    ("BTC/USDC", "long"): 5,
    ("BTC/USDC", "short"): 5,
    ("BTC/USDT", "long"): 5,
    ("BTC/USDT", "short"): 5,

    # ETH - LOSER but keep original 5 bars
    ("ETH/USDC", "long"): 5,
    ("ETH/USDC", "short"): 2,
    ("ETH/USDT", "long"): 5,
    ("ETH/USDT", "short"): 2,

    # LINK - WINNER: optimized to 4 bars
    ("LINK/USDC", "long"): 4,
    ("LINK/USDC", "short"): 2,
    ("LINK/USDT", "long"): 4,
    ("LINK/USDT", "short"): 2,

    # LUNC - WINNER: optimized to 4 bars (57% win rate, +4376 PnL)
    ("LUNC/USDC", "long"): 4,
    ("LUNC/USDC", "short"): 2,
    ("LUNC/USDT", "long"): 4,
    ("LUNC/USDT", "short"): 2,

    # SOL - WINNER: optimized to 4 bars
    ("SOL/USDC", "long"): 4,
    ("SOL/USDC", "short"): 2,
    ("SOL/USDT", "long"): 4,
    ("SOL/USDT", "short"): 2,

    # SUI - WINNER: optimized to 4 bars
    ("SUI/USDC", "long"): 4,
    ("SUI/USDC", "short"): 2,
    ("SUI/USDT", "long"): 4,
    ("SUI/USDT", "short"): 2,

    # TNSR - BIGGEST WINNER: optimized to 4 bars (59% win rate, +8552 PnL)
    ("TNSR/USDC", "long"): 4,
    ("TNSR/USDC", "short"): 2,
    ("TNSR/USDT", "long"): 4,
    ("TNSR/USDT", "short"): 2,

    # XRP - LOSER but keep original 5 bars
    ("XRP/USDC", "long"): 5,
    ("XRP/USDC", "short"): 2,
    ("XRP/USDT", "long"): 5,
    ("XRP/USDT", "short"): 2,

    # ZEC - WINNER: optimized to 4 bars (56% win rate, +1648 PnL)
    ("ZEC/USDC", "long"): 4,
    ("ZEC/USDC", "short"): 4,
    ("ZEC/USDT", "long"): 4,
    ("ZEC/USDT", "short"): 4,

    # ADA - no data yet, use default 4
    ("ADA/USDC", "long"): 4,
    ("ADA/USDC", "short"): 3,
    ("ADA/USDT", "long"): 4,
    ("ADA/USDT", "short"): 3,

    # ICP - no data yet, use default 4
    ("ICP/USDC", "long"): 4,
    ("ICP/USDC", "short"): 3,
    ("ICP/USDT", "long"): 4,
    ("ICP/USDT", "short"): 3,

    # BNB - no data yet, use default 4
    ("BNB/USDC", "long"): 4,
    ("BNB/USDC", "short"): 3,
    ("BNB/USDT", "long"): 4,
    ("BNB/USDT", "short"): 3,

    # TAO - no data yet, use default 4
    ("TAO/USDC", "long"): 4,
    ("TAO/USDC", "short"): 3,
    ("TAO/USDT", "long"): 4,
    ("TAO/USDT", "short"): 3,
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
