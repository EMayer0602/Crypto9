"""
Optimal hold times based on peak profit analysis.

Long trades: Avg optimal 5 bars
Short trades: Avg optimal 3 bars

Updated for USDC pairs.
"""

# Optimal hold times from real data analysis
# Format: (symbol, direction) -> bars
OPTIMAL_HOLD_BARS = {
    # BTC
    ("BTC/USDC", "long"): 5,
    ("BTC/USDC", "short"): 5,

    # ETH
    ("ETH/USDC", "long"): 5,
    ("ETH/USDC", "short"): 2,

    # LINK
    ("LINK/USDC", "long"): 3,
    ("LINK/USDC", "short"): 2,

    # LUNC
    ("LUNC/USDT", "long"): 4,
    ("LUNC/USDT", "short"): 2,

    # SOL
    ("SOL/USDC", "long"): 3,
    ("SOL/USDC", "short"): 2,

    # SUI
    ("SUI/USDC", "long"): 5,
    ("SUI/USDC", "short"): 2,

    # TNSR
    ("TNSR/USDC", "long"): 2,
    ("TNSR/USDC", "short"): 2,

    # XRP
    ("XRP/USDC", "long"): 5,
    ("XRP/USDC", "short"): 2,

    # ZEC
    ("ZEC/USDC", "long"): 2,
    ("ZEC/USDC", "short"): 4,

    # ADA
    ("ADA/USDC", "long"): 5,
    ("ADA/USDC", "short"): 3,

    # ICP
    ("ICP/USDC", "long"): 5,
    ("ICP/USDC", "short"): 3,

    # BNB
    ("BNB/USDC", "long"): 5,
    ("BNB/USDC", "short"): 3,

    # TAO
    ("TAO/USDC", "long"): 5,
    ("TAO/USDC", "short"): 3,
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
