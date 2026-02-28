"""
Asset definitions and default portfolio weights for the trading engine.
"""
from typing import Dict, List

# Perpetual futures traded on Binance testnet
TRADING_SYMBOLS: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# Default capital allocation weights (must sum to 1.0)
ASSET_WEIGHTS_DEFAULT: Dict[str, float] = {
    "BTC/USDT": 0.50,
    "ETH/USDT": 0.30,
    "SOL/USDT": 0.20,
}
