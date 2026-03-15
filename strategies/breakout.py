"""
Volatility Breakout Strategy.

Range: Upper = max(High, 20 bars),  Lower = min(Low, 20 bars)
ATR ratio: ATR(14) / ATR(50)

Long entry (all must be true):
  1. Price > Upper                       — breakout above 20-bar range
  2. ATR_ratio > 1.2                     — volatility expanding
  3. Volume > 1.5 × SMA(Volume, 20)     — volume surge confirms move

Short entry: mirror (Price < Lower, same vol/volatility filters).

Stop  : min(candle_low, price − 1.0 × ATR(14))  for longs
Target: 2.0 × ATR(14)  (2 R — breakouts run further)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from config.settings import (
    ATR_EXPANSION_RATIO,
    ATR_PERIOD,
    BREAKOUT_LOOKBACK,
    VOLUME_SPIKE_MULTIPLIER,
)
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


class BreakoutStrategy(BaseStrategy):
    """Volatility expansion / breakout strategy."""

    def get_name(self) -> str:
        return "Volatility Breakout"

    def get_type(self) -> str:
        return "breakout"

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Return long/short signal dict or *None*."""
        required = max(BREAKOUT_LOOKBACK, 50) + 5
        if len(df) < required:
            logger.debug("Insufficient candles (%d) for BreakoutStrategy", len(df))
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 20-bar range (exclude the current bar)
        upper = high.iloc[-BREAKOUT_LOOKBACK - 1 : -1].max()
        lower = low.iloc[-BREAKOUT_LOOKBACK - 1 : -1].min()

        atr14 = _atr(df, ATR_PERIOD)
        atr50 = _atr(df, 50)
        atr_ratio = atr14.iloc[-1] / (atr50.iloc[-1] if atr50.iloc[-1] != 0 else 1e-8)

        vol_ma = volume.rolling(BREAKOUT_LOOKBACK).mean()

        price = close.iloc[-1]
        atr_val = atr14.iloc[-1]
        vol_val = volume.iloc[-1]
        vol_avg = vol_ma.iloc[-1]

        volatility_ok = atr_ratio > ATR_EXPANSION_RATIO
        volume_ok = vol_val > VOLUME_SPIKE_MULTIPLIER * vol_avg

        if price > upper and volatility_ok and volume_ok:
            stop = min(low.iloc[-1], price - atr_val)
            return {"side": "long", "stop": stop, "confidence": 0.75}

        if price < lower and volatility_ok and volume_ok:
            stop = max(high.iloc[-1], price + atr_val)
            return {"side": "short", "stop": stop, "confidence": 0.75}

        return None
