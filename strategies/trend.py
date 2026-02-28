"""
EMA Trend Momentum Strategy.

Entry logic (long):
  1. EMA(20) > EMA(50)                — bullish crossover
  2. Price > EMA(200)                 — above long-term trend
  3. ADX(14) > 25                     — strong trend confirmed
  4. Slope(EMA20, 5 bars) > 0         — momentum accelerating
  5. Volume > 1.2 × SMA(Volume, 20)  — volume confirmation

Short entry: mirror opposite.

Stop-loss : Entry ∓ 1.2 × ATR(14)
Take-profit: 1.5 R
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import (
    ADX_PERIOD,
    ADX_TREND_THRESHOLD,
    ATR_PERIOD,
    ATR_STOP_MULTIPLIER,
    EMA_FAST,
    EMA_FILTER,
    EMA_SLOW,
    VOLUME_TREND_MULTIPLIER,
)
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    idx = df.index
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    prev_high, prev_low = df["high"].shift(1), df["low"].shift(1)

    plus_dm = pd.Series(
        np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0),
        index=idx,
    )
    minus_dm = pd.Series(
        np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0),
        index=idx,
    )

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(span=period, adjust=False).mean()


class TrendStrategy(BaseStrategy):
    """EMA Momentum trend-following strategy."""

    def get_name(self) -> str:
        return "EMA Trend Momentum"

    def get_type(self) -> str:
        return "trend"

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Return long/short signal dict or *None*."""
        if len(df) < EMA_FILTER + 10:
            logger.debug("Insufficient candles (%d) for TrendStrategy", len(df))
            return None

        close = df["close"]
        volume = df["volume"]

        ema_fast = _ema(close, EMA_FAST)
        ema_slow = _ema(close, EMA_SLOW)
        ema_filter = _ema(close, EMA_FILTER)
        atr = _atr(df, ATR_PERIOD)
        adx = _adx(df, ADX_PERIOD)
        vol_ma = volume.rolling(20).mean()

        # Most-recent values
        price = close.iloc[-1]
        ef = ema_fast.iloc[-1]
        es = ema_slow.iloc[-1]
        ef200 = ema_filter.iloc[-1]
        adx_val = adx.iloc[-1]
        atr_val = atr.iloc[-1]
        vol_val = volume.iloc[-1]
        vol_avg = vol_ma.iloc[-1]

        # EMA-20 percentage slope over last 5 bars (normalised for cross-asset consistency)
        slope = (ema_fast.iloc[-1] - ema_fast.iloc[-5]) / ema_fast.iloc[-5]

        long_conditions = (
            ef > es,
            price > ef200,
            adx_val > ADX_TREND_THRESHOLD,
            slope > 0,
            vol_val > VOLUME_TREND_MULTIPLIER * vol_avg,
        )
        short_conditions = (
            ef < es,
            price < ef200,
            adx_val > ADX_TREND_THRESHOLD,
            slope < 0,
            vol_val > VOLUME_TREND_MULTIPLIER * vol_avg,
        )

        if all(long_conditions):
            stop = price - ATR_STOP_MULTIPLIER * atr_val
            return {"side": "long", "stop": stop, "confidence": 0.7}

        if all(short_conditions):
            stop = price + ATR_STOP_MULTIPLIER * atr_val
            return {"side": "short", "stop": stop, "confidence": 0.7}

        return None
