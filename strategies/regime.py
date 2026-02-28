"""
Market Regime Classifier.

Uses ADX + ATR expansion ratio + volume to classify market state:

  TREND      → ADX(14) > 25  AND  |EMA20 − EMA50| / EMA50 > 0.4 %
  CHOP       → ADX(14) < 20  AND  ATR_ratio < 1.0
  EXPANSION  → ATR_ratio > 1.3  AND  Volume > 1.5 × avg
  UNCERTAIN  → everything else (no trading)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    ADX_CHOP_THRESHOLD,
    ADX_PERIOD,
    ADX_TREND_THRESHOLD,
    ATR_EXPANSION_RATIO,
    ATR_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    VOLUME_SPIKE_MULTIPLIER,
)

logger = logging.getLogger(__name__)

REGIME_TREND = "TREND"
REGIME_CHOP = "CHOP"
REGIME_EXPANSION = "EXPANSION"
REGIME_UNCERTAIN = "UNCERTAIN"


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


class RegimeClassifier:
    """Classify the current market regime from a candle DataFrame."""

    def classify(self, df: pd.DataFrame) -> str:
        """Return one of ``TREND``, ``CHOP``, ``EXPANSION``, ``UNCERTAIN``."""
        required = max(EMA_SLOW, 50) + 5
        if len(df) < required:
            logger.warning("Insufficient candles (%d) for regime classification", len(df))
            return REGIME_UNCERTAIN

        close = df["close"]
        volume = df["volume"]

        ema_fast = _ema(close, EMA_FAST)
        ema_slow = _ema(close, EMA_SLOW)
        atr14 = _atr(df, ATR_PERIOD)
        atr50 = _atr(df, 50)
        adx = _adx(df, ADX_PERIOD)
        vol_ma = volume.rolling(20).mean()

        adx_val = adx.iloc[-1]
        ef = ema_fast.iloc[-1]
        es = ema_slow.iloc[-1]
        atr_ratio = atr14.iloc[-1] / (atr50.iloc[-1] if atr50.iloc[-1] != 0 else 1e-8)
        vol_val = volume.iloc[-1]
        vol_avg = vol_ma.iloc[-1]

        ema_spread = abs(ef - es) / es if es != 0 else 0.0

        if adx_val > ADX_TREND_THRESHOLD and ema_spread > 0.004:
            return REGIME_TREND

        if adx_val < ADX_CHOP_THRESHOLD and atr_ratio < 1.0:
            return REGIME_CHOP

        if atr_ratio > ATR_EXPANSION_RATIO and vol_val > VOLUME_SPIKE_MULTIPLIER * vol_avg:
            return REGIME_EXPANSION

        return REGIME_UNCERTAIN
