"""
VWAP Mean-Reversion Strategy.

Long entry (all must be true):
  1. (Price − VWAP) / VWAP < −0.8 %  — price well below VWAP
  2. RSI(14) < 30                     — oversold
  3. Volume declining (selling exhaustion)

Short entry: mirror opposite.

Stop  : 1.0 × ATR(14)
Target: price returns to VWAP
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from config.settings import (
    ATR_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    VWAP_DEVIATION_THRESHOLD,
)
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


class MeanReversionStrategy(BaseStrategy):
    """VWAP mean-reversion strategy."""

    def get_name(self) -> str:
        return "VWAP Mean Reversion"

    def get_type(self) -> str:
        return "mean_reversion"

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Return long/short signal dict or *None*."""
        if len(df) < RSI_PERIOD + 10:
            logger.debug("Insufficient candles (%d) for MeanReversionStrategy", len(df))
            return None

        close = df["close"]
        volume = df["volume"]

        # VWAP must exist (added by DataLoader); fall back to cumulative calc
        if "vwap" in df.columns:
            vwap = df["vwap"]
        else:
            tp = (df["high"] + df["low"] + df["close"]) / 3.0
            vwap = (tp * volume).cumsum() / volume.cumsum()

        rsi = _rsi(close, RSI_PERIOD)
        atr = _atr(df, ATR_PERIOD)

        price = close.iloc[-1]
        vwap_val = vwap.iloc[-1]
        rsi_val = rsi.iloc[-1]
        atr_val = atr.iloc[-1]

        deviation = (price - vwap_val) / vwap_val

        # Volume declining: 3-bar average is falling (more robust than single-bar comparison)
        vol_declining = (
            volume.rolling(3).mean().iloc[-1] < volume.rolling(3).mean().iloc[-3]
        )

        if deviation < VWAP_DEVIATION_THRESHOLD and rsi_val < RSI_OVERSOLD and vol_declining:
            stop = price - atr_val
            return {"side": "long", "stop": stop, "confidence": 0.65}

        if deviation > -VWAP_DEVIATION_THRESHOLD and rsi_val > RSI_OVERBOUGHT and vol_declining:
            stop = price + atr_val
            return {"side": "short", "stop": stop, "confidence": 0.65}

        return None
