"""
Market data loader: OHLCV fetcher, VWAP calculator, in-memory cache.

Uses ccxt to connect to Binance Futures (testnet or live).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import CANDLE_LIMIT, DEFAULT_TIMEFRAME

logger = logging.getLogger(__name__)

# Simple in-memory cache: (symbol, timeframe) → (timestamp, DataFrame)
_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
CACHE_TTL_SECONDS: float = 60.0  # re-fetch after 60 s


class DataLoader:
    """Fetches and caches OHLCV data from Binance Futures via ccxt."""

    def __init__(self, exchange) -> None:
        """
        Parameters
        ----------
        exchange:
            An initialised ccxt exchange instance (sandbox mode already set).
        """
        self.exchange = exchange

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = CANDLE_LIMIT,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles for *symbol* and return an enriched DataFrame.

        Columns returned: ``open``, ``high``, ``low``, ``close``, ``volume``,
        ``typical_price``, ``vwap``.

        Parameters
        ----------
        symbol:
            E.g. ``"BTC/USDT"``.
        timeframe:
            Candle timeframe such as ``"5m"``, ``"15m"``, ``"1h"``.
        limit:
            Number of candles to fetch.
        use_cache:
            When *True* cached data younger than :data:`CACHE_TTL_SECONDS`
            is returned without hitting the exchange.
        """
        cache_key = (symbol, timeframe)
        if use_cache:
            cached = _cache.get(cache_key)
            if cached is not None:
                ts, df = cached
                if time.time() - ts < CACHE_TTL_SECONDS:
                    logger.debug("Cache hit for %s %s", symbol, timeframe)
                    return df

        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch OHLCV for %s %s: %s", symbol, timeframe, exc)
            return None

        if not raw:
            logger.warning("Empty OHLCV response for %s %s", symbol, timeframe)
            return None

        df = self._parse_ohlcv(raw)
        df = self._enrich(df)
        _cache[cache_key] = (time.time(), df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ohlcv(raw: list) -> pd.DataFrame:
        """Convert raw ccxt OHLCV list to a typed DataFrame."""
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``typical_price`` and cumulative ``vwap`` columns."""
        df = df.copy()
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

        # VWAP = Σ(TypicalPrice × Volume) / Σ(Volume)  (session-cumulative)
        cum_tp_vol = (df["typical_price"] * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        df["vwap"] = cum_tp_vol / cum_vol
        return df
