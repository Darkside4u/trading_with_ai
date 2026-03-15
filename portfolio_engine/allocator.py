"""
Dynamic weight allocator using risk parity + Sharpe-ratio blending.

Risk parity:
    Vol_i  = StdDev(Returns_i, 20 periods)
    Weight_i = (1 / Vol_i) / Σ(1 / Vol_j)

Sharpe-adjusted update (optional, replaces risk parity when Sharpe history available):
    sharpe_clipped = max(sharpe_i, 0)
    Weight_i = sharpe_clipped_i / Σ(sharpe_clipped_j)

Falls back to equal weights when there is insufficient history.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.symbols import ASSET_WEIGHTS_DEFAULT, TRADING_SYMBOLS

logger = logging.getLogger(__name__)

_VOLATILITY_WINDOW: int = 20


class DynamicAllocator:
    """Allocates capital across assets using risk parity or Sharpe weighting."""

    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self.symbols = symbols or TRADING_SYMBOLS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def risk_parity_weights(
        self, returns_map: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """Compute risk-parity weights from recent return series.

        Parameters
        ----------
        returns_map:
            ``{symbol: pd.Series of returns}``

        Returns
        -------
        dict
            ``{symbol: weight}`` summing to 1.0.
        """
        inv_vols: Dict[str, float] = {}
        missing_data = False
        for sym in self.symbols:
            series = returns_map.get(sym)
            if series is None or len(series) < _VOLATILITY_WINDOW:
                logger.warning("Insufficient return history for %s — using default", sym)
                missing_data = True
                break
            vol = float(series.tail(_VOLATILITY_WINDOW).std())
            inv_vols[sym] = 1.0 / vol if vol > 0 else 0.0

        # Fall back to default weights if any asset lacks data
        if missing_data:
            return dict(ASSET_WEIGHTS_DEFAULT)

        total_inv = sum(inv_vols.values())
        if total_inv == 0:
            return self._equal_weights()

        return {sym: inv_vols[sym] / total_inv for sym in self.symbols}

    def sharpe_weights(
        self, sharpe_map: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute weights proportional to clipped Sharpe ratios.

        Parameters
        ----------
        sharpe_map:
            ``{symbol: sharpe_ratio}`` — negative Sharpe assets receive 0 weight.

        Returns
        -------
        dict
            ``{symbol: weight}`` summing to 1.0.
        """
        clipped = {sym: max(sharpe_map.get(sym, 0.0), 0.0) for sym in self.symbols}
        total = sum(clipped.values())
        if total == 0:
            logger.warning("All Sharpe ratios ≤ 0 — falling back to equal weights")
            return self._equal_weights()
        return {sym: clipped[sym] / total for sym in self.symbols}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _equal_weights(self) -> Dict[str, float]:
        n = len(self.symbols)
        return {sym: 1.0 / n for sym in self.symbols}
