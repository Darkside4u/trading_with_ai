"""
Portfolio tracker: open positions, correlation guard, exposure control.

Correlation guard:
    If corr(asset_A, asset_B) > 0.8 and both positions are the same direction
    → reduce the weight of the newer position by 50 %.

Max concurrent positions and total exposure are enforced here.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import MAX_CONCURRENT_RISK, MAX_PER_ASSET_EXPOSURE

logger = logging.getLogger(__name__)

CORRELATION_THRESHOLD: float = 0.8
MAX_CONCURRENT_POSITIONS: int = 3  # no more than 3 open positions at once


class Portfolio:
    """Tracks open positions and enforces portfolio-level constraints."""

    def __init__(self, account_size: float) -> None:
        self.account_size = account_size
        # {symbol: {'side': str, 'quantity': float, 'entry': float,
        #           'stop': float, 'risk_amount': float}}
        self._positions: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(self, symbol: str, position: dict) -> None:
        """Record a new open position."""
        self._positions[symbol] = position
        logger.info("Portfolio: opened %s %s", symbol, position.get("side"))

    def remove_position(self, symbol: str) -> Optional[dict]:
        """Remove and return the closed position, or *None* if not found."""
        pos = self._positions.pop(symbol, None)
        if pos:
            logger.info("Portfolio: closed %s", symbol)
        return pos

    def get_position(self, symbol: str) -> Optional[dict]:
        """Return the open position for *symbol*, or *None*."""
        return self._positions.get(symbol)

    @property
    def open_symbols(self) -> List[str]:
        return list(self._positions.keys())

    @property
    def open_count(self) -> int:
        return len(self._positions)

    # ------------------------------------------------------------------
    # Guard checks
    # ------------------------------------------------------------------

    def can_open_position(
        self,
        symbol: str,
        side: str,
        risk_amount: float,
        returns_map: Optional[Dict[str, pd.Series]] = None,
    ) -> Tuple[bool, float]:
        """Determine whether a new position is allowed and return an adjusted weight.

        Parameters
        ----------
        symbol:
            The asset to trade.
        side:
            ``'long'`` or ``'short'``.
        risk_amount:
            Dollar risk for this trade.
        returns_map:
            Recent returns series keyed by symbol, used for correlation calculation.

        Returns
        -------
        (allowed, weight_multiplier)
            *allowed* is *True* if the trade may proceed.
            *weight_multiplier* is 1.0 normally, 0.5 if a correlated position exists.
        """
        # Already open
        if symbol in self._positions:
            logger.info("Position already open for %s", symbol)
            return False, 0.0

        # Concurrent positions cap
        if self.open_count >= MAX_CONCURRENT_POSITIONS:
            logger.warning(
                "Max concurrent positions (%d) reached", MAX_CONCURRENT_POSITIONS
            )
            return False, 0.0

        # Total open risk
        total_risk = sum(p.get("risk_amount", 0) for p in self._positions.values())
        if (total_risk + risk_amount) / self.account_size > MAX_CONCURRENT_RISK:
            logger.warning("Adding trade would exceed MAX_CONCURRENT_RISK")
            return False, 0.0

        # Correlation guard
        weight_multiplier = 1.0
        if returns_map and symbol in returns_map:
            for open_sym, pos in self._positions.items():
                if open_sym not in returns_map:
                    continue
                if pos.get("side") != side:
                    continue  # Different directions don't compound correlation risk
                corr = self._correlation(returns_map[symbol], returns_map[open_sym])
                if corr > CORRELATION_THRESHOLD:
                    logger.info(
                        "High correlation (%.2f) between %s and %s — reducing weight 50%%",
                        corr, symbol, open_sym,
                    )
                    weight_multiplier = 0.5
                    break

        return True, weight_multiplier

    @property
    def total_exposure(self) -> float:
        """Sum of all notional values as a fraction of account size."""
        notional = sum(
            p.get("quantity", 0) * p.get("entry", 0) for p in self._positions.values()
        )
        return notional / self.account_size if self.account_size > 0 else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation(a: pd.Series, b: pd.Series) -> float:
        """Pearson correlation of two return series (last 20 overlapping values)."""
        combined = pd.DataFrame({"a": a, "b": b}).dropna().tail(20)
        if len(combined) < 5:
            return 0.0
        return float(combined["a"].corr(combined["b"]))
