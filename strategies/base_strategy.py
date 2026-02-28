"""
Abstract base class for all trading strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BaseStrategy(ABC):
    """All concrete strategies must extend this class."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Analyse *df* and return a trade signal or ``None``.

        Returns
        -------
        dict or None
            ``{'side': 'long'|'short', 'stop': float, 'confidence': float}``
            or ``None`` when no trade is warranted.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def get_type(self) -> str:
        """Strategy type: ``'trend'``, ``'mean_reversion'``, or ``'breakout'``."""
