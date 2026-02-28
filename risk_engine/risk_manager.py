"""
Risk Manager: position sizing, dynamic leverage, liquidation-safety checks.

Position sizing formula:
    Size = (Equity × RISK_PER_TRADE) / StopDistance

Dynamic leverage:
    Leverage = TargetRisk / AssetVolatility
    Clamped so that: Leverage × StopPct ≤ LEVERAGE_STOP_PRODUCT_LIMIT
    And capped at MAX_LEVERAGE.

Liquidation buffer:
    liquidation_distance must be > 5 × stop_distance.

Funding rate filter:
    If abs(funding_rate) > 0.05 % → halve position size.
"""
from __future__ import annotations

import logging
from typing import Optional

from config.settings import (
    LEVERAGE_STOP_PRODUCT_LIMIT,
    MAX_LEVERAGE,
    MAX_PER_ASSET_EXPOSURE,
    RISK_PER_TRADE,
)

logger = logging.getLogger(__name__)

FUNDING_RATE_THRESHOLD: float = 0.0005  # 0.05 %
LIQUIDATION_BUFFER_FACTOR: float = 5.0   # liq_dist must be > 5 × stop_dist


class RiskManager:
    """Calculates safe position sizes and leverage for a trade."""

    def __init__(self, account_size: float) -> None:
        """
        Parameters
        ----------
        account_size:
            Current equity in USDT.
        """
        self.account_size = account_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        side: str,
        asset_volatility: Optional[float] = None,
        funding_rate: Optional[float] = None,
    ) -> dict:
        """Compute safe position size, leverage, and notional value.

        Parameters
        ----------
        entry_price:
            Anticipated entry price.
        stop_price:
            Hard stop-loss price.
        side:
            ``'long'`` or ``'short'``.
        asset_volatility:
            Annualised or per-period volatility used for leverage calculation.
            If *None*, leverage defaults to 1.
        funding_rate:
            Current perpetual funding rate.  If ``abs(rate) > 0.05 %`` the
            position size is halved.

        Returns
        -------
        dict
            ``{'quantity': float, 'leverage': int, 'notional': float,
               'risk_amount': float, 'safe': bool}``
        """
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            logger.warning("Zero stop distance — refusing to size position")
            return self._zero_result()

        stop_pct = stop_distance / entry_price

        # --- Risk-based position size (contracts = USDT units at 1× leverage) ---
        risk_amount = self.account_size * RISK_PER_TRADE
        quantity = risk_amount / stop_distance  # in base-asset units

        # --- Dynamic leverage ---
        leverage = self._calculate_leverage(stop_pct, asset_volatility)

        # --- Funding rate filter ---
        if funding_rate is not None and abs(funding_rate) > FUNDING_RATE_THRESHOLD:
            logger.info(
                "High funding rate (%.4f%%) — halving position size", funding_rate * 100
            )
            quantity *= 0.5

        # --- Max per-asset exposure cap ---
        notional = quantity * entry_price
        max_notional = self.account_size * MAX_PER_ASSET_EXPOSURE
        if notional > max_notional:
            quantity = max_notional / entry_price
            notional = max_notional

        # --- Liquidation safety check ---
        safe = self._liquidation_safe(entry_price, stop_distance, leverage, side)
        if not safe:
            logger.warning(
                "Liquidation too close to stop (lev=%d, stop_pct=%.3f%%). Reducing leverage.",
                leverage,
                stop_pct * 100,
            )
            leverage = max(1, leverage // 2)
            safe = self._liquidation_safe(entry_price, stop_distance, leverage, side)

        return {
            "quantity": round(quantity, 6),
            "leverage": leverage,
            "notional": round(notional, 2),
            "risk_amount": round(risk_amount, 2),
            "safe": safe,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_leverage(
        self, stop_pct: float, asset_volatility: Optional[float]
    ) -> int:
        """Return an integer leverage satisfying the product-limit constraint."""
        if asset_volatility and asset_volatility > 0:
            raw_leverage = RISK_PER_TRADE / asset_volatility
        else:
            raw_leverage = 1.0

        # Enforce leverage × stop_pct ≤ LEVERAGE_STOP_PRODUCT_LIMIT
        if stop_pct > 0:
            max_lev_from_stop = LEVERAGE_STOP_PRODUCT_LIMIT / stop_pct
            raw_leverage = min(raw_leverage, max_lev_from_stop)

        leverage = int(min(raw_leverage, MAX_LEVERAGE))
        return max(1, leverage)

    @staticmethod
    def _liquidation_safe(
        entry_price: float, stop_distance: float, leverage: int, side: str
    ) -> bool:
        """Return True if liquidation distance > 5 × stop distance."""
        # Approximate liquidation distance for isolated margin
        # Long:  liquidation occurs below entry at entry_price / leverage
        # Short: liquidation occurs above entry at entry_price * leverage / (leverage - 1)
        if side == "long":
            liq_distance = entry_price / leverage
        else:
            liq_distance = (
                entry_price * leverage / (leverage - 1) - entry_price
                if leverage > 1
                else entry_price
            )
        return liq_distance > LIQUIDATION_BUFFER_FACTOR * stop_distance

    @staticmethod
    def _zero_result() -> dict:
        return {
            "quantity": 0.0,
            "leverage": 1,
            "notional": 0.0,
            "risk_amount": 0.0,
            "safe": False,
        }
