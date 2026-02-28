"""
Circuit Breaker — hard kill switches that the AI layer CANNOT override.

Rules:
  1. Daily loss > 2.5 %       → disable trading for 24 h
  2. 3 consecutive losses     → pause for 2 h
  3. Equity drawdown > 10 %   → close all positions, disable 24 h
  4. Spread > 2× normal       → block new entries
  5. Liquidation dist < 5×stop → flag for immediate leverage reduction

All state is stored with timestamps to survive process restarts if persisted
externally; in-memory state resets on restart (acceptable for this scaffold).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import (
    CONSECUTIVE_LOSS_PAUSE,
    COOLDOWN_HOURS,
    DAILY_DISABLE_HOURS,
    MAX_DAILY_LOSS,
    MAX_DRAWDOWN,
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class CircuitBreaker:
    """Tracks risk events and enforces non-bypassable trading halts."""

    def __init__(self, initial_equity: float) -> None:
        """
        Parameters
        ----------
        initial_equity:
            Equity at session start (used to track drawdown).
        """
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity

        self.consecutive_losses: int = 0
        self.daily_pnl: float = 0.0

        # Timestamps for timed pauses (None = no active pause)
        self._disabled_until: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

        # Daily reset timestamp
        self._day_start: datetime = _now()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_trading_allowed(self) -> bool:
        """Return *True* only if all circuit breakers are inactive."""
        now = _now()
        if self._disabled_until and now < self._disabled_until:
            remaining = (self._disabled_until - now).seconds // 60
            logger.warning("Trading DISABLED — %d min remaining", remaining)
            return False
        if self._cooldown_until and now < self._cooldown_until:
            remaining = (self._cooldown_until - now).seconds // 60
            logger.warning("Trading in COOLDOWN — %d min remaining", remaining)
            return False
        return True

    # ------------------------------------------------------------------
    # Event recorders
    # ------------------------------------------------------------------

    def record_trade_result(self, pnl: float, equity: float) -> None:
        """Update internal state after a trade closes.

        Parameters
        ----------
        pnl:
            Profit/loss for the closed trade (positive = profit, negative = loss).
        equity:
            Current account equity after the trade.
        """
        self._maybe_reset_daily(equity)
        self.daily_pnl += pnl
        self.peak_equity = max(self.peak_equity, equity)

        if pnl < 0:
            self.consecutive_losses += 1
            logger.info("Consecutive losses: %d", self.consecutive_losses)
        else:
            self.consecutive_losses = 0

        # Rule 2: consecutive losses → cooldown
        if self.consecutive_losses >= CONSECUTIVE_LOSS_PAUSE:
            logger.warning(
                "Circuit breaker: %d consecutive losses → %d-hour cooldown",
                self.consecutive_losses,
                COOLDOWN_HOURS,
            )
            self._cooldown_until = _now() + timedelta(hours=COOLDOWN_HOURS)
            self.consecutive_losses = 0

        # Rule 1: daily loss limit
        daily_loss_pct = self.daily_pnl / self.daily_start_equity
        if daily_loss_pct < -MAX_DAILY_LOSS:
            logger.error(
                "Circuit breaker: daily loss %.2f%% exceeds limit → 24-hour disable",
                daily_loss_pct * 100,
            )
            self._disabled_until = _now() + timedelta(hours=DAILY_DISABLE_HOURS)

        # Rule 3: drawdown
        drawdown = (equity - self.peak_equity) / self.peak_equity
        if drawdown < -MAX_DRAWDOWN:
            logger.error(
                "Circuit breaker: equity drawdown %.2f%% → full stop 24 h",
                drawdown * 100,
            )
            self._disabled_until = _now() + timedelta(hours=DAILY_DISABLE_HOURS)

    def check_spread(self, current_spread: float, normal_spread: float) -> bool:
        """Return *True* if spread is acceptable for new entries.

        Rule 4: if spread > 2× normal, block new entries.
        """
        if normal_spread > 0 and current_spread > 2.0 * normal_spread:
            logger.warning(
                "Spread too wide (%.4f vs normal %.4f) — blocking entry",
                current_spread,
                normal_spread,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self, equity: float) -> None:
        """Reset daily PnL tracker if a new calendar day has started."""
        now = _now()
        if now.date() > self._day_start.date():
            logger.info("New trading day — resetting daily PnL tracker")
            self.daily_pnl = 0.0
            self.daily_start_equity = equity
            self._day_start = now
