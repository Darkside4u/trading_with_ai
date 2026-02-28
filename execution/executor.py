"""
Order Execution Layer — Binance Futures TESTNET via ccxt.

SAFETY: exchange.set_sandbox_mode(True) is called in __init__.
        All API keys must be provided via environment variables.

Supports:
  - Market orders
  - Limit orders
  - Stop-loss orders (reduce-only)
  - Fetching account balance, positions, and funding rates
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Executor:
    """Wraps ccxt Binance Futures for testnet order execution."""

    def __init__(self) -> None:
        """Initialise ccxt exchange in sandbox (testnet) mode.

        API credentials are read from environment variables:
          - ``BINANCE_TESTNET_KEY``
          - ``BINANCE_TESTNET_SECRET``
        """
        try:
            import ccxt  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("ccxt is required: pip install ccxt") from exc

        api_key = os.environ.get("BINANCE_TESTNET_KEY", "")
        api_secret = os.environ.get("BINANCE_TESTNET_SECRET", "")

        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "future"},
            }
        )
        # MANDATORY: testnet only
        self.exchange.set_sandbox_mode(True)
        logger.info("Executor initialised in SANDBOX (testnet) mode")

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_balance(self) -> Optional[Dict[str, Any]]:
        """Fetch USDT balance from the testnet account."""
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            logger.info(
                "Balance — free: %.2f, total: %.2f",
                usdt.get("free", 0),
                usdt.get("total", 0),
            )
            return balance
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch balance: %s", exc)
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all open positions on the testnet account."""
        try:
            positions = self.exchange.fetch_positions()
            open_pos = [p for p in positions if float(p.get("contracts", 0) or 0) != 0]
            logger.info("Open positions: %d", len(open_pos))
            return open_pos
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch positions: %s", exc)
            return []

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current funding rate for *symbol*."""
        try:
            data = self.exchange.fetch_funding_rate(symbol)
            rate = float(data.get("fundingRate", 0) or 0)
            logger.debug("Funding rate for %s: %.5f", symbol, rate)
            return rate
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch funding rate for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Place a market order.

        Parameters
        ----------
        symbol:
            E.g. ``"BTC/USDT"``.
        side:
            ``'buy'`` or ``'sell'``.
        quantity:
            Base-asset quantity.
        reduce_only:
            When *True* the order will only reduce an existing position.
        """
        params = {"reduceOnly": reduce_only} if reduce_only else {}
        try:
            order = self.exchange.create_order(
                symbol, "market", side, quantity, params=params
            )
            logger.info(
                "Market order placed: %s %s %s qty=%.6f",
                side.upper(), symbol, "REDUCE" if reduce_only else "", quantity,
            )
            return order
        except Exception as exc:  # noqa: BLE001
            logger.error("Market order failed (%s %s): %s", side, symbol, exc)
            return None

    def limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Place a limit order."""
        params = {"reduceOnly": reduce_only} if reduce_only else {}
        try:
            order = self.exchange.create_order(
                symbol, "limit", side, quantity, price, params=params
            )
            logger.info(
                "Limit order placed: %s %s qty=%.6f @ %.4f",
                side.upper(), symbol, quantity, price,
            )
            return order
        except Exception as exc:  # noqa: BLE001
            logger.error("Limit order failed (%s %s @ %.4f): %s", side, symbol, price, exc)
            return None

    def stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> Optional[Dict[str, Any]]:
        """Place a reduce-only stop-market order as stop-loss."""
        params = {
            "stopPrice": stop_price,
            "reduceOnly": True,
        }
        try:
            order = self.exchange.create_order(
                symbol, "stop_market", side, quantity, None, params=params
            )
            logger.info(
                "Stop-loss order placed: %s %s qty=%.6f stop=%.4f",
                side.upper(), symbol, quantity, stop_price,
            )
            return order
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Stop-loss order failed (%s %s stop=%.4f): %s",
                side, symbol, stop_price, exc,
            )
            return None

    def close_position(
        self, symbol: str, side: str, quantity: float
    ) -> Optional[Dict[str, Any]]:
        """Close an open position with a reduce-only market order."""
        exit_side = "sell" if side == "long" else "buy"
        return self.market_order(symbol, exit_side, quantity, reduce_only=True)
