"""
Walk-forward backtesting engine.

Simulates trading fees (0.04 % maker / 0.06 % taker) and slippage (0.05 %).
Iterates through historical OHLCV data candle-by-candle.
Stores results in the SQLite database via the database module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtester.metrics import calculate_metrics
from config.settings import ACCOUNT_SIZE, RISK_PER_TRADE

logger = logging.getLogger(__name__)

MAKER_FEE: float = 0.0004   # 0.04 %
TAKER_FEE: float = 0.0006   # 0.06 %
SLIPPAGE: float = 0.0005    # 0.05 %


class BacktestEngine:
    """Candle-by-candle walk-forward backtesting engine."""

    def __init__(
        self,
        strategy,
        initial_capital: float = ACCOUNT_SIZE,
        fee: float = TAKER_FEE,
        slippage: float = SLIPPAGE,
    ) -> None:
        """
        Parameters
        ----------
        strategy:
            A :class:`~strategies.base_strategy.BaseStrategy` instance.
        initial_capital:
            Starting equity in USDT.
        fee:
            Per-side trading fee as a fraction.
        slippage:
            One-way price slippage as a fraction.
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee = fee
        self.slippage = slippage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run a full backtest over *df* and return metrics + trade log.

        Parameters
        ----------
        df:
            Full OHLCV DataFrame (must include 'open', 'high', 'low',
            'close', 'volume').

        Returns
        -------
        dict
            Combined metrics dict plus a ``'trades'`` list and
            ``'equity_curve'`` list.
        """
        equity = self.initial_capital
        equity_curve: List[float] = [equity]
        trades: List[Dict[str, Any]] = []

        open_trade: Optional[Dict[str, Any]] = None
        warmup = 210  # ensure all indicators are seeded

        for i in range(warmup, len(df)):
            candle = df.iloc[i]
            window = df.iloc[: i + 1]

            # --- Manage open trade ---
            if open_trade is not None:
                equity, open_trade, closed = self._update_open_trade(
                    open_trade, candle, equity
                )
                if closed:
                    closed["duration_bars"] = i - closed.get("entry_bar", i)
                    trades.append(closed)
                    open_trade = None

            # --- Signal generation (only when flat) ---
            if open_trade is None:
                try:
                    signal = self.strategy.generate_signal(window)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Signal error at bar %d: %s", i, exc)
                    signal = None

                if signal:
                    open_trade = self._open_trade(signal, candle, equity, i)

            equity_curve.append(equity)

        # --- Force-close any remaining position ---
        if open_trade is not None:
            last = df.iloc[-1]
            exit_price = last["close"]
            pnl = self._calc_pnl(open_trade, exit_price)
            equity += pnl
            open_trade["pnl"] = pnl
            open_trade["duration_bars"] = len(df) - 1 - open_trade["entry_bar"]
            trades.append(open_trade)
            equity_curve[-1] = equity

        metrics = calculate_metrics(equity_curve, trades)
        metrics["trades"] = trades
        metrics["equity_curve"] = equity_curve
        return metrics

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.70,
        n_splits: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run N walk-forward splits and return a list of out-of-sample results.

        Parameters
        ----------
        df:
            Full historical DataFrame.
        train_pct:
            Fraction of each window used for in-sample optimisation.
        n_splits:
            Number of walk-forward folds.
        """
        results = []
        window_size = len(df) // n_splits
        for i in range(n_splits):
            start = i * window_size
            end = start + window_size if i < n_splits - 1 else len(df)
            fold_df = df.iloc[start:end].copy()
            split = int(len(fold_df) * train_pct)
            test_df = fold_df.iloc[split:].copy()
            logger.info(
                "Walk-forward fold %d/%d — test bars: %d", i + 1, n_splits, len(test_df)
            )
            result = self.run(test_df)
            result["fold"] = i + 1
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_trade(
        self, signal: dict, candle: pd.Series, equity: float, bar_idx: int
    ) -> Dict[str, Any]:
        side = signal["side"]
        slippage_factor = 1 + self.slippage if side == "long" else 1 - self.slippage
        entry_price = candle["close"] * slippage_factor
        stop = signal["stop"]
        stop_distance = abs(entry_price - stop)
        risk_amount = equity * RISK_PER_TRADE
        quantity = risk_amount / stop_distance if stop_distance > 0 else 0.0

        # Deduct entry fee
        entry_fee = quantity * entry_price * self.fee

        return {
            "side": side,
            "entry_price": entry_price,
            "stop": stop,
            "quantity": quantity,
            "entry_fee": entry_fee,
            "entry_bar": bar_idx,
            "confidence": signal.get("confidence", 0.5),
        }

    def _update_open_trade(
        self,
        trade: Dict[str, Any],
        candle: pd.Series,
        equity: float,
    ) -> Tuple[float, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Check for stop hit; return (equity, open_trade, closed_trade)."""
        side = trade["side"]
        stop = trade["stop"]
        low = candle["low"]
        high = candle["high"]

        hit = (side == "long" and low <= stop) or (side == "short" and high >= stop)

        if hit:
            exit_price = stop * (1 - self.slippage if side == "long" else 1 + self.slippage)
            pnl = self._calc_pnl(trade, exit_price)
            equity += pnl
            closed = dict(trade)
            closed["pnl"] = pnl
            closed["exit_price"] = exit_price
            # duration_bars will be patched in the main loop
            closed["duration_bars"] = 0
            return equity, None, closed

        return equity, trade, None

    @staticmethod
    def _calc_pnl(trade: Dict[str, Any], exit_price: float) -> float:
        qty = trade["quantity"]
        entry = trade["entry_price"]
        fee = trade.get("entry_fee", 0.0)
        exit_fee = qty * exit_price * TAKER_FEE
        if trade["side"] == "long":
            return qty * (exit_price - entry) - fee - exit_fee
        return qty * (entry - exit_price) - fee - exit_fee
