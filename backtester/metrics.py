"""
Performance metrics for backtesting.

Calculated metrics:
  - Total return %
  - Annualised return %
  - Sharpe ratio
  - Sortino ratio
  - Max drawdown %
  - Win rate %
  - Profit factor (gross profit / gross loss)
  - Calmar ratio (annualised return / |max drawdown|)
  - Volatility %
  - Total trades
  - Average trade duration
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Number of 5-minute candles in a trading year (365 days × 24 h × 12)
PERIODS_PER_YEAR_5M: int = 365 * 24 * 12
RISK_FREE_RATE: float = 0.0


def calculate_metrics(
    equity_curve: List[float],
    trades: List[Dict[str, Any]],
    periods_per_year: int = PERIODS_PER_YEAR_5M,
) -> Dict[str, Any]:
    """Compute a comprehensive set of backtest performance metrics.

    Parameters
    ----------
    equity_curve:
        List of equity values sampled every candle.
    trades:
        List of closed trade dicts with at least keys:
        ``pnl`` (float), ``duration_bars`` (int).
    periods_per_year:
        Annualisation factor matching the candle timeframe.

    Returns
    -------
    dict
        Keyed metric results.
    """
    if not equity_curve or len(equity_curve) < 2:
        return _empty_metrics()

    equity = np.array(equity_curve, dtype=float)
    returns = np.diff(equity) / equity[:-1]

    # --- Basic ---
    initial = equity[0]
    final = equity[-1]
    total_return_pct = (final - initial) / initial * 100.0

    # --- Annualised return ---
    n_periods = len(equity) - 1
    annualised_return = (
        ((final / initial) ** (periods_per_year / n_periods) - 1) * 100.0
        if n_periods > 0
        else 0.0
    )

    # --- Volatility ---
    vol_pct = float(np.std(returns, ddof=1) * math.sqrt(periods_per_year) * 100)

    # --- Sharpe ---
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    sharpe = (
        (mean_r - RISK_FREE_RATE / periods_per_year) / std_r * math.sqrt(periods_per_year)
        if std_r > 0
        else 0.0
    )

    # --- Sortino ---
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        (mean_r - RISK_FREE_RATE / periods_per_year) / downside_std * math.sqrt(periods_per_year)
        if downside_std > 0
        else 0.0
    )

    # --- Max drawdown ---
    roll_max = np.maximum.accumulate(equity)
    drawdowns = (equity - roll_max) / roll_max
    max_drawdown_pct = float(np.min(drawdowns) * 100)

    # --- Calmar ---
    calmar = (
        annualised_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0.0
    )

    # --- Trade statistics ---
    total_trades = len(trades)
    if total_trades == 0:
        win_rate_pct = 0.0
        profit_factor = 0.0
        avg_duration = "N/A"
    else:
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        win_rate_pct = len(wins) / total_trades * 100.0

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        durations = [t.get("duration_bars", 0) for t in trades]
        avg_bars = float(np.mean(durations)) if durations else 0.0
        avg_duration = f"{avg_bars:.1f} bars"

    return {
        "total_return_pct": round(total_return_pct, 4),
        "annualized_return_pct": round(annualised_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "win_rate_pct": round(win_rate_pct, 4),
        "profit_factor": round(profit_factor, 4),
        "calmar_ratio": round(calmar, 4),
        "volatility_pct": round(vol_pct, 4),
        "total_trades": total_trades,
        "avg_trade_duration": avg_duration,
        "initial_capital": round(initial, 2),
        "final_capital": round(final, 2),
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "calmar_ratio": 0.0,
        "volatility_pct": 0.0,
        "total_trades": 0,
        "avg_trade_duration": "N/A",
        "initial_capital": 0.0,
        "final_capital": 0.0,
    }
