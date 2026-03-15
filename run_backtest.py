"""
Automated Backtest Runner + LLM-Driven Strategy Optimizer.

Run:
    python run_backtest.py                     # Backtest all strategies, all symbols
    python run_backtest.py --optimize          # + LLM suggests improved parameters
    python run_backtest.py --symbol BTC/USDT   # Single symbol
    python run_backtest.py --timeframe 15m     # Different timeframe

What it does:
    1. Fetches historical OHLCV from Binance (up to 1500 candles)
    2. Runs walk-forward backtests for each strategy × symbol combo
    3. Saves results to SQLite database
    4. Prints a performance summary table
    5. (--optimize) Sends results to local LLM → suggests parameter tweaks
       → re-runs backtest with tweaked params → compares
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import ccxt
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from backtester.engine import BacktestEngine
from config.settings import ACCOUNT_SIZE, DEFAULT_TIMEFRAME
from config.symbols import TRADING_SYMBOLS
from database.db import DatabaseManager
from strategies.trend import TrendStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest_runner")


# ═══════════════════════════════════════════════════════════════════════════
# Historical data fetcher
# ═══════════════════════════════════════════════════════════════════════════

def fetch_historical(
    symbol: str,
    timeframe: str = "5m",
    limit: int = 1500,
) -> pd.DataFrame:
    """Fetch historical OHLCV from Binance Futures (public, no API key)."""
    exchange = ccxt.binanceusdm({
        "options": {"defaultType": "future", "fetchCurrencies": False},
    })
    logger.info("Fetching %d candles of %s %s ...", limit, symbol, timeframe)

    all_data: list = []
    since = None
    batch = min(limit, 1000)

    while len(all_data) < limit:
        remaining = limit - len(all_data)
        fetch_count = min(batch, remaining)
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_count, since=since)
        if not raw:
            break
        all_data.extend(raw)
        since = raw[-1][0] + 1  # next ms after last candle
        if len(raw) < fetch_count:
            break
        time.sleep(0.2)  # rate limit courtesy

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add VWAP (cumulative)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["typical_price"] = tp
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()

    logger.info("Fetched %d candles for %s (%s → %s)",
                len(df), symbol, df.index[0], df.index[-1])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Run backtests
# ═══════════════════════════════════════════════════════════════════════════

ALL_STRATEGIES = [
    ("EMA Trend Momentum",   TrendStrategy),
    ("VWAP Mean Reversion",  MeanReversionStrategy),
    ("Volatility Breakout",  BreakoutStrategy),
]


def run_all_backtests(
    symbols: List[str],
    timeframe: str,
    candle_limit: int = 1500,
    walk_forward: bool = False,
) -> List[Dict[str, Any]]:
    """Run every strategy on every symbol and return results."""
    db = DatabaseManager()
    results: List[Dict[str, Any]] = []

    for symbol in symbols:
        df = fetch_historical(symbol, timeframe, candle_limit)
        if df is None or len(df) < 250:
            logger.warning("Insufficient data for %s — skipping", symbol)
            continue

        for strat_name, strat_cls in ALL_STRATEGIES:
            strategy = strat_cls()
            engine = BacktestEngine(strategy, initial_capital=ACCOUNT_SIZE)

            logger.info("─" * 60)
            logger.info("Backtesting: %s on %s (%s)", strat_name, symbol, timeframe)

            if walk_forward:
                fold_results = engine.walk_forward(df, train_pct=0.7, n_splits=3)
                for fold in fold_results:
                    fold["strategy"] = strat_name
                    fold["symbol"] = symbol
                    fold["timeframe"] = timeframe
                    results.append(fold)
                    _save_to_db(db, strat_name, symbol, timeframe, fold)
            else:
                result = engine.run(df)
                result["strategy"] = strat_name
                result["symbol"] = symbol
                result["timeframe"] = timeframe
                results.append(result)
                _save_to_db(db, strat_name, symbol, timeframe, result)

            logger.info(
                "  → Return: %.2f%%  Sharpe: %.2f  DD: %.2f%%  Win: %.1f%%  Trades: %d",
                result["total_return_pct"],
                result["sharpe_ratio"],
                result["max_drawdown_pct"],
                result["win_rate_pct"],
                result["total_trades"],
            )

    return results


def _save_to_db(
    db: DatabaseManager,
    strat_name: str,
    symbol: str,
    timeframe: str,
    result: Dict[str, Any],
) -> None:
    """Persist backtest result in SQLite."""
    strat_id = db.get_strategy_id(strat_name)
    if strat_id is None:
        strat_id = db.upsert_strategy(
            name=strat_name,
            strategy_type="auto",
            parameters={},
            asset_classes=[symbol],
        )
    db.save_backtest_result(
        strategy_id=strat_id,
        result={
            "asset_symbol": symbol,
            "timeframe": timeframe,
            "start_date": "",
            "end_date": "",
            "initial_capital": result.get("initial_capital", ACCOUNT_SIZE),
            "final_capital": result.get("final_capital", ACCOUNT_SIZE),
            "total_return_pct": result.get("total_return_pct", 0),
            "annualized_return_pct": result.get("annualized_return_pct", 0),
            "sharpe_ratio": result.get("sharpe_ratio", 0),
            "sortino_ratio": result.get("sortino_ratio", 0),
            "max_drawdown_pct": result.get("max_drawdown_pct", 0),
            "win_rate_pct": result.get("win_rate_pct", 0),
            "profit_factor": result.get("profit_factor", 0),
            "total_trades": result.get("total_trades", 0),
            "avg_trade_duration": result.get("avg_trade_duration", "N/A"),
            "calmar_ratio": result.get("calmar_ratio", 0),
            "volatility_pct": result.get("volatility_pct", 0),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table."""
    print("\n" + "═" * 100)
    print("  BACKTEST RESULTS SUMMARY")
    print("═" * 100)
    print(f"{'Strategy':<25} {'Symbol':<12} {'Return %':>10} {'Sharpe':>8} "
          f"{'Max DD %':>10} {'Win %':>8} {'Trades':>8} {'PF':>8}")
    print("─" * 100)

    for r in results:
        pf = r.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf < 999 else "∞"
        print(f"{r.get('strategy','?'):<25} {r.get('symbol','?'):<12} "
              f"{r.get('total_return_pct',0):>+10.2f} "
              f"{r.get('sharpe_ratio',0):>8.2f} "
              f"{r.get('max_drawdown_pct',0):>10.2f} "
              f"{r.get('win_rate_pct',0):>8.1f} "
              f"{r.get('total_trades',0):>8d} "
              f"{pf_str:>8}")

    print("═" * 100)

    # Best strategy
    if results:
        best = max(results, key=lambda r: r.get("sharpe_ratio", 0))
        print(f"\n  🏆 Best: {best['strategy']} on {best['symbol']} "
              f"(Sharpe {best['sharpe_ratio']:.2f}, Return {best['total_return_pct']:+.2f}%)")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# LLM-Driven Parameter Optimization
# ═══════════════════════════════════════════════════════════════════════════

def llm_optimize(results: List[Dict[str, Any]], max_rounds: int = 3) -> None:
    """Use Ollama local LLM to suggest parameter improvements, re-backtest."""
    import requests as req

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

    # Import current config for baseline
    from config import settings

    current_params = {
        "EMA_FAST": settings.EMA_FAST,
        "EMA_SLOW": settings.EMA_SLOW,
        "EMA_FILTER": settings.EMA_FILTER,
        "ADX_PERIOD": settings.ADX_PERIOD,
        "ADX_TREND_THRESHOLD": settings.ADX_TREND_THRESHOLD,
        "ATR_PERIOD": settings.ATR_PERIOD,
        "ATR_STOP_MULTIPLIER": settings.ATR_STOP_MULTIPLIER,
        "RSI_PERIOD": settings.RSI_PERIOD,
        "RSI_OVERSOLD": settings.RSI_OVERSOLD,
        "RSI_OVERBOUGHT": settings.RSI_OVERBOUGHT,
        "VWAP_DEVIATION_THRESHOLD": settings.VWAP_DEVIATION_THRESHOLD,
        "BREAKOUT_LOOKBACK": settings.BREAKOUT_LOOKBACK,
        "VOLUME_SPIKE_MULTIPLIER": settings.VOLUME_SPIKE_MULTIPLIER,
        "VOLUME_TREND_MULTIPLIER": settings.VOLUME_TREND_MULTIPLIER,
    }

    # Summarize backtest results for the LLM
    summary_rows = []
    for r in results:
        summary_rows.append({
            "strategy": r.get("strategy"),
            "symbol": r.get("symbol"),
            "return_pct": r.get("total_return_pct"),
            "sharpe": r.get("sharpe_ratio"),
            "max_dd_pct": r.get("max_drawdown_pct"),
            "win_rate": r.get("win_rate_pct"),
            "trades": r.get("total_trades"),
            "profit_factor": r.get("profit_factor"),
        })

    prompt = (
        "You are a quantitative trading researcher. Here are the backtest results "
        "from our 3 crypto trading strategies:\n\n"
        f"Results:\n{json.dumps(summary_rows, indent=2)}\n\n"
        f"Current parameters:\n{json.dumps(current_params, indent=2)}\n\n"
        "Analyse the performance and suggest improved parameter values to:\n"
        "1. Increase Sharpe ratio\n"
        "2. Reduce max drawdown\n"
        "3. Improve win rate without sacrificing too many trades\n\n"
        "Respond ONLY with a JSON object containing the parameter names as keys "
        "and their new suggested values. Include ONLY parameters you want to change. "
        "Also include a 'reasoning' key with a brief explanation.\n"
        "Example: {\"EMA_FAST\": 15, \"ADX_TREND_THRESHOLD\": 28, "
        "\"reasoning\": \"Faster EMA for...\"}"
    )

    logger.info("─" * 60)
    logger.info("Sending backtest results to local LLM for optimization...")

    try:
        resp = req.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        logger.info("Tip: Make sure Ollama is running → 'ollama serve'")
        return

    # Parse JSON from LLM response
    try:
        # Extract JSON from potentially wrapped response
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start < 0 or end <= 0:
            raise ValueError("No JSON found in LLM response")
        suggestions = json.loads(raw_text[start:end])
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Could not parse LLM response: %s", exc)
        logger.info("Raw response:\n%s", raw_text[:500])
        return

    reasoning = suggestions.pop("reasoning", "No reasoning provided")
    logger.info("\n📊 LLM Optimization Suggestions:")
    logger.info("  Reasoning: %s", reasoning)

    param_changes = {}
    for key, val in suggestions.items():
        if key in current_params:
            old = current_params[key]
            if val != old:
                param_changes[key] = (old, val)
                logger.info("  %s: %s → %s", key, old, val)

    if not param_changes:
        logger.info("  No parameter changes suggested — current params look good!")
        return

    # Apply changes temporarily and re-backtest
    logger.info("\nRe-running backtests with suggested parameters...")
    for key, (_, new_val) in param_changes.items():
        setattr(settings, key, type(current_params[key])(new_val))

    new_results = run_all_backtests(
        symbols=[r["symbol"] for r in results[:len(TRADING_SYMBOLS)]],
        timeframe=results[0].get("timeframe", DEFAULT_TIMEFRAME),
    )

    # Compare
    print("\n" + "═" * 100)
    print("  OPTIMIZATION COMPARISON: BEFORE vs AFTER")
    print("═" * 100)
    print(f"{'Metric':<30} {'Before':>15} {'After':>15} {'Change':>15}")
    print("─" * 100)

    def _avg(lst, key):
        vals = [r.get(key, 0) for r in lst]
        return sum(vals) / len(vals) if vals else 0

    for metric, label in [
        ("total_return_pct", "Avg Return %"),
        ("sharpe_ratio", "Avg Sharpe"),
        ("max_drawdown_pct", "Avg Max DD %"),
        ("win_rate_pct", "Avg Win Rate %"),
        ("profit_factor", "Avg Profit Factor"),
    ]:
        before = _avg(results, metric)
        after = _avg(new_results, metric)
        diff = after - before
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<28} {before:>15.2f} {after:>15.2f} {sign}{diff:>14.2f}")

    print("═" * 100)

    # Revert parameters (don't persist bad changes)
    for key, (old, _) in param_changes.items():
        setattr(settings, key, old)

    # Check if improvement
    old_sharpe = _avg(results, "sharpe_ratio")
    new_sharpe = _avg(new_results, "sharpe_ratio")
    old_dd = _avg(results, "max_drawdown_pct")
    new_dd = _avg(new_results, "max_drawdown_pct")

    if new_sharpe > old_sharpe and new_dd >= old_dd:
        print("\n  ✅ Suggested parameters IMPROVE performance!")
        print("  To apply permanently, update config/settings.py with:")
        for key, (_, new_val) in param_changes.items():
            print(f"    {key} = {new_val}")
    else:
        print("\n  ❌ Suggested parameters did NOT improve — keeping originals.")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backtest all trading strategies on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py                           # Backtest all strategies × all symbols
  python run_backtest.py --symbol BTC/USDT         # Single symbol
  python run_backtest.py --timeframe 15m           # Different timeframe
  python run_backtest.py --walk-forward            # Walk-forward analysis (3 folds)
  python run_backtest.py --optimize                # + LLM parameter optimization
  python run_backtest.py --candles 1500            # More history
        """,
    )
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol to test (default: all)")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
                        help=f"Candle timeframe (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument("--candles", type=int, default=1500,
                        help="Number of historical candles (default: 1500)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Use walk-forward analysis instead of full backtest")
    parser.add_argument("--optimize", action="store_true",
                        help="Use local LLM to suggest parameter improvements")

    args = parser.parse_args()
    symbols = [args.symbol] if args.symbol else TRADING_SYMBOLS

    print()
    print("═" * 60)
    print("  AI Trading Engine — Automated Backtester")
    print("═" * 60)
    print(f"  Symbols:    {symbols}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Candles:    {args.candles}")
    print(f"  Mode:       {'Walk-Forward' if args.walk_forward else 'Full Backtest'}")
    print(f"  Optimize:   {'Yes (LLM)' if args.optimize else 'No'}")
    print(f"  Capital:    ${ACCOUNT_SIZE:,.0f}")
    print("═" * 60)
    print()

    results = run_all_backtests(
        symbols=symbols,
        timeframe=args.timeframe,
        candle_limit=args.candles,
        walk_forward=args.walk_forward,
    )

    print_summary(results)

    if args.optimize and results:
        llm_optimize(results)

    logger.info("All results saved to database: trading_strategies.db")


if __name__ == "__main__":
    main()
