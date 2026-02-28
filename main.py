"""
Main 24/7 trading loop for the AI-assisted quantitative intraday trading engine.

Architecture:
    Market Data → Regime Detector → Strategy Stack → Signal Confidence
    → Risk Engine → Portfolio Correlation Filter → Execution (testnet)
    → Monitoring & Kill Switch

SAFETY DISCLAIMER:
    This system is designed for TESTNET use ONLY.
    No real funds should be used without extensive review and testing.
    Past performance of backtests does NOT guarantee future results.
    Crypto trading carries significant risk of total capital loss.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Dict, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables FIRST (before any config imports that might
# read env vars at import time)
# ---------------------------------------------------------------------------
load_dotenv()

from ai_layer.local_llm import LocalLLM
from config.settings import ACCOUNT_SIZE, DEFAULT_TIMEFRAME
from config.symbols import TRADING_SYMBOLS
from database.db import DatabaseManager
from execution.executor import Executor
from market_data.data_loader import DataLoader
from portfolio_engine.allocator import DynamicAllocator
from portfolio_engine.portfolio import Portfolio
from risk_engine.circuit_breaker import CircuitBreaker
from risk_engine.risk_manager import RiskManager
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.regime import (
    REGIME_CHOP,
    REGIME_EXPANSION,
    REGIME_TREND,
    RegimeClassifier,
)
from strategies.trend import TrendStrategy

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_engine.log"),
    ],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_running = True


def _handle_shutdown(signum, frame) -> None:  # noqa: ANN001
    global _running
    logger.warning("Shutdown signal received (%s) — stopping after current cycle", signum)
    _running = False


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

# ---------------------------------------------------------------------------
# Timeframe → sleep duration map (seconds)
# ---------------------------------------------------------------------------
TIMEFRAME_SLEEP: Dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}


def _candle_sleep(timeframe: str) -> int:
    return TIMEFRAME_SLEEP.get(timeframe, 300)


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------

_STRATEGIES = {
    REGIME_TREND: TrendStrategy(),
    REGIME_CHOP: MeanReversionStrategy(),
    REGIME_EXPANSION: BreakoutStrategy(),
}


def _select_strategy(regime: str):
    """Return the strategy instance for the given regime, or None."""
    return _STRATEGIES.get(regime)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Initialise all components and run the 24/7 trading loop."""

    logger.info("=" * 70)
    logger.info("RISK DISCLAIMER: This system trades on TESTNET only.")
    logger.info("Crypto trading carries significant risk of total capital loss.")
    logger.info("Never risk funds you cannot afford to lose.")
    logger.info("=" * 70)

    # --- Initialise components ---
    account_size = float(os.environ.get("ACCOUNT_SIZE", ACCOUNT_SIZE))

    try:
        executor = Executor()
    except Exception as exc:
        logger.error("Failed to initialise exchange executor: %s", exc)
        sys.exit(1)

    data_loader = DataLoader(executor.exchange)
    regime_classifier = RegimeClassifier()
    risk_manager = RiskManager(account_size)
    circuit_breaker = CircuitBreaker(account_size)
    portfolio = Portfolio(account_size)
    allocator = DynamicAllocator(TRADING_SYMBOLS)
    local_llm = LocalLLM()
    db = DatabaseManager()

    # Register strategies in database
    for name, strat in [
        ("EMA Trend Momentum", TrendStrategy()),
        ("VWAP Mean Reversion", MeanReversionStrategy()),
        ("Volatility Breakout", BreakoutStrategy()),
    ]:
        db.upsert_strategy(
            name=name,
            strategy_type=strat.get_type(),
            parameters={},
            asset_classes=TRADING_SYMBOLS,
            description=f"Auto-registered: {name}",
        )

    logger.info("Trading engine initialised — watching %s", TRADING_SYMBOLS)

    # --- Main loop ---
    while _running:
        try:
            _run_cycle(
                data_loader=data_loader,
                regime_classifier=regime_classifier,
                risk_manager=risk_manager,
                circuit_breaker=circuit_breaker,
                portfolio=portfolio,
                allocator=allocator,
                local_llm=local_llm,
                executor=executor,
                db=db,
                account_size=account_size,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Unhandled error in trading cycle: %s", exc, exc_info=True)

        sleep_secs = _candle_sleep(DEFAULT_TIMEFRAME)
        logger.info("Sleeping %d s until next candle…", sleep_secs)
        time.sleep(sleep_secs)

    logger.info("Trading engine stopped cleanly.")


def _run_cycle(
    data_loader: DataLoader,
    regime_classifier: RegimeClassifier,
    risk_manager: RiskManager,
    circuit_breaker: CircuitBreaker,
    portfolio: Portfolio,
    allocator: DynamicAllocator,
    local_llm: LocalLLM,
    executor: Executor,
    db: DatabaseManager,
    account_size: float,
) -> None:
    """Execute a single candle cycle across all symbols."""

    # 1. Circuit breaker gate (checked once per cycle)
    if not circuit_breaker.is_trading_allowed:
        return

    # 2. Fetch candles for all symbols
    candles: Dict[str, pd.DataFrame] = {}
    for symbol in TRADING_SYMBOLS:
        df = data_loader.fetch_ohlcv(symbol, DEFAULT_TIMEFRAME)
        if df is not None:
            candles[symbol] = df

    if not candles:
        logger.warning("No candle data available — skipping cycle")
        return

    # 3. Build returns map for correlation / allocation
    returns_map = {
        sym: candles[sym]["close"].pct_change().dropna()
        for sym in candles
    }

    for symbol, df in candles.items():
        _process_symbol(
            symbol=symbol,
            df=df,
            returns_map=returns_map,
            regime_classifier=regime_classifier,
            risk_manager=risk_manager,
            circuit_breaker=circuit_breaker,
            portfolio=portfolio,
            local_llm=local_llm,
            executor=executor,
            db=db,
        )


def _process_symbol(
    symbol: str,
    df,
    returns_map: dict,
    regime_classifier: RegimeClassifier,
    risk_manager: RiskManager,
    circuit_breaker: CircuitBreaker,
    portfolio: Portfolio,
    local_llm: LocalLLM,
    executor: Executor,
    db: DatabaseManager,
) -> None:
    """Evaluate one symbol and execute a trade if all checks pass."""

    # a. Detect regime
    regime = regime_classifier.classify(df)
    logger.info("[%s] Regime: %s", symbol, regime)

    # b. Select strategy
    strategy = _select_strategy(regime)
    if strategy is None:
        logger.info("[%s] No strategy for regime %s — sitting out", symbol, regime)
        return

    # c. Generate signal
    signal = strategy.generate_signal(df)
    if signal is None:
        logger.info("[%s] No signal from %s", symbol, strategy.get_name())
        return

    side = signal["side"]
    stop_price = signal["stop"]
    base_confidence = signal.get("confidence", 0.5)
    entry_price = float(df["close"].iloc[-1])

    logger.info(
        "[%s] Signal: %s @ %.4f  stop=%.4f  conf=%.2f",
        symbol, side, entry_price, stop_price, base_confidence,
    )

    # d. Optional — AI confidence scoring
    try:
        rolling_std = float(df["close"].pct_change().rolling(14).std().iloc[-1])
        market_summary = {
            symbol.split("/")[0]: {
                "regime": regime,
                "signal": 1 if side == "long" else -1,
                "vol": round(rolling_std, 5),
                "rsi": round(float(df["close"].pct_change().mean() * 1000), 1),
            }
        }
        ai_result = local_llm.score_signal(market_summary)
        ai_confidence = ai_result.get("signal_confidence", base_confidence)
        if ai_result.get("anomaly", False):
            logger.warning("[%s] AI detected anomaly — skipping trade", symbol)
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] AI scoring failed: %s — using base confidence", symbol, exc)
        ai_confidence = base_confidence

    # e. Risk sizing
    funding_rate = executor.get_funding_rate(symbol)
    sizing = risk_manager.calculate_position_size(
        entry_price=entry_price,
        stop_price=stop_price,
        side=side,
        funding_rate=funding_rate,
    )

    if not sizing["safe"] or sizing["quantity"] <= 0:
        logger.warning("[%s] Risk check failed — skipping", symbol)
        return

    # f. Circuit breaker — check spread
    ticker = None
    try:
        ticker = executor.exchange.fetch_ticker(symbol)
    except Exception:  # noqa: BLE001
        pass

    if ticker:
        spread = ticker.get("ask", entry_price) - ticker.get("bid", entry_price)
        if not circuit_breaker.check_spread(spread, entry_price * 0.0005):
            logger.warning("[%s] Wide spread — skipping", symbol)
            return

    # g. Portfolio constraints
    allowed, weight_mult = portfolio.can_open_position(
        symbol=symbol,
        side=side,
        risk_amount=sizing["risk_amount"],
        returns_map=returns_map,
    )
    if not allowed:
        logger.info("[%s] Portfolio constraint blocks new position", symbol)
        return

    adjusted_qty = sizing["quantity"] * weight_mult

    # h. Set leverage on exchange
    try:
        executor.exchange.set_leverage(sizing["leverage"], symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Could not set leverage: %s", symbol, exc)

    # i. Execute
    order_side = "buy" if side == "long" else "sell"
    order = executor.market_order(symbol, order_side, adjusted_qty)
    if order is None:
        logger.error("[%s] Order placement failed", symbol)
        return

    # j. Place stop-loss
    stop_side = "sell" if side == "long" else "buy"
    executor.stop_loss_order(symbol, stop_side, adjusted_qty, stop_price)

    # k. Track position in portfolio
    portfolio.add_position(
        symbol,
        {
            "side": side,
            "quantity": adjusted_qty,
            "entry": entry_price,
            "stop": stop_price,
            "risk_amount": sizing["risk_amount"],
        },
    )

    # l. Log to database
    strategy_id = db.get_strategy_id(strategy.get_name()) or -1
    db.log_trade(
        strategy_id=strategy_id,
        asset_symbol=symbol,
        side=side,
        quantity=adjusted_qty,
        price=entry_price,
        order_type="market",
        notes=f"regime={regime} conf={ai_confidence:.2f} lev={sizing['leverage']}",
    )

    logger.info(
        "[%s] TRADE OPENED — %s qty=%.6f @ %.4f  lev=%d  conf=%.2f",
        symbol, side.upper(), adjusted_qty, entry_price,
        sizing["leverage"], ai_confidence,
    )


if __name__ == "__main__":
    main()
