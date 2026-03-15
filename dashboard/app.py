"""
Flask dashboard server for the AI Trading Engine.

Endpoints:
  GET /              → Dashboard HTML page
  GET /api/snapshot  → Engine state + live prices (JSON)
  GET /api/trades    → Last 50 trades from SQLite (JSON)
  GET /api/prices    → Live ticker for all symbols (JSON)

The server is started in a daemon thread by main.py so it does not
block the trading loop.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template

from config.settings import DB_PATH
from config.symbols import TRADING_SYMBOLS
from dashboard.state import DashboardState

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
app.logger.setLevel(logging.WARNING)  # suppress Flask request logs

# ---------------------------------------------------------------------------
# Live price cache — avoid hammering Binance on every browser refresh
# ---------------------------------------------------------------------------
_price_cache: Dict[str, Any] = {}
_price_cache_ts: float = 0.0
_PRICE_CACHE_TTL: float = 4.0  # seconds

_exchange = None


def _get_exchange():
    global _exchange
    if _exchange is None:
        try:
            import ccxt  # type: ignore[import-untyped]
            _exchange = ccxt.binanceusdm({
                "options": {
                    "defaultType": "future",
                    "fetchCurrencies": False,
                },
            })
        except Exception as exc:
            logger.error("Dashboard exchange init failed: %s", exc)
    return _exchange


def _fetch_live_prices() -> Dict[str, Any]:
    """Fetch live tickers for all trading symbols, with caching."""
    global _price_cache, _price_cache_ts
    now = time.time()
    if now - _price_cache_ts < _PRICE_CACHE_TTL and _price_cache:
        return _price_cache

    ex = _get_exchange()
    if ex is None:
        return {}

    result: Dict[str, Any] = {}
    for symbol in TRADING_SYMBOLS:
        try:
            ticker = ex.fetch_ticker(symbol)
            result[symbol] = {
                "last": ticker.get("last") or ticker.get("close"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "change_pct": ticker.get("percentage"),      # 24h % change
                "high_24h": ticker.get("high"),
                "low_24h": ticker.get("low"),
                "volume_24h": ticker.get("quoteVolume"),
            }
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", symbol, exc)
            result[symbol] = {}

    _price_cache = result
    _price_cache_ts = now
    return result


def _fetch_recent_trades(limit: int = 50) -> List[Dict[str, Any]]:
    """Read last *limit* trades from the SQLite trade_log table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT tl.id, tl.asset_symbol, tl.side, tl.quantity, tl.price,
                   tl.timestamp, tl.order_type, tl.status, tl.pnl, tl.notes,
                   s.name AS strategy_name
            FROM trade_log tl
            LEFT JOIN strategies s ON s.id = tl.strategy_id
            ORDER BY tl.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        logger.error("Trade log query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", symbols=TRADING_SYMBOLS)


@app.route("/api/snapshot")
def api_snapshot():
    """Return full engine state + live prices in one payload."""
    state = DashboardState.get().snapshot()
    live_prices = _fetch_live_prices()

    # Enrich open positions with live PnL and estimated TP
    positions_enriched = {}
    for symbol, pos in state["positions"].items():
        ticker = live_prices.get(symbol, {})
        current = ticker.get("last") or pos.get("entry", 0)
        entry = pos.get("entry", 0)
        stop = pos.get("stop", 0)
        side = pos.get("side", "long")

        # Risk / Reward → TP at 2× R
        risk = abs(entry - stop)
        if side == "long":
            tp = entry + 2.0 * risk
            pnl_pct = ((current - entry) / entry * 100) if entry else 0
        else:
            tp = entry - 2.0 * risk
            pnl_pct = ((entry - current) / entry * 100) if entry else 0

        pnl_usd = (pnl_pct / 100) * pos.get("risk_amount", 0) * 20  # approx

        positions_enriched[symbol] = {
            **pos,
            "current_price": round(current, 4) if current else None,
            "tp": round(tp, 4),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2),
        }

    return jsonify({
        **state,
        "positions": positions_enriched,
        "live_prices": live_prices,
    })


@app.route("/api/trades")
def api_trades():
    return jsonify(_fetch_recent_trades(50))


@app.route("/api/prices")
def api_prices():
    return jsonify(_fetch_live_prices())


# ---------------------------------------------------------------------------
# Server launcher (called from main.py)
# ---------------------------------------------------------------------------

def start_dashboard(host: str = "0.0.0.0", port: int = 5050) -> None:
    """Start the Flask server in a background daemon thread."""
    import threading

    def _run():
        logger.info("Dashboard started at http://%s:%d", host, port)
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="dashboard")
    t.start()
    logger.info("Dashboard thread launched → http://localhost:%d", port)
