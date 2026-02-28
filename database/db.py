"""
SQLite database manager.

Tables:
  - strategies
  - backtest_results
  - trade_log

Views:
  - best_strategies  (Sharpe > 1.0, DD > −25%, win_rate > 45%, trades ≥ 30)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

_CREATE_STRATEGIES = """
CREATE TABLE IF NOT EXISTS strategies (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    type         TEXT    NOT NULL,
    parameters   TEXT    NOT NULL,
    asset_classes TEXT   NOT NULL,
    description  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active    BOOLEAN DEFAULT 1
);
"""

_CREATE_BACKTEST_RESULTS = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id          INTEGER NOT NULL REFERENCES strategies(id),
    asset_symbol         TEXT    NOT NULL,
    timeframe            TEXT    NOT NULL,
    start_date           TEXT    NOT NULL,
    end_date             TEXT    NOT NULL,
    initial_capital      REAL    NOT NULL,
    final_capital        REAL    NOT NULL,
    total_return_pct     REAL,
    annualized_return_pct REAL,
    sharpe_ratio         REAL,
    sortino_ratio        REAL,
    max_drawdown_pct     REAL,
    win_rate_pct         REAL,
    profit_factor        REAL,
    total_trades         INTEGER,
    avg_trade_duration   TEXT,
    calmar_ratio         REAL,
    volatility_pct       REAL,
    run_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata             TEXT
);
"""

_CREATE_BEST_STRATEGIES_VIEW = """
CREATE VIEW IF NOT EXISTS best_strategies AS
SELECT
    s.id, s.name, s.type, s.parameters, s.asset_classes,
    br.asset_symbol, br.timeframe,
    br.sharpe_ratio, br.sortino_ratio,
    br.total_return_pct, br.max_drawdown_pct,
    br.win_rate_pct, br.profit_factor,
    br.total_trades, br.run_at
FROM strategies s
JOIN backtest_results br ON s.id = br.strategy_id
WHERE br.sharpe_ratio    >  1.0
  AND br.max_drawdown_pct > -25.0
  AND br.win_rate_pct     >  45.0
  AND br.total_trades     >= 30
ORDER BY br.sharpe_ratio DESC, br.sortino_ratio DESC
LIMIT 20;
"""

_CREATE_TRADE_LOG = """
CREATE TABLE IF NOT EXISTS trade_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    asset_symbol TEXT   NOT NULL,
    side        TEXT    NOT NULL,
    quantity    REAL    NOT NULL,
    price       REAL    NOT NULL,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_type  TEXT,
    status      TEXT    DEFAULT 'FILLED',
    pnl         REAL,
    notes       TEXT
);
"""


class DatabaseManager:
    """SQLite database interface for strategies, backtest results, and trade logs."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and view if they don't exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_STRATEGIES)
            conn.execute(_CREATE_BACKTEST_RESULTS)
            conn.execute(_CREATE_TRADE_LOG)
            # View must be created after both tables
            conn.execute(_CREATE_BEST_STRATEGIES_VIEW)
            conn.commit()
        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def upsert_strategy(
        self,
        name: str,
        strategy_type: str,
        parameters: dict,
        asset_classes: list,
        description: str = "",
    ) -> int:
        """Insert or update a strategy record; return its id."""
        params_json = json.dumps(parameters)
        assets_json = json.dumps(asset_classes)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO strategies (name, type, parameters, asset_classes, description)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    type=excluded.type,
                    parameters=excluded.parameters,
                    asset_classes=excluded.asset_classes,
                    description=excluded.description,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (name, strategy_type, params_json, assets_json, description),
            )
            conn.commit()
            # Retrieve the id
            row = conn.execute(
                "SELECT id FROM strategies WHERE name = ?", (name,)
            ).fetchone()
            return int(row[0]) if row else -1

    def get_strategy_id(self, name: str) -> Optional[int]:
        """Return the id for a strategy by name, or *None*."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM strategies WHERE name = ?", (name,)
            ).fetchone()
            return int(row[0]) if row else None

    # ------------------------------------------------------------------
    # Backtest results
    # ------------------------------------------------------------------

    def save_backtest_result(self, strategy_id: int, result: Dict[str, Any]) -> int:
        """Persist a backtest result dict; return the new record id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO backtest_results (
                    strategy_id, asset_symbol, timeframe,
                    start_date, end_date,
                    initial_capital, final_capital,
                    total_return_pct, annualized_return_pct,
                    sharpe_ratio, sortino_ratio,
                    max_drawdown_pct, win_rate_pct,
                    profit_factor, total_trades,
                    avg_trade_duration, calmar_ratio,
                    volatility_pct, metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    strategy_id,
                    result.get("asset_symbol", "UNKNOWN"),
                    result.get("timeframe", "5m"),
                    result.get("start_date", ""),
                    result.get("end_date", ""),
                    result.get("initial_capital", 0.0),
                    result.get("final_capital", 0.0),
                    result.get("total_return_pct"),
                    result.get("annualized_return_pct"),
                    result.get("sharpe_ratio"),
                    result.get("sortino_ratio"),
                    result.get("max_drawdown_pct"),
                    result.get("win_rate_pct"),
                    result.get("profit_factor"),
                    result.get("total_trades"),
                    result.get("avg_trade_duration"),
                    result.get("calmar_ratio"),
                    result.get("volatility_pct"),
                    json.dumps(result.get("metadata", {})),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # Trade log
    # ------------------------------------------------------------------

    def log_trade(
        self,
        strategy_id: int,
        asset_symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "market",
        status: str = "FILLED",
        pnl: Optional[float] = None,
        notes: str = "",
    ) -> int:
        """Insert a trade log entry; return the new record id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trade_log
                    (strategy_id, asset_symbol, side, quantity, price,
                     order_type, status, pnl, notes)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (strategy_id, asset_symbol, side, quantity, price,
                 order_type, status, pnl, notes),
            )
            conn.commit()
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_best_strategies(self) -> List[Dict[str, Any]]:
        """Return rows from the best_strategies view."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM best_strategies")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
