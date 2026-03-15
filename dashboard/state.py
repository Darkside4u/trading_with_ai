"""
Shared in-process state between the trading engine (writer)
and the Flask dashboard server (reader).

Thread-safe singleton — both sides import and call DashboardState.get().
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict


class DashboardState:
    """Singleton that holds live engine data for the dashboard."""

    _instance: "DashboardState | None" = None
    _class_lock: Lock = Lock()

    def __init__(self) -> None:
        self._lock: Lock = Lock()
        self.positions: Dict[str, dict] = {}       # symbol → position dict
        self.regimes: Dict[str, str] = {}          # symbol → regime string
        self.weights: Dict[str, float] = {}        # symbol → alloc weight
        self.circuit_breaker_ok: bool = True
        self.engine_running: bool = True
        self.last_cycle_utc: str = ""
        self.account_size: float = 5000.0

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "DashboardState":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Writer API (called by main.py trading engine)
    # ------------------------------------------------------------------

    def update_positions(self, positions: Dict[str, dict]) -> None:
        with self._lock:
            self.positions = dict(positions)

    def update_regime(self, symbol: str, regime: str) -> None:
        with self._lock:
            self.regimes[symbol] = regime

    def update_weights(self, weights: Dict[str, float]) -> None:
        with self._lock:
            self.weights = dict(weights)

    def update_circuit_breaker(self, ok: bool) -> None:
        with self._lock:
            self.circuit_breaker_ok = ok

    def mark_cycle_complete(self) -> None:
        with self._lock:
            self.last_cycle_utc = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

    def set_account_size(self, size: float) -> None:
        with self._lock:
            self.account_size = size

    def set_engine_running(self, running: bool) -> None:
        with self._lock:
            self.engine_running = running

    # ------------------------------------------------------------------
    # Reader API (called by Flask)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "positions": dict(self.positions),
                "regimes": dict(self.regimes),
                "weights": dict(self.weights),
                "circuit_breaker_ok": self.circuit_breaker_ok,
                "engine_running": self.engine_running,
                "last_cycle_utc": self.last_cycle_utc,
                "account_size": self.account_size,
            }
