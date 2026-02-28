"""
Local LLM integration via Ollama (llama3.1:8b).

Sends a compact market summary to the local model and receives:
  - signal_confidence : float [0, 1]
  - anomaly           : bool
  - position_adjustment: float [-1, 1]  (−1 = reduce, +1 = increase)

IMPORTANT: The AI response NEVER overrides the risk engine.
           It can only adjust confidence within bounds [0, 1].
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import requests

from config.settings import OLLAMA_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE: float = 0.5
REQUEST_TIMEOUT: int = 10  # seconds — keep short to avoid blocking the trading loop


class LocalLLM:
    """Interface to the local Ollama LLM for real-time signal scoring."""

    def __init__(
        self,
        url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
    ) -> None:
        self.url = url
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_signal(self, market_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Send a compressed market summary and return AI-scored confidence.

        Parameters
        ----------
        market_summary:
            E.g.::

                {
                    "BTC": {"regime": "trend", "signal": 1,
                            "vol": 0.018, "rsi": 55}
                }

        Returns
        -------
        dict
            ``{'signal_confidence': float, 'anomaly': bool,
               'position_adjustment': float}``
        """
        prompt = self._build_prompt(market_summary)
        raw = self._call_ollama(prompt)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, summary: Dict[str, Any]) -> str:
        summary_json = json.dumps(summary, separators=(",", ":"))
        return (
            f"You are a crypto trading assistant. Analyse this market data:\n"
            f"{summary_json}\n\n"
            "Respond ONLY with a JSON object with these keys:\n"
            "  signal_confidence (float 0-1),\n"
            "  anomaly (bool),\n"
            "  position_adjustment (float -1 to 1, where -1=reduce, +1=increase).\n"
            "Example: {\"signal_confidence\": 0.72, \"anomaly\": false, "
            "\"position_adjustment\": 0.1}"
        )

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("response", ""))
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama request failed: %s", exc)
            return ""

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """Extract structured fields from the LLM response text."""
        defaults: Dict[str, Any] = {
            "signal_confidence": DEFAULT_CONFIDENCE,
            "anomaly": False,
            "position_adjustment": 0.0,
        }
        if not raw:
            return defaults

        # Try to extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("No JSON found in LLM response — using defaults")
            return defaults

        try:
            parsed = json.loads(raw[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error: %s — using defaults", exc)
            return defaults

        # Validate and clamp values
        confidence = float(parsed.get("signal_confidence", DEFAULT_CONFIDENCE))
        confidence = max(0.0, min(1.0, confidence))

        adj = float(parsed.get("position_adjustment", 0.0))
        adj = max(-1.0, min(1.0, adj))

        return {
            "signal_confidence": confidence,
            "anomaly": bool(parsed.get("anomaly", False)),
            "position_adjustment": adj,
        }
