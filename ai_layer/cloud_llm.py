"""
Cloud LLM integration via NVIDIA API (OpenAI-compatible endpoint).

Called WEEKLY (not per-trade) for strategic portfolio review.

SAFETY: All suggestions from the cloud LLM must pass the Python promotion gate:
    Accept ONLY IF: Sharpe_new > Sharpe_old AND Drawdown_new <= Drawdown_old
    The AI can NEVER directly modify the trading engine parameters.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from config.settings import NVIDIA_API_URL

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT: int = 180  # seconds — 253B reasoning model needs more time
CLOUD_MODEL: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"  # NVIDIA Nemotron Ultra 253B reasoning model


class CloudLLM:
    """Interface to the NVIDIA cloud LLM for weekly strategic review."""

    def __init__(
        self,
        api_url: str = NVIDIA_API_URL,
        model: str = CLOUD_MODEL,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self._api_key: str = os.environ.get("NVIDIA_API_KEY", "")
        self._last_call: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def weekly_review(
        self,
        performance_summary: Dict[str, Any],
        current_sharpe: float,
        current_drawdown: float,
    ) -> Dict[str, Any]:
        """Send weekly performance summary and return validated suggestions.

        Parameters
        ----------
        performance_summary:
            E.g.::

                {
                    "week": 12,
                    "equity_change": "+2.3%",
                    "strategies": {"trend": {"sharpe": 1.2, "trades": 45}}
                }
        current_sharpe:
            Current portfolio Sharpe ratio (gate threshold).
        current_drawdown:
            Current max drawdown percentage (gate threshold, negative value).

        Returns
        -------
        dict
            Validated and filtered suggestions, or empty dict if gate fails.
        """
        self._last_call = datetime.now(tz=timezone.utc)
        prompt = self._build_prompt(performance_summary)
        raw = self._call_api(prompt)
        suggestions = self._parse_response(raw)
        return self._apply_promotion_gate(suggestions, current_sharpe, current_drawdown)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, summary: Dict[str, Any]) -> str:
        summary_json = json.dumps(summary, separators=(",", ":"))
        return (
            "You are a quantitative trading strategy advisor.\n"
            f"Weekly performance summary:\n{summary_json}\n\n"
            "Provide strategy weight adjustments and parameter suggestions "
            "as a JSON object with keys:\n"
            "  strategy_weights (dict symbol→float),\n"
            "  parameter_adjustments (dict param_name→new_value),\n"
            "  expected_sharpe (float),\n"
            "  expected_max_drawdown_pct (float, negative),\n"
            "  rationale (str).\n"
            "Keep suggestions conservative and data-driven."
        )

    def _call_api(self, prompt: str) -> str:
        if not self._api_key:
            logger.warning("NVIDIA_API_KEY not set — skipping cloud LLM call")
            return ""

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2048,  # extra tokens for chain-of-thought reasoning
        }
        try:
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return str(content)
        except Exception as exc:  # noqa: BLE001
            logger.error("Cloud LLM request failed: %s", exc)
            return ""

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """Extract JSON suggestions from LLM response.

        Nemotron Ultra (reasoning model) may wrap its chain-of-thought in
        ``<think>...</think>`` tags before emitting the final JSON answer.
        We strip those tags so only the answer portion is parsed.
        """
        if not raw:
            return {}

        # Strip <think>...</think> reasoning block emitted by Nemotron Ultra
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("No JSON in cloud LLM response")
            return {}
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("Cloud LLM JSON parse error: %s", exc)
            return {}

    @staticmethod
    def _apply_promotion_gate(
        suggestions: Dict[str, Any],
        current_sharpe: float,
        current_drawdown: float,
    ) -> Dict[str, Any]:
        """Accept suggestions ONLY IF Sharpe improves AND drawdown does not worsen.

        This is a HARD Python gate — the AI cannot bypass it.
        """
        if not suggestions:
            return {}

        expected_sharpe = float(suggestions.get("expected_sharpe", 0.0))
        expected_dd = float(suggestions.get("expected_max_drawdown_pct", current_drawdown))

        sharpe_improves = expected_sharpe > current_sharpe
        dd_ok = expected_dd >= current_drawdown  # drawdown is negative; less negative = better

        if sharpe_improves and dd_ok:
            logger.info(
                "Cloud LLM suggestions ACCEPTED "
                "(Sharpe %.2f → %.2f, DD %.2f%% → %.2f%%)",
                current_sharpe, expected_sharpe,
                current_drawdown, expected_dd,
            )
            return suggestions

        logger.warning(
            "Cloud LLM suggestions REJECTED by promotion gate "
            "(Sharpe %.2f → %.2f, DD %.2f%% → %.2f%%)",
            current_sharpe, expected_sharpe,
            current_drawdown, expected_dd,
        )
        return {}
