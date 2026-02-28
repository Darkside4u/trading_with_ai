"""
Global configuration constants for the trading engine.
All system parameters are centralized here to avoid magic numbers scattered in code.
"""

# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
RISK_PER_TRADE: float = 0.005        # 0.5% of equity risked per trade
MAX_DAILY_LOSS: float = 0.025        # 2.5% daily max loss → trigger daily disable
MAX_PORTFOLIO_LEVERAGE: float = 3.0  # 3× total portfolio leverage cap
MAX_PER_ASSET_EXPOSURE: float = 0.20 # 20% of capital per single asset
MAX_CONCURRENT_RISK: float = 0.015   # 1.5% total open risk across all positions
MAX_DRAWDOWN: float = 0.10           # 10% equity drawdown → full stop + 24 h disable
CONSECUTIVE_LOSS_PAUSE: int = 3      # N consecutive losses → cooldown
COOLDOWN_HOURS: int = 2              # Hours to pause after consecutive losses
DAILY_DISABLE_HOURS: int = 24        # Hours to disable after daily loss / drawdown

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
EMA_FAST: int = 20
EMA_SLOW: int = 50
EMA_FILTER: int = 200

ADX_PERIOD: int = 14
ADX_TREND_THRESHOLD: float = 25.0
ADX_CHOP_THRESHOLD: float = 20.0

ATR_PERIOD: int = 14
ATR_EXPANSION_RATIO: float = 1.3
ATR_STOP_MULTIPLIER: float = 1.2

RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0

VWAP_DEVIATION_THRESHOLD: float = -0.008  # −0.8% deviation for mean-reversion long

BREAKOUT_LOOKBACK: int = 20
VOLUME_SPIKE_MULTIPLIER: float = 1.5    # Volume must exceed N × average for breakout
VOLUME_TREND_MULTIPLIER: float = 1.2    # Volume must exceed N × average for trend

# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------
MAX_LEVERAGE: int = 15
LEVERAGE_STOP_PRODUCT_LIMIT: float = 0.05  # leverage × stop_pct ≤ 5 %

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
DEFAULT_TIMEFRAME: str = "5m"
CANDLE_LIMIT: int = 500
ACCOUNT_SIZE: float = 5000.0  # Default simulated account size in USDT

# ---------------------------------------------------------------------------
# AI Layer
# ---------------------------------------------------------------------------
OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "llama3.1:8b"
NVIDIA_API_URL: str = "https://integrate.api.nvidia.com/v1"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = "trading_strategies.db"
