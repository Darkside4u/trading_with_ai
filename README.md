# trading_with_ai

> **⚠️ RISK DISCLAIMER**: This software is designed for **TESTNET / PAPER TRADING ONLY**.
> Crypto perpetual futures carry extreme risk of total capital loss due to leverage.
> Never use this system with real funds without extensive independent testing, review by
> qualified financial professionals, and full understanding of the risks involved.
> Past backtesting performance does **not** guarantee future results.
> The authors accept no responsibility for financial losses.

---

## Overview

A production-grade, multi-strategy, multi-asset quantitative intraday trading engine for
crypto perpetual futures, targeting the Binance Futures **testnet**.

The system follows a strict layered safety architecture where **the LLM never directly
executes trades** — all money operations are controlled by deterministic Python logic:

```
Market Data (BTC, ETH, SOL)
        │
        ▼
Regime Detector (ADX + ATR)
        │
        ▼
Strategy Stack (Trend / Mean-Reversion / Breakout)
        │
        ▼
Signal Confidence (optional local LLM scoring)
        │
        ▼
Risk Engine (position sizing, leverage, liquidation buffer)
        │
        ▼
Circuit Breakers (daily loss, drawdown, consecutive losses)
        │
        ▼
Portfolio Correlation Filter (exposure cap, correlation guard)
        │
        ▼
Execution Layer — Binance Futures TESTNET via ccxt
        │
        ▼
SQLite Logging + Monitoring
```

---

## Target Configuration

| Parameter | Value |
|-----------|-------|
| Assets | BTC/USDT, ETH/USDT, SOL/USDT perpetual futures |
| Timeframe | Intraday (5m default; 15m / 1h supported) |
| Leverage | Dynamic, volatility-adaptive (not fixed) |
| Capital simulation | $1k–$10k (testnet) |
| AI | Local 8B (Ollama) for signal scoring; Cloud 400B (NVIDIA) for weekly review |
| Database | SQLite — strategies, backtest results, trade logs |
| Deployment | Ubuntu VM, 24/7 autonomous via systemd |

---

## Project Structure

```
├── config/
│   ├── __init__.py
│   ├── settings.py          # All constants, thresholds, API config
│   └── symbols.py           # Asset definitions & exchange mappings
├── market_data/
│   ├── __init__.py
│   └── data_loader.py       # OHLCV fetcher, VWAP calculator, caching
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py     # Abstract base class
│   ├── trend.py             # EMA Momentum strategy
│   ├── mean_reversion.py    # VWAP Mean Reversion strategy
│   ├── breakout.py          # Volatility Breakout strategy
│   └── regime.py            # Regime classifier (ADX + ATR ratio)
├── risk_engine/
│   ├── __init__.py
│   ├── risk_manager.py      # Position sizing, leverage calculator
│   └── circuit_breaker.py   # Kill switches, drawdown protection
├── portfolio_engine/
│   ├── __init__.py
│   ├── portfolio.py         # Correlation guard, exposure control
│   └── allocator.py         # Risk parity + Sharpe weighting
├── execution/
│   ├── __init__.py
│   └── executor.py          # Testnet order placement via ccxt
├── backtester/
│   ├── __init__.py
│   ├── engine.py            # Walk-forward backtesting engine
│   └── metrics.py           # Sharpe, Sortino, Calmar, profit factor…
├── ai_layer/
│   ├── __init__.py
│   ├── local_llm.py         # Ollama 8B integration
│   └── cloud_llm.py         # NVIDIA API integration (weekly)
├── database/
│   ├── __init__.py
│   └── db.py                # SQLite manager
├── main.py                  # 24/7 trading loop
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Darkside4u/trading_with_ai.git
cd trading_with_ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Testnet Keys

```bash
cp .env.example .env
# Edit .env and fill in your Binance Futures TESTNET credentials
# Get testnet keys at: https://testnet.binancefuture.com/
```

### 3. (Optional) Start Ollama Local LLM

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1:8b
ollama serve
```

### 4. Run

```bash
python main.py
```

---

## Strategies

### EMA Trend Momentum (`strategies/trend.py`)

Follows established price trends using exponential moving averages.

**Long entry** (all conditions must be true):
1. `EMA(20) > EMA(50)` — bullish momentum crossover
2. `Price > EMA(200)` — above long-term trend filter
3. `ADX(14) > 25` — confirmed strong trend (not choppy)
4. `Slope(EMA20, 5 bars) > 0` — momentum accelerating
5. `Volume > 1.2 × SMA(Volume, 20)` — volume confirmation

**Stop-loss**: `Entry − 1.2 × ATR(14)` for longs  
**Take-profit**: 1.5R target

### VWAP Mean Reversion (`strategies/mean_reversion.py`)

Fades extreme deviations from fair value (VWAP).

**Long entry** (all conditions must be true):
1. `(Price − VWAP) / VWAP < −0.8%` — price significantly below VWAP
2. `RSI(14) < 30` — oversold
3. Volume declining — selling exhaustion

**Stop-loss**: `Entry − 1.0 × ATR(14)`  
**Target**: Price returns to VWAP

### Volatility Breakout (`strategies/breakout.py`)

Trades explosive breakouts from 20-bar consolidation ranges.

**Long entry** (all conditions must be true):
1. `Price > max(High, 20 bars)` — breakout above range
2. `ATR(14) / ATR(50) > 1.2` — volatility expanding
3. `Volume > 1.5 × SMA(Volume, 20)` — volume surge confirms

**Stop-loss**: `min(candle low, Entry − 1.0 × ATR)`  
**Target**: `2.0 × ATR(14)` — breakouts run further

### Regime Classifier (`strategies/regime.py`)

```
IF ADX(14) > 25 AND |EMA20 − EMA50| / EMA50 > 0.4%:
    → TREND      (activate EMA Trend strategy)

ELIF ADX(14) < 20 AND ATR_ratio < 1.0:
    → CHOP       (activate VWAP Mean Reversion)

ELIF ATR_ratio > 1.3 AND Volume > 1.5× avg:
    → EXPANSION  (activate Volatility Breakout)

ELSE:
    → UNCERTAIN  (no trading — sit in cash)
```

---

## Risk Management

### Position Sizing

```
Size = (Equity × 0.5%) / StopDistance
```

### Dynamic Leverage

```
Leverage = TargetRisk / AssetVolatility
```
Constrained so that: `Leverage × Stop% ≤ 5%`, capped at 15×.

### Liquidation Buffer

```
liquidation_distance > 5 × stop_distance  (always enforced)
```

### Circuit Breakers (Non-Bypassable)

| Rule | Trigger | Action |
|------|---------|--------|
| Daily loss | > 2.5% | Disable 24 h |
| Consecutive losses | 3 in a row | Pause 2 h |
| Equity drawdown | > 10% | Close all, disable 24 h |
| Wide spread | > 2× normal | Block new entries |

---

## AI Integration

### Local LLM (Ollama — per signal)

- Runs `llama3.1:8b` locally at `localhost:11434`
- Receives compact JSON market summary (~200 tokens)
- Returns `signal_confidence`, `anomaly` flag, `position_adjustment`
- **AI output NEVER overrides the risk engine**
- Defaults to 0.5 confidence on any error

### Cloud LLM (NVIDIA — weekly)

- Calls `meta/llama-3.1-405b-instruct` via NVIDIA API
- Receives weekly performance summary (~500 tokens)
- Suggests strategy weight adjustments and parameter changes
- **Hard Python promotion gate**: suggestions accepted only if
  `Sharpe_new > Sharpe_old AND Drawdown_new ≤ Drawdown_old`

---

## Database Schema

```sql
-- Active strategies
strategies (id, name, type, parameters JSON, asset_classes JSON, ...)

-- Backtest results (one row per run)
backtest_results (strategy_id, asset_symbol, timeframe,
                  sharpe_ratio, sortino_ratio, max_drawdown_pct, ...)

-- Live trade log
trade_log (strategy_id, asset_symbol, side, quantity, price, pnl, ...)

-- Best performers view (Sharpe>1, DD>-25%, win_rate>45%, trades>=30)
best_strategies VIEW
```

---

## Backtesting

```python
from backtester.engine import BacktestEngine
from strategies.trend import TrendStrategy
import pandas as pd

# Load your OHLCV data
df = pd.read_csv("btc_5m.csv", index_col="timestamp", parse_dates=True)

engine = BacktestEngine(strategy=TrendStrategy(), initial_capital=5000)
results = engine.run(df)

print(f"Sharpe: {results['sharpe_ratio']:.2f}")
print(f"Max DD: {results['max_drawdown_pct']:.2f}%")
print(f"Win Rate: {results['win_rate_pct']:.1f}%")

# Walk-forward
wf_results = engine.walk_forward(df, train_pct=0.70, n_splits=3)
```

---

## Deployment (Ubuntu VM — systemd)

Create `/etc/systemd/system/trading_engine.service`:

```ini
[Unit]
Description=AI Trading Engine
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading_with_ai
ExecStart=/home/ubuntu/trading_with_ai/venv/bin/python main.py
Restart=on-failure
RestartSec=30
EnvironmentFile=/home/ubuntu/trading_with_ai/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading_engine
sudo systemctl start trading_engine
sudo journalctl -u trading_engine -f
```

---

## Safety Rules

1. **Testnet ONLY**: `exchange.set_sandbox_mode(True)` is called unconditionally
2. **No hardcoded secrets**: all keys from environment variables via `.env`
3. **AI cannot override risk**: all AI outputs are filtered through Python risk checks
4. **Circuit breakers are non-negotiable**: coded in Python, not LLM-controlled
5. **Liquidation buffer**: `liquidation_distance > 5 × stop_distance` always enforced
6. **Correlation guard**: correlated positions automatically reduced 50%
7. **Graceful shutdown**: SIGTERM/SIGINT handled for clean position state

---

## License

MIT — see LICENSE file for details.

---

> **Final reminder**: This is an experimental research tool. Always trade on testnet first.
> Understand every line of code before connecting to live markets.