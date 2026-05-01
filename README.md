# ICT/SMC Crypto Trade Assistant

A terminal-based trade assistant for **ETHUSDT** and **BTCUSDT** using ICT/Smart Money Concepts methodology. Designed for limit-order traders using multi-timeframe confluence.

---

## What it does

Analyzes real-time Binance data and tells you:

- **Signal direction** (LONG / SHORT / WAIT) based on SMC confluence
- **Exact limit order entry price** — derived from Order Block mid or FVG mid
- **Stop Loss** placement (below OB low / above OB high)
- **3 Take Profit targets** (TP1: 1.5R, TP2: liquidity pool, TP3: range extreme)
- **Win probability** (20–85%) based on scored confluence
- **Volume confirmation** (delta bias + relative volume)

---

## ICT/SMC Components Analyzed

| Component | What it detects |
|---|---|
| **Market Structure** | BOS (Break of Structure), CHoCH (Change of Character), HH/LH/HL/LL swings |
| **Order Blocks** | Last opposing candle before a displacement. Strength-rated. |
| **Fair Value Gaps** | 3-candle imbalances (bullish/bearish). Filters filled gaps. |
| **Liquidity** | Equal highs/lows (BSL/SSL), swept vs. unswept. Bonus for double-tapped levels. |
| **Premium/Discount** | Fibonacci zones: OTE (0.618–0.79), equilibrium (0.5), fib 0.705 |
| **Volume** | Relative volume vs 20-bar avg, delta proxy (buying/selling pressure) |
| **Multi-TF Bias** | HTF (4H or 1H) + LTF (1H or 15m) confluence required |

---

## Setup

```bash
# Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# Run interactively (recommended first run)
python main.py

# Direct flags
python main.py --symbol ETHUSDT --tf 1H-15m
python main.py --symbol BTCUSDT --tf 4H-1H

# Auto-refresh every 5 minutes (watch mode)
python main.py --symbol ETHUSDT --tf 1H-15m --watch --interval 300

# Demo mode — no internet required
python main.py --demo
```

---

## Timeframe Modes

| Mode | HTF (Bias) | LTF (Entry) | Best for |
|---|---|---|---|
| `1H-15m` | 1H | 15m | Intraday scalps, 2–8 hour trades |
| `4H-1H` | 4H | 1H | Swing trades, 1–5 day holds |

---

## Reading the Output

```
  Entry (Limit):      3,447.2100  USDT  ← place limit order here
  Stop Loss:          3,428.5000  USDT
  Take Profit 1:      3,474.3000  USDT  1.5R — partial exit 40%
  Take Profit 2:      3,501.8000  USDT  liquidity target — 40%
  Take Profit 3:      3,538.4000  USDT  range extreme — 20%

  Risk:  0.543%   R:R  1:1.58
```

- **Entry**: The limit order price. Price must come TO you — never chase.
- **Stop Loss**: Hard invalidation. If price closes beyond this, the setup is broken.
- **TP2** is the primary target (liquidity pool — where institutional orders likely sit).
- **Win Probability** factors: HTF alignment (25pts), LTF structure (15pts), OB quality (20pts), FVG (15pts), liquidity (10pts), P/D zone (10pts), volume (5pts).

---

## Files

```
main.py          — CLI entry point, demo mode, watch mode
analyzer.py      — Pure SMC engine (swing detection, OB, FVG, liquidity, scoring)
fetcher.py       — Binance REST API wrapper (no API key needed)
renderer.py      — ANSI terminal report renderer
requirements.txt
```

---

## Notes

- Uses **Binance Futures** data (higher liquidity, more representative of smart money flows)
- Falls back to **Binance Spot** if futures endpoint is unavailable
- No API key required (public endpoints only)
- Volume delta is a **proxy** (close position in range × volume) — not true tape delta
- Probability range is intentionally capped at 85% — no setup is ever a certainty

---

*Not financial advice. Manage your own risk.*
