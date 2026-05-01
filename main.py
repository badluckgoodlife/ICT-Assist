#!/usr/bin/env python3
"""
ICT/SMC Crypto Trade Assistant
================================
Usage:
    python main.py                          # Interactive menu
    python main.py --symbol ETHUSDT --tf 1H-15m
    python main.py --symbol BTCUSDT --tf 4H-1H
    python main.py --watch --interval 300   # Auto-refresh every 5 minutes
    python main.py --demo                   # Run with synthetic data (no internet required)

ICT/SMC methodology by Michael J. Huddleston.
"""

import argparse
import sys
import time
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from analyzer import SMCEngine
from renderer import render_full_report, c, GREEN, RED, YELLOW, CYAN, GRAY, BOLD, RESET


SYMBOLS    = ["ETHUSDT", "BTCUSDT"]
TF_MODES   = ["1H-15m", "4H-1H"]


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generator (for offline / testing use)
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_data(symbol: str, n: int = 200, base_price: float = 0,
                        trend: str = "bullish") -> pd.DataFrame:
    """
    Synthetic OHLCV data with realistic SMC structure:
    - Creates swing highs/lows
    - Adds realistic OB-like moves
    - Adds volume variation
    """
    if base_price == 0:
        base_price = 3450 if "ETH" in symbol else 65000

    np.random.seed(42)
    prices = [base_price]
    vols   = []

    # Generate price path with trend and noise
    for i in range(1, n):
        drift = 0.0003 if trend == "bullish" else -0.0003
        sigma = 0.008
        ret   = drift + np.random.normal(0, sigma)

        # Add occasional impulse moves (simulate OBs / FVGs)
        if i % 20 == 0:
            impulse = 0.012 if trend == "bullish" else -0.012
            ret += impulse
        if i % 33 == 0:
            impulse = -0.007 if trend == "bullish" else 0.007
            ret += impulse  # retracement

        prices.append(prices[-1] * (1 + ret))

        # Volume: higher on impulse moves
        base_vol = base_price * 0.5
        vol_mult = 2.5 if abs(ret) > 0.008 else (1.2 if abs(ret) > 0.003 else 1.0)
        vols.append(base_vol * vol_mult * np.random.uniform(0.8, 1.2))

    vols.insert(0, vols[0] if vols else base_price * 0.5)

    closes = pd.Series(prices)
    opens  = closes.shift(1).fillna(closes[0])
    noise  = pd.Series(np.abs(np.random.normal(0, closes * 0.002)))
    highs  = closes.where(closes >= opens, opens) + noise
    lows   = closes.where(closes <= opens, opens) - noise

    idx = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=n, freq="1h", tz="UTC"
    )

    return pd.DataFrame({
        'open':   opens.values,
        'high':   highs.values,
        'low':    lows.values,
        'close':  closes.values,
        'volume': vols,
    }, index=idx)


def run_demo(symbol: str = "ETHUSDT", tf_mode: str = "1H-15m") -> None:
    """Run analysis using synthetic data."""
    print(f"\n{c(YELLOW, '  [DEMO MODE] Using synthetic data — no internet required')}\n")

    base = 3450 if "ETH" in symbol else 65000
    trend = "bullish"  # can change for different scenarios

    htf_df = generate_demo_data(symbol, n=200, base_price=base, trend=trend)
    ltf_df = generate_demo_data(symbol, n=200, base_price=float(htf_df['close'].iloc[-1]),
                                  trend=trend)

    engine = SMCEngine(swing_lookback=5)
    setup  = engine.analyze(ltf_df, htf_df)

    current_price = float(ltf_df['close'].iloc[-1])
    ticker = {
        'price':       current_price,
        'change_pct':  1.23,
        'volume_24h':  2_400_000_000,
        'high_24h':    current_price * 1.02,
        'low_24h':     current_price * 0.98,
    }

    report = render_full_report(symbol, tf_mode, current_price, setup, ticker)
    print(report)


# ─────────────────────────────────────────────────────────────────────────────
# Live run
# ─────────────────────────────────────────────────────────────────────────────

def run_live(symbol: str, tf_mode: str) -> bool:
    """Fetch live data and run analysis. Returns True on success."""
    try:
        from fetcher import fetch_pair, get_ticker
    except ImportError:
        print(c(RED, "  [error] fetcher.py not found"), file=sys.stderr)
        return False

    print(f"\n{c(CYAN, f'  Analyzing {symbol}  [{tf_mode}]')}")
    print(c(GRAY, "  " + "─" * 40))

    result = fetch_pair(symbol, tf_mode)
    if result is None:
        print(c(RED, "  [error] Failed to fetch market data. Check internet connection."))
        print(c(YELLOW, "  Tip: Run with --demo to test without internet."))
        return False

    htf_df, ltf_df = result

    ticker = get_ticker(symbol)
    current_price = float(ltf_df['close'].iloc[-1])

    engine = SMCEngine(swing_lookback=5)
    setup  = engine.analyze(ltf_df, htf_df)

    report = render_full_report(symbol, tf_mode, current_price, setup, ticker)
    print(report)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────────────────────────────────────

def interactive_menu() -> tuple[str, str]:
    print(f"\n{c(BOLD + CYAN, '  ICT/SMC Trade Assistant')}")
    print(c(GRAY, "  ─────────────────────────"))

    print(f"\n  {c(GRAY, 'Select symbol:')}")
    for i, s in enumerate(SYMBOLS, 1):
        print(f"  {c(CYAN, str(i))}  {s}")
    while True:
        choice = input(f"\n  Enter choice [1-{len(SYMBOLS)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(SYMBOLS):
            symbol = SYMBOLS[int(choice) - 1]
            break
        print(c(RED, "  Invalid choice"))

    print(f"\n  {c(GRAY, 'Select timeframe mode:')}")
    tf_descriptions = {
        "1H-15m": "Intraday scalp / swing  (HTF: 1H → Entry: 15m)",
        "4H-1H":  "Swing / position        (HTF: 4H → Entry: 1H)",
    }
    for i, tf in enumerate(TF_MODES, 1):
        print(f"  {c(CYAN, str(i))}  {tf}  {c(GRAY, tf_descriptions[tf])}")

    while True:
        choice = input(f"\n  Enter choice [1-{len(TF_MODES)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(TF_MODES):
            tf_mode = TF_MODES[int(choice) - 1]
            break
        print(c(RED, "  Invalid choice"))

    return symbol, tf_mode


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICT/SMC Crypto Trade Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--symbol",   choices=SYMBOLS,   default=None, help="Trading pair")
    parser.add_argument("--tf",       choices=TF_MODES,  default=None, help="Timeframe mode")
    parser.add_argument("--watch",    action="store_true",              help="Auto-refresh mode")
    parser.add_argument("--interval", type=int, default=300,            help="Refresh interval in seconds (default: 300)")
    parser.add_argument("--demo",     action="store_true",              help="Run with synthetic data")

    args = parser.parse_args()

    # Demo mode
    if args.demo:
        symbol  = args.symbol  or "ETHUSDT"
        tf_mode = args.tf      or "1H-15m"
        run_demo(symbol, tf_mode)
        return

    # Symbol / TF from args or interactive
    if args.symbol and args.tf:
        symbol, tf_mode = args.symbol, args.tf
    else:
        symbol, tf_mode = interactive_menu()

    # Watch mode (auto-refresh)
    if args.watch:
        print(f"\n{c(CYAN, f'  Watch mode: {symbol} [{tf_mode}] — refreshing every {args.interval}s')}")
        print(c(GRAY, "  Press Ctrl+C to stop\n"))
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                run_live(symbol, tf_mode)
                print(c(GRAY, f"\n  Next refresh in {args.interval}s..."))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(c(GRAY, "\n  Stopped."))
        return

    # Single run
    success = run_live(symbol, tf_mode)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
