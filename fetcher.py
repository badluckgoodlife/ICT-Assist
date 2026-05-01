"""
Binance REST API data fetcher (no API key required for public OHLCV).
Supports ETHUSDT and BTCUSDT with multi-timeframe fetching.
"""

import requests
import pandas as pd
import time
import sys
from typing import Optional


BINANCE_BASE = "https://fapi.binance.com"  # Futures (higher volume, preferred)
BINANCE_SPOT = "https://api.binance.com"


TF_MAP = {
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

TF_PAIRS = {
    "4H-1H":  ("4h", "1h"),
    "1H-15m": ("1h", "15m"),
}


def fetch_klines(symbol: str, interval: str,
                  limit: int = 200,
                  use_futures: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Binance.
    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (UTC)
    """
    base = BINANCE_BASE if use_futures else BINANCE_SPOT
    endpoint = f"{base}/fapi/v1/klines" if use_futures else f"{base}/api/v3/klines"

    params = {
        "symbol":   symbol.upper(),
        "interval": interval,
        "limit":    limit,
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print(f"  [network] Cannot reach Binance. Check your internet connection.", file=sys.stderr)
        return None
    except requests.exceptions.Timeout:
        print(f"  [timeout] Binance request timed out.", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e:
        # Fall back to spot if futures 400/404
        if use_futures and resp.status_code in (400, 404):
            return fetch_klines(symbol, interval, limit, use_futures=False)
        print(f"  [http error] {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [fetch error] {e}", file=sys.stderr)
        return None

    rows = []
    for k in data:
        rows.append({
            'timestamp': pd.to_datetime(int(k[0]), unit='ms', utc=True),
            'open':   float(k[1]),
            'high':   float(k[2]),
            'low':    float(k[3]),
            'close':  float(k[4]),
            'volume': float(k[5]),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index('timestamp')
    return df


def fetch_pair(symbol: str, tf_mode: str = "1H-15m") -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fetch HTF and LTF DataFrames for a given symbol and timeframe mode.
    tf_mode: '4H-1H' or '1H-15m'
    Returns: (htf_df, ltf_df) or None on failure
    """
    if tf_mode not in TF_PAIRS:
        raise ValueError(f"Unknown TF mode: {tf_mode}. Choose from: {list(TF_PAIRS.keys())}")

    htf_tf, ltf_tf = TF_PAIRS[tf_mode]

    print(f"  Fetching {symbol} {htf_tf} (200 bars)...", end=" ", flush=True)
    htf = fetch_klines(symbol, htf_tf, limit=200)
    print("OK" if htf is not None else "FAILED")

    time.sleep(0.2)  # polite rate-limit

    print(f"  Fetching {symbol} {ltf_tf} (200 bars)...", end=" ", flush=True)
    ltf = fetch_klines(symbol, ltf_tf, limit=200)
    print("OK" if ltf is not None else "FAILED")

    if htf is None or ltf is None:
        return None

    return htf, ltf


def get_ticker(symbol: str) -> Optional[dict]:
    """Get latest price and 24h stats."""
    url = f"{BINANCE_SPOT}/api/v3/ticker/24hr"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=5)
        data = resp.json()
        return {
            'price':     float(data['lastPrice']),
            'change_pct': float(data['priceChangePercent']),
            'volume_24h': float(data['quoteVolume']),
            'high_24h':  float(data['highPrice']),
            'low_24h':   float(data['lowPrice']),
        }
    except Exception:
        return None
