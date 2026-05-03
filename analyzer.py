"""
ICT/SMC Crypto Trade Assistant — Analyzer v2
=============================================
Analyzes ETHUSDT / BTCUSDT using ICT/SMC methodology.

NEW in v2 (based on ICT 2022 Mentorship textbook):
- Breaker Blocks  — failed OBs that flip to support/resistance (Ch. 29)
- Mitigation Blocks — failure-swing reversal setups (Ch. 29)
- Previous Day High/Low (PDH/PDL) as primary liquidity targets (Ch. 27)
- Asian Session Range as intraday context (Ch. 11)
- Power of Three (PO3) phase detection vs. midnight open (Ch. 13)
- Draw on Liquidity (DOL) identification (Ch. 3)
- OTE zone entry preference (0.618–0.79 fib, sweet-spot 0.705) (Ch. 16)
- Displacement validation — FVGs/BOS confirmed by aggressive candles (Ch. 10)
- Asian Killzone (20:00–00:00 EST) added to session tracking (Ch. 11)
- OB + FVG stacking bonus — "high-probability OBs have FVG paired" (Ch. 15)
- Consequent Encroachment (CE) — midpoint of FVG used as primary entry (Ch. 24)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class Bias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL  = "NEUTRAL"


class SignalType(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    WAIT  = "WAIT"


@dataclass
class SwingPoint:
    index:     int
    price:     float
    kind:      str       # 'HH' | 'LH' | 'HL' | 'LL'
    timestamp: pd.Timestamp


@dataclass
class OrderBlock:
    high:      float
    low:       float
    mid:       float
    kind:      str       # 'bullish' | 'bearish'
    index:     int
    mitigated: bool  = False
    volume:    float = 0.0
    strength:  float = 0.0   # 0-1 based on displacement after OB
    has_fvg:   bool  = False  # True when a FVG is stacked inside/adjacent (high-prob)


@dataclass
class FairValueGap:
    high:          float
    low:           float
    mid:           float      # Consequent Encroachment (CE) — 50% of gap
    kind:          str        # 'bullish' | 'bearish'
    index:         int
    filled:        bool  = False
    size_pct:      float = 0.0
    displacement:  bool  = False  # True when the gap was formed by a displacement candle


@dataclass
class BreakerBlock:
    """
    A Breaker Block is a former Order Block that price has broken through.
    It flips polarity: a bearish OB broken to the upside becomes bullish support
    on the return.  A bullish OB broken to the downside becomes bearish resistance.

    ICT Pattern (bearish breaker):  High → Low → Higher High → Lower Low
    ICT Pattern (bullish breaker):  Low  → High → Lower Low  → Higher High
    """
    high:     float
    low:      float
    mid:      float
    kind:     str        # 'bullish' | 'bearish'
    index:    int
    strength: float = 0.0


@dataclass
class MitigationBlock:
    """
    A Mitigation Block forms from a FAILURE swing (no stop-run on old high/low).
    Bearish: rally to near old high but fails (lower high), then breaks down.
    Bullish: drop to near old low but fails (higher low), then breaks up.

    ICT Pattern (bearish): High → Low → Lower High → Lower Low
    ICT Pattern (bullish): Low  → High → Higher Low  → Higher High
    """
    high:  float
    low:   float
    mid:   float
    kind:  str        # 'bullish' | 'bearish'
    index: int


@dataclass
class LiquidityLevel:
    price:  float
    kind:   str        # 'BSL' | 'SSL' | 'PDH' | 'PDL'
    index:  int
    swept:  bool = False
    touches: int = 1
    label:  str  = ""  # e.g. "Equal Highs", "Previous Day High"


@dataclass
class StructurePoint:
    price: float
    kind:  str         # 'BOS_bull' | 'BOS_bear' | 'CHoCH_bull' | 'CHoCH_bear'
    index: int


@dataclass
class TradeSetup:
    signal:          SignalType
    entry_price:     float
    stop_loss:       float
    take_profit_1:   float
    take_profit_2:   float
    take_profit_3:   float
    risk_reward:     float
    probability:     float          # 0-100
    confluence_score: float         # 0-100
    reasons:         list[str]
    warnings:        list[str]
    ob:              Optional[OrderBlock]        = None
    fvg:             Optional[FairValueGap]      = None
    nearest_liquidity: Optional[LiquidityLevel] = None
    htf_bias:        Bias = Bias.NEUTRAL
    ltf_bias:        Bias = Bias.NEUTRAL
    volume_confirmed: bool = False
    # ── v2 additions ──────────────────────────────────────────────────────────
    breaker:          Optional[BreakerBlock]     = None
    mitigation:       Optional[MitigationBlock]  = None
    draw_on_liquidity: str  = ""     # Textual description of the current DOL
    po3_phase:         str  = ""     # "Accumulation" | "Manipulation" | "Distribution"
    pdh:               Optional[float] = None   # Previous Day High
    pdl:               Optional[float] = None   # Previous Day Low
    asian_high:        Optional[float] = None   # Asian session range high
    asian_low:         Optional[float] = None   # Asian session range low
    midnight_open:     Optional[float] = None   # NY midnight open price (00:00 EST)


# ─────────────────────────────────────────────────────────────────────────────
# Core SMC Engine
# ─────────────────────────────────────────────────────────────────────────────

class SMCEngine:
    """
    Pure computation engine — takes OHLCV DataFrames, returns structured SMC analysis.
    All prices are raw floats.  No I/O here.
    """

    MIN_CONFLUENCE_SCORE = 65
    MIN_OB_DISTANCE_PCT  = 0.003   # 0.3 %
    MIN_OB_STRENGTH      = 0.20

    def __init__(self, swing_lookback: int = 8):
        self.swing_lookback = swing_lookback

    # ── Swing Detection ──────────────────────────────────────────────────────

    def find_swings(self, df: pd.DataFrame) -> tuple[list[int], list[int]]:
        """Return indices of swing highs and lows using fractals."""
        n = self.swing_lookback
        highs, lows = [], []
        for i in range(n, len(df) - n):
            if df['high'].iloc[i] == df['high'].iloc[i-n:i+n+1].max():
                highs.append(i)
            if df['low'].iloc[i] == df['low'].iloc[i-n:i+n+1].min():
                lows.append(i)
        return highs, lows

    def classify_swings(self, df: pd.DataFrame,
                         swing_highs: list[int],
                         swing_lows:  list[int]) -> list[SwingPoint]:
        """Classify swing points as HH/LH/HL/LL."""
        points: list[SwingPoint] = []
        prev_h = prev_l = None
        for i in sorted(swing_highs + swing_lows):
            ts = df.index[i]
            if i in swing_highs:
                price = df['high'].iloc[i]
                kind  = 'HH' if (prev_h is None or price > prev_h) else 'LH'
                prev_h = price
                points.append(SwingPoint(i, price, kind, ts))
            else:
                price = df['low'].iloc[i]
                kind  = 'HL' if (prev_l is None or price > prev_l) else 'LL'
                prev_l = price
                points.append(SwingPoint(i, price, kind, ts))
        return points

    def market_bias(self, swings: list[SwingPoint]) -> Bias:
        """Determine bias from last 6 swing classifications."""
        if len(swings) < 4:
            return Bias.NEUTRAL
        recent = [s.kind for s in swings[-6:]]
        bull = recent.count('HH') + recent.count('HL')
        bear = recent.count('LH') + recent.count('LL')
        if bull > bear:   return Bias.BULLISH
        if bear > bull:   return Bias.BEARISH
        return Bias.NEUTRAL

    # ── Market Structure ─────────────────────────────────────────────────────

    def find_structure_breaks(self, df: pd.DataFrame,
                               swings: list[SwingPoint]) -> list[StructurePoint]:
        """
        Detect BOS and CHoCH.  Each swing level is consumed exactly once
        to prevent phantom breaks firing on the same static level.
        """
        breaks: list[StructurePoint] = []
        bias   = Bias.NEUTRAL

        sh_list = sorted([s for s in swings if s.kind in ('HH','LH')], key=lambda s: s.index)
        sl_list = sorted([s for s in swings if s.kind in ('HL','LL')], key=lambda s: s.index)

        sh_ptr, sl_ptr   = 0, 0
        active_sh: Optional[SwingPoint] = None
        active_sl: Optional[SwingPoint] = None

        for i in range(len(df)):
            close = df['close'].iloc[i]

            if active_sh is None:
                while sh_ptr < len(sh_list) and sh_list[sh_ptr].index < i:
                    active_sh = sh_list[sh_ptr]; sh_ptr += 1

            if active_sl is None:
                while sl_ptr < len(sl_list) and sl_list[sl_ptr].index < i:
                    active_sl = sl_list[sl_ptr]; sl_ptr += 1

            if active_sh and close > active_sh.price:
                kind = 'BOS_bull' if bias == Bias.BULLISH else 'CHoCH_bull'
                breaks.append(StructurePoint(active_sh.price, kind, i))
                bias      = Bias.BULLISH
                active_sh = None

            if active_sl and close < active_sl.price:
                kind = 'BOS_bear' if bias == Bias.BEARISH else 'CHoCH_bear'
                breaks.append(StructurePoint(active_sl.price, kind, i))
                bias      = Bias.BEARISH
                active_sl = None

        return breaks

    # ── Order Blocks ─────────────────────────────────────────────────────────

    def find_order_blocks(self, df: pd.DataFrame,
                           swing_highs: list[int],
                           swing_lows:  list[int]) -> list[OrderBlock]:
        """
        Bullish OB: last bearish candle before a bullish impulse to a new SH.
        Bearish OB: last bullish candle before a bearish impulse to a new SL.
        """
        obs: list[OrderBlock] = []

        for sh in swing_highs:
            for j in range(sh - 1, max(sh - 15, 0), -1):
                o, c = df['open'].iloc[j], df['close'].iloc[j]
                if c < o:
                    vol      = df['volume'].iloc[j] if 'volume' in df else 0
                    ob_size  = o - c
                    post_move = df['high'].iloc[sh] - df['high'].iloc[j]
                    strength = round(min(min(post_move / max(ob_size, 1e-8), 1.0) * 0.5, 1.0), 2)
                    if strength < self.MIN_OB_STRENGTH:
                        break
                    obs.append(OrderBlock(
                        high=max(o, c), low=min(o, c),
                        mid=(o + c) / 2, kind='bullish',
                        index=j, volume=vol, strength=strength
                    ))
                    break

        for sl in swing_lows:
            for j in range(sl - 1, max(sl - 15, 0), -1):
                o, c = df['open'].iloc[j], df['close'].iloc[j]
                if c > o:
                    vol      = df['volume'].iloc[j] if 'volume' in df else 0
                    ob_size  = c - o
                    post_move = df['low'].iloc[j] - df['low'].iloc[sl]
                    strength = round(min(min(post_move / max(ob_size, 1e-8), 1.0) * 0.5, 1.0), 2)
                    if strength < self.MIN_OB_STRENGTH:
                        break
                    obs.append(OrderBlock(
                        high=max(o, c), low=min(o, c),
                        mid=(o + c) / 2, kind='bearish',
                        index=j, volume=vol, strength=strength
                    ))
                    break

        for ob in obs:
            post = df.iloc[ob.index + 1:]
            if ob.kind == 'bullish':
                if not post.empty and (post['low'] <= ob.high).any():
                    ob.mitigated = True
            else:
                if not post.empty and (post['high'] >= ob.low).any():
                    ob.mitigated = True

        return [ob for ob in obs if not ob.mitigated]

    # ── NEW: Breaker Blocks ───────────────────────────────────────────────────

    def find_breaker_blocks(self, df: pd.DataFrame,
                             swings: list[SwingPoint]) -> list[BreakerBlock]:
        """
        ICT Breaker Block — Ch. 29.

        Bearish breaker pattern:  High → Low → Higher High (BSL taken) → Lower Low
          The down-close candle forming the Low becomes a bearish breaker.
          When price returns to it from below, sell there.

        Bullish breaker pattern:  Low → High → Lower Low (SSL taken) → Higher High
          The up-close candle forming the High becomes a bullish breaker.
          When price returns to it from above, buy there.
        """
        breakers: list[BreakerBlock] = []
        s = sorted(swings, key=lambda x: x.index)

        for i in range(len(s) - 3):
            s1, s2, s3, s4 = s[i], s[i+1], s[i+2], s[i+3]

            # ── Bearish breaker: SH → SL → HH (stop run) → LL (breaks below SL)
            if (s1.kind in ('HH', 'LH') and
                s2.kind in ('HL', 'LL') and
                s3.kind == 'HH' and s3.price > s1.price and   # BSL taken
                s4.kind == 'LL' and s4.price < s2.price):     # breaks below SL

                # Find the last down-close candle around the SL (s2)
                bb_idx = self._last_directional_candle(df, s2.index, direction='down')
                if bb_idx is not None:
                    o, c = df['open'].iloc[bb_idx], df['close'].iloc[bb_idx]
                    # Only valid if price is currently ABOVE this zone (pullback expected)
                    if df['close'].iloc[-1] > max(o, c):
                        breakers.append(BreakerBlock(
                            high=max(o, c), low=min(o, c),
                            mid=(o + c) / 2, kind='bearish', index=bb_idx,
                            strength=round(abs(s3.price - s4.price) /
                                           max(abs(s1.price - s2.price), 1e-8), 2)
                        ))

            # ── Bullish breaker: SL → SH → LL (stop run) → HH (breaks above SH)
            if (s1.kind in ('HL', 'LL') and
                s2.kind in ('HH', 'LH') and
                s3.kind == 'LL' and s3.price < s1.price and   # SSL taken
                s4.kind == 'HH' and s4.price > s2.price):     # breaks above SH

                bb_idx = self._last_directional_candle(df, s2.index, direction='up')
                if bb_idx is not None:
                    o, c = df['open'].iloc[bb_idx], df['close'].iloc[bb_idx]
                    # Only valid if price is currently BELOW this zone (return expected)
                    if df['close'].iloc[-1] < min(o, c):
                        breakers.append(BreakerBlock(
                            high=max(o, c), low=min(o, c),
                            mid=(o + c) / 2, kind='bullish', index=bb_idx,
                            strength=round(abs(s4.price - s3.price) /
                                           max(abs(s2.price - s1.price), 1e-8), 2)
                        ))

        return breakers

    def _last_directional_candle(self, df: pd.DataFrame,
                                   around_idx: int,
                                   direction:   str,
                                   lookback:    int = 6) -> Optional[int]:
        """Find the most recent up/down-close candle near a swing index."""
        start = max(around_idx - lookback, 0)
        for j in range(around_idx, start, -1):
            o, c = df['open'].iloc[j], df['close'].iloc[j]
            if direction == 'down' and c < o:
                return j
            if direction == 'up'   and c > o:
                return j
        return None

    # ── NEW: Mitigation Blocks ────────────────────────────────────────────────

    def find_mitigation_blocks(self, df: pd.DataFrame,
                                swings: list[SwingPoint]) -> list[MitigationBlock]:
        """
        ICT Mitigation Block — Ch. 29.

        Bearish mitigation pattern: High → Low → Lower High (fails) → Lower Low
          The down-close candle at the Low is the bearish mitigation block.

        Bullish mitigation pattern: Low → High → Higher Low (fails) → Higher High
          The up-close candle at the High is the bullish mitigation block.

        Unlike a breaker, there is NO stop run on the original high/low.
        """
        blocks: list[MitigationBlock] = []
        s = sorted(swings, key=lambda x: x.index)

        for i in range(len(s) - 3):
            s1, s2, s3, s4 = s[i], s[i+1], s[i+2], s[i+3]

            # ── Bearish mitigation: SH → SL → LH (failure, no new HH) → LL
            if (s1.kind in ('HH', 'LH') and
                s2.kind in ('HL', 'LL') and
                s3.kind == 'LH' and s3.price < s1.price and    # lower high = failure swing
                s4.kind == 'LL' and s4.price < s2.price):      # breaks below SL

                mb_idx = self._last_directional_candle(df, s2.index, direction='down')
                if mb_idx is not None:
                    o, c = df['open'].iloc[mb_idx], df['close'].iloc[mb_idx]
                    if df['close'].iloc[-1] > max(o, c):
                        blocks.append(MitigationBlock(
                            high=max(o, c), low=min(o, c),
                            mid=(o + c) / 2, kind='bearish', index=mb_idx
                        ))

            # ── Bullish mitigation: SL → SH → HL (failure) → HH
            if (s1.kind in ('HL', 'LL') and
                s2.kind in ('HH', 'LH') and
                s3.kind == 'HL' and s3.price > s1.price and    # higher low = failure swing
                s4.kind == 'HH' and s4.price > s2.price):      # breaks above SH

                mb_idx = self._last_directional_candle(df, s2.index, direction='up')
                if mb_idx is not None:
                    o, c = df['open'].iloc[mb_idx], df['close'].iloc[mb_idx]
                    if df['close'].iloc[-1] < min(o, c):
                        blocks.append(MitigationBlock(
                            high=max(o, c), low=min(o, c),
                            mid=(o + c) / 2, kind='bullish', index=mb_idx
                        ))

        return blocks

    # ── Fair Value Gaps ───────────────────────────────────────────────────────

    def find_fvgs(self, df: pd.DataFrame,
                   min_size_pct: float = 0.15) -> list[FairValueGap]:
        """
        FVG: 3-candle imbalance.  Bullish: c[i-2].high < c[i].low.
        Bearish: c[i-2].low > c[i].high.

        v2: adds displacement flag — FVGs formed by large aggressive candles
        (candle[i-1] body > 1.5× 10-bar average body) are marked as displacement-
        confirmed, which the book says are of higher probability.
        """
        fvgs: list[FairValueGap] = []
        current_price = df['close'].iloc[-1]

        # Compute rolling average body size for displacement check
        body_size = (df['close'] - df['open']).abs()
        avg_body  = body_size.rolling(10).mean()

        for i in range(2, len(df)):
            c0h, c0l = df['high'].iloc[i-2], df['low'].iloc[i-2]
            c2h, c2l = df['high'].iloc[i],   df['low'].iloc[i]
            avg = avg_body.iloc[i-1] if pd.notna(avg_body.iloc[i-1]) else 0
            displacement = bool(body_size.iloc[i-1] > avg * 1.5) if avg > 0 else False

            # Bullish FVG
            if c2l > c0h:
                size_pct = (c2l - c0h) / c0h * 100
                if size_pct >= min_size_pct:
                    filled = current_price <= c0h
                    fvgs.append(FairValueGap(
                        high=c2l, low=c0h, mid=(c2l + c0h) / 2,
                        kind='bullish', index=i, filled=filled,
                        size_pct=round(size_pct, 4), displacement=displacement
                    ))

            # Bearish FVG
            if c2h < c0l:
                size_pct = (c0l - c2h) / c0l * 100
                if size_pct >= min_size_pct:
                    filled = current_price >= c0l
                    fvgs.append(FairValueGap(
                        high=c0l, low=c2h, mid=(c0l + c2h) / 2,
                        kind='bearish', index=i, filled=filled,
                        size_pct=round(size_pct, 4), displacement=displacement
                    ))

        return [f for f in fvgs if not f.filled]

    # ── Liquidity ─────────────────────────────────────────────────────────────

    def find_liquidity(self, df: pd.DataFrame,
                        swing_highs: list[int],
                        swing_lows:  list[int]) -> list[LiquidityLevel]:
        """
        BSL: above equal highs / swing highs (buy-stops resting above)
        SSL: below equal lows  / swing lows  (sell-stops resting below)
        """
        levels: list[LiquidityLevel] = []
        current_price = df['close'].iloc[-1]
        tol = current_price * 0.001

        high_prices = [(i, df['high'].iloc[i]) for i in swing_highs]
        low_prices  = [(i, df['low'].iloc[i])  for i in swing_lows]

        # Equal highs → BSL
        for i in range(len(high_prices)):
            for j in range(i+1, len(high_prices)):
                if abs(high_prices[i][1] - high_prices[j][1]) < tol:
                    swept = current_price > high_prices[j][1]
                    levels.append(LiquidityLevel(
                        price=max(high_prices[i][1], high_prices[j][1]),
                        kind='BSL', index=high_prices[j][0],
                        swept=swept, touches=2, label="Equal Highs"
                    ))

        # Single swing highs above current price → BSL
        for idx, price in high_prices:
            if price > current_price * 1.002:
                levels.append(LiquidityLevel(price=price, kind='BSL', index=idx))

        # Equal lows → SSL
        for i in range(len(low_prices)):
            for j in range(i+1, len(low_prices)):
                if abs(low_prices[i][1] - low_prices[j][1]) < tol:
                    swept = current_price < low_prices[j][1]
                    levels.append(LiquidityLevel(
                        price=min(low_prices[i][1], low_prices[j][1]),
                        kind='SSL', index=low_prices[j][0],
                        swept=swept, touches=2, label="Equal Lows"
                    ))

        # Single swing lows below current price → SSL
        for idx, price in low_prices:
            if price < current_price * 0.998:
                levels.append(LiquidityLevel(price=price, kind='SSL', index=idx))

        return [l for l in levels if not l.swept]

    # ── NEW: Previous Day High / Low ──────────────────────────────────────────

    def find_pdh_pdl(self, df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        """
        Find the Previous Day High (PDH) and Previous Day Low (PDL) in EST.

        ICT (Ch. 27): PDH/PDL are the most accessible liquidity pools.
        Every day, the market tends to gravitate toward one of these levels.
        """
        try:
            from zoneinfo import ZoneInfo
            tz_est = ZoneInfo('America/New_York')
            df_est = df.copy()
            # Convert index to EST
            if df_est.index.tzinfo is None:
                df_est.index = df_est.index.tz_localize('UTC')
            df_est.index = df_est.index.tz_convert(tz_est)

            today = df_est.index[-1].date()
            prev_day = df_est[df_est.index.date < today]
            if prev_day.empty:
                return None, None

            # Get the most recent full previous day
            prev_date = prev_day.index[-1].date()
            prev_day_data = prev_day[prev_day.index.date == prev_date]
            if prev_day_data.empty:
                # Fall back: use data before today's session
                return float(prev_day['high'].max()), float(prev_day['low'].min())

            return float(prev_day_data['high'].max()), float(prev_day_data['low'].min())

        except Exception:
            # Fallback: approximate PDH/PDL using the 24-48 candle range
            if len(df) < 24:
                return None, None
            lookback = min(48, len(df) - 1)
            prev_slice = df.iloc[-lookback:-1]
            return float(prev_slice['high'].max()), float(prev_slice['low'].min())

    # ── NEW: Session Ranges ───────────────────────────────────────────────────

    def get_session_ranges(self, df: pd.DataFrame) -> dict:
        """
        Extract Asian, London, and NY session range highs and lows.

        ICT (Ch. 11):
        - Asian session (20:00–00:00 EST): sets the consolidation, gives context.
        - London session (02:00–05:00 EST): usually makes the HOD/LOD.
        - NY session    (07:00–10:00 EST): continuation or reversal.

        Returns dict with keys: asian_high, asian_low, london_high, london_low,
                                 ny_high, ny_low, midnight_open
        """
        result = {}
        try:
            from zoneinfo import ZoneInfo
            tz_est = ZoneInfo('America/New_York')
            df_est = df.copy()
            if df_est.index.tzinfo is None:
                df_est.index = df_est.index.tz_localize('UTC')
            df_est.index = df_est.index.tz_convert(tz_est)

            hours = df_est.index.hour + df_est.index.minute / 60.0

            # Midnight open — the 00:00 EST candle open price
            midnight_mask = (hours >= 0.0) & (hours < 1.0)
            midnight_bars  = df_est[midnight_mask]
            if not midnight_bars.empty:
                result['midnight_open'] = float(midnight_bars['open'].iloc[-1])

            # Asian session: 20:00 – 00:00 EST
            asian_mask = (hours >= 20.0) | (hours < 0.5)
            asian_bars  = df_est[asian_mask]
            if not asian_bars.empty:
                result['asian_high'] = float(asian_bars['high'].max())
                result['asian_low']  = float(asian_bars['low'].min())

            # London session: 02:00 – 05:00 EST
            london_mask = (hours >= 2.0) & (hours < 5.0)
            london_bars  = df_est[london_mask]
            if not london_bars.empty:
                result['london_high'] = float(london_bars['high'].max())
                result['london_low']  = float(london_bars['low'].min())

            # NY session: 07:00 – 10:00 EST
            ny_mask = (hours >= 7.0) & (hours < 10.0)
            ny_bars  = df_est[ny_mask]
            if not ny_bars.empty:
                result['ny_high'] = float(ny_bars['high'].max())
                result['ny_low']  = float(ny_bars['low'].min())

        except Exception:
            pass

        return result

    # ── NEW: Power of Three (PO3) Phase ──────────────────────────────────────

    def detect_power_of_three(self, df: pd.DataFrame,
                               session: dict) -> str:
        """
        ICT Power of Three — Ch. 13: Accumulation → Manipulation → Distribution.

        Using the midnight open price as the reference:
        - If price is near midnight open and ranging → Accumulation
        - If price has moved against the bias from midnight open → Manipulation (Judas Swing)
        - If price has moved with the bias significantly → Distribution

        Returns: "Accumulation" | "Manipulation" | "Distribution" | ""
        """
        midnight_open = session.get('midnight_open')
        if midnight_open is None or midnight_open == 0:
            return ""

        current = float(df['close'].iloc[-1])
        move_pct = abs(current - midnight_open) / midnight_open * 100

        # Small move relative to midnight open = still in accumulation/consolidation
        if move_pct < 0.3:
            return "Accumulation"

        # Detect Judas Swing: price moved away from midnight open but we're
        # in an opposing structure relative to the HTF bias.
        # Simplified: if current session high is above midnight open but bias is bearish
        # (or vice versa), label as Manipulation.
        asian_high = session.get('asian_high')
        asian_low  = session.get('asian_low')
        if asian_high and asian_low:
            asian_range = asian_high - asian_low
            if asian_range > 0:
                price_in_asian = asian_low <= current <= asian_high
                if price_in_asian:
                    return "Accumulation"
                if current > asian_high:
                    # Price broke above Asian range — could be Manipulation (bearish day)
                    # or real expansion (bullish day). Without full context, call it Manipulation.
                    return "Manipulation (Judas Swing — above Asian range)"
                if current < asian_low:
                    return "Manipulation (Judas Swing — below Asian range)"

        if move_pct > 0.8:
            return "Distribution"

        return "Accumulation"

    # ── NEW: Draw on Liquidity ────────────────────────────────────────────────

    def identify_draw_on_liquidity(self,
                                    signal:    SignalType,
                                    liq:       list[LiquidityLevel],
                                    fvgs:      list[FairValueGap],
                                    pdh:       Optional[float],
                                    pdl:       Optional[float],
                                    current:   float) -> str:
        """
        ICT (Ch. 3): Price is always either rebalancing or taking liquidity.
        Identify the current Draw on Liquidity (DOL).

        Priority: PDH/PDL > Equal Highs/Lows > Single swing highs/lows > FVG
        """
        if signal == SignalType.LONG:
            targets = []
            if pdh and pdh > current:
                targets.append((pdh, f"Previous Day High @ {pdh:,.2f}"))
            bsl = sorted(
                [(l.price, f"{l.label or 'Swing High'} BSL @ {l.price:,.2f}")
                 for l in liq if l.kind == 'BSL' and l.price > current],
                key=lambda x: x[0]
            )
            targets.extend(bsl[:2])
            fvg_t = sorted(
                [(f.high, f"Bearish FVG CE @ {f.mid:,.2f}")
                 for f in fvgs if f.kind == 'bearish' and f.low > current],
                key=lambda x: x[0]
            )
            targets.extend(fvg_t[:1])
            if targets:
                return targets[0][1]
            return "No clear DOL identified above price"

        else:  # SHORT
            targets = []
            if pdl and pdl < current:
                targets.append((pdl, f"Previous Day Low @ {pdl:,.2f}"))
            ssl = sorted(
                [(l.price, f"{l.label or 'Swing Low'} SSL @ {l.price:,.2f}")
                 for l in liq if l.kind == 'SSL' and l.price < current],
                key=lambda x: x[0], reverse=True
            )
            targets.extend(ssl[:2])
            fvg_t = sorted(
                [(f.low, f"Bullish FVG CE @ {f.mid:,.2f}")
                 for f in fvgs if f.kind == 'bullish' and f.high < current],
                key=lambda x: x[0], reverse=True
            )
            targets.extend(fvg_t[:1])
            if targets:
                return targets[0][1]
            return "No clear DOL identified below price"

    # ── Premium / Discount ────────────────────────────────────────────────────

    def pd_zones(self, df: pd.DataFrame,
                  swing_highs: list[int],
                  swing_lows:  list[int]) -> dict:
        """
        Fibonacci-based premium/discount zones.
        Equilibrium = 0.5.  OTE = 0.618–0.79.  Sweet spot = 0.705.
        """
        if not swing_highs or not swing_lows:
            return {}

        sh_pairs = [(i, df['high'].iloc[i]) for i in swing_highs]
        sl_pairs = [(i, df['low'].iloc[i])  for i in swing_lows]

        last_sh_idx, rng_high = max(sh_pairs, key=lambda x: x[0])
        last_sl_idx, rng_low  = max(sl_pairs, key=lambda x: x[0])

        if rng_high <= rng_low:
            rng_high = max(p for _, p in sh_pairs)
            rng_low  = min(p for _, p in sl_pairs)

        rng = rng_high - rng_low
        if rng == 0 or rng / max(rng_low, 1e-8) < 0.005:
            return {}

        current = df['close'].iloc[-1]
        fib_pos = round(max(0.0, min(1.0, (current - rng_low) / rng)), 4)

        return {
            'range_high':  rng_high,
            'range_low':   rng_low,
            'equilibrium': rng_low + rng * 0.50,
            'ote_high':    rng_low + rng * 0.79,   # OTE zone top
            'ote_low':     rng_low + rng * 0.618,  # OTE zone bottom
            'fib_705':     rng_low + rng * 0.705,  # sweet spot
            'fib_pos':     fib_pos,
            'zone':        'DISCOUNT' if fib_pos < 0.5 else 'PREMIUM',
        }

    # ── Volume Analysis ───────────────────────────────────────────────────────

    def volume_analysis(self, df: pd.DataFrame) -> dict:
        """Volume metrics: relative volume, delta proxy, trend."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return {'available': False}

        vols    = df['volume']
        avg_vol = vols.rolling(20).mean()
        last_vol = vols.iloc[-1]
        rel_vol  = last_vol / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1.0

        def delta_proxy(row):
            rng = row['high'] - row['low']
            return (row['close'] - row['low']) / rng if rng > 0 else 0.5

        df2 = df.copy()
        df2['delta']     = df2.apply(delta_proxy, axis=1)
        df2['vol_delta'] = df2['delta'] * df2['volume']

        recent    = df2.iloc[-5:]
        cum_delta = recent['vol_delta'].sum()
        total_vol = recent['volume'].sum()
        avg_delta = cum_delta / total_vol if total_vol > 0 else 0.5

        v5  = vols.iloc[-5:].mean()
        v20 = vols.iloc[-20:].mean()
        trend = ('INCREASING' if v5 > v20 * 1.1 else
                 'DECREASING' if v5 < v20 * 0.9 else 'NEUTRAL')

        return {
            'available':   True,
            'rel_vol':     round(float(rel_vol), 2),
            'avg_delta':   round(float(avg_delta), 3),
            'vol_trend':   trend,
            'is_high_vol': rel_vol > 1.3,
            'bias': ('BULLISH' if avg_delta > 0.55 else
                     'BEARISH' if avg_delta < 0.45 else 'NEUTRAL')
        }

    # ── Killzone Detection ────────────────────────────────────────────────────

    def in_killzone(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Check whether the most recent candle falls inside an ICT killzone.

        v2: Added Asian Killzone (20:00–00:00 EST).
        Killzones:
          Asian        20:00 – 00:00 EST
          London Open  02:00 – 05:00 EST
          NY AM Open   07:00 – 10:00 EST
          London Close 10:00 – 12:00 EST
        """
        from datetime import timezone
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        last_ts = df.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        est_ts = last_ts.astimezone(ZoneInfo('America/New_York'))
        h      = est_ts.hour + est_ts.minute / 60.0

        if 20.0 <= h < 24.0 or 0.0 <= h < 1.0:
            return True,  "Asian KZ        (20:00–00:00 EST)"
        if  2.0 <= h <  5.0:
            return True,  "London Open KZ  (02:00–05:00 EST)"
        if  7.0 <= h < 10.0:
            return True,  "NY AM Open KZ   (07:00–10:00 EST)"
        if 10.0 <= h < 12.0:
            return True,  "London Close KZ (10:00–12:00 EST)"

        return False, ""

    # ── Setup Scoring ─────────────────────────────────────────────────────────

    def score_setup(self,
                    bias:             Bias,
                    htf_bias:         Bias,
                    ob:               Optional[OrderBlock],
                    fvg:              Optional[FairValueGap],
                    liq:              Optional[LiquidityLevel],
                    pd_data:          dict,
                    vol:              dict,
                    structure_breaks: list[StructurePoint],
                    signal:           SignalType,
                    breaker:          Optional[BreakerBlock]   = None,
                    mitigation:       Optional[MitigationBlock] = None,
                    pdh:              Optional[float]           = None,
                    pdl:              Optional[float]           = None,
                    current_price:    float                     = 0.0,
                    ) -> tuple[float, list[str], list[str]]:
        """
        Score confluence 0-100.
        v2: adds Breaker Block, Mitigation Block, OB+FVG stacking, PDH/PDL scores.
        Returns (score, reasons[], warnings[])
        """
        score    = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        # ── HTF alignment (25 pts) ────────────────────────────────────────────
        if htf_bias == Bias.BULLISH and signal == SignalType.LONG:
            score += 25
            reasons.append("HTF bias BULLISH — trading with institutional trend")
        elif htf_bias == Bias.BEARISH and signal == SignalType.SHORT:
            score += 25
            reasons.append("HTF bias BEARISH — trading with institutional trend")
        elif htf_bias == Bias.NEUTRAL:
            score += 10
            warnings.append("HTF bias NEUTRAL — reduced directional edge")
        else:
            warnings.append("Counter-trend trade — HTF bias opposes signal direction")

        # ── LTF structure (15 pts) ────────────────────────────────────────────
        if bias == Bias.BULLISH and signal == SignalType.LONG:
            score += 15
            reasons.append("LTF structure: higher highs and higher lows")
        elif bias == Bias.BEARISH and signal == SignalType.SHORT:
            score += 15
            reasons.append("LTF structure: lower highs and lower lows")
        elif bias == Bias.NEUTRAL:
            score += 5

        # ── Order Block (20 pts) ──────────────────────────────────────────────
        if ob:
            ob_score = 10 + int(ob.strength * 10)
            score += ob_score
            reasons.append(
                f"{'Bullish' if ob.kind=='bullish' else 'Bearish'} Order Block "
                f"[{ob.low:.2f}–{ob.high:.2f}] strength={ob.strength:.0%}"
            )
            if ob.volume > 0:
                reasons.append(f"OB backed by volume: {ob.volume:,.0f}")

        # ── FVG (15 pts) ──────────────────────────────────────────────────────
        if fvg:
            fvg_pts = 15
            score += fvg_pts
            disp_note = " [displacement-confirmed]" if fvg.displacement else ""
            reasons.append(
                f"{'Bullish' if fvg.kind=='bullish' else 'Bearish'} FVG "
                f"[{fvg.low:.2f}–{fvg.high:.2f}] "
                f"CE={fvg.mid:.2f} size={fvg.size_pct:.3f}%{disp_note}"
            )

        # ── OB + FVG stacking bonus (+5) — "high-probability OBs have FVG paired" ──
        if ob and fvg:
            # Check if FVG falls within or near the OB zone
            overlap = (fvg.low <= ob.high and fvg.high >= ob.low)
            if overlap:
                score += 5
                reasons.append("OB + FVG stacked at same zone — A+ confluence")

        # ── Breaker Block (10 pts) ────────────────────────────────────────────
        if breaker:
            score += 10
            reasons.append(
                f"{'Bullish' if breaker.kind=='bullish' else 'Bearish'} Breaker Block "
                f"[{breaker.low:.2f}–{breaker.high:.2f}] "
                f"— failed OB, now flipped polarity (strength={breaker.strength:.1f}x)"
            )

        # ── Mitigation Block (5 pts) ──────────────────────────────────────────
        if mitigation:
            score += 5
            reasons.append(
                f"{'Bullish' if mitigation.kind=='bullish' else 'Bearish'} Mitigation Block "
                f"[{mitigation.low:.2f}–{mitigation.high:.2f}] — failure-swing setup"
            )

        # ── Liquidity target (10 pts) ─────────────────────────────────────────
        if liq:
            score += 10
            kind_name = "Buy-side" if liq.kind == 'BSL' else "Sell-side"
            reasons.append(
                f"{kind_name} liquidity target at {liq.price:.2f} "
                f"({'double top/bottom' if liq.touches > 1 else 'single swing'})"
            )

        # ── PDH / PDL alignment (+5) ──────────────────────────────────────────
        if current_price > 0:
            tol = current_price * 0.005  # 0.5% proximity
            if signal == SignalType.LONG and pdl and abs(current_price - pdl) < tol:
                score += 5
                reasons.append(f"Price near Previous Day Low ({pdl:,.2f}) — strong SSL sweep zone")
            elif signal == SignalType.SHORT and pdh and abs(current_price - pdh) < tol:
                score += 5
                reasons.append(f"Price near Previous Day High ({pdh:,.2f}) — strong BSL sweep zone")

        # ── Premium/Discount zone (10 pts) ────────────────────────────────────
        if pd_data:
            fib  = pd_data.get('fib_pos', 0.5)
            zone = pd_data.get('zone', '')
            ote_low  = pd_data.get('ote_low',  0)
            ote_high = pd_data.get('ote_high', 0)
            in_ote   = ote_low <= current_price <= ote_high if current_price > 0 else False

            if signal == SignalType.LONG and zone == 'DISCOUNT':
                score += 10
                ote_note = " — INSIDE OTE ZONE" if in_ote else ""
                reasons.append(f"Price in DISCOUNT zone (fib={fib:.2%}){ote_note}")
            elif signal == SignalType.SHORT and zone == 'PREMIUM':
                score += 10
                ote_note = " — INSIDE OTE ZONE" if in_ote else ""
                reasons.append(f"Price in PREMIUM zone (fib={fib:.2%}){ote_note}")
            elif signal == SignalType.LONG and zone == 'PREMIUM':
                warnings.append(f"Buying in PREMIUM zone (fib={fib:.2%}) — suboptimal entry")
            elif signal == SignalType.SHORT and zone == 'DISCOUNT':
                warnings.append(f"Shorting in DISCOUNT zone (fib={fib:.2%}) — suboptimal entry")

        # ── Volume (display only — ICT is price-action, not volume-based) ─────
        if vol.get('available'):
            rv    = vol.get('rel_vol', 1.0)
            vbias = vol.get('bias', 'NEUTRAL')
            if rv > 1.5:
                reasons.append(f"[info] Relative volume elevated ({rv:.1f}×avg) — not scored")
            if vbias != 'NEUTRAL':
                reasons.append(f"[info] Volume-delta bias: {vbias} — not scored")

        # ── CHoCH bonus (5 pts) ───────────────────────────────────────────────
        recent_choch = [s for s in structure_breaks[-5:] if 'CHoCH' in s.kind]
        if recent_choch:
            choch = recent_choch[-1]
            if 'bull' in choch.kind and signal == SignalType.LONG:
                score += 5
                reasons.append("Recent bullish CHoCH — market structure flip confirmed")
            elif 'bear' in choch.kind and signal == SignalType.SHORT:
                score += 5
                reasons.append("Recent bearish CHoCH — market structure flip confirmed")

        return round(min(score, 100), 1), reasons, warnings

    # ── Entry Computation ─────────────────────────────────────────────────────

    def compute_entry(self,
                       signal:       SignalType,
                       ob:           Optional[OrderBlock],
                       fvg:          Optional[FairValueGap],
                       current_price: float,
                       swing_highs:  list[int],
                       swing_lows:   list[int],
                       df:           pd.DataFrame,
                       pd_data:      dict = None,
                       breaker:      Optional[BreakerBlock]   = None,
                       mitigation:   Optional[MitigationBlock] = None,
                       ) -> tuple[float, float]:
        """
        Entry priority (v2):
        1. Breaker Block mid (flipped OB — high-probability)
        2. OTE zone (0.705 fib) when price is in OTE range
        3. OB mid (CE of OB)
        4. FVG mid (CE of gap)
        5. Nearest swing + buffer

        Stop: just below OB low (long) or above OB high (short).
        """
        # ── Determine OTE entry if price is inside the zone ──
        ote_entry = None
        if pd_data:
            ote_low  = pd_data.get('ote_low',  0)
            ote_high = pd_data.get('ote_high', 0)
            fib_705  = pd_data.get('fib_705',  0)
            rng_low  = pd_data.get('range_low',  0)
            rng_high = pd_data.get('range_high', 0)
            if ote_low > 0 and ote_low <= current_price <= ote_high:
                if signal == SignalType.LONG:
                    ote_entry = fib_705  # enter at sweet-spot pullback
                else:
                    # For short, OTE is in premium (above EQ)
                    # Flip: 1 - fib_705 equivalent
                    ote_entry = rng_low + (rng_high - rng_low) * (1 - 0.705)

        if signal == SignalType.LONG:
            # Priority: Breaker → OTE → OB → FVG → swing
            if breaker and breaker.kind == 'bullish':
                entry = breaker.mid
                stop  = breaker.low * 0.999
            elif ote_entry is not None:
                entry = ote_entry
                lows  = [df['low'].iloc[i] for i in swing_lows if df['low'].iloc[i] < ote_entry]
                stop  = max(lows) * 0.999 if lows else ote_entry * 0.995
            elif ob:
                entry = ob.mid
                stop  = ob.low * 0.999
            elif fvg:
                entry = fvg.mid   # Consequent Encroachment (CE)
                lows  = [df['low'].iloc[i] for i in swing_lows if df['low'].iloc[i] < fvg.low]
                stop  = max(lows) * 0.999 if lows else fvg.low * 0.995
            else:
                entry = current_price
                lows  = [df['low'].iloc[i] for i in swing_lows]
                stop  = max([l for l in lows if l < current_price],
                             default=current_price * 0.99) * 0.999

        else:  # SHORT
            if breaker and breaker.kind == 'bearish':
                entry = breaker.mid
                stop  = breaker.high * 1.001
            elif ote_entry is not None:
                entry = ote_entry
                highs = [df['high'].iloc[i] for i in swing_highs if df['high'].iloc[i] > ote_entry]
                stop  = min(highs) * 1.001 if highs else ote_entry * 1.005
            elif ob:
                entry = ob.mid
                stop  = ob.high * 1.001
            elif fvg:
                entry = fvg.mid   # Consequent Encroachment (CE)
                highs = [df['high'].iloc[i] for i in swing_highs if df['high'].iloc[i] > fvg.high]
                stop  = min(highs) * 1.001 if highs else fvg.high * 1.005
            else:
                entry = current_price
                highs = [df['high'].iloc[i] for i in swing_highs]
                stop  = min([h for h in highs if h > current_price],
                             default=current_price * 1.01) * 1.001

        return round(entry, 6), round(stop, 6)

    # ── Target Computation ────────────────────────────────────────────────────

    def compute_targets(self,
                         signal:     SignalType,
                         entry:      float,
                         stop:       float,
                         liq_levels: list[LiquidityLevel],
                         pd_data:    dict,
                         pdh:        Optional[float] = None,
                         pdl:        Optional[float] = None,
                         ) -> tuple[float, float, float]:
        """
        TP1: 1.5R  (partial exit 40%)
        TP2: nearest opposing liquidity pool — PDH/PDL preferred over swing highs (Ch. 27)
        TP3: 1:3R or range extreme
        """
        risk = abs(entry - stop)

        if signal == SignalType.LONG:
            tp1 = entry + risk * 1.5
            # TP2: prefer PDH over generic BSL
            candidates = []
            if pdh and pdh > entry:
                candidates.append(pdh)
            bsl = [l.price for l in liq_levels if l.kind == 'BSL' and l.price > entry]
            candidates.extend(bsl)
            tp2 = min(candidates) if candidates else entry + risk * 2.5
            rng_high = pd_data.get('range_high', entry + risk * 3) if pd_data else entry + risk * 3
            tp3 = max(rng_high, entry + risk * 3)

        else:  # SHORT
            tp1 = entry - risk * 1.5
            candidates = []
            if pdl and pdl < entry:
                candidates.append(pdl)
            ssl = [l.price for l in liq_levels if l.kind == 'SSL' and l.price < entry]
            candidates.extend(ssl)
            tp2 = max(candidates) if candidates else entry - risk * 2.5
            rng_low = pd_data.get('range_low', entry - risk * 3) if pd_data else entry - risk * 3
            tp3 = min(rng_low, entry - risk * 3)

        rr = abs(tp2 - entry) / risk if risk > 0 else 0
        return round(tp1, 6), round(tp2, 6), round(tp3, 6)

    # ── Full Analysis ──────────────────────────────────────────────────────────

    def analyze(self, ltf_df: pd.DataFrame,
                 htf_df: pd.DataFrame) -> TradeSetup:
        """
        Full ICT/SMC analysis (v2).
        ltf_df: lower timeframe (1H or 15m)
        htf_df: higher timeframe (4H or 1H)
        Returns: TradeSetup with signal, entry, stops, targets, probability.
        """
        current_price = float(ltf_df['close'].iloc[-1])

        # ── HTF Analysis ──────────────────────────────────────────────────────
        htf_sh, htf_sl = self.find_swings(htf_df)
        htf_swings     = self.classify_swings(htf_df, htf_sh, htf_sl)
        htf_bias       = self.market_bias(htf_swings)

        # ── LTF Analysis ──────────────────────────────────────────────────────
        ltf_sh, ltf_sl   = self.find_swings(ltf_df)
        ltf_swings       = self.classify_swings(ltf_df, ltf_sh, ltf_sl)
        ltf_bias         = self.market_bias(ltf_swings)
        structure_breaks = self.find_structure_breaks(ltf_df, ltf_swings)

        # ── SMC Components ────────────────────────────────────────────────────
        obs        = self.find_order_blocks(ltf_df, ltf_sh, ltf_sl)
        fvgs       = self.find_fvgs(ltf_df)
        liq        = self.find_liquidity(ltf_df, ltf_sh, ltf_sl)
        pd_d       = self.pd_zones(ltf_df, ltf_sh, ltf_sl)
        vol        = self.volume_analysis(ltf_df)

        # ── v2: New Components ────────────────────────────────────────────────
        breakers   = self.find_breaker_blocks(ltf_df, ltf_swings)
        mitigations = self.find_mitigation_blocks(ltf_df, ltf_swings)
        pdh, pdl   = self.find_pdh_pdl(ltf_df)
        session    = self.get_session_ranges(ltf_df)
        po3_phase  = self.detect_power_of_three(ltf_df, session)

        # Annotate OBs that have an FVG stacked in/near them
        for ob in obs:
            for fvg in fvgs:
                if fvg.kind == ob.kind and fvg.low <= ob.high and fvg.high >= ob.low:
                    ob.has_fvg = True
                    break

        # ── Signal Direction ──────────────────────────────────────────────────
        if htf_bias == Bias.BULLISH and ltf_bias in (Bias.BULLISH, Bias.NEUTRAL):
            signal = SignalType.LONG
        elif htf_bias == Bias.BEARISH and ltf_bias in (Bias.BEARISH, Bias.NEUTRAL):
            signal = SignalType.SHORT
        elif htf_bias == Bias.BULLISH and ltf_bias == Bias.BEARISH:
            recent_choch = [s for s in structure_breaks[-3:] if 'CHoCH_bull' in s.kind]
            signal = SignalType.LONG if recent_choch else SignalType.WAIT
        elif htf_bias == Bias.BEARISH and ltf_bias == Bias.BULLISH:
            recent_choch = [s for s in structure_breaks[-3:] if 'CHoCH_bear' in s.kind]
            signal = SignalType.SHORT if recent_choch else SignalType.WAIT
        else:
            signal = SignalType.WAIT

        if signal == SignalType.WAIT:
            return TradeSetup(
                signal=SignalType.WAIT,
                entry_price=current_price,
                stop_loss=0, take_profit_1=0, take_profit_2=0, take_profit_3=0,
                risk_reward=0, probability=0, confluence_score=0,
                reasons=["No high-probability confluence detected — standing aside"],
                warnings=["HTF and LTF biases are conflicting or neutral"],
                htf_bias=htf_bias, ltf_bias=ltf_bias,
                pdh=pdh, pdl=pdl,
                asian_high=session.get('asian_high'),
                asian_low=session.get('asian_low'),
                midnight_open=session.get('midnight_open'),
                po3_phase=po3_phase,
            )

        # ── Select Best OB ────────────────────────────────────────────────────
        target_obs = [
            o for o in obs
            if o.kind == ('bullish' if signal == SignalType.LONG else 'bearish')
            and (
                (signal == SignalType.LONG  and o.high < current_price * (1 - self.MIN_OB_DISTANCE_PCT))
                or
                (signal == SignalType.SHORT and o.low  > current_price * (1 + self.MIN_OB_DISTANCE_PCT))
            )
        ]
        # Prefer OBs with FVG stacked (highest quality), then by strength
        best_ob = max(target_obs, key=lambda o: (o.has_fvg, o.strength)) if target_obs else None

        # ── Select Best FVG ───────────────────────────────────────────────────
        target_fvgs = [
            f for f in fvgs
            if f.kind == ('bullish' if signal == SignalType.LONG else 'bearish')
            and (
                (signal == SignalType.LONG  and f.high < current_price * (1 - self.MIN_OB_DISTANCE_PCT))
                or
                (signal == SignalType.SHORT and f.low  > current_price * (1 + self.MIN_OB_DISTANCE_PCT))
            )
        ]
        # Prefer displacement-confirmed FVGs, then by size
        best_fvg = max(target_fvgs,
                       key=lambda f: (f.displacement, f.size_pct)) if target_fvgs else None

        # ── Select Best Breaker Block ─────────────────────────────────────────
        target_breakers = [
            b for b in breakers
            if b.kind == ('bullish' if signal == SignalType.LONG else 'bearish')
        ]
        best_breaker = max(target_breakers, key=lambda b: b.strength) if target_breakers else None

        # ── Select Best Mitigation Block ──────────────────────────────────────
        target_mit = [
            m for m in mitigations
            if m.kind == ('bullish' if signal == SignalType.LONG else 'bearish')
        ]
        best_mit = target_mit[-1] if target_mit else None  # most recent

        # ── Nearest Opposing Liquidity ────────────────────────────────────────
        if signal == SignalType.LONG:
            targets = [l for l in liq if l.kind == 'BSL' and l.price > current_price]
        else:
            targets = [l for l in liq if l.kind == 'SSL' and l.price < current_price]
        nearest_liq = (sorted(targets, key=lambda l: abs(l.price - current_price))[0]
                       if targets else None)

        # ── Score ─────────────────────────────────────────────────────────────
        score, reasons, warnings = self.score_setup(
            ltf_bias, htf_bias, best_ob, best_fvg, nearest_liq, pd_d, vol,
            structure_breaks, signal,
            breaker=best_breaker, mitigation=best_mit,
            pdh=pdh, pdl=pdl, current_price=current_price
        )

        # ── Killzone Check ────────────────────────────────────────────────────
        in_kz, kz_name = self.in_killzone(ltf_df)
        if in_kz:
            reasons.append(f"✓ Inside ICT killzone: {kz_name}")
        else:
            warnings.append(
                "Outside ICT killzone — London Open (02–05 EST), NY AM (07–10 EST), "
                "and Asian (20–00 EST) are the high-probability windows. Consider waiting."
            )

        # ── DOL Identification ────────────────────────────────────────────────
        draw_on_liq = self.identify_draw_on_liquidity(
            signal, liq, fvgs, pdh, pdl, current_price
        )

        # ── Entry & Stops ─────────────────────────────────────────────────────
        entry, stop = self.compute_entry(
            signal, best_ob, best_fvg, current_price,
            ltf_sh, ltf_sl, ltf_df,
            pd_data=pd_d,
            breaker=best_breaker,
            mitigation=best_mit,
        )

        tp1, tp2, tp3 = self.compute_targets(
            signal, entry, stop, liq, pd_d, pdh=pdh, pdl=pdl
        )
        risk = abs(entry - stop)
        rr   = round(abs(tp2 - entry) / risk, 2) if risk > 0 else 0

        # ── Probability ───────────────────────────────────────────────────────
        prob = 35 + (score / 100) * 50  # 35–85% range
        if not in_kz:             prob -= 10
        if best_ob and best_ob.strength > 0.5: prob += 5
        if best_breaker:          prob += 3   # breaker = higher-quality setup
        if best_ob and best_ob.has_fvg:       prob += 2   # OB+FVG stack
        prob = round(min(max(prob, 20), 85), 1)

        # ── A+ Gate ───────────────────────────────────────────────────────────
        if score < self.MIN_CONFLUENCE_SCORE or rr < 2.0:
            gate_reason = (
                f"Confluence score {score:.0f}/100 below A+ threshold ({self.MIN_CONFLUENCE_SCORE})"
                if score < self.MIN_CONFLUENCE_SCORE
                else f"Risk/reward {rr:.1f} below minimum 2.0 — setup not worth the risk"
            )
            return TradeSetup(
                signal=SignalType.WAIT,
                entry_price=current_price,
                stop_loss=0, take_profit_1=0, take_profit_2=0, take_profit_3=0,
                risk_reward=0, probability=prob, confluence_score=score,
                reasons=reasons,
                warnings=[gate_reason] + warnings,
                htf_bias=htf_bias, ltf_bias=ltf_bias,
                pdh=pdh, pdl=pdl,
                asian_high=session.get('asian_high'),
                asian_low=session.get('asian_low'),
                midnight_open=session.get('midnight_open'),
                po3_phase=po3_phase,
                draw_on_liquidity=draw_on_liq,
            )

        vol_confirmed = (
            vol.get('available', False) and
            vol.get('is_high_vol', False) and
            vol.get('bias', 'NEUTRAL') != 'NEUTRAL'
        )

        return TradeSetup(
            signal=signal,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_reward=rr,
            probability=prob,
            confluence_score=score,
            reasons=reasons,
            warnings=warnings,
            ob=best_ob,
            fvg=best_fvg,
            nearest_liquidity=nearest_liq,
            htf_bias=htf_bias,
            ltf_bias=ltf_bias,
            volume_confirmed=vol_confirmed,
            # v2 fields
            breaker=best_breaker,
            mitigation=best_mit,
            draw_on_liquidity=draw_on_liq,
            po3_phase=po3_phase,
            pdh=pdh,
            pdl=pdl,
            asian_high=session.get('asian_high'),
            asian_low=session.get('asian_low'),
            midnight_open=session.get('midnight_open'),
        )