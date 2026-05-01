"""
ICT/SMC Crypto Trade Assistant
================================
Analyzes ETHUSDT / BTCUSDT using ICT/SMC methodology:
- Market Structure (BOS / CHoCH)
- Order Blocks (Bullish / Bearish)
- Fair Value Gaps (FVG / Imbalances)
- Liquidity Sweeps (Buy-side / Sell-side)
- Premium / Discount Zones (Fibonacci)
- Volume Confirmation (delta, volume profile)
- Multi-timeframe confluence (HTF 4H/1H → LTF 1H/15m)
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
    NEUTRAL = "NEUTRAL"


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # 'HH', 'LH', 'HL', 'LL'
    timestamp: pd.Timestamp


@dataclass
class OrderBlock:
    high: float
    low: float
    mid: float
    kind: str          # 'bullish' | 'bearish'
    index: int
    mitigated: bool = False
    volume: float = 0.0
    strength: float = 0.0   # 0-1 based on displacement after OB


@dataclass
class FairValueGap:
    high: float
    low: float
    mid: float
    kind: str          # 'bullish' | 'bearish'
    index: int
    filled: bool = False
    size_pct: float = 0.0


@dataclass
class LiquidityLevel:
    price: float
    kind: str          # 'BSL' (buy-side) | 'SSL' (sell-side)
    index: int
    swept: bool = False
    touches: int = 1


@dataclass
class StructurePoint:
    price: float
    kind: str          # 'BOS_bull' | 'BOS_bear' | 'CHoCH_bull' | 'CHoCH_bear'
    index: int


@dataclass
class TradeSetup:
    signal: SignalType
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float
    probability: float          # 0-100
    confluence_score: float     # 0-100
    reasons: list[str]
    warnings: list[str]
    ob: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    nearest_liquidity: Optional[LiquidityLevel] = None
    htf_bias: Bias = Bias.NEUTRAL
    ltf_bias: Bias = Bias.NEUTRAL
    volume_confirmed: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Core SMC Engine
# ─────────────────────────────────────────────────────────────────────────────

class SMCEngine:
    """
    Pure computation engine — takes OHLCV DataFrames, returns structured SMC analysis.
    All prices are raw floats. No I/O here.
    """

    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback

    # ── Swing Detection ──────────────────────────────────────────────────────

    def find_swings(self, df: pd.DataFrame) -> tuple[list[int], list[int]]:
        """Return indices of swing highs and swing lows using fractals."""
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
                         swing_lows: list[int]) -> list[SwingPoint]:
        """Classify swing points as HH/LH/HL/LL."""
        points: list[SwingPoint] = []

        prev_h = prev_l = None
        for i in sorted(swing_highs + swing_lows):
            ts = df.index[i]
            if i in swing_highs:
                price = df['high'].iloc[i]
                if prev_h is None:
                    kind = 'HH'
                else:
                    kind = 'HH' if price > prev_h else 'LH'
                prev_h = price
                points.append(SwingPoint(i, price, kind, ts))
            else:
                price = df['low'].iloc[i]
                if prev_l is None:
                    kind = 'HL'
                else:
                    kind = 'HL' if price > prev_l else 'LL'
                prev_l = price
                points.append(SwingPoint(i, price, kind, ts))

        return points

    def market_bias(self, swings: list[SwingPoint]) -> Bias:
        """Determine HTF bias from last 4 swing classifications."""
        if len(swings) < 4:
            return Bias.NEUTRAL
        recent = [s.kind for s in swings[-6:]]
        bull = recent.count('HH') + recent.count('HL')
        bear = recent.count('LH') + recent.count('LL')
        if bull > bear:
            return Bias.BULLISH
        elif bear > bull:
            return Bias.BEARISH
        return Bias.NEUTRAL

    # ── Market Structure ─────────────────────────────────────────────────────

    def find_structure_breaks(self, df: pd.DataFrame,
                               swings: list[SwingPoint]) -> list[StructurePoint]:
        """Detect BOS and CHoCH from close prices vs swing levels."""
        breaks: list[StructurePoint] = []
        highs = [s for s in swings if s.kind in ('HH', 'LH')]
        lows  = [s for s in swings if s.kind in ('HL', 'LL')]

        bias = Bias.NEUTRAL

        for i in range(len(df)):
            close = df['close'].iloc[i]

            # Bullish break above last swing high
            if highs:
                last_h = highs[-1]
                if close > last_h.price and i > last_h.index:
                    kind = 'BOS_bull' if bias == Bias.BULLISH else 'CHoCH_bull'
                    breaks.append(StructurePoint(last_h.price, kind, i))
                    bias = Bias.BULLISH

            # Bearish break below last swing low
            if lows:
                last_l = lows[-1]
                if close < last_l.price and i > last_l.index:
                    kind = 'BOS_bear' if bias == Bias.BEARISH else 'CHoCH_bear'
                    breaks.append(StructurePoint(last_l.price, kind, i))
                    bias = Bias.BEARISH

        return breaks

    # ── Order Blocks ─────────────────────────────────────────────────────────

    def find_order_blocks(self, df: pd.DataFrame,
                           swing_highs: list[int],
                           swing_lows: list[int]) -> list[OrderBlock]:
        """
        Bullish OB: last bearish candle before a bullish impulse that creates a new swing high.
        Bearish OB: last bullish candle before a bearish impulse that creates a new swing low.
        """
        obs: list[OrderBlock] = []

        # Bullish OBs — look for bearish candles before upswing
        for sh in swing_highs:
            # Find last bearish candle before this swing high
            for j in range(sh - 1, max(sh - 10, 0), -1):
                o, c = df['open'].iloc[j], df['close'].iloc[j]
                if c < o:  # bearish candle
                    vol = df['volume'].iloc[j] if 'volume' in df else 0
                    # Displacement: candle body after OB vs OB size
                    ob_size = o - c
                    post_move = df['high'].iloc[sh] - df['high'].iloc[j]
                    strength = min(post_move / max(ob_size, 1e-8), 1.0) * 0.5
                    strength = round(min(strength, 1.0), 2)
                    ob = OrderBlock(
                        high=max(o, c), low=min(o, c),
                        mid=(o + c) / 2, kind='bullish',
                        index=j, volume=vol, strength=strength
                    )
                    obs.append(ob)
                    break

        # Bearish OBs — look for bullish candles before downswing
        for sl in swing_lows:
            for j in range(sl - 1, max(sl - 10, 0), -1):
                o, c = df['open'].iloc[j], df['close'].iloc[j]
                if c > o:  # bullish candle
                    vol = df['volume'].iloc[j] if 'volume' in df else 0
                    ob_size = c - o
                    post_move = df['low'].iloc[j] - df['low'].iloc[sl]
                    strength = min(post_move / max(ob_size, 1e-8), 1.0) * 0.5
                    strength = round(min(strength, 1.0), 2)
                    ob = OrderBlock(
                        high=max(o, c), low=min(o, c),
                        mid=(o + c) / 2, kind='bearish',
                        index=j, volume=vol, strength=strength
                    )
                    obs.append(ob)
                    break

        # Mark mitigated OBs (price has returned into them)
        current_price = df['close'].iloc[-1]
        for ob in obs:
            if ob.kind == 'bullish' and current_price < ob.low:
                ob.mitigated = True
            elif ob.kind == 'bearish' and current_price > ob.high:
                ob.mitigated = True

        return [ob for ob in obs if not ob.mitigated]

    # ── Fair Value Gaps ───────────────────────────────────────────────────────

    def find_fvgs(self, df: pd.DataFrame, min_size_pct: float = 0.03) -> list[FairValueGap]:
        """
        FVG: 3-candle imbalance.
        Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
        Bearish FVG: candle[i-2].low  > candle[i].high (gap down)
        """
        fvgs: list[FairValueGap] = []
        current_price = df['close'].iloc[-1]

        for i in range(2, len(df)):
            c0h = df['high'].iloc[i-2]
            c0l = df['low'].iloc[i-2]
            c2h = df['high'].iloc[i]
            c2l = df['low'].iloc[i]

            # Bullish FVG
            if c2l > c0h:
                size_pct = (c2l - c0h) / c0h * 100
                if size_pct >= min_size_pct:
                    filled = current_price <= c0h
                    fvgs.append(FairValueGap(
                        high=c2l, low=c0h, mid=(c2l + c0h) / 2,
                        kind='bullish', index=i, filled=filled,
                        size_pct=round(size_pct, 4)
                    ))

            # Bearish FVG
            if c2h < c0l:
                size_pct = (c0l - c2h) / c0l * 100
                if size_pct >= min_size_pct:
                    filled = current_price >= c0l
                    fvgs.append(FairValueGap(
                        high=c0l, low=c2h, mid=(c0l + c2h) / 2,
                        kind='bearish', index=i, filled=filled,
                        size_pct=round(size_pct, 4)
                    ))

        return [f for f in fvgs if not f.filled]

    # ── Liquidity ─────────────────────────────────────────────────────────────

    def find_liquidity(self, df: pd.DataFrame,
                        swing_highs: list[int],
                        swing_lows: list[int]) -> list[LiquidityLevel]:
        """
        BSL (buy-side liquidity): above equal highs / swing highs  → targets for shorts
        SSL (sell-side liquidity): below equal lows  / swing lows  → targets for longs
        """
        levels: list[LiquidityLevel] = []
        current_price = df['close'].iloc[-1]
        tol = current_price * 0.001  # 0.1% tolerance for "equal" levels

        # Equal highs → BSL
        high_prices = [(i, df['high'].iloc[i]) for i in swing_highs]
        for i in range(len(high_prices)):
            for j in range(i+1, len(high_prices)):
                if abs(high_prices[i][1] - high_prices[j][1]) < tol:
                    swept = current_price > high_prices[j][1]
                    lev = LiquidityLevel(
                        price=max(high_prices[i][1], high_prices[j][1]),
                        kind='BSL', index=high_prices[j][0],
                        swept=swept, touches=2
                    )
                    levels.append(lev)

        # Single swing highs above current price → BSL
        for idx, price in high_prices:
            if price > current_price * 1.002:
                levels.append(LiquidityLevel(price=price, kind='BSL', index=idx))

        # Equal lows → SSL
        low_prices = [(i, df['low'].iloc[i]) for i in swing_lows]
        for i in range(len(low_prices)):
            for j in range(i+1, len(low_prices)):
                if abs(low_prices[i][1] - low_prices[j][1]) < tol:
                    swept = current_price < low_prices[j][1]
                    lev = LiquidityLevel(
                        price=min(low_prices[i][1], low_prices[j][1]),
                        kind='SSL', index=low_prices[j][0],
                        swept=swept, touches=2
                    )
                    levels.append(lev)

        # Single swing lows below current price → SSL
        for idx, price in low_prices:
            if price < current_price * 0.998:
                levels.append(LiquidityLevel(price=price, kind='SSL', index=idx))

        return [l for l in levels if not l.swept]

    # ── Premium / Discount ────────────────────────────────────────────────────

    def pd_zones(self, df: pd.DataFrame,
                  swing_highs: list[int],
                  swing_lows: list[int]) -> dict:
        """
        Fibonacci-based premium/discount zones from recent range.
        Equilibrium = 0.5, Discount < 0.5, Premium > 0.5
        """
        if not swing_highs or not swing_lows:
            return {}

        # Use most recent significant range
        last_n = 50
        sub = df.iloc[-last_n:]
        rng_high = sub['high'].max()
        rng_low  = sub['low'].min()
        rng      = rng_high - rng_low
        if rng == 0:
            return {}

        current = df['close'].iloc[-1]
        fib_pos  = (current - rng_low) / rng  # 0 = at low, 1 = at high

        return {
            'range_high':   rng_high,
            'range_low':    rng_low,
            'equilibrium':  rng_low + rng * 0.5,
            'ote_high':     rng_low + rng * 0.79,   # OTE zone
            'ote_low':      rng_low + rng * 0.618,
            'fib_705':      rng_low + rng * 0.705,  # Sweet spot
            'fib_pos':      round(fib_pos, 4),
            'zone':         'DISCOUNT' if fib_pos < 0.5 else 'PREMIUM'
        }

    # ── Volume Analysis ───────────────────────────────────────────────────────

    def volume_analysis(self, df: pd.DataFrame) -> dict:
        """
        Compute volume metrics:
        - Relative volume vs 20-bar average
        - Volume trend (increasing / decreasing into structure)
        - Delta proxy (close position within candle range × volume)
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return {'available': False}

        vols = df['volume']
        avg_vol = vols.rolling(20).mean()
        last_vol = vols.iloc[-1]
        rel_vol = last_vol / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1.0

        # Delta proxy: (close - low) / (high - low) → 1 = full buy, 0 = full sell
        def delta_proxy(row):
            rng = row['high'] - row['low']
            if rng == 0:
                return 0.5
            return (row['close'] - row['low']) / rng

        df2 = df.copy()
        df2['delta'] = df2.apply(delta_proxy, axis=1)
        df2['vol_delta'] = df2['delta'] * df2['volume']

        recent = df2.iloc[-5:]
        cum_delta = recent['vol_delta'].sum()
        total_vol = recent['volume'].sum()
        avg_delta = cum_delta / total_vol if total_vol > 0 else 0.5

        # Volume trend
        v5  = vols.iloc[-5:].mean()
        v20 = vols.iloc[-20:].mean()
        trend = 'INCREASING' if v5 > v20 * 1.1 else ('DECREASING' if v5 < v20 * 0.9 else 'NEUTRAL')

        return {
            'available':  True,
            'rel_vol':    round(float(rel_vol), 2),
            'avg_delta':  round(float(avg_delta), 3),
            'vol_trend':  trend,
            'is_high_vol': rel_vol > 1.3,
            'bias':       'BULLISH' if avg_delta > 0.55 else ('BEARISH' if avg_delta < 0.45 else 'NEUTRAL')
        }

    # ── Setup Scoring ─────────────────────────────────────────────────────────

    def score_setup(self,
                    bias: Bias,
                    htf_bias: Bias,
                    ob: Optional[OrderBlock],
                    fvg: Optional[FairValueGap],
                    liq: Optional[LiquidityLevel],
                    pd_data: dict,
                    vol: dict,
                    structure_breaks: list[StructurePoint],
                    signal: SignalType) -> tuple[float, list[str], list[str]]:
        """
        Score confluence 0-100.
        Returns (score, reasons[], warnings[])
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        # ── HTF alignment (25 pts) ──
        if htf_bias == Bias.BULLISH and signal == SignalType.LONG:
            score += 25
            reasons.append("HTF bias is BULLISH — trading with the trend")
        elif htf_bias == Bias.BEARISH and signal == SignalType.SHORT:
            score += 25
            reasons.append("HTF bias is BEARISH — trading with the trend")
        elif htf_bias == Bias.NEUTRAL:
            score += 10
            warnings.append("HTF bias is NEUTRAL — reduced edge")
        else:
            warnings.append("Counter-trend trade — HTF bias opposes signal")

        # ── LTF structure alignment (15 pts) ──
        if bias == Bias.BULLISH and signal == SignalType.LONG:
            score += 15
            reasons.append("LTF market structure: higher highs and higher lows")
        elif bias == Bias.BEARISH and signal == SignalType.SHORT:
            score += 15
            reasons.append("LTF market structure: lower highs and lower lows")
        elif bias == Bias.NEUTRAL:
            score += 5

        # ── Order Block (20 pts) ──
        if ob:
            ob_score = 10 + int(ob.strength * 10)
            score += ob_score
            reasons.append(
                f"{'Bullish' if ob.kind=='bullish' else 'Bearish'} Order Block "
                f"[{ob.low:.2f} – {ob.high:.2f}] "
                f"strength={ob.strength:.0%}"
            )
            if ob.volume > 0:
                reasons.append(f"OB backed by volume: {ob.volume:,.0f}")

        # ── FVG (15 pts) ──
        if fvg:
            score += 15
            reasons.append(
                f"{'Bullish' if fvg.kind=='bullish' else 'Bearish'} FVG "
                f"[{fvg.low:.2f} – {fvg.high:.2f}] "
                f"size={fvg.size_pct:.3f}%"
            )

        # ── Liquidity target (10 pts) ──
        if liq:
            score += 10
            kind_name = "Buy-side" if liq.kind == 'BSL' else "Sell-side"
            reasons.append(
                f"{kind_name} liquidity target at {liq.price:.2f} "
                f"({'double top/bottom' if liq.touches > 1 else 'single swing'})"
            )

        # ── Premium/Discount zone (10 pts) ──
        if pd_data:
            fib = pd_data.get('fib_pos', 0.5)
            zone = pd_data.get('zone', '')
            if signal == SignalType.LONG and zone == 'DISCOUNT':
                score += 10
                reasons.append(f"Price in DISCOUNT zone (fib={fib:.2%}) — optimal long entry area")
            elif signal == SignalType.SHORT and zone == 'PREMIUM':
                score += 10
                reasons.append(f"Price in PREMIUM zone (fib={fib:.2%}) — optimal short entry area")
            elif signal == SignalType.LONG and zone == 'PREMIUM':
                warnings.append(f"Buying in PREMIUM zone (fib={fib:.2%}) — suboptimal entry")
            elif signal == SignalType.SHORT and zone == 'DISCOUNT':
                warnings.append(f"Shorting in DISCOUNT zone (fib={fib:.2%}) — suboptimal entry")

        # ── Volume confirmation (5 pts) ──
        if vol.get('available'):
            rv = vol.get('rel_vol', 1.0)
            vbias = vol.get('bias', 'NEUTRAL')
            if rv > 1.3:
                score += 3
                reasons.append(f"High relative volume ({rv:.1f}x average)")
            if vbias == 'BULLISH' and signal == SignalType.LONG:
                score += 2
                reasons.append("Volume delta confirms bullish pressure")
            elif vbias == 'BEARISH' and signal == SignalType.SHORT:
                score += 2
                reasons.append("Volume delta confirms bearish pressure")
            elif vbias != 'NEUTRAL':
                warnings.append(f"Volume delta ({vbias}) diverges from signal")
        else:
            warnings.append("Volume data unavailable — unconfirmed setup")

        # Recent CHoCH bonus
        recent_choch = [s for s in structure_breaks[-5:] if 'CHoCH' in s.kind]
        if recent_choch:
            choch = recent_choch[-1]
            if 'bull' in choch.kind and signal == SignalType.LONG:
                score += 5
                reasons.append("Recent bullish CHoCH — structure flip confirmed")
            elif 'bear' in choch.kind and signal == SignalType.SHORT:
                score += 5
                reasons.append("Recent bearish CHoCH — structure flip confirmed")

        score = round(min(score, 100), 1)
        return score, reasons, warnings

    # ── Entry Computation ─────────────────────────────────────────────────────

    def compute_entry(self,
                       signal: SignalType,
                       ob: Optional[OrderBlock],
                       fvg: Optional[FairValueGap],
                       current_price: float,
                       swing_highs: list[int],
                       swing_lows: list[int],
                       df: pd.DataFrame) -> tuple[float, float]:
        """
        Returns (entry_price, stop_loss).
        Entry: OB mid > FVG mid > nearest swing + buffer
        Stop: just below OB low (long) or above OB high (short)
        """
        if signal == SignalType.LONG:
            if ob:
                entry = ob.mid
                stop  = ob.low * 0.999
            elif fvg:
                entry = fvg.mid
                # Nearest swing low for stop
                lows = [df['low'].iloc[i] for i in swing_lows if df['low'].iloc[i] < fvg.low]
                stop = max(lows) * 0.999 if lows else fvg.low * 0.995
            else:
                entry = current_price
                lows = [df['low'].iloc[i] for i in swing_lows]
                stop = max([l for l in lows if l < current_price], default=current_price * 0.99) * 0.999
        else:  # SHORT
            if ob:
                entry = ob.mid
                stop  = ob.high * 1.001
            elif fvg:
                entry = fvg.mid
                highs = [df['high'].iloc[i] for i in swing_highs if df['high'].iloc[i] > fvg.high]
                stop = min(highs) * 1.001 if highs else fvg.high * 1.005
            else:
                entry = current_price
                highs = [df['high'].iloc[i] for i in swing_highs]
                stop = min([h for h in highs if h > current_price], default=current_price * 1.01) * 1.001

        return round(entry, 6), round(stop, 6)

    def compute_targets(self,
                         signal: SignalType,
                         entry: float,
                         stop: float,
                         liq_levels: list[LiquidityLevel],
                         pd_data: dict) -> tuple[float, float, float]:
        """
        TP1: 1:1.5 R
        TP2: Nearest opposing liquidity pool
        TP3: 1:3 R or range extreme
        """
        risk = abs(entry - stop)

        if signal == SignalType.LONG:
            tp1 = entry + risk * 1.5
            # TP2: nearest BSL above entry
            bsl = [l.price for l in liq_levels if l.kind == 'BSL' and l.price > entry]
            tp2 = min(bsl) if bsl else entry + risk * 2.5
            # TP3: range high or 1:3
            rng_high = pd_data.get('range_high', entry + risk * 3)
            tp3 = max(rng_high, entry + risk * 3)
        else:
            tp1 = entry - risk * 1.5
            ssl = [l.price for l in liq_levels if l.kind == 'SSL' and l.price < entry]
            tp2 = max(ssl) if ssl else entry - risk * 2.5
            rng_low = pd_data.get('range_low', entry - risk * 3)
            tp3 = min(rng_low, entry - risk * 3)

        rr = abs(tp2 - entry) / risk if risk > 0 else 0

        return round(tp1, 6), round(tp2, 6), round(tp3, 6)

    # ── Full Analysis ──────────────────────────────────────────────────────────

    def analyze(self, ltf_df: pd.DataFrame,
                 htf_df: pd.DataFrame) -> TradeSetup:
        """
        Full ICT/SMC analysis.
        ltf_df: lower timeframe (1H or 15m)
        htf_df: higher timeframe (4H or 1H)
        Returns: TradeSetup with signal, entry, stops, targets, probability
        """
        current_price = ltf_df['close'].iloc[-1]

        # ── HTF Analysis ──
        htf_sh, htf_sl = self.find_swings(htf_df)
        htf_swings = self.classify_swings(htf_df, htf_sh, htf_sl)
        htf_bias = self.market_bias(htf_swings)

        # ── LTF Analysis ──
        ltf_sh, ltf_sl = self.find_swings(ltf_df)
        ltf_swings = self.classify_swings(ltf_df, ltf_sh, ltf_sl)
        ltf_bias = self.market_bias(ltf_swings)
        structure_breaks = self.find_structure_breaks(ltf_df, ltf_swings)

        # ── SMC Components ──
        obs   = self.find_order_blocks(ltf_df, ltf_sh, ltf_sl)
        fvgs  = self.find_fvgs(ltf_df)
        liq   = self.find_liquidity(ltf_df, ltf_sh, ltf_sl)
        pd_d  = self.pd_zones(ltf_df, ltf_sh, ltf_sl)
        vol   = self.volume_analysis(ltf_df)

        # ── Determine Signal Direction ──
        if htf_bias == Bias.BULLISH and ltf_bias in (Bias.BULLISH, Bias.NEUTRAL):
            signal = SignalType.LONG
        elif htf_bias == Bias.BEARISH and ltf_bias in (Bias.BEARISH, Bias.NEUTRAL):
            signal = SignalType.SHORT
        elif htf_bias == Bias.BULLISH and ltf_bias == Bias.BEARISH:
            # HTF bull, LTF bear → wait for CHoCH bullish
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
                htf_bias=htf_bias, ltf_bias=ltf_bias
            )

        # ── Find Best OB ──
        target_obs = [o for o in obs if o.kind == ('bullish' if signal == SignalType.LONG else 'bearish')]
        # Prefer closest OB to current price
        if target_obs:
            best_ob = sorted(
                target_obs,
                key=lambda o: abs(o.mid - current_price)
            )[0]
        else:
            best_ob = None

        # ── Find Best FVG ──
        target_fvgs = [f for f in fvgs if f.kind == ('bullish' if signal == SignalType.LONG else 'bearish')]
        if target_fvgs:
            best_fvg = sorted(target_fvgs, key=lambda f: abs(f.mid - current_price))[0]
        else:
            best_fvg = None

        # ── Find Nearest Opposing Liquidity ──
        if signal == SignalType.LONG:
            targets = [l for l in liq if l.kind == 'BSL' and l.price > current_price]
        else:
            targets = [l for l in liq if l.kind == 'SSL' and l.price < current_price]
        nearest_liq = sorted(targets, key=lambda l: abs(l.price - current_price))[0] if targets else None

        # ── Score ──
        score, reasons, warnings = self.score_setup(
            ltf_bias, htf_bias, best_ob, best_fvg, nearest_liq, pd_d, vol,
            structure_breaks, signal
        )

        # ── Entry & Stops ──
        entry, stop = self.compute_entry(
            signal, best_ob, best_fvg, current_price,
            ltf_sh, ltf_sl, ltf_df
        )

        tp1, tp2, tp3 = self.compute_targets(signal, entry, stop, liq, pd_d)
        risk = abs(entry - stop)
        rr = round(abs(tp2 - entry) / risk, 2) if risk > 0 else 0

        # ── Probability Estimate ──
        # Bayesian-style: base rate 45%, adjust for confluence
        prob = 35 + (score / 100) * 50  # 35-85% range
        if vol.get('available') and not vol.get('is_high_vol'):
            prob -= 5  # penalise low-volume setups
        if best_ob and best_ob.strength > 0.5:
            prob += 5
        prob = round(min(max(prob, 20), 85), 1)

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
            volume_confirmed=vol_confirmed
        )
