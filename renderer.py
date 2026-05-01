"""
Terminal renderer for ICT/SMC Trade Assistant.
Produces rich, readable trade reports in the terminal using ANSI escape codes.
No external dependencies required.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

from analyzer import TradeSetup, SignalType, Bias, OrderBlock, FairValueGap, LiquidityLevel


# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

# Foreground
WHITE   = "\033[97m"
GRAY    = "\033[90m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"

# Backgrounds
BG_RED    = "\033[41m"
BG_GREEN  = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE   = "\033[44m"
BG_DARK   = "\033[40m"

W = 72  # terminal width


def _supports_color() -> bool:
    return sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"


def c(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"{code}{text}{RESET}"


def line(char: str = "─", color: str = GRAY) -> str:
    return c(color, char * W)


def center(text: str, width: int = W) -> str:
    clean = _strip_ansi(text)
    pad = max(0, (width - len(clean)) // 2)
    return " " * pad + text


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r'\033\[[0-9;]*m', '', s)


def pad_right(text: str, width: int) -> str:
    clean_len = len(_strip_ansi(text))
    return text + " " * max(0, width - clean_len)


# ─────────────────────────────────────────────────────────────────────────────
# Report sections
# ─────────────────────────────────────────────────────────────────────────────

def render_header(symbol: str, tf_mode: str, current_price: float,
                   ticker: Optional[dict] = None) -> str:
    lines = []
    lines.append(line("═", CYAN))
    title = f"  ICT/SMC Trade Assistant  │  {symbol}  │  {tf_mode}"
    lines.append(c(BOLD + CYAN, title))
    lines.append(c(GRAY, f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC')}"))

    price_str = f"  Current Price: {c(BOLD + WHITE, f'{current_price:,.4f}')}"
    if ticker:
        pct = ticker.get('change_pct', 0)
        col = GREEN if pct >= 0 else RED
        price_str += f"  {c(col, f'{pct:+.2f}% 24h')}"
        vol = ticker.get('volume_24h', 0)
        price_str += f"  {c(GRAY, f'Vol: {vol/1e6:.1f}M USDT')}"
    lines.append(price_str)
    lines.append(line("═", CYAN))
    return "\n".join(lines)


def render_bias_bar(htf_bias: Bias, ltf_bias: Bias, tf_mode: str) -> str:
    tfs = tf_mode.split("-")
    htf_label = tfs[0] if len(tfs) == 2 else "HTF"
    ltf_label = tfs[1] if len(tfs) == 2 else "LTF"

    def bias_colored(b: Bias) -> str:
        if b == Bias.BULLISH: return c(BOLD + GREEN,   "▲ BULLISH")
        if b == Bias.BEARISH: return c(BOLD + RED,     "▼ BEARISH")
        return c(BOLD + YELLOW, "◆ NEUTRAL")

    htf_s = bias_colored(htf_bias)
    ltf_s = bias_colored(ltf_bias)

    lines = []
    lines.append(c(GRAY, "  Market Bias"))
    lines.append(f"  {c(GRAY, htf_label+':')}  {htf_s}    {c(GRAY, ltf_label+':')}  {ltf_s}")
    lines.append(line())
    return "\n".join(lines)


def render_signal_box(setup: TradeSetup) -> str:
    lines = []

    if setup.signal == SignalType.LONG:
        header = c(BOLD + BG_GREEN + WHITE, f"  ▲  LONG SIGNAL  ▲  ".center(W))
    elif setup.signal == SignalType.SHORT:
        header = c(BOLD + BG_RED + WHITE,   f"  ▼  SHORT SIGNAL  ▼  ".center(W))
    else:
        header = c(BOLD + BG_YELLOW + WHITE, f"  ◆  STAND ASIDE — NO SETUP  ◆  ".center(W))

    lines.append(header)

    if setup.signal == SignalType.WAIT:
        lines.append("")
        for w in setup.warnings:
            lines.append(c(YELLOW, f"  ⚠  {w}"))
        lines.append("")
        return "\n".join(lines)

    lines.append("")

    # ── Entry / Targets table ──
    is_long = setup.signal == SignalType.LONG
    entry_col = GREEN if is_long else RED
    sl_col    = RED   if is_long else GREEN

    def price_row(label: str, price: float, color: str, extra: str = "") -> str:
        lbl = c(GRAY, f"  {label:<18}")
        val = c(BOLD + color, f"{price:>14,.4f}  USDT")
        return lbl + val + (f"  {c(GRAY, extra)}" if extra else "")

    lines.append(price_row("Entry (Limit):",  setup.entry_price, entry_col, "← place limit order here"))
    lines.append(price_row("Stop Loss:",       setup.stop_loss,   sl_col))
    lines.append("")
    lines.append(price_row("Take Profit 1:",  setup.take_profit_1, CYAN, "1.5R — partial exit 40%"))
    lines.append(price_row("Take Profit 2:",  setup.take_profit_2, CYAN, "liquidity target — 40%"))
    lines.append(price_row("Take Profit 3:",  setup.take_profit_3, CYAN, "range extreme — 20%"))
    lines.append("")

    # Risk metrics
    risk_pct = abs(setup.entry_price - setup.stop_loss) / setup.entry_price * 100
    lines.append(
        f"  {c(GRAY, 'Risk:')}"
        f"  {c(YELLOW, f'{risk_pct:.3f}%')} from entry"
        f"   {c(GRAY, 'R:R')}  {c(BOLD + CYAN, f'1:{setup.risk_reward:.2f}')}"
    )
    lines.append("")
    return "\n".join(lines)


def render_probability_meter(score: float, probability: float) -> str:
    lines = []
    lines.append(line())

    # Confluence score bar
    filled = int(score / 100 * (W - 20))
    empty  = (W - 20) - filled
    col = GREEN if score >= 70 else (YELLOW if score >= 45 else RED)
    bar = c(col, "█" * filled) + c(GRAY, "░" * empty)
    lines.append(f"  Confluence   [{bar}]  {c(BOLD + col, f'{score:.0f}/100')}")

    # Win probability bar
    filled2 = int(probability / 100 * (W - 20))
    empty2  = (W - 20) - filled2
    col2 = GREEN if probability >= 65 else (YELLOW if probability >= 50 else RED)
    bar2 = c(col2, "█" * filled2) + c(GRAY, "░" * empty2)
    lines.append(f"  Win Prob     [{bar2}]  {c(BOLD + col2, f'{probability:.1f}%')}")

    lines.append("")
    return "\n".join(lines)


def render_components(setup: TradeSetup) -> str:
    lines = []
    lines.append(c(GRAY, "  SMC Components Detected"))
    lines.append("")

    def tag(label: str, text: str, color: str) -> str:
        return f"  {c(color, f'[{label}]'):<28}  {text}"

    # Order Block
    if setup.ob:
        ob = setup.ob
        kind_col = GREEN if ob.kind == 'bullish' else RED
        strength_col = GREEN if ob.strength > 0.5 else YELLOW
        lines.append(tag(
            "ORDER BLOCK",
            f"{c(kind_col, ob.kind.upper())}  "
            f"{c(GRAY, 'Zone:')} {c(WHITE, f'{ob.low:.2f}–{ob.high:.2f}')}  "
            f"{c(GRAY, 'Strength:')} {c(strength_col, f'{ob.strength:.0%}')}",
            kind_col
        ))
    else:
        lines.append(tag("ORDER BLOCK", c(GRAY, "none identified"), GRAY))

    # FVG
    if setup.fvg:
        fvg = setup.fvg
        kind_col = GREEN if fvg.kind == 'bullish' else RED
        lines.append(tag(
            "FAIR VALUE GAP",
            f"{c(kind_col, fvg.kind.upper())}  "
            f"{c(GRAY, 'Zone:')} {c(WHITE, f'{fvg.low:.2f}–{fvg.high:.2f}')}  "
            f"{c(GRAY, 'Size:')} {c(YELLOW, f'{fvg.size_pct:.3f}%')}",
            kind_col
        ))
    else:
        lines.append(tag("FAIR VALUE GAP", c(GRAY, "none in range"), GRAY))

    # Liquidity
    if setup.nearest_liquidity:
        liq = setup.nearest_liquidity
        liq_col = CYAN
        touches = f"  ({liq.touches}x touched)" if liq.touches > 1 else ""
        lines.append(tag(
            "LIQUIDITY TARGET",
            f"{c(liq_col, liq.kind)}  "
            f"{c(WHITE, f'@ {liq.price:.2f}')}{c(GRAY, touches)}",
            liq_col
        ))
    else:
        lines.append(tag("LIQUIDITY TARGET", c(GRAY, "none mapped"), GRAY))

    # Volume
    vol_col = GREEN if setup.volume_confirmed else (YELLOW if setup.confluence_score > 0 else RED)
    vol_text = c(GREEN, "CONFIRMED ✓") if setup.volume_confirmed else c(YELLOW, "NOT CONFIRMED ⚠")
    lines.append(tag("VOLUME", vol_text, vol_col))

    lines.append("")
    return "\n".join(lines)


def render_reasons(setup: TradeSetup) -> str:
    lines = []
    lines.append(line())
    if setup.reasons:
        lines.append(c(GRAY, "  Confluence Factors"))
        for r in setup.reasons:
            lines.append(f"  {c(GREEN, '✓')}  {r}")
        lines.append("")

    if setup.warnings:
        lines.append(c(GRAY, "  Warnings"))
        for w in setup.warnings:
            lines.append(f"  {c(YELLOW, '⚠')}  {w}")
        lines.append("")

    return "\n".join(lines)


def render_execution_guide(setup: TradeSetup) -> str:
    if setup.signal == SignalType.WAIT:
        return ""

    lines = []
    lines.append(line())
    lines.append(c(BOLD + GRAY, "  Execution Guide (Limit Orders)"))
    lines.append("")

    is_long = setup.signal == SignalType.LONG

    steps = [
        ("1", "Place LIMIT order at entry price",
         f"{setup.entry_price:,.4f}  USDT"),
        ("2", "Set Stop Loss immediately after fill",
         f"{setup.stop_loss:,.4f}  USDT"),
        ("3", "Set TP1 for 40% of position",
         f"{setup.take_profit_1:,.4f}  USDT  — move SL to breakeven"),
        ("4", "Set TP2 for 40% of position",
         f"{setup.take_profit_2:,.4f}  USDT  — trail remaining stop"),
        ("5", "Remaining 20% targets TP3",
         f"{setup.take_profit_3:,.4f}  USDT"),
    ]

    for num, action, value in steps:
        lines.append(
            f"  {c(CYAN, num+'.')}  {c(WHITE, action):<45}  {c(YELLOW, value)}"
        )

    lines.append("")

    # Invalidation note
    invalidation = setup.stop_loss
    if is_long:
        lines.append(c(GRAY, f"  Setup invalidates if price closes below {invalidation:,.4f}"))
    else:
        lines.append(c(GRAY, f"  Setup invalidates if price closes above {invalidation:,.4f}"))

    lines.append("")
    return "\n".join(lines)


def render_footer() -> str:
    lines = []
    lines.append(line("═", CYAN))
    disclaimer = "Not financial advice. Always manage your own risk."
    lines.append(c(GRAY, center(disclaimer, W)))
    lines.append(line("═", CYAN))
    return "\n".join(lines)


def render_full_report(symbol: str, tf_mode: str, current_price: float,
                        setup: TradeSetup,
                        ticker: Optional[dict] = None) -> str:
    parts = [
        render_header(symbol, tf_mode, current_price, ticker),
        render_bias_bar(setup.htf_bias, setup.ltf_bias, tf_mode),
        render_signal_box(setup),
    ]

    if setup.signal != SignalType.WAIT:
        parts += [
            render_probability_meter(setup.confluence_score, setup.probability),
            render_components(setup),
        ]

    parts += [
        render_reasons(setup),
        render_execution_guide(setup),
        render_footer(),
    ]

    return "\n".join(parts)
