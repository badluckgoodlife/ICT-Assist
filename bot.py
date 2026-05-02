#!/usr/bin/env python3
"""
ICT/SMC Trade Assistant — Telegram Bot
========================================
Setup:
  1. Message @BotFather on Telegram → /newbot → copy your token
  2. Set TELEGRAM_BOT_TOKEN in .env or as an environment variable
  3. pip install -r requirements.txt
  4. python bot.py

Commands:
  /eth          — Analyze ETHUSDT 1H-15m
  /btc          — Analyze BTCUSDT 1H-15m
  /eth4h        — Analyze ETHUSDT 4H-1H
  /btc4h        — Analyze BTCUSDT 4H-1H
  /watch on     — Enable auto-alerts (every 30 min, only high-probability setups)
  /watch off    — Disable auto-alerts
  /status       — Show current watch settings
  /help         — Command list
"""

import os
import sys
import time
import threading
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

import requests

# ── load .env if present ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional dependency

# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from analyzer import SMCEngine, SignalType, Bias
from fetcher  import fetch_pair, get_ticker

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_BASE = f"https://api.telegram.org/bot{TOKEN}"

WATCH_INTERVAL  = int(os.environ.get("WATCH_INTERVAL_SECONDS", "1800"))  # 30 min default
MIN_PROBABILITY = float(os.environ.get("MIN_ALERT_PROBABILITY", "65"))   # only alert if ≥ 65%
MIN_CONFLUENCE  = float(os.environ.get("MIN_ALERT_CONFLUENCE", "60"))    # and confluence ≥ 60

WATCH_PAIRS = [
    ("ETHUSDT", "1H-15m"),
    ("BTCUSDT", "1H-15m"),
    ("ETHUSDT", "4H-1H"),
    ("BTCUSDT", "4H-1H"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ict_bot")

engine = SMCEngine(swing_lookback=5)

# ─────────────────────────────────────────────────────────────────────────────
# Telegram API helpers
# ─────────────────────────────────────────────────────────────────────────────

def tg_get(method: str, params: dict = {}) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}/{method}", params=params, timeout=10)
        return r.json()
    except Exception as e:
        log.error(f"tg_get {method}: {e}")
        return None


def send_message(chat_id: int, text: str, parse_mode: str = "HTML") -> None:
    try:
        requests.post(f"{API_BASE}/sendMessage", json={
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }, timeout=15)
    except Exception as e:
        log.error(f"send_message: {e}")


def send_typing(chat_id: int) -> None:
    try:
        requests.post(f"{API_BASE}/sendChatAction", json={
            "chat_id": chat_id,
            "action":  "typing",
        }, timeout=5)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter (Telegram HTML)
# ─────────────────────────────────────────────────────────────────────────────

def fmt_price(p: float) -> str:
    if p > 1000:
        return f"{p:,.2f}"
    return f"{p:,.4f}"


def bias_emoji(b: Bias) -> str:
    return {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(b.value, "⚪")


def signal_emoji(s: SignalType) -> str:
    return {"LONG": "📈", "SHORT": "📉", "WAIT": "⏸"}.get(s.value, "❓")


def prob_bar(pct: float, width: int = 10) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def build_report(symbol: str, tf_mode: str, setup, ticker: Optional[dict]) -> str:
    """Build a Telegram HTML-formatted trade report."""
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    tfs = tf_mode.split("-")
    htf_label = tfs[0] if len(tfs) == 2 else "HTF"
    ltf_label = tfs[1] if len(tfs) == 2 else "LTF"

    current = ticker['price'] if ticker else 0
    change  = ticker.get('change_pct', 0) if ticker else 0
    vol24   = ticker.get('volume_24h', 0) if ticker else 0

    sig_icon = signal_emoji(setup.signal)

    lines = []

    # ── Header ──
    lines.append(f"<b>{'━'*30}</b>")
    lines.append(f"<b>{sig_icon} ICT/SMC │ {symbol} │ {tf_mode}</b>")
    lines.append(f"<code>{now}</code>")
    if ticker:
        chg_icon = "▲" if change >= 0 else "▼"
        lines.append(
            f"<b>${fmt_price(current)}</b>  "
            f"{chg_icon} {change:+.2f}%  │  Vol ${vol24/1e9:.2f}B"
        )
    lines.append("")

    # ── Bias ──
    hb = bias_emoji(setup.htf_bias)
    lb = bias_emoji(setup.ltf_bias)
    lines.append(
        f"{hb} <b>{htf_label} Bias:</b> {setup.htf_bias.value}    "
        f"{lb} <b>{ltf_label} Bias:</b> {setup.ltf_bias.value}"
    )
    lines.append("")

    # ── Signal ──
    if setup.signal == SignalType.WAIT:
        lines.append("⏸ <b>NO SETUP — STAND ASIDE</b>")
        lines.append("")
        if setup.warnings:
            for w in setup.warnings:
                lines.append(f"⚠️ {w}")
        return "\n".join(lines)

    dir_word = "LONG 📈" if setup.signal == SignalType.LONG else "SHORT 📉"
    lines.append(f"<b>{'━'*30}</b>")
    lines.append(f"<b>SIGNAL: {dir_word}</b>")
    lines.append(f"<b>{'━'*30}</b>")
    lines.append("")

    # ── Prices ──
    lines.append(f"🎯 <b>Entry (Limit):</b>  <code>${fmt_price(setup.entry_price)}</code>")
    lines.append(f"🛑 <b>Stop Loss:</b>      <code>${fmt_price(setup.stop_loss)}</code>")
    lines.append("")
    lines.append(f"✅ <b>TP1</b> (1.5R / 40%): <code>${fmt_price(setup.take_profit_1)}</code>")
    lines.append(f"✅ <b>TP2</b> (pool / 40%): <code>${fmt_price(setup.take_profit_2)}</code>")
    lines.append(f"✅ <b>TP3</b> (ext / 20%):  <code>${fmt_price(setup.take_profit_3)}</code>")
    lines.append("")

    # ── Risk metrics ──
    risk_pct = abs(setup.entry_price - setup.stop_loss) / setup.entry_price * 100
    lines.append(
        f"📊 Risk: <b>{risk_pct:.3f}%</b>   R:R: <b>1:{setup.risk_reward:.2f}</b>"
    )
    lines.append("")

    # ── Probability bars ──
    lines.append(f"<b>{'─'*30}</b>")
    conf_bar = prob_bar(setup.confluence_score)
    prob_bar_ = prob_bar(setup.probability)
    lines.append(f"Confluence  <code>[{conf_bar}]</code> <b>{setup.confluence_score:.0f}/100</b>")
    lines.append(f"Win Prob    <code>[{prob_bar_}]</code> <b>{setup.probability:.1f}%</b>")
    lines.append("")

    # ── SMC Components ──
    lines.append(f"<b>{'─'*30}</b>")
    lines.append("<b>SMC Components</b>")

    if setup.ob:
        ob = setup.ob
        icon = "🟢" if ob.kind == "bullish" else "🔴"
        lines.append(
            f"{icon} <b>Order Block:</b> {ob.kind.upper()}  "
            f"<code>${fmt_price(ob.low)} – ${fmt_price(ob.high)}</code>  "
            f"str={ob.strength:.0%}"
        )
    else:
        lines.append("⚪ <b>Order Block:</b> none")

    if setup.fvg:
        fvg = setup.fvg
        icon = "🟢" if fvg.kind == "bullish" else "🔴"
        lines.append(
            f"{icon} <b>FVG:</b> {fvg.kind.upper()}  "
            f"<code>${fmt_price(fvg.low)} – ${fmt_price(fvg.high)}</code>  "
            f"{fvg.size_pct:.3f}%"
        )
    else:
        lines.append("⚪ <b>FVG:</b> none in range")

    if setup.nearest_liquidity:
        liq = setup.nearest_liquidity
        kind = "Buy-side (BSL)" if liq.kind == "BSL" else "Sell-side (SSL)"
        lines.append(
            f"💧 <b>Liquidity:</b> {kind}  <code>${fmt_price(liq.price)}</code>"
            + (f"  (double-tapped)" if liq.touches > 1 else "")
        )
    else:
        lines.append("⚪ <b>Liquidity:</b> none mapped")

    vol_icon = "✅" if setup.volume_confirmed else "⚠️"
    vol_text = "confirmed" if setup.volume_confirmed else "NOT confirmed"
    lines.append(f"{vol_icon} <b>Volume:</b> {vol_text}")
    lines.append("")

    # ── Confluence factors ──
    if setup.reasons:
        lines.append(f"<b>{'─'*30}</b>")
        lines.append("<b>Confluence</b>")
        for r in setup.reasons:
            lines.append(f"✓ {r}")
        lines.append("")

    if setup.warnings:
        lines.append("<b>Warnings</b>")
        for w in setup.warnings:
            lines.append(f"⚠️ {w}")
        lines.append("")

    # ── Invalidation ──
    is_long = setup.signal == SignalType.LONG
    inv_word = "below" if is_long else "above"
    lines.append(
        f"<i>Invalidates on close {inv_word} ${fmt_price(setup.stop_loss)}</i>"
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(symbol: str, tf_mode: str) -> tuple[Optional[object], Optional[dict]]:
    """Fetch data and run SMC analysis. Returns (setup, ticker)."""
    result = fetch_pair(symbol, tf_mode)
    if result is None:
        return None, None
    htf_df, ltf_df = result
    ticker = get_ticker(symbol)
    setup  = engine.analyze(ltf_df, htf_df)
    return setup, ticker


def analyze_and_reply(chat_id: int, symbol: str, tf_mode: str) -> None:
    """Run analysis and send formatted report to chat."""
    send_typing(chat_id)
    send_message(chat_id, f"⏳ Fetching {symbol} [{tf_mode}]...")

    setup, ticker = run_analysis(symbol, tf_mode)

    if setup is None:
        send_message(chat_id, "❌ Failed to fetch data. Check your internet connection.")
        return

    current = ticker['price'] if ticker else 0
    report  = build_report(symbol, tf_mode, setup, ticker)
    send_message(chat_id, report)

    log.info(f"Sent report: {symbol} {tf_mode}  signal={setup.signal.value}  prob={setup.probability}%")


# ─────────────────────────────────────────────────────────────────────────────
# Watch mode (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class WatchState:
    def __init__(self):
        self.active_chats: set[int] = set()
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def add(self, chat_id: int) -> None:
        with self.lock:
            self.active_chats.add(chat_id)

    def remove(self, chat_id: int) -> None:
        with self.lock:
            self.active_chats.discard(chat_id)

    def has(self, chat_id: int) -> bool:
        with self.lock:
            return chat_id in self.active_chats

    def count(self) -> int:
        with self.lock:
            return len(self.active_chats)


watch = WatchState()


def watch_loop() -> None:
    log.info("Watch loop started")
    while watch.running:
        chats = list(watch.active_chats)
        if not chats:
            time.sleep(30)
            continue

        log.info(f"Watch scan: {len(WATCH_PAIRS)} pairs × {len(chats)} chats")
        alerts_sent = 0

        for symbol, tf_mode in WATCH_PAIRS:
            try:
                setup, ticker = run_analysis(symbol, tf_mode)
                if setup is None:
                    continue

                # Only push high-probability actionable setups
                if (setup.signal != SignalType.WAIT
                        and setup.probability >= MIN_PROBABILITY
                        and setup.confluence_score >= MIN_CONFLUENCE):

                    report = build_report(symbol, tf_mode, setup, ticker)
                    header = (
                        f"🔔 <b>AUTO-ALERT</b>  |  "
                        f"{signal_emoji(setup.signal)} {setup.signal.value}  "
                        f"{symbol} [{tf_mode}]\n\n"
                    )

                    for chat_id in chats:
                        send_message(chat_id, header + report)
                        alerts_sent += 1
                    log.info(f"Alert: {symbol} {tf_mode}  {setup.signal.value}  {setup.probability:.0f}%")

                time.sleep(2)  # polite between pair scans

            except Exception:
                log.error(traceback.format_exc())

        if alerts_sent == 0:
            log.info("Watch scan complete — no qualifying setups")

        # Sleep until next scan
        for _ in range(WATCH_INTERVAL):
            if not watch.running:
                break
            time.sleep(1)

    log.info("Watch loop stopped")


def start_watch_thread() -> None:
    if watch.thread and watch.thread.is_alive():
        return
    watch.running = True
    watch.thread = threading.Thread(target=watch_loop, daemon=True)
    watch.thread.start()


def stop_watch_thread() -> None:
    watch.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────────────────────

HELP_TEXT = """
<b>ICT/SMC Trade Assistant</b>

<b>On-demand analysis:</b>
/eth      — ETHUSDT  1H → 15m entry
/btc      — BTCUSDT  1H → 15m entry
/eth4h    — ETHUSDT  4H → 1H  entry
/btc4h    — BTCUSDT  4H → 1H  entry

<b>Auto-alerts (watch mode):</b>
/watch on   — Push alerts when prob ≥ {min_prob}%
/watch off  — Stop alerts
/status     — Show current watch state

<b>What each signal includes:</b>
• Limit entry price (OB mid or FVG mid)
• Stop loss (below/above OB)
• TP1 / TP2 / TP3 (1.5R, liquidity pool, range extreme)
• Confluence score &amp; win probability
• Volume confirmation
• HTF + LTF bias

<i>Signals only fire when HTF and LTF align.
Volume must back the setup or a warning is shown.</i>
""".format(min_prob=int(MIN_PROBABILITY))


def handle_command(chat_id: int, text: str) -> None:
    cmd = text.strip().lower().split()[0].lstrip("/").split("@")[0]
    args = text.strip().lower().split()[1:]

    if cmd == "start" or cmd == "help":
        send_message(chat_id, HELP_TEXT)

    elif cmd == "eth":
        threading.Thread(
            target=analyze_and_reply,
            args=(chat_id, "ETHUSDT", "1H-15m"),
            daemon=True
        ).start()

    elif cmd == "btc":
        threading.Thread(
            target=analyze_and_reply,
            args=(chat_id, "BTCUSDT", "1H-15m"),
            daemon=True
        ).start()

    elif cmd == "eth4h":
        threading.Thread(
            target=analyze_and_reply,
            args=(chat_id, "ETHUSDT", "4H-1H"),
            daemon=True
        ).start()

    elif cmd == "btc4h":
        threading.Thread(
            target=analyze_and_reply,
            args=(chat_id, "BTCUSDT", "4H-1H"),
            daemon=True
        ).start()

    elif cmd == "watch":
        if not args:
            send_message(chat_id, "Usage: /watch on  or  /watch off")
            return

        if args[0] == "on":
            watch.add(chat_id)
            start_watch_thread()
            send_message(
                chat_id,
                f"🟢 <b>Watch mode ON</b>\n\n"
                f"Scanning all pairs every {WATCH_INTERVAL // 60} min.\n"
                f"You'll be alerted when probability ≥ {MIN_PROBABILITY:.0f}% "
                f"AND confluence ≥ {MIN_CONFLUENCE:.0f}.\n\n"
                f"<i>Use /watch off to stop.</i>"
            )
            log.info(f"Chat {chat_id} enabled watch mode")

        elif args[0] == "off":
            watch.remove(chat_id)
            send_message(chat_id, "🔴 <b>Watch mode OFF.</b>  No more auto-alerts.")
            log.info(f"Chat {chat_id} disabled watch mode")
            if watch.count() == 0:
                stop_watch_thread()
        else:
            send_message(chat_id, "Usage: /watch on  or  /watch off")

    elif cmd == "status":
        on = watch.has(chat_id)
        status = "🟢 ON" if on else "🔴 OFF"
        pairs_str = "\n".join(f"  • {s} [{tf}]" for s, tf in WATCH_PAIRS)
        send_message(
            chat_id,
            f"<b>Watch status:</b> {status}\n\n"
            f"<b>Monitored pairs:</b>\n{pairs_str}\n\n"
            f"Scan interval: every {WATCH_INTERVAL // 60} min\n"
            f"Alert threshold: prob ≥ {MIN_PROBABILITY:.0f}%  confluence ≥ {MIN_CONFLUENCE:.0f}"
        )

    else:
        send_message(chat_id, "Unknown command. Use /help to see available commands.")


# ─────────────────────────────────────────────────────────────────────────────
# Polling loop
# ─────────────────────────────────────────────────────────────────────────────

def poll() -> None:
    """Long-poll Telegram for updates."""
    log.info("Bot polling started")
    offset = 0

    while True:
        try:
            resp = tg_get("getUpdates", {
                "offset":          offset,
                "timeout":         30,
                "allowed_updates": ["message"],
            })

            if not resp or not resp.get("ok"):
                time.sleep(5)
                continue

            for update in resp.get("result", []):
                offset = update["update_id"] + 1
                msg    = update.get("message", {})
                text   = msg.get("text", "")
                chat_id = msg.get("chat", {}).get("id")

                if not text or not chat_id:
                    continue
                if not text.startswith("/"):
                    continue

                log.info(f"Command from {chat_id}: {text.split()[0]}")
                threading.Thread(
                    target=handle_command,
                    args=(chat_id, text),
                    daemon=True
                ).start()

        except KeyboardInterrupt:
            log.info("Shutting down...")
            stop_watch_thread()
            break
        except Exception:
            log.error(traceback.format_exc())
            time.sleep(5)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TOKEN:
        print(
            "\n  ERROR: TELEGRAM_BOT_TOKEN is not set.\n\n"
            "  1. Message @BotFather on Telegram\n"
            "  2. Send /newbot and follow the steps\n"
            "  3. Copy the token and either:\n"
            "       export TELEGRAM_BOT_TOKEN=your_token_here\n"
            "     or create a .env file:\n"
            "       echo 'TELEGRAM_BOT_TOKEN=your_token_here' > .env\n"
            "  4. Run this script again.\n"
        )
        sys.exit(1)

    # Verify token
    me = tg_get("getMe")
    if not me or not me.get("ok"):
        print(f"\n  ERROR: Invalid token or Telegram unreachable.\n  Response: {me}\n")
        sys.exit(1)

    bot_name = me["result"]["username"]
    log.info(f"Logged in as @{bot_name}")
    log.info(f"Watch interval: {WATCH_INTERVAL}s  |  Min prob: {MIN_PROBABILITY}%")
    log.info("Send /help in Telegram to get started")

    poll()


if __name__ == "__main__":
    main()
