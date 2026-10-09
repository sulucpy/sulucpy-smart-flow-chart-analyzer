
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import websocket

# ============================================================
# Smart Flow Gold — Live XAUUSD + Big Candle + Telegram Alerts
# ============================================================
# IMPORTANT:
# - Put secrets in Streamlit -> Manage app -> Settings -> Secrets.
# - Never put Telegram token or SiftingIO key in GitHub code.
#
# Required secrets:
# SIFTINGIO_API_KEY = "..."
# TELEGRAM_BOT_TOKEN = "..."
# TELEGRAM_CHAT_ID = "..."
#
# The app uses SiftingIO's live XAUUSD WebSocket to build the
# current 5-minute candle locally, so a stale REST 5M bar does
# not block the early-entry engine.
# ============================================================

st.set_page_config(
    page_title="Smart Flow Gold Alerts",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = "https://api.sifting.io"
WS_URL = "wss://stream.sifting.io/ws/v1"
SYMBOL = "XAUUSD"


# -----------------------------
# Secrets / configuration
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


SIFTING_KEY = get_secret("SIFTINGIO_API_KEY")
TELEGRAM_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")


# -----------------------------
# Live WebSocket feed
# -----------------------------
class GoldFeed:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.lock = threading.Lock()
        self.ticks = deque(maxlen=6000)
        self.completed = deque(maxlen=300)
        self.current = None
        self.last_tick = None
        self.last_error = ""
        self.connected = False
        self.stop_event = threading.Event()
        self.thread = None

    @staticmethod
    def bucket(ts_ms: int) -> int:
        return (ts_ms // 300_000) * 300_000

    def _update_candle(self, price: float, ts_ms: int):
        bucket = self.bucket(ts_ms)
        with self.lock:
            if self.current is None or bucket != self.current["t"]:
                if self.current is not None:
                    self.completed.append(dict(self.current))
                self.current = {
                    "t": bucket,
                    "o": price,
                    "h": price,
                    "l": price,
                    "c": price,
                    "v": 1,
                }
            else:
                self.current["h"] = max(self.current["h"], price)
                self.current["l"] = min(self.current["l"], price)
                self.current["c"] = price
                self.current["v"] += 1

            self.last_tick = {
                "price": price,
                "t": ts_ms,
                "received": time.time(),
            }

    def _handle(self, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return

        if msg.get("f") == "tick" and msg.get("s") == SYMBOL:
            try:
                price = float(msg["p"])
                ts_ms = int(msg["t"])
                self.ticks.append((ts_ms, price))
                self._update_candle(price, ts_ms)
            except Exception:
                pass

        elif msg.get("f") == "error":
            with self.lock:
                self.last_error = f'{msg.get("code", "error")}: {msg.get("message", "")}'

    def _run(self):
        while not self.stop_event.is_set():
            ws = None
            ping_stop = threading.Event()
            try:
                ws = websocket.create_connection(
                    f"{WS_URL}?key={self.api_key}",
                    timeout=20,
                    enable_multithread=True,
                )
                ws.settimeout(5)
                ws.send(json.dumps({
                    "op": "subscribe",
                    "product": "com",
                    "symbols": [SYMBOL],
                }))
                with self.lock:
                    self.connected = True
                    self.last_error = ""

                def pinger():
                    while not ping_stop.wait(30):
                        try:
                            ws.send(json.dumps({"op": "ping"}))
                        except Exception:
                            break

                threading.Thread(target=pinger, daemon=True).start()

                while not self.stop_event.is_set():
                    try:
                        raw = ws.recv()
                        if raw:
                            self._handle(raw)
                    except websocket.WebSocketTimeoutException:
                        continue
                    except Exception:
                        break

            except Exception as e:
                with self.lock:
                    self.last_error = str(e)[:240]
            finally:
                ping_stop.set()
                try:
                    if ws:
                        ws.close()
                except Exception:
                    pass
                with self.lock:
                    self.connected = False

            self.stop_event.wait(2)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return {
                "connected": self.connected,
                "current": dict(self.current) if self.current else None,
                "completed": list(self.completed),
                "last_tick": dict(self.last_tick) if self.last_tick else None,
                "error": self.last_error,
            }


@st.cache_resource(show_spinner=False)
def get_feed(api_key: str):
    feed = GoldFeed(api_key)
    feed.start()
    return feed


@st.cache_resource(show_spinner=False)
def get_alert_state():
    return {"last_key": "", "last_sent": 0.0, "last_result": ""}


# -----------------------------
# REST historical data
# -----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_bars(api_key: str, interval: str, limit: int = 250) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    url = f"{BASE}/v1/hist/commodities/{SYMBOL}/bars"
    # First request needs a start. SiftingIO's first page is ordered from
    # the requested start, so choose a recent window that is just large
    # enough to include the latest bars we need for the indicators.
    # This avoids accidentally loading an old first page (which previously
    # made the 1H/15M/5M REST data appear weeks behind the live WebSocket).
    span_days = {
        "1h": 14,
        "15m": 4,
        "5m": 2,
    }.get(interval, 4)
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=span_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "start": start,
        "end": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval": interval,
        "limit": min(max(int(limit), 320), 2000),
    }
    headers = {
        "X-API-Key": api_key,
        "Accept-Encoding": "gzip",
    }
    r = requests.get(url, params=params, headers=headers, timeout=8)
    r.raise_for_status()
    body = r.json()
    rows = body.get("data", body if isinstance(body, list) else [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=rename)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").drop_duplicates("time")
    return df.tail(limit).reset_index(drop=True)


# -----------------------------
# Live REST quote fallback
# -----------------------------
@st.cache_data(ttl=5, show_spinner=False)
def get_live_quote(api_key: str):
    """Get the latest SiftingIO XAUUSD quote when WebSocket is reconnecting.

    This is a price fallback only. It must NOT be used to build the current
    5M candle or trigger live signals because it is a snapshot, not a tick stream.
    """
    if not api_key:
        return None
    url = f"{BASE}/v1/last/quote/commodities/{SYMBOL}"
    try:
        r = requests.get(
            url,
            headers={"X-API-Key": api_key, "Accept-Encoding": "gzip"},
            timeout=4,
        )
        r.raise_for_status()
        body = r.json()
        row = body.get("data", body) if isinstance(body, dict) else body
        if isinstance(row, list):
            row = row[0] if row else {}
        bid = float(row.get("b")) if row.get("b") is not None else None
        ask = float(row.get("a")) if row.get("a") is not None else None
        last = row.get("p")
        if last is not None:
            last = float(last)
        elif bid is not None and ask is not None:
            last = (bid + ask) / 2.0
        elif bid is not None:
            last = bid
        elif ask is not None:
            last = ask
        if last is None:
            return None
        ts = row.get("t")
        if ts is not None:
            try:
                ts = int(ts)
                if ts < 10_000_000_000:
                    ts *= 1000
            except Exception:
                ts = None
        return {"price": last, "bid": bid, "ask": ask, "t": ts, "received": time.time()}
    except Exception:
        return None


# -----------------------------
# Indicators
# -----------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx_di(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    atrv = tr.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atrv
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atrv
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx.fillna(0), pdi.fillna(0), mdi.fillna(0)


def enrich(df):
    if df.empty:
        return df
    x = df.copy()
    x["ema20"] = ema(x.close, 20)
    x["ema50"] = ema(x.close, 50)
    x["ema200"] = ema(x.close, 200)
    x["rsi"] = rsi(x.close, 14)
    x["atr"] = atr(x, 14)
    x["adx"], x["pdi"], x["mdi"] = adx_di(x, 14)
    # Session VWAP approximation from available bars.
    day = x["time"].dt.date
    tp = (x.high + x.low + x.close) / 3
    x["vwap"] = (tp * x.get("volume", pd.Series(1, index=x.index)).fillna(1)).groupby(day).cumsum() / (
        x.get("volume", pd.Series(1, index=x.index)).fillna(1).groupby(day).cumsum()
    )
    return x


def trend_state(row):
    if row.empty:
        return "NEUTRAL"
    if row.ema20.iloc[-1] > row.ema50.iloc[-1] > row.ema200.iloc[-1] and row.close.iloc[-1] > row.ema20.iloc[-1]:
        return "BULL"
    if row.ema20.iloc[-1] < row.ema50.iloc[-1] < row.ema200.iloc[-1] and row.close.iloc[-1] < row.ema20.iloc[-1]:
        return "BEAR"
    return "NEUTRAL"


def structure(df):
    if len(df) < 25:
        return {"sweep": False, "choch": False, "bos_bull": False, "bos_bear": False}
    x = df
    last = x.iloc[-1]
    prev_hi = x.high.iloc[-11:-1].max()
    prev_lo = x.low.iloc[-11:-1].min()
    prior_hi = x.high.iloc[-21:-11].max()
    prior_lo = x.low.iloc[-21:-11].min()

    bos_bull = bool(last.close > prev_hi)
    bos_bear = bool(last.close < prev_lo)

    sweep_low = bool(last.low < prev_lo and last.close > prev_lo)
    sweep_high = bool(last.high > prev_hi and last.close < prev_hi)

    choch = bool(
        (last.close > prior_hi and last.close > prev_hi) or
        (last.close < prior_lo and last.close < prev_lo)
    )
    return {
        "sweep": sweep_low or sweep_high,
        "choch": choch,
        "bos_bull": bos_bull,
        "bos_bear": bos_bear,
        "sweep_low": sweep_low,
        "sweep_high": sweep_high,
    }


def support_resistance(df, max_levels=4):
    if len(df) < 30:
        return {"support": [], "resistance": []}
    x = df.copy()
    a = float(x["atr"].iloc[-1]) if "atr" in x and pd.notna(x["atr"].iloc[-1]) else float(x["close"].iloc[-1]) * 0.002
    tol = max(a * 0.55, 0.8)
    highs, lows = [], []
    for i in range(3, len(x) - 3):
        if x.high.iloc[i] == x.high.iloc[i-3:i+4].max():
            highs.append(float(x.high.iloc[i]))
        if x.low.iloc[i] == x.low.iloc[i-3:i+4].min():
            lows.append(float(x.low.iloc[i]))

    def cluster(vals):
        out = []
        for v in sorted(vals):
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
            else:
                out[-1] = (out[-1] + v) / 2
        return out

    price = float(x.close.iloc[-1])
    sup = [v for v in cluster(lows) if v <= price]
    res = [v for v in cluster(highs) if v >= price]
    return {"support": sup[-max_levels:], "resistance": res[:max_levels]}


def nearest(levels, price, direction):
    arr = levels.get(direction, [])
    if not arr:
        return None
    if direction == "support":
        return max([x for x in arr if x <= price], default=None)
    return min([x for x in arr if x >= price], default=None)


# -----------------------------
# Order Block Detection (5M)
# -----------------------------
def order_block_map(df, price, lookback=80):
    """Detect recent, unmitigated 5M order-block candidates.

    A bullish candidate is the last bearish candle before a bullish
    displacement that closes above a recent swing high. Bearish is inverse.
    This is a price-action heuristic, not exchange order-book data.
    """
    empty = {
        "bullish": None, "bearish": None,
        "near_bullish": False, "near_bearish": False,
        "note": "Not enough candles to detect order blocks."
    }
    if df.empty or len(df) < 25:
        return empty

    x = enrich(df).tail(lookback).reset_index(drop=True)
    last = x.iloc[-1]
    atrv = float(last["atr"]) if pd.notna(last["atr"]) and last["atr"] > 0 else max(float(price) * 0.001, 1.0)
    found = {"bullish": None, "bearish": None}

    # Search only completed/prior candles; current candle is excluded from
    # zone creation so the live candle cannot repaint the origin candle.
    end = len(x) - 1
    start = max(3, end - 35)
    for i in range(end - 1, start, -1):
        c = x.iloc[i]
        nxt = x.iloc[i + 1]
        prev_window = x.iloc[max(0, i - 10):i]
        if len(prev_window) < 5:
            continue
        swing_hi = float(prev_window["high"].max())
        swing_lo = float(prev_window["low"].min())
        body = abs(float(nxt["close"] - nxt["open"]))
        nxt_range = max(float(nxt["high"] - nxt["low"]), 1e-9)
        displacement = body >= 0.55 * atrv and body / nxt_range >= 0.55

        if found["bullish"] is None and c["close"] < c["open"] and nxt["close"] > nxt["open"] and displacement and nxt["close"] > swing_hi:
            low, high = float(c["low"]), float(max(c["open"], c["close"]))
            # A zone is invalidated when a later candle closes decisively below it.
            later = x.iloc[i + 2:]
            invalid = not later.empty and bool((later["close"] < low - 0.05 * atrv).any())
            touched = not later.empty and bool(((later["low"] <= high) & (later["high"] >= low)).any())
            if not invalid:
                found["bullish"] = {
                    "low": low, "high": high, "time": str(c["time"]),
                    "status": "MITIGATED / RETESTED" if touched else "FRESH",
                    "fresh": not touched,
                }

        if found["bearish"] is None and c["close"] > c["open"] and nxt["close"] < nxt["open"] and displacement and nxt["close"] < swing_lo:
            low, high = float(min(c["open"], c["close"])), float(c["high"])
            later = x.iloc[i + 2:]
            invalid = not later.empty and bool((later["close"] > high + 0.05 * atrv).any())
            touched = not later.empty and bool(((later["high"] >= low) & (later["low"] <= high)).any())
            if not invalid:
                found["bearish"] = {
                    "low": low, "high": high, "time": str(c["time"]),
                    "status": "MITIGATED / RETESTED" if touched else "FRESH",
                    "fresh": not touched,
                }

        if found["bullish"] is not None and found["bearish"] is not None:
            break

    bull = found["bullish"]
    bear = found["bearish"]
    found["near_bullish"] = bool(bull and bull["low"] <= price <= bull["high"])
    found["near_bearish"] = bool(bear and bear["low"] <= price <= bear["high"])
    found["note"] = "5M price-action zones; candidates are heuristic, not guaranteed institutional orders."
    return found


# -----------------------------
# Live 5M engine
# -----------------------------
def make_live_df(snap):
    rows = list(snap.get("completed", []))
    cur = snap.get("current")
    if cur:
        rows.append(cur)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["time", "o", "h", "l", "c", "v"]].rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )


def early_engine(live5, h1, m15, price):
    if live5.empty or len(live5) < 15:
        return {
            "status": "WAIT", "direction": "NEUTRAL", "score": 0,
            "quality": "C", "up_trigger": price, "down_trigger": price,
            "reason": "Waiting for live 5M candles.",
        }

    x = enrich(live5)
    cur = x.iloc[-1]
    atrv = float(x.atr.iloc[-1]) if pd.notna(x.atr.iloc[-1]) else max(price * 0.001, 1)
    prior = x.iloc[-2]
    micro_hi = float(x.high.iloc[-6:-1].max())
    micro_lo = float(x.low.iloc[-6:-1].min())

    body = abs(float(cur.close - cur.open))
    rng = max(float(cur.high - cur.low), 1e-9)
    body_ratio = body / rng
    speed = 0.0
    if len(live5) >= 3:
        dt = max((live5.time.iloc[-1] - live5.time.iloc[-3]).total_seconds(), 1)
        speed = float((live5.close.iloc[-1] - live5.close.iloc[-3]) / dt)

    atr_ratio = rng / max(atrv, 1e-9)
    h1e = enrich(h1) if not h1.empty else h1
    m15e = enrich(m15) if not m15.empty else m15
    h1trend = trend_state(h1e)
    m15trend = trend_state(m15e)

    bull = 0
    bear = 0
    bull_reasons, bear_reasons = [], []

    if h1trend == "BULL":
        bull += 20; bull_reasons.append("1H bullish")
    elif h1trend == "BEAR":
        bear += 20; bear_reasons.append("1H bearish")

    if m15trend == "BULL":
        bull += 15; bull_reasons.append("15M bullish")
    elif m15trend == "BEAR":
        bear += 15; bear_reasons.append("15M bearish")

    if cur.close > cur.ema20:
        bull += 10; bull_reasons.append("5M above EMA20")
    else:
        bear += 10; bear_reasons.append("5M below EMA20")

    if cur.pdi > cur.mdi:
        bull += 10; bull_reasons.append("+DI pressure")
    else:
        bear += 10; bear_reasons.append("-DI pressure")

    if cur.rsi >= 52:
        bull += 8; bull_reasons.append("RSI pressure")
    elif cur.rsi <= 48:
        bear += 8; bear_reasons.append("RSI pressure")

    if body_ratio >= 0.60:
        if cur.close > cur.open:
            bull += 12; bull_reasons.append("strong bullish body")
        else:
            bear += 12; bear_reasons.append("strong bearish body")

    if atr_ratio >= 0.65:
        if cur.close > cur.open:
            bull += 10; bull_reasons.append("range expansion UP")
        else:
            bear += 10; bear_reasons.append("range expansion DOWN")

    if cur.close > micro_hi:
        bull += 15; bull_reasons.append("micro-high breakout")
    if cur.close < micro_lo:
        bear += 15; bear_reasons.append("micro-low breakout")

    up_trigger = micro_hi + max(atrv * 0.03, 0.05)
    down_trigger = micro_lo - max(atrv * 0.03, 0.05)

    sr = support_resistance(x)
    sup = nearest(sr, price, "support")
    res = nearest(sr, price, "resistance")

    if sup is not None and abs(price - sup) <= atrv * 0.30:
        bull += 8; bull_reasons.append("near support")
    if res is not None and abs(price - res) <= atrv * 0.30:
        bear += 8; bear_reasons.append("near resistance")

    best = max(bull, bear)
    direction = "UP" if bull > bear else "DOWN" if bear > bull else "NEUTRAL"
    score = int(min(100, best))

    # Anti-chase: a large move already far from its trigger is not a new entry.
    extension = abs(price - (micro_hi if direction == "UP" else micro_lo if direction == "DOWN" else price))
    chase = extension > atrv * 0.90 and atr_ratio > 1.15

    if direction == "UP" and bull >= 70 and not chase:
        status = "BIG CANDLE BUY NOW" if cur.close > micro_hi else "READY UP"
    elif direction == "DOWN" and bear >= 70 and not chase:
        status = "BIG CANDLE SELL NOW" if cur.close < micro_lo else "READY DOWN"
    elif direction == "UP" and bull >= 55:
        status = "BUILDING UP"
    elif direction == "DOWN" and bear >= 55:
        status = "BUILDING DOWN"
    else:
        status = "WAIT"

    if chase:
        status = "EXTENDED — DON'T CHASE"

    quality = "A+" if score >= 85 else "A" if score >= 70 else "B" if score >= 55 else "C"

    return {
        "status": status,
        "direction": direction,
        "score": score,
        "quality": quality,
        "up_trigger": up_trigger,
        "down_trigger": down_trigger,
        "body_ratio": body_ratio,
        "atr_ratio": atr_ratio,
        "speed": speed,
        "micro_hi": micro_hi,
        "micro_lo": micro_lo,
        "h1trend": h1trend,
        "m15trend": m15trend,
        "support": sup,
        "resistance": res,
        "reasons": bull_reasons if direction == "UP" else bear_reasons,
        "atr": atrv,
    }


# -----------------------------
# Institutional / Confluence Engine
# -----------------------------
def _session_levels(df, start_hour, end_hour):
    if df.empty:
        return None, None
    x = df.copy()
    h = x["time"].dt.hour
    if start_hour < end_hour:
        z = x[(h >= start_hour) & (h < end_hour)]
    else:
        z = x[(h >= start_hour) | (h < end_hour)]
    if z.empty:
        return None, None
    return float(z.high.max()), float(z.low.min())


def institutional_engine(h1, m15, m5, live_engine, price):
    """Rule-based institutional-style confluence.

    Important: this is NOT exchange order-book data. XAUUSD spot volume and
    price/volume pressure are treated as proxies. True COMEX depth/order flow
    requires a dedicated futures market-depth feed.
    """
    if h1.empty or m15.empty or m5.empty:
        return {"signal":"WAIT", "score":0, "buy":0, "sell":0,
                "reason":"Insufficient data for institutional confluence."}

    H, M, X = enrich(h1), enrich(m15), enrich(m5)
    ht, mt, xt = trend_state(H), trend_state(M), trend_state(X)
    stc = structure(X)
    sr = support_resistance(X)
    last = X.iloc[-1]
    atrv = float(last.atr) if pd.notna(last.atr) else max(price*0.001, 1.0)

    buy = sell = 0
    br, srx = [], []

    # 1) Higher timeframe bias — 15 points
    if ht == "BULL": buy += 8; br.append("1H bullish")
    elif ht == "BEAR": sell += 8; srx.append("1H bearish")
    if mt == "BULL": buy += 7; br.append("15M bullish")
    elif mt == "BEAR": sell += 7; srx.append("15M bearish")

    # 2) Structure — 15 points
    if stc.get("sweep_low"): buy += 5; br.append("sell-side liquidity sweep")
    if stc.get("sweep_high"): sell += 5; srx.append("buy-side liquidity sweep")
    if stc.get("choch"):
        if stc.get("bos_bull"): buy += 4; br.append("bullish CHoCH")
        elif stc.get("bos_bear"): sell += 4; srx.append("bearish CHoCH")
    if stc.get("bos_bull"): buy += 6; br.append("BOS up")
    if stc.get("bos_bear"): sell += 6; srx.append("BOS down")

    # 3) Liquidity map — previous UTC day + major sessions, 15 points
    day = X["time"].dt.date
    prev_day = X[day < day.iloc[-1]]
    pdh = float(prev_day.high.max()) if not prev_day.empty else None
    pdl = float(prev_day.low.min()) if not prev_day.empty else None
    asia_hi, asia_lo = _session_levels(X, 0, 8)
    lon_hi, lon_lo = _session_levels(X, 8, 13)
    ny_hi, ny_lo = _session_levels(X, 13, 21)
    levels = [v for v in [pdh,pdl,asia_hi,asia_lo,lon_hi,lon_lo,ny_hi,ny_lo] if v is not None]
    near = min((abs(price-v),v) for v in levels) if levels else (None,None)
    if near[0] is not None and near[0] <= atrv*0.45:
        if price >= near[1]:
            buy += 7; br.append("near liquidity above")
        else:
            sell += 7; srx.append("near liquidity below")
    if pdl is not None and price <= pdl + atrv*0.35:
        buy += 4; br.append("near previous-day low")
    if pdh is not None and price >= pdh - atrv*0.35:
        sell += 4; srx.append("near previous-day high")
    if asia_hi is not None and price > asia_hi: buy += 2
    if asia_lo is not None and price < asia_lo: sell += 2

    # 4) VWAP — 10 points
    vwap = float(last.vwap) if pd.notna(last.vwap) else price
    if price > vwap and last.close >= last.open:
        buy += 10; br.append("above session VWAP")
    elif price < vwap and last.close <= last.open:
        sell += 10; srx.append("below session VWAP")
    elif price > vwap:
        buy += 5
    elif price < vwap:
        sell += 5

    # 5) Volume expansion — 15 points (SiftingIO spot volume proxy)
    vol = pd.to_numeric(X["volume"], errors="coerce").fillna(0)
    if len(vol) >= 21 and float(vol.iloc[-1]) > float(vol.iloc[-21:-1].median()) * 1.35:
        if last.close > last.open:
            buy += 15; br.append("volume expansion UP (proxy)")
        elif last.close < last.open:
            sell += 15; srx.append("volume expansion DOWN (proxy)")
    elif len(vol) >= 21 and float(vol.iloc[-1]) > float(vol.iloc[-21:-1].median()) * 1.10:
        if last.close > last.open: buy += 7
        elif last.close < last.open: sell += 7

    # 6) Price/volume pressure proxy — 10 points
    recent = X.tail(8)
    up = int((recent.close > recent.open).sum())
    down = int((recent.close < recent.open).sum())
    if up >= 6 and price > vwap:
        buy += 10; br.append("bullish pressure proxy")
    elif down >= 6 and price < vwap:
        sell += 10; srx.append("bearish pressure proxy")
    elif up > down: buy += 5
    elif down > up: sell += 5

    # 7) Big-move engine — 10 points
    if live_engine.get("direction") == "UP":
        buy += min(10, int(live_engine.get("score",0) * 0.10))
        if live_engine.get("score",0) >= 70: br.append("early expansion UP")
    elif live_engine.get("direction") == "DOWN":
        sell += min(10, int(live_engine.get("score",0) * 0.10))
        if live_engine.get("score",0) >= 70: srx.append("early expansion DOWN")

    # 8) S/R location — 10 points
    ns = nearest(sr, price, "support")
    nr = nearest(sr, price, "resistance")
    if ns is not None and abs(price-ns) <= atrv*0.35:
        buy += 10; br.append("near support")
    if nr is not None and abs(price-nr) <= atrv*0.35:
        sell += 10; srx.append("near resistance")

    buy = min(100, buy); sell = min(100, sell)
    best = max(buy, sell)
    direction = "BUY" if buy > sell else "SELL" if sell > buy else "WAIT"
    margin = abs(buy-sell)

    # Conservative gate: avoid calling a trade A+ when the higher timeframe
    # is strongly opposite or when the two sides are too close.
    if best >= 80 and margin >= 10:
        if direction == "BUY" and ht != "BEAR" and mt != "BEAR":
            signal, label = "BUY", "A+ BUY"
        elif direction == "SELL" and ht != "BULL" and mt != "BULL":
            signal, label = "SELL", "A+ SELL"
        else:
            signal, label = "WAIT", "CONFLICT"
    elif best >= 70 and margin >= 8:
        signal, label = "WATCH", "WATCH"
    else:
        signal, label = "WAIT", "NO TRADE"

    return {
        "signal": signal, "label": label, "score": int(best), "buy": int(buy),
        "sell": int(sell), "margin": int(margin), "vwap": vwap,
        "pdh": pdh, "pdl": pdl, "asia_hi": asia_hi, "asia_lo": asia_lo,
        "london_hi": lon_hi, "london_lo": lon_lo, "ny_hi": ny_hi, "ny_lo": ny_lo,
        "support": ns, "resistance": nr,
        "buy_reasons": br, "sell_reasons": srx,
        "data_note": "Volume/order-flow values are XAUUSD spot proxies; true COMEX depth requires a futures depth feed.",
    }


# -----------------------------
# Next Trigger Map + Liquidity Sweep / Retest Engine
# -----------------------------
def next_trigger_map(h1, m15, m5, engine, institutional, price):
    """Build a live roadmap for the next high-quality move.

    This does not predict the next candle. It answers:
      1) Where is the nearest meaningful breakout level?
      2) Was liquidity swept?
      3) What structure/price confirmation is still missing?
      4) Is a retest preferable to chasing?

    XAUUSD volume/order-flow remains a spot proxy; true COMEX depth requires
    a dedicated futures market-depth feed.
    """
    out = {
        "state": "NO CLEAR TRIGGER",
        "buy_trigger": float(price), "sell_trigger": float(price),
        "buy_break": float(price), "sell_break": float(price),
        "buy_retest_low": float(price), "buy_retest_high": float(price),
        "sell_retest_low": float(price), "sell_retest_high": float(price),
        "buy_state": "BREAKOUT PENDING", "sell_state": "BREAKDOWN PENDING",
        "buy_missing": [], "sell_missing": [],
        "buy_score": 0, "sell_score": 0,
        "buy_sweep": False, "sell_sweep": False,
        "buy_ready": False, "sell_ready": False,
        "vwap": float(institutional.get("vwap", price) or price),
        "support": None, "resistance": None,
        "reason": "Waiting for a clean structural trigger."
    }
    if m5.empty:
        out["reason"] = "Waiting for live 5M structure."
        return out

    # Always initialize local state variables before the decision tree.
    # Without this, a run where none of the later branches matches can raise
    # UnboundLocalError when the function builds its return dictionary.
    state = out["state"]
    buy_state = out["buy_state"]
    sell_state = out["sell_state"]
    reason = out["reason"]
    # Initialize return values before the decision tree so every Streamlit
    # rerun/market state has a safe value.
    vwap = out["vwap"]
    s5 = out["support"]
    r5 = out["resistance"]

    X = enrich(m5)
    M = enrich(m15) if not m15.empty else m15
    H = enrich(h1) if not h1.empty else h1
    st5 = structure(X)
    ht = trend_state(H) if not H.empty else "NEUTRAL"
    mt = trend_state(M) if not M.empty else "NEUTRAL"
    last = X.iloc[-1]
    atrv = float(last.atr) if pd.notna(last.atr) else max(price * 0.001, 1.0)
    vwap = float(institutional.get("vwap", last.vwap if pd.notna(last.vwap) else price) or price)

    sr5 = support_resistance(X)
    sr15 = support_resistance(M) if not M.empty else {"support": [], "resistance": []}
    r5 = nearest(sr5, price, "resistance")
    s5 = nearest(sr5, price, "support")
    r15 = nearest(sr15, price, "resistance")
    s15 = nearest(sr15, price, "support")

    # Confirmed micro levels are based on completed bars before the current bar.
    look = X.tail(8)
    if len(look) >= 3:
        micro_hi = float(look.high.iloc[:-1].max())
        micro_lo = float(look.low.iloc[:-1].min())
    else:
        micro_hi = price + 0.20 * atrv
        micro_lo = price - 0.20 * atrv

    # Use the closest meaningful barrier on each side.
    buy_candidates = [v for v in [micro_hi, r5, r15] if v is not None and v > price]
    sell_candidates = [v for v in [micro_lo, s5, s15] if v is not None and v < price]
    buy_break = min(buy_candidates) if buy_candidates else price + 0.20 * atrv
    sell_break = max(sell_candidates) if sell_candidates else price - 0.20 * atrv

    # Small buffer prevents a touch of the level from becoming a false trigger.
    buffer = max(0.04 * atrv, 0.05)
    buy_trigger = float(buy_break + buffer)
    sell_trigger = float(sell_break - buffer)

    # Retest zones are deliberately narrow: broken level ± 0.15 ATR.
    retest_pad = max(0.15 * atrv, 0.10)
    buy_retest_low = float(buy_break - retest_pad)
    buy_retest_high = float(buy_break + retest_pad)
    sell_retest_low = float(sell_break - retest_pad)
    sell_retest_high = float(sell_break + retest_pad)

    # Liquidity sweep flags from the 5M structure engine.
    buy_sweep = bool(st5.get("sweep_low"))
    sell_sweep = bool(st5.get("sweep_high"))

    # ---------------- BUY confirmation score ----------------
    bs = 0
    bmiss = []
    if ht == "BULL": bs += 20
    elif ht == "NEUTRAL": bs += 8; bmiss.append("1H bias bullish/neutral")
    else: bmiss.append("resolve 1H bearish conflict")
    if mt == "BULL": bs += 15
    else: bmiss.append("15M bullish structure")
    if st5.get("bos_bull"): bs += 15
    elif st5.get("choch") and st5.get("sweep_low"): bs += 10
    else: bmiss.append("5M BOS UP or sweep + CHoCH")
    if buy_sweep: bs += 10
    else: bmiss.append("sell-side liquidity sweep")
    if price > vwap: bs += 10
    else: bmiss.append(f"reclaim VWAP {vwap:,.2f}")
    if float(last.pdi) > float(last.mdi): bs += 10
    else: bmiss.append("+DI pressure")
    if engine.get("direction") == "UP" and engine.get("score", 0) >= 60: bs += 10
    else: bmiss.append("early expansion UP ≥60")
    if price >= buy_trigger: bs += 10
    else: bmiss.append(f"break {buy_trigger:,.2f}")

    # ---------------- SELL confirmation score ----------------
    ss = 0
    smiss = []
    if ht == "BEAR": ss += 20
    elif ht == "NEUTRAL": ss += 8; smiss.append("1H bias bearish/neutral")
    else: smiss.append("resolve 1H bullish conflict")
    if mt == "BEAR": ss += 15
    else: smiss.append("15M bearish structure")
    if st5.get("bos_bear"): ss += 15
    elif st5.get("choch") and st5.get("sweep_high"): ss += 10
    else: smiss.append("5M BOS DOWN or sweep + CHoCH")
    if sell_sweep: ss += 10
    else: smiss.append("buy-side liquidity sweep")
    if price < vwap: ss += 10
    else: smiss.append(f"lose VWAP {vwap:,.2f}")
    if float(last.mdi) > float(last.pdi): ss += 10
    else: smiss.append("-DI pressure")
    if engine.get("direction") == "DOWN" and engine.get("score", 0) >= 60: ss += 10
    else: smiss.append("early expansion DOWN ≥60")
    if price <= sell_trigger: ss += 10
    else: smiss.append(f"break {sell_trigger:,.2f}")

    # A trigger can be active without becoming an A+ trade. The institutional
    # engine remains the final gate for actual BUY/SELL levels.
    buy_ready = price >= buy_trigger and bs >= 65 and not (ht == "BEAR" and mt == "BEAR")
    sell_ready = price <= sell_trigger and ss >= 65 and not (ht == "BULL" and mt == "BULL")

    # Retest confirmation: after a trigger, prefer a return to the broken level
    # instead of entering after a large extension.
    buy_retest = price >= buy_trigger and buy_retest_low <= price <= buy_retest_high
    sell_retest = price <= sell_trigger and sell_retest_low <= price <= sell_retest_high

    if buy_retest and bs >= ss:
        state = "BUY RETEST — WATCH"
        reason = "BUY trigger broke and price is testing the breakout zone; this is preferable to chasing a stretched candle."
        buy_state = "RETEST CONFIRMATION"
    elif sell_retest and ss > bs:
        state = "SELL RETEST — WATCH"
        reason = "SELL trigger broke and price is testing the breakdown zone; this is preferable to chasing a stretched candle."
        sell_state = "RETEST CONFIRMATION"
    elif buy_ready and bs > ss:
        state = "BUY TRIGGER ACTIVE"
        reason = "BUY trigger is active. Require the 5M confirmation/retest before treating it as a trade."
        buy_state = "TRIGGER ACTIVE"
    elif sell_ready and ss > bs:
        state = "SELL TRIGGER ACTIVE"
        reason = "SELL trigger is active. Require the 5M confirmation/retest before treating it as a trade."
        sell_state = "TRIGGER ACTIVE"
    elif engine.get("status") == "EXTENDED — DON'T CHASE":
        state = "MOVE EXTENDED — WAIT RETEST"
        reason = "The move is already extended. Do not chase; wait for the marked breakout level to be retested."
    elif bs > ss:
        state = "BUY SETUP BUILDING"
        reason = "BUY side has more confirmation, but the trigger has not cleared all required conditions."
    elif ss > bs:
        state = "SELL SETUP BUILDING"
        reason = "SELL side has more confirmation, but the trigger has not cleared all required conditions."

    return {
        **out,
        "state": state,
        "buy_trigger": buy_trigger, "sell_trigger": sell_trigger,
        "buy_break": float(buy_break), "sell_break": float(sell_break),
        "buy_retest_low": buy_retest_low, "buy_retest_high": buy_retest_high,
        "sell_retest_low": sell_retest_low, "sell_retest_high": sell_retest_high,
        "buy_state": buy_state, "sell_state": sell_state,
        "buy_missing": bmiss[:6], "sell_missing": smiss[:6],
        "buy_score": int(min(bs, 100)), "sell_score": int(min(ss, 100)),
        "buy_sweep": buy_sweep, "sell_sweep": sell_sweep,
        "buy_ready": bool(buy_ready), "sell_ready": bool(sell_ready),
        "vwap": vwap, "support": s5, "resistance": r5, "reason": reason,
    }


# -----------------------------
# Strict A+ confirmation: OB + recent liquidity sweep + structure break
# -----------------------------
def order_block_confirmation(df, price, ob_map, lookback=4):
    """Return a conservative directional gate for A+ signals.

    Requires price to be in/near the relevant 5M order-block zone, plus a
    recent same-direction liquidity sweep and BOS/CHoCH-style structure break.
    This is a price-action heuristic and must be validated with backtesting.
    """
    result = {
        "buy_ok": False, "sell_ok": False,
        "buy_ob_near": False, "sell_ob_near": False,
        "buy_sweep": False, "sell_sweep": False,
        "buy_break": False, "sell_break": False,
        "buy_reason": "Insufficient 5M candles",
        "sell_reason": "Insufficient 5M candles",
    }
    if df is None or df.empty or len(df) < 25:
        return result

    x = enrich(df).reset_index(drop=True)
    atrv = float(x.atr.iloc[-1]) if pd.notna(x.atr.iloc[-1]) and float(x.atr.iloc[-1]) > 0 else max(float(price) * 0.001, 1.0)
    pad = max(0.20 * atrv, 0.10)
    bull = ob_map.get("bullish") if ob_map else None
    bear = ob_map.get("bearish") if ob_map else None
    result["buy_ob_near"] = bool(bull and bull["low"] - pad <= price <= bull["high"] + pad)
    result["sell_ob_near"] = bool(bear and bear["low"] - pad <= price <= bear["high"] + pad)

    recent = x.tail(max(lookback, 1))
    # Evaluate each recent candle against its own preceding structure window.
    for end in range(max(24, len(x) - lookback), len(x)):
        stc = structure(x.iloc[:end + 1])
        if stc.get("sweep_low"):
            result["buy_sweep"] = True
        if stc.get("sweep_high"):
            result["sell_sweep"] = True
        if stc.get("bos_bull"):
            result["buy_break"] = True
        if stc.get("bos_bear"):
            result["sell_break"] = True

    result["buy_ok"] = bool(result["buy_ob_near"] and result["buy_sweep"] and result["buy_break"])
    result["sell_ok"] = bool(result["sell_ob_near"] and result["sell_sweep"] and result["sell_break"])

    def reason(side):
        ob_key, sweep_key, break_key = f"{side}_ob_near", f"{side}_sweep", f"{side}_break"
        missing = []
        if not result[ob_key]: missing.append("price not at matching OB")
        if not result[sweep_key]: missing.append("no recent liquidity sweep")
        if not result[break_key]: missing.append("no recent BOS")
        return "Confirmed: OB + sweep + BOS" if not missing else "Waiting: " + ", ".join(missing)
    result["buy_reason"] = reason("buy")
    result["sell_reason"] = reason("sell")
    return result


# -----------------------------
# Smart Flow signal
# -----------------------------
def smart_flow_signal(h1, m15, m5, live_engine):
    if h1.empty or m15.empty or m5.empty:
        return {"signal": "WAIT", "score": 0, "reason": "Insufficient market data."}

    H, M, X = enrich(h1), enrich(m15), enrich(m5)
    ht, mt, xt = trend_state(H), trend_state(M), trend_state(X)
    stc = structure(X)
    sr = support_resistance(X)
    price = float(X.close.iloc[-1])

    score_b = 0
    score_s = 0
    br, sr_reasons = [], []

    if ht == "BULL": score_b += 25; br.append("1H BULL")
    if ht == "BEAR": score_s += 25; sr_reasons.append("1H BEAR")
    if mt == "BULL": score_b += 20; br.append("15M BULL")
    if mt == "BEAR": score_s += 20; sr_reasons.append("15M BEAR")
    if xt == "BULL": score_b += 10; br.append("5M BULL")
    if xt == "BEAR": score_s += 10; sr_reasons.append("5M BEAR")

    if stc["sweep_low"]: score_b += 12; br.append("liquidity sweep low")
    if stc["sweep_high"]: score_s += 12; sr_reasons.append("liquidity sweep high")
    if stc["bos_bull"]: score_b += 18; br.append("BOS UP")
    if stc["bos_bear"]: score_s += 18; sr_reasons.append("BOS DOWN")

    last = X.iloc[-1]
    if last.rsi > 52: score_b += 5
    if last.rsi < 48: score_s += 5

    ns = nearest(sr, price, "support")
    nr = nearest(sr, price, "resistance")
    if ns is not None and abs(price - ns) <= last.atr * 0.35:
        score_b += 8; br.append("near support")
    if nr is not None and abs(price - nr) <= last.atr * 0.35:
        score_s += 8; sr_reasons.append("near resistance")

    if live_engine["direction"] == "UP" and live_engine["score"] >= 70:
        score_b += 10; br.append("live early UP")
    if live_engine["direction"] == "DOWN" and live_engine["score"] >= 70:
        score_s += 10; sr_reasons.append("live early DOWN")

    if score_b >= 70 and score_b > score_s:
        return {"signal": "BUY", "score": min(100, score_b), "reason": ", ".join(br)}
    if score_s >= 70 and score_s > score_b:
        return {"signal": "SELL", "score": min(100, score_s), "reason": ", ".join(sr_reasons)}
    return {
        "signal": "WAIT",
        "score": max(score_b, score_s),
        "reason": "15M/5M setup not sufficiently confirmed.",
    }


def trade_levels(signal, price, m5):
    x = enrich(m5)
    if x.empty:
        return None
    a = float(x.atr.iloc[-1])
    if signal == "BUY":
        entry = price
        sl = price - 1.15 * a
        risk = entry - sl
        return {
            "entry": entry, "sl": sl,
            "tp1": entry + 1.0 * risk,
            "tp2": entry + 1.7 * risk,
            "tp3": entry + 2.4 * risk,
            "tp4": entry + 3.2 * risk,
        }
    if signal == "SELL":
        entry = price
        sl = price + 1.15 * a
        risk = sl - entry
        return {
            "entry": entry, "sl": sl,
            "tp1": entry - 1.0 * risk,
            "tp2": entry - 1.7 * risk,
            "tp3": entry - 2.4 * risk,
            "tp4": entry - 3.2 * risk,
        }
    return None


# -----------------------------
# Backtest / Best Filter Engine
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_backtest_5m(api_key: str, days: int = 30) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    end = datetime.now(timezone.utc)
    start_all = end - timedelta(days=days)
    frames = []
    cursor_end = end
    while cursor_end > start_all:
        cursor_start = max(start_all, cursor_end - timedelta(days=6))
        url = f"{BASE}/v1/hist/commodities/{SYMBOL}/bars"
        params = {
            "start": cursor_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": cursor_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "interval": "5m",
            "limit": 2000,
        }
        try:
            r = requests.get(url, params=params, headers={"X-API-Key": api_key, "Accept-Encoding": "gzip"}, timeout=20)
            r.raise_for_status()
            body = r.json()
            rows = body.get("data", body if isinstance(body, list) else [])
            if rows:
                d = pd.DataFrame(rows).rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})
                for c in ["open","high","low","close"]:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                # SiftingIO timestamps are epoch milliseconds.
                d["time"] = pd.to_datetime(d["time"], unit="ms", utc=True)
                frames.append(d.dropna(subset=["time","open","high","low","close"]))
        except Exception:
            break
        cursor_end = cursor_start - timedelta(minutes=5)
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
            .sort_values("time").drop_duplicates("time").reset_index(drop=True))

@st.cache_data(ttl=900, show_spinner=False)
def prepare_backtest(df: pd.DataFrame):
    """Prepare indicators once. The old version rebuilt these for every candidate,
    which made the Backtest Lab look frozen on iPhone."""
    x = enrich(df.copy()).reset_index(drop=True)
    h15 = (x.set_index("time").resample("15min", label="left", closed="left")
           .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
           .dropna().reset_index())
    h1 = (x.set_index("time").resample("1h", label="left", closed="left")
          .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
          .dropna().reset_index())
    h15 = enrich(h15)
    h1 = enrich(h1)
    # Align higher-timeframe context once. The previous version did a
    # DataFrame .loc[:t] search inside every trade/candidate, which could
    # make the Backtest Lab appear frozen on mobile.
    c15 = h15[["time","ema20","ema50","close"]].rename(columns={"ema20":"m15_ema20","ema50":"m15_ema50","close":"m15_close"})
    c1 = h1[["time","ema20","ema50","close"]].rename(columns={"ema20":"h1_ema20","ema50":"h1_ema50","close":"h1_close"})
    aligned = pd.merge_asof(x.sort_values("time"), c15.sort_values("time"), on="time", direction="backward")
    aligned = pd.merge_asof(aligned.sort_values("time"), c1.sort_values("time"), on="time", direction="backward")
    return aligned.reset_index(drop=True), h15.set_index("time"), h1.set_index("time")

def _backtest_metrics(trades):
    if not trades:
        return {"trades":0,"win_rate":0.0,"pf":0.0,"avg_r":0.0,"net_r":0.0,"max_dd":0.0}
    rs = np.array([t["r"] for t in trades], dtype=float)
    wins = rs[rs > 0].sum()
    losses = -rs[rs < 0].sum()
    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = np.maximum(0.0, peak[1:] - equity)
    return {
        "trades": int(len(rs)),
        "win_rate": float((rs > 0).mean() * 100),
        "pf": float(wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0),
        "avg_r": float(rs.mean()),
        "net_r": float(rs.sum()),
        "max_dd": float(dd.max()) if len(dd) else 0.0,
    }

def _run_prepared(prep, mode="COMPOSITE", adx_min=20, body_min=0.50, rr=1.7, start_i=250, end_i=None):
    x, _, _ = prep
    if x.empty or len(x) < 300:
        return _backtest_metrics([])
    end_i = min(len(x)-2, end_i if end_i is not None else len(x)-2)
    start_i = max(250, int(start_i))

    # Vectorized signal conditions: this is much faster than doing DataFrame
    # slicing for every bar/candidate.
    bull15 = (x.m15_ema20 > x.m15_ema50) & (x.m15_close > x.m15_ema20)
    bear15 = (x.m15_ema20 < x.m15_ema50) & (x.m15_close < x.m15_ema20)
    bull1 = x.h1_ema20 > x.h1_ema50
    bear1 = x.h1_ema20 < x.h1_ema50
    body = (x.close - x.open).abs() / (x.high - x.low).clip(lower=1e-9)
    micro_hi = x.high.shift(1).rolling(5).max()
    micro_lo = x.low.shift(1).rolling(5).min()

    long = bull15 & bull1 & (x.close > x.ema20)
    short = bear15 & bear1 & (x.close < x.ema20)
    if mode in ("COMPOSITE", "EMA_RSI_ADX"):
        long &= (x.rsi >= 52) & (x.adx >= adx_min) & (x.pdi > x.mdi)
        short &= (x.rsi <= 48) & (x.adx >= adx_min) & (x.mdi > x.pdi)
    elif mode == "PULLBACK":
        long &= (x.adx >= adx_min) & (body >= body_min) & (x.low <= x.ema20) & (x.close > x.open) & (x.close > x.ema20)
        short &= (x.adx >= adx_min) & (body >= body_min) & (x.high >= x.ema20) & (x.close < x.open) & (x.close < x.ema20)
    elif mode == "BREAKOUT":
        long &= (x.adx >= adx_min) & (body >= body_min) & (x.close > micro_hi)
        short &= (x.adx >= adx_min) & (body >= body_min) & (x.close < micro_lo)
    if mode == "COMPOSITE":
        long &= (x.close > micro_hi) & (body >= body_min)
        short &= (x.close < micro_lo) & (body >= body_min)

    # Make an independent writable NumPy array before masking.
    # Some pandas/NumPy combinations can return a read-only view here.
    mask = np.asarray((long | short).to_numpy(dtype=bool), dtype=bool).copy()
    mask[:start_i] = False
    mask[end_i+1:] = False
    entries = np.flatnonzero(mask)
    if len(entries) == 0:
        return _backtest_metrics([])

    # Entry-by-entry execution is retained for conservative same-bar handling,
    # but only actual candidate bars are scanned, not every historical bar.
    trades=[]; next_allowed=start_i
    highs=x.high.to_numpy(dtype=float); lows=x.low.to_numpy(dtype=float)
    closes=x.close.to_numpy(dtype=float); atrs=x.atr.to_numpy(dtype=float)
    long_arr=long.to_numpy();
    for i in entries:
        if i < next_allowed: continue
        atrv=atrs[i]
        if not np.isfinite(atrv) or atrv <= 0: continue
        side=1 if long_arr[i] else -1
        entry=float(closes[i]); risk=1.15*float(atrv)
        sl=entry-side*risk; tp=entry+side*risk*rr
        result=None; exit_i=i
        stop=min(i+61, len(x))
        for j in range(i+1, stop):
            hi=float(highs[j]); lo=float(lows[j])
            hit_sl = lo <= sl if side==1 else hi >= sl
            hit_tp = hi >= tp if side==1 else lo <= tp
            if hit_sl:
                result=-1.0; exit_i=j; break
            if hit_tp:
                result=rr; exit_i=j; break
        if result is not None:
            trades.append({"r":result}); next_allowed=exit_i+4
    return _backtest_metrics(trades)

def _run_filter_backtest(df, mode="COMPOSITE", adx_min=20, body_min=0.50, rr=1.7):
    return _run_prepared(prepare_backtest(df), mode, adx_min, body_min, rr)

def run_best_filter_search(df):
    prep = prepare_backtest(df)
    modes = ["PULLBACK", "EMA_RSI_ADX", "BREAKOUT", "COMPOSITE"]
    candidates = []
    for mode in modes:
        adxs = [18,22] if mode == "PULLBACK" else [18,20,22,25]
        bodies = [0.50] if mode == "PULLBACK" else ([0.45,0.50,0.60] if mode == "COMPOSITE" else [0.50])
        for adx_min in adxs:
            for body_min in bodies:
                for rr in [1.5,1.7,2.0]:
                    m = _run_prepared(prep, mode, adx_min, body_min, rr)
                    if m["trades"] >= 20:
                        score = m["pf"] * max(m["avg_r"], 0) * 100 + m["win_rate"] * 0.15 - m["max_dd"] * 0.08
                        candidates.append((score, mode, adx_min, body_min, rr, m))
    candidates.sort(reverse=True, key=lambda z: z[0])
    return candidates[:10]


# -----------------------------
# Forward / Paper Trading Log
# -----------------------------
def _paper_trades():
    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []
    return st.session_state["paper_trades"]

def update_paper_trades(price):
    """Close open paper trades on first touch of SL or TP1.
    Conservative rule: first level touched wins; no hindsight.
    """
    trades = _paper_trades()
    for t in trades:
        if t.get("status") != "OPEN":
            continue
        side = t["side"]
        if side == "BUY":
            if price <= t["sl"]:
                t.update(status="LOSS", exit=price, r=-1.0)
            elif price >= t["tp1"]:
                t.update(status="TP1", exit=price, r=1.0)
        else:
            if price >= t["sl"]:
                t.update(status="LOSS", exit=price, r=-1.0)
            elif price <= t["tp1"]:
                t.update(status="TP1", exit=price, r=1.0)

def maybe_log_paper_signal(signal, engine, price, levels):
    if not levels or signal.get("signal") not in ("BUY", "SELL"):
        return
    if signal.get("score", 0) < 70 or engine.get("score", 0) < 70:
        return
    trades = _paper_trades()
    key = f'{signal["signal"]}|{round(levels["entry"],2)}|{round(levels["sl"],2)}|{round(levels["tp1"],2)}'
    if any(t.get("key") == key for t in trades if t.get("status") == "OPEN"):
        return
    trades.append({
        "key": key,
        "time": datetime.now(timezone.utc),
        "side": signal["signal"],
        "entry": float(levels["entry"]),
        "sl": float(levels["sl"]),
        "tp1": float(levels["tp1"]),
        "tp2": float(levels["tp2"]),
        "tp3": float(levels["tp3"]),
        "tp4": float(levels["tp4"]),
        "signal_score": int(signal.get("score", 0)),
        "early_score": int(engine.get("score", 0)),
        "status": "OPEN",
        "exit": None,
        "r": None,
    })
    # Keep the in-session paper log bounded.
    if len(trades) > 100:
        del trades[:-100]


# -----------------------------
# Telegram
# -----------------------------
def telegram_send(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Telegram secrets not configured."

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID.strip(),
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        try:
            data = r.json()
        except Exception:
            data = {}
        if r.ok and data.get("ok") is True:
            return True, "Telegram message sent successfully."
        desc = data.get("description", r.text[:240])
        return False, f"Telegram HTTP {r.status_code}: {desc}"
    except requests.RequestException as e:
        return False, f"Telegram network error: {e}"
    except Exception as e:
        return False, f"Telegram error: {e}"


def telegram_diagnostics():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in Streamlit Secrets."
    try:
        me = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe",
            timeout=15,
        )
        try:
            md = me.json()
        except Exception:
            md = {}
        if not (me.ok and md.get("ok") is True):
            return False, f"Bot token check failed: HTTP {me.status_code}: {md.get('description', me.text[:200])}"
        bot_name = md.get("result", {}).get("username", "unknown")
        return True, f"Bot connected: @{bot_name} • Chat ID configured: {TELEGRAM_CHAT_ID.strip()}"
    except Exception as e:
        return False, f"Telegram connection check failed: {e}"


def maybe_alert(signal, engine, price, levels):
    # Only high-quality signals. This is an alert, not an automatic order.
    alert_state = get_alert_state()
    if signal["signal"] not in ("BUY", "SELL"):
        return
    if signal["score"] < 70:
        return
    if engine["score"] < 70:
        return

    key = f'{signal["signal"]}|{round(price,2)}|{engine["status"]}|{signal["score"]}'
    now = time.time()

    # Avoid repeated alerts for the same state for 10 minutes.
    if alert_state["last_key"] == key and now - alert_state["last_sent"] < 600:
        return

    if not levels:
        return

    emoji = "🟢" if signal["signal"] == "BUY" else "🔴"
    text = (
        f"{emoji} SMART FLOW GOLD — {signal['signal']} A+\n\n"
        f"XAUUSD: {price:.2f}\n"
        f"Signal score: {signal['score']}/100\n"
        f"Early score: {engine['score']}/100\n"
        f"Live engine: {engine['status']}\n"
        f"1H: {engine['h1trend']} | 15M: {engine['m15trend']}\n\n"
        f"ENTRY: {levels['entry']:.2f}\n"
        f"SL: {levels['sl']:.2f}\n"
        f"TP1: {levels['tp1']:.2f}\n"
        f"TP2: {levels['tp2']:.2f}\n"
        f"TP3: {levels['tp3']:.2f}\n"
        f"TP4: {levels['tp4']:.2f}\n\n"
        f"⚠️ Alert only — not an automatic order."
    )
    ok, result = telegram_send(text)
    if ok:
        alert_state["last_key"] = key
        alert_state["last_sent"] = now
        alert_state["last_result"] = "Telegram alert sent"
    else:
        alert_state["last_result"] = result


# -----------------------------
# UI
# -----------------------------

# -----------------------------
# Professional terminal styling
# -----------------------------
PRO_CSS = """
<style>
:root { --bg:#071019; --panel:#0d1722; --panel2:#101d2a; --line:#203142; --text:#eef4f8; --muted:#8ea0b2; --green:#23d18b; --red:#ff5c67; --amber:#f4b740; --blue:#55a7ff; }
.stApp { background: radial-gradient(circle at 15% 0%, #112233 0%, #071019 42%, #050b11 100%); color:var(--text); }
.block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { background:#071019; border-right:1px solid var(--line); }
.terminal-header { display:flex; justify-content:space-between; align-items:flex-end; gap:18px; padding:18px 20px; border:1px solid var(--line); border-radius:18px; background:linear-gradient(135deg,rgba(16,29,42,.98),rgba(7,16,25,.96)); box-shadow:0 14px 38px rgba(0,0,0,.22); margin-bottom:14px; }
.terminal-title { font-size:1.55rem; font-weight:800; letter-spacing:.2px; margin:0; }
.terminal-sub { color:var(--muted); font-size:.86rem; margin-top:5px; }
.live-pill { border:1px solid rgba(35,209,139,.45); color:#9ff1ce; background:rgba(35,209,139,.09); padding:7px 11px; border-radius:999px; font-weight:700; white-space:nowrap; }
.command { border:1px solid var(--line); border-radius:20px; background:linear-gradient(145deg,#0f1d2a,#09131d); padding:18px; box-shadow:0 16px 44px rgba(0,0,0,.24); margin:8px 0 16px; }
.command-grid { display:grid; grid-template-columns:1.25fr .8fr .8fr .8fr; gap:10px; margin-top:13px; }
.kpi { background:rgba(255,255,255,.025); border:1px solid #1d2d3c; border-radius:14px; padding:12px 14px; }
.kpi-label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { color:var(--text); font-size:1.18rem; font-weight:800; margin-top:4px; }
.kpi-note { color:var(--muted); font-size:.75rem; margin-top:3px; }
.action { margin-top:12px; border-radius:14px; padding:12px 14px; border:1px solid #2a3c4d; background:rgba(255,255,255,.025); }
.action b { color:#fff; }
.action-green { border-color:rgba(35,209,139,.45); background:rgba(35,209,139,.07); }
.action-red { border-color:rgba(255,92,103,.45); background:rgba(255,92,103,.07); }
.action-amber { border-color:rgba(244,183,64,.45); background:rgba(244,183,64,.06); }
.action-neutral { border-color:#2a3c4d; }
.path-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
.path { border-radius:15px; padding:14px; border:1px solid #243747; background:#0b1620; }
.path.buy { border-color:rgba(35,209,139,.35); }
.path.sell { border-color:rgba(255,92,103,.35); }
.path-head { font-weight:800; font-size:.9rem; }
.path-price { font-size:1.25rem; font-weight:850; margin:5px 0; }
.path-muted { color:var(--muted); font-size:.76rem; }
.section-head { margin-top:18px; margin-bottom:7px; font-size:1.02rem; font-weight:800; letter-spacing:.01em; }
@media (max-width: 800px) { .terminal-header { align-items:flex-start; flex-direction:column; } .command-grid,.path-grid { grid-template-columns:1fr 1fr; } }
</style>
"""
st.markdown(PRO_CSS, unsafe_allow_html=True)

st.markdown("""<div class="terminal-header"><div><div class="terminal-title">🟡 Smart Flow Gold <span style="color:#8ea0b2;font-weight:600">/ XAUUSD</span></div><div class="terminal-sub">Professional live decision terminal • Smart Flow • Liquidity • Institutional Confluence • Early Move Engine</div></div><div class="live-pill">● LIVE MARKET MODE</div></div>""", unsafe_allow_html=True)
st.caption("Live WebSocket • Smart Flow • Big Candle / Early Entry • Support/Resistance • Telegram A+ Alerts")

with st.sidebar:
    st.header("⚙️ Connection")
    if SIFTING_KEY:
        st.success("SiftingIO key: loaded from Secrets")
    else:
        SIFTING_KEY = st.text_input("SiftingIO API key", type="password")

    st.markdown("**Telegram Secrets**")
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        st.success("🔔 Telegram A/A+ alerts: ON")
    else:
        st.warning("Telegram alerts are OFF")
        st.caption("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Streamlit Secrets.")

    refresh = st.slider("Live refresh (seconds)", 5, 15, 5)
    view_mode = st.radio("Interface", ["⚡ Quick Trade", "🧭 Full Pro"], index=0)
    st.caption("Quick Trade = one-screen decision view. Full Pro = all diagnostics.")
    st.caption("Only the live dashboard refreshes; Backtest/Telegram sections stay stable.")
    st.caption("A/A+ alert rule: Smart Flow ≥70 AND Early Engine ≥70.")
    st.info("🟢 Support / 🔴 Resistance are rule-based pivot estimates, not a private TradingView indicator copy.")
    st.caption("SiftingIO XAUUSD is an aggregated/reference price feed. Test before real-money use.")

if not SIFTING_KEY:
    st.error("Add SIFTINGIO_API_KEY in Streamlit Secrets first.")
    st.stop()

# Stable live refresh architecture:
# The old version used a full-page st_autorefresh every few seconds. That
# caused the entire app (including heavy sections) to rerun repeatedly.
# Streamlit fragments now refresh only the live dashboard section, while
# Backtest/Telegram controls remain stable.

@st.fragment(run_every=refresh)
def live_dashboard():
    feed = get_feed(SIFTING_KEY)
    snap = feed.snapshot()

    # History (cached REST calls; these do not refetch on every live tick).
    try:
        h1 = get_bars(SIFTING_KEY, "1h", 260)
        m15 = get_bars(SIFTING_KEY, "15m", 300)
        rest5 = get_bars(SIFTING_KEY, "5m", 300)

        if not rest5.empty:
            rest5_age_min = (datetime.now(timezone.utc) - rest5.time.iloc[-1].to_pydatetime()).total_seconds() / 60
        else:
            rest5_age_min = float("inf")

        if rest5.empty or rest5_age_min > 15:
            one_min = get_bars(SIFTING_KEY, "1m", 2000)
            if not one_min.empty:
                one_min = one_min.set_index("time").sort_index()
                rest5_fallback = one_min.resample("5min", label="left", closed="left").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
                }).dropna(subset=["open", "high", "low", "close"]).reset_index()
                rest5 = rest5_fallback.tail(350).reset_index(drop=True)
    except Exception as e:
        st.error(f"SiftingIO history error: {e}")
        h1 = m15 = rest5 = pd.DataFrame()

    live5 = make_live_df(snap)
    if not live5.empty and not rest5.empty:
        base5 = pd.concat([rest5, live5], ignore_index=True).drop_duplicates("time", keep="last").sort_values("time").tail(350).reset_index(drop=True)
    else:
        base5 = live5 if not live5.empty else rest5

    h1e = enrich(h1) if not h1.empty else h1
    m15e = enrich(m15) if not m15.empty else m15
    m5e = enrich(base5) if not base5.empty else base5

    price = None
    last_tick_age = None
    price_source = "NONE"
    quote = None

    if snap["last_tick"]:
        last_tick_age = max(0, time.time() - snap["last_tick"]["received"])
        if last_tick_age <= 8:
            price = float(snap["last_tick"]["price"])
            price_source = "WEBSOCKET"

    if price is None:
        quote = get_live_quote(SIFTING_KEY)
        if quote and quote.get("price") is not None:
            price = float(quote["price"])
            price_source = "REST_QUOTE_FALLBACK"

    if price is None and not base5.empty:
        price = float(base5.close.iloc[-1])
        price_source = "HISTORICAL_FALLBACK"

    if price is None:
        st.warning("Waiting for live XAUUSD price…")
        return

    feed_live = price_source == "WEBSOCKET" and last_tick_age is not None and last_tick_age <= 8
    engine = early_engine(base5, h1, m15, price)
    institutional = institutional_engine(h1, m15, base5, engine, price)
    trigger_map = next_trigger_map(h1, m15, base5, engine, institutional, price)
    ob_map = order_block_map(base5, price)
    ob_confirmation = order_block_confirmation(base5, price, ob_map)
    if feed_live:
        signal = smart_flow_signal(h1, m15, base5, engine)
        proposed = institutional.get("signal")
        ob_gate = ob_confirmation.get("buy_ok") if proposed == "BUY" else ob_confirmation.get("sell_ok") if proposed == "SELL" else False
        final_signal = proposed if proposed in ("BUY", "SELL") and signal["score"] >= 60 and engine["score"] >= 60 and ob_gate else "WAIT"
        gate_reason = ob_confirmation.get("buy_reason") if proposed == "BUY" else ob_confirmation.get("sell_reason") if proposed == "SELL" else "Waiting for institutional direction."
        final_reason = institutional.get("label", "NO TRADE") if final_signal in ("BUY", "SELL") else f"NO TRADE — {gate_reason}"
        final = {"signal": final_signal, "score": institutional["score"], "reason": final_reason}
        levels = trade_levels(final_signal, price, base5)
        update_paper_trades(price)
        maybe_log_paper_signal(final, engine, price, levels)
        maybe_alert(final, engine, price, levels)
    else:
        signal = {"signal": "WAIT", "score": 0, "reason": "Live WebSocket is not fresh; waiting for live ticks."}
        institutional = {"signal":"WAIT","label":"NO TRADE","score":0,"buy":0,"sell":0,"margin":0,"buy_reasons":[],"sell_reasons":[],"data_note":"Live feed unavailable."}
        final = signal
        levels = None

    # --------------------------------------------------------
    # COMMAND CENTER — one master snapshot for the whole screen
    # --------------------------------------------------------
    master_score = int(institutional.get("score", 0))
    master_signal = final.get("signal", "WAIT") if feed_live else "WAIT"
    if master_signal == "BUY" and master_score >= 80:
        decision_label, action_class, action_text = "A+ BUY", "action-green", "BUY confirmation is active — use the displayed Entry / SL / TP plan."
    elif master_signal == "SELL" and master_score >= 80:
        decision_label, action_class, action_text = "A+ SELL", "action-red", "SELL confirmation is active — use the displayed Entry / SL / TP plan."
    elif trigger_map["state"].startswith("BUY"):
        decision_label, action_class, action_text = "WATCH BUY", "action-green", "Do not chase. Wait for the BUY break + retest + confirmation gate."
    elif trigger_map["state"].startswith("SELL"):
        decision_label, action_class, action_text = "WATCH SELL", "action-red", "Do not chase. Wait for the SELL break + retest + confirmation gate."
    else:
        decision_label, action_class, action_text = "NO TRADE", "action-neutral", "Stay flat. Let price reach a trigger and prove direction before entering."

    now_utc = datetime.now(timezone.utc)
    elapsed = now_utc.minute * 60 + now_utc.second
    remain = 300 - (elapsed % 300)
    mm, ss = divmod(remain, 60)

    st.markdown(f"""
    <div class="command">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
        <div><div style="color:#8ea0b2;font-size:.72rem;text-transform:uppercase;letter-spacing:.09em">Trade Command Center</div><div style="font-size:1.65rem;font-weight:900;margin-top:2px">{decision_label}</div></div>
        <div style="text-align:right"><div style="color:#8ea0b2;font-size:.72rem;text-transform:uppercase;letter-spacing:.09em">Master Score</div><div style="font-size:1.45rem;font-weight:900">{master_score}<span style="font-size:.8rem;color:#8ea0b2">/100</span></div></div>
      </div>
      <div class="action {action_class}"><b>WHAT TO DO NOW:</b> {action_text}</div>
      <div style="color:#718096;font-size:.72rem;margin:8px 0 0">Single master snapshot • all trigger/score sections below use this same calculation</div>
      <div class="command-grid">
        <div class="kpi"><div class="kpi-label">XAUUSD</div><div class="kpi-value">{price:,.2f}</div><div class="kpi-note">{price_source.replace('_',' ')}</div></div>
        <div class="kpi"><div class="kpi-label">BUY Trigger</div><div class="kpi-value">{trigger_map['buy_break']:,.2f}</div><div class="kpi-note">{abs(trigger_map['buy_break']-price):.2f} away</div></div>
        <div class="kpi"><div class="kpi-label">SELL Trigger</div><div class="kpi-value">{trigger_map['sell_break']:,.2f}</div><div class="kpi-note">{abs(price-trigger_map['sell_break']):.2f} away</div></div>
        <div class="kpi"><div class="kpi-label">5M Candle</div><div class="kpi-value">{mm:02d}:{ss:02d}</div><div class="kpi-note">time remaining</div></div>
      </div>
      <div class="path-grid">
        <div class="path buy"><div class="path-head">🟢 BUY PLAN</div><div class="path-price">Break {trigger_map['buy_break']:,.2f}</div><div class="path-muted">Retest {trigger_map['buy_retest_low']:,.2f} – {trigger_map['buy_retest_high']:,.2f} • score {trigger_map['buy_score']}/100</div></div>
        <div class="path sell"><div class="path-head">🔴 SELL PLAN</div><div class="path-price">Break {trigger_map['sell_break']:,.2f}</div><div class="path-muted">Retest {trigger_map['sell_retest_low']:,.2f} – {trigger_map['sell_retest_high']:,.2f} • score {trigger_map['sell_score']}/100</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if feed_live:
        st.success(f"🟢 LIVE WebSocket • tick {last_tick_age:.1f}s ago")
    elif price_source == "REST_QUOTE_FALLBACK":
        st.warning("🟡 WebSocket reconnecting — live REST quote shown. Signals/alerts are paused until ticks resume.")
    elif price_source == "HISTORICAL_FALLBACK":
        st.error("🔴 Live feed unavailable — historical price only. Signals/alerts are paused.")
    else:
        st.warning(f"🟠 WebSocket reconnecting… {snap['error']}")

    # Quick Trade mode: keep the mobile screen focused on the decision.
    if view_mode == "⚡ Quick Trade":
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Action", decision_label)
        q2.metric("Smart Flow", f"{signal['score']}/100")
        q3.metric("Big Move", f"{engine['score']}/100")
        q4.metric("5M ADX", f"{float(m5e.adx.iloc[-1]):.1f}" if not m5e.empty and pd.notna(m5e.adx.iloc[-1]) else "—")
        if decision_label == "A+ BUY":
            st.success("🟢 TRADE NOW — BUY confirmation is active.")
        elif decision_label == "A+ SELL":
            st.error("🔴 TRADE NOW — SELL confirmation is active.")
        elif decision_label == "WATCH SELL":
            st.warning(f"🔴 WATCH SELL — wait for break {trigger_map['sell_break']:,.2f} + retest + confirmation.")
        elif decision_label == "WATCH BUY":
            st.success(f"🟢 WATCH BUY — wait for break {trigger_map['buy_break']:,.2f} + retest + confirmation.")
        else:
            st.info("⚪ WAIT — no clean trade setup yet.")
        with st.expander("🧱 Order Block Map"):
            bull_ob = ob_map.get("bullish")
            bear_ob = ob_map.get("bearish")
            st.write(
                "🟢 Bullish OB: " + (
                    f"{bull_ob['low']:,.2f} – {bull_ob['high']:,.2f} • {bull_ob['status']}"
                    if bull_ob else "No valid recent zone"
                )
            )
            st.write(
                "🔴 Bearish OB: " + (
                    f"{bear_ob['low']:,.2f} – {bear_ob['high']:,.2f} • {bear_ob['status']}"
                    if bear_ob else "No valid recent zone"
                )
            )
            st.write(f"Price inside bullish/bearish zone: **{'YES' if ob_map.get('near_bullish') else 'NO'} / {'YES' if ob_map.get('near_bearish') else 'NO'}**")
            st.caption(ob_map.get("note", ""))
        with st.expander("📋 Confirmation checklist"):
            st.write(f"1H: **{trend_state(h1e) if not h1e.empty else '—'}**")
            st.write(f"5M BOS: **{'UP' if structure(m5e).get('bos_bull') else 'DOWN' if structure(m5e).get('bos_bear') else 'NO'}**" if not m5e.empty else "5M BOS: **—**")
            st.write(f"Liquidity sweep: **{'YES' if trigger_map['buy_sweep'] or trigger_map['sell_sweep'] else 'NO'}**")
            st.write(f"VWAP: **{trigger_map['vwap']:,.2f}**")
            st.write(f"Institutional: **{institutional.get('label','NO TRADE')} • {institutional.get('score',0)}/100**")
            st.caption("Trigger touch alone is not an entry. Wait for break + retest + structure confirmation.")
        with st.expander("📊 Detailed diagnostics"):
            st.caption("Switch the sidebar Interface to 🧭 Full Pro to keep all sections permanently visible.")
        return

    if final["signal"] == "BUY" and levels:
        st.success(f"🟢 A+ BUY — {master_score}/100")
    elif final["signal"] == "SELL" and levels:
        st.error(f"🔴 A+ SELL — {master_score}/100")
    else:
        st.info(f"⚪ {decision_label} • Smart Flow {signal['score']}/100 • Big Move {engine['score']}/100")

    st.markdown('<div class="section-head">🏦 Institutional Confluence</div>', unsafe_allow_html=True)

    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
    ic1.metric("Decision", institutional.get("label","NO TRADE"))
    ic2.metric("BUY pressure", institutional.get("buy",0))
    ic3.metric("SELL pressure", institutional.get("sell",0))
    ic4.metric("Confluence", f"{institutional.get('score',0)}/100")
    ic5.metric("Margin", institutional.get("margin",0))
    if institutional.get("signal") == "BUY":
        st.success("🟢 A+ BUY — major confluence is aligned.")
    elif institutional.get("signal") == "SELL":
        st.error("🔴 A+ SELL — major confluence is aligned.")
    elif institutional.get("signal") == "WATCH":
        st.warning("🟡 WATCH — pressure is building, but the A+ gate is not confirmed.")
    else:
        st.info("⚪ NO TRADE — the confluence gate is intentionally strict.")
    with st.expander("🏦 Institutional map"):
        st.write(f"VWAP: **{institutional.get('vwap', price):,.2f}**")
        st.write(f"Previous day High / Low: **{institutional.get('pdh') or 0:,.2f} / {institutional.get('pdl') or 0:,.2f}**")
        st.write(f"Asia H/L: **{institutional.get('asia_hi') or 0:,.2f} / {institutional.get('asia_lo') or 0:,.2f}**")
        st.write(f"London H/L: **{institutional.get('london_hi') or 0:,.2f} / {institutional.get('london_lo') or 0:,.2f}**")
        st.write(f"NY H/L: **{institutional.get('ny_hi') or 0:,.2f} / {institutional.get('ny_lo') or 0:,.2f}**")
        st.write("BUY factors: " + (" • ".join(institutional.get("buy_reasons",[])) or "—"))
        st.write("SELL factors: " + (" • ".join(institutional.get("sell_reasons",[])) or "—"))
        st.caption(institutional.get("data_note",""))

    # --------------------------------------------------------
    # Next Trigger Map — use the SAME master snapshot calculated above.
    # Never recalculate it inside the same dashboard render, otherwise
    # live values can disagree between the Command Center and this section.
    # --------------------------------------------------------
    st.subheader("🎯 Next Trigger + Liquidity Sweep / Retest Map")
    st.caption("Condition map only — it does not predict the future. The final BUY/SELL gate remains Institutional Confluence.")

    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("Market state", trigger_map["state"])
    tm2.metric("BUY trigger", f"{trigger_map['buy_trigger']:,.2f}", f"score {trigger_map['buy_score']}/100")
    tm3.metric("SELL trigger", f"{trigger_map['sell_trigger']:,.2f}", f"score {trigger_map['sell_score']}/100")
    tm4.metric("VWAP", f"{trigger_map['vwap']:,.2f}")

    bcol, scol = st.columns(2)
    with bcol:
        st.success(f"🟢 BUY path • {trigger_map['buy_state']}")
        st.write(f"Break level: **{trigger_map['buy_break']:,.2f}**")
        st.write(f"Retest zone: **{trigger_map['buy_retest_low']:,.2f} – {trigger_map['buy_retest_high']:,.2f}**")
        st.write(f"Liquidity sweep: **{'YES' if trigger_map['buy_sweep'] else 'NO'}**")
        if trigger_map["buy_missing"]:
            st.caption("BUY still needs: " + " • ".join(trigger_map["buy_missing"]))
        else:
            st.caption("BUY conditions are aligned — wait for trigger/retest confirmation.")
    with scol:
        st.error(f"🔴 SELL path • {trigger_map['sell_state']}")
        st.write(f"Break level: **{trigger_map['sell_break']:,.2f}**")
        st.write(f"Retest zone: **{trigger_map['sell_retest_low']:,.2f} – {trigger_map['sell_retest_high']:,.2f}**")
        st.write(f"Liquidity sweep: **{'YES' if trigger_map['sell_sweep'] else 'NO'}**")
        if trigger_map["sell_missing"]:
            st.caption("SELL still needs: " + " • ".join(trigger_map["sell_missing"]))
        else:
            st.caption("SELL conditions are aligned — wait for trigger/retest confirmation.")

    if trigger_map["state"].startswith("BUY"):
        st.info("🟢 " + trigger_map["reason"])
    elif trigger_map["state"].startswith("SELL"):
        st.info("🔴 " + trigger_map["reason"])
    elif "EXTENDED" in trigger_map["state"]:
        st.warning("🟡 " + trigger_map["reason"])
    else:
        st.info("⚪ " + trigger_map["reason"])

    st.subheader("🧱 Order Block Map — 5M")
    st.caption("Order Block is an additional location filter, not a standalone BUY/SELL signal.")
    ob1, ob2 = st.columns(2)
    bull_ob = ob_map.get("bullish")
    bear_ob = ob_map.get("bearish")
    with ob1:
        if bull_ob:
            st.success(f"🟢 Bullish OB • {bull_ob['status']}")
            st.write(f"Zone: **{bull_ob['low']:,.2f} – {bull_ob['high']:,.2f}**")
            st.write(f"Current price in zone: **{'YES' if ob_map['near_bullish'] else 'NO'}**")
            st.caption(f"Origin candle: {bull_ob['time']}")
        else:
            st.info("No valid bullish Order Block found in recent 5M candles.")
    with ob2:
        if bear_ob:
            st.error(f"🔴 Bearish OB • {bear_ob['status']}")
            st.write(f"Zone: **{bear_ob['low']:,.2f} – {bear_ob['high']:,.2f}**")
            st.write(f"Current price in zone: **{'YES' if ob_map['near_bearish'] else 'NO'}**")
            st.caption(f"Origin candle: {bear_ob['time']}")
        else:
            st.info("No valid bearish Order Block found in recent 5M candles.")
    st.caption(ob_map.get("note", ""))
    with st.expander("🛡️ A+ False-Signal Filter"):
        fg1, fg2 = st.columns(2)
        with fg1:
            st.write("🟢 **BUY filter**")
            st.write(f"OB location: **{'YES' if ob_confirmation.get('buy_ob_near') else 'NO'}**")
            st.write(f"Recent liquidity sweep: **{'YES' if ob_confirmation.get('buy_sweep') else 'NO'}**")
            st.write(f"Recent BOS UP: **{'YES' if ob_confirmation.get('buy_break') else 'NO'}**")
            st.caption(ob_confirmation.get("buy_reason", "—"))
        with fg2:
            st.write("🔴 **SELL filter**")
            st.write(f"OB location: **{'YES' if ob_confirmation.get('sell_ob_near') else 'NO'}**")
            st.write(f"Recent liquidity sweep: **{'YES' if ob_confirmation.get('sell_sweep') else 'NO'}**")
            st.write(f"Recent BOS DOWN: **{'YES' if ob_confirmation.get('sell_break') else 'NO'}**")
            st.caption(ob_confirmation.get("sell_reason", "—"))
        st.warning("A+ BUY/SELL is blocked unless the matching Order Block, recent liquidity sweep, and directional BOS all agree. This may reduce signal frequency; it does not guarantee accuracy.")

    st.subheader("Smart Flow Checklist")
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("1H trend", trend_state(h1e) if not h1e.empty else "—")
    s5 = structure(m5e) if not m5e.empty else {}
    cc2.metric("Sweep", "YES" if s5.get("sweep") else "NO")
    cc3.metric("CHoCH", "YES" if s5.get("choch") else "NO")
    cc4.metric("BOS", "UP" if s5.get("bos_bull") else "DOWN" if s5.get("bos_bear") else "NO")
    cc5.metric("ADX", f"{m5e.adx.iloc[-1]:.1f}" if not m5e.empty else "—")

    st.subheader("Key Support / Resistance")
    sr5 = support_resistance(m5e) if not m5e.empty else {"support": [], "resistance": []}
    sr15 = support_resistance(m15e) if not m15e.empty else {"support": [], "resistance": []}
    sr1 = support_resistance(h1e) if not h1e.empty else {"support": [], "resistance": []}
    a, b = st.columns(2)
    with a:
        st.markdown("🟢 **Support**")
        st.write("1H:", ", ".join(f"{x:,.2f}" for x in sr1["support"]) or "—")
        st.write("15M:", ", ".join(f"{x:,.2f}" for x in sr15["support"]) or "—")
        st.write("5M:", ", ".join(f"{x:,.2f}" for x in sr5["support"]) or "—")
    with b:
        st.markdown("🔴 **Resistance**")
        st.write("1H:", ", ".join(f"{x:,.2f}" for x in sr1["resistance"]) or "—")
        st.write("15M:", ", ".join(f"{x:,.2f}" for x in sr15["resistance"]) or "—")
        st.write("5M:", ", ".join(f"{x:,.2f}" for x in sr5["resistance"]) or "—")

    st.subheader("⚡ Big Candle / Early Entry Engine")
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Status", engine["status"]); ec2.metric("Direction", engine["direction"]); ec3.metric("Early Score", f"{engine['score']}/100"); ec4.metric("Quality", engine["quality"])
    st.info(f"🔴/🟢 {engine['status']} • UP trigger {engine['up_trigger']:,.2f} • DOWN trigger {engine['down_trigger']:,.2f}")
    e1, e2, e3 = st.columns(3)
    e1.metric("Live candle range / ATR", f"{engine.get('atr_ratio', 0):.2f}x")
    e2.metric("Body / range", f"{engine.get('body_ratio', 0):.0%}")
    e3.metric("Tick speed", f"{engine.get('speed', 0):+.3f}/s")
    if engine.get("reasons"):
        st.caption("Why: " + " • ".join(engine["reasons"]))

    with st.expander("📡 Data health / freshness"):
        st.write(f"WebSocket connected: **{snap['connected']}**")
        st.write(f"Last tick age: **{last_tick_age:.1f}s**" if last_tick_age is not None else "Last tick age: —")
        st.write(f"Displayed price source: **{price_source}**")
        for label, df in [("1H REST", h1), ("15M REST", m15), ("5M REST / 1M→5M fallback", rest5), ("Live 5M", live5)]:
            if not df.empty:
                ts = df.time.iloc[-1]
                age = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds() / 60
                st.write(f"{label}: {ts.strftime('%Y-%m-%d %H:%M UTC')} • {age:.1f} min old")
            else:
                st.write(f"{label}: unavailable")
        if snap["error"]:
            st.warning(snap["error"])

# Fragment reruns only the live dashboard. No full-page refresh loop.
live_dashboard()

# Research/backtest UI intentionally separated from the live dashboard in V2.


if view_mode == "🧭 Full Pro":

    # Forward / Paper Testing
    st.subheader("📋 Forward / Paper Trading")
    st.caption("Live signals are recorded here without placing broker orders. This replaces the unstable OOS validation.")
    trades = _paper_trades()
    closed = [t for t in trades if t.get("status") != "OPEN"]
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    if closed:
        wins = sum(1 for t in closed if t.get("status") == "TP1")
        losses = sum(1 for t in closed if t.get("status") == "LOSS")
        total_r = sum(float(t.get("r") or 0) for t in closed)
        pf_num = sum(max(float(t.get("r") or 0),0) for t in closed)
        pf_den = -sum(min(float(t.get("r") or 0),0) for t in closed)
        pf = pf_num / pf_den if pf_den else (999.0 if pf_num else 0.0)
        f1,f2,f3,f4 = st.columns(4)
        f1.metric("Closed", len(closed))
        f2.metric("Win %", f"{(wins/len(closed)*100):.1f}%")
        f3.metric("Paper PF", f"{pf:.2f}")
        f4.metric("Net R", f"{total_r:+.1f}")
    else:
        st.info("No A+ paper trades recorded yet. The log starts only when Smart Flow ≥70 AND Early Engine ≥70.")
    if open_trades:
        st.write(f"🟡 Open paper trades: **{len(open_trades)}**")
    if trades:
        rows=[]
        for t in reversed(trades[-20:]):
            rows.append({
                "Time": t["time"].strftime("%d-%m %H:%M UTC"),
                "Side": t["side"],
                "Entry": round(t["entry"],2),
                "SL": round(t["sl"],2),
                "TP1": round(t["tp1"],2),
                "Signal": t["signal_score"],
                "Early": t["early_score"],
                "Status": t["status"],
                "R": t.get("r"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear paper-trading log"):
        st.session_state["paper_trades"] = []
        st.rerun()


    # Telegram status
    # Get a lightweight current price for the manual Telegram test without
    # depending on a variable inside the live fragment.
    test_price = None
    try:
        _q = get_live_quote(SIFTING_KEY)
        if _q and _q.get("price") is not None:
            test_price = float(_q["price"])
    except Exception:
        pass
    if test_price is None:
        test_price = 0.0

    with st.expander("🔔 Telegram A/A+ alerts"):
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            st.success("Telegram configuration loaded.")
            st.caption("Alerts are sent only when Smart Flow ≥70 AND Early Engine ≥70. Duplicate states are throttled.")

            if st.button("🔎 Check Telegram connection"):
                ok, result = telegram_diagnostics()
                st.session_state["telegram_diag"] = (ok, result)

            if st.button("📩 Send Telegram test alert"):
                ok, result = telegram_send(
                    f"🟡 Smart Flow Gold TEST\nXAUUSD: {test_price:.2f}\nTelegram connection is working."
                )
                st.session_state["telegram_test"] = (ok, result)

            if "telegram_diag" in st.session_state:
                ok, result = st.session_state["telegram_diag"]
                (st.success if ok else st.error)(result)
            if "telegram_test" in st.session_state:
                ok, result = st.session_state["telegram_test"]
                (st.success if ok else st.error)(result)
        else:
            st.warning("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to Streamlit Secrets.")

    st.divider()
    st.caption(
        "Institutional Engine uses rule-based confluence. XAUUSD volume/order-flow are proxies, not COMEX order-book data. "
        "Alerts are not guaranteed predictions or automatic orders; forward-test before real-money use."
    )
