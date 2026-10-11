"""
FX Treasury Copilot - 7-day dynamic FX projection with Historical Backtesting Module.

Hugging Face pipelines used (course requirement: at least two):
  1. text-classification : fine-tuned sentiment model          (Student A, Model 1)
  2. summarization       : fine-tuned news-briefing model       (Student B, Model 2)
  3. time-series forecast: amazon/chronos-bolt-small            (pre-trained, zero-shot)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import requests
import urllib.parse
import feedparser
import gc
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SENTIMENT_MODEL_ID = "chena2339/finbert-fx-sentiment"   # Student A - Model 1
SUMMARY_MODEL_ID   = "sshleifer/distilbart-cnn-12-6"    # Pretrained winner
CHRONOS_MODEL_ID   = "amazon/chronos-bolt-small"        # pre-trained, NOT fine-tuned

FX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "HKD/USD": "HKDUSD=X",
    "INR/USD": "INRUSD=X",
}

VOL_BENCHMARKS = {"^VIX": 20.0, "^V2TX": 20.0, "^VFTSE": 15.0}
ESI_BASELINE = 100.0

NEWS_QUERIES = {
    "EUR/USD": "euro dollar ECB exchange rate",
    "GBP/USD": "pound sterling dollar Bank of England",
    "HKD/USD": "Hong Kong dollar peg HKMA",
    "INR/USD": "rupee dollar RBI exchange rate",
}

FORECAST_DAYS   = 7
WEEKEND_FACTOR  = 0.995   
EVENT_SPIKE     = 0.002   
EVENT_PAIRS     = {"EUR/USD", "GBP/USD", "INR/USD"}   
SENTIMENT_WEIGHT = 0.0015  
VOL_WEIGHT       = 0.0005  
ESI_WEIGHT       = 0.0005  

st.set_page_config(page_title="FX Treasury Copilot", layout="wide")

# ---------------------------------------------------------------------------
# Data loaders (cached) - Extended to 5y history for backtesting support
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_fx_history(ticker: str, period: str = "5y") -> pd.Series:
    """Download daily closes with multi-source fallback (yfinance -> Frankfurter API -> static default)."""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) > 0:
                close.index = pd.to_datetime(close.index).tz_localize(None)
                return close
    except Exception:
        pass

    is_fx_ticker = ticker in FX_PAIRS.values()

    if is_fx_ticker:
        try:
            base_curr = ticker[:3]
            quote_curr = ticker[3:6]
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=1825)).strftime('%Y-%m-%d') # ~5 years
            url_hist = f"https://frankfurter.app{start_date}..?from={base_curr}&to={quote_curr}"
            resp = requests.get(url_hist, timeout=5).json()
            rates = resp.get("rates", {})
            if rates:
                data = {pd.to_datetime(dt): val[quote_curr] for dt, val in rates.items()}
                s = pd.Series(data).sort_index()
                if len(s) > 0:
                    return s
        except Exception:
            pass

    if is_fx_ticker:
        defaults = {"EURUSD=X": 1.0850, "GBPUSD=X": 1.2700, "HKDUSD=X": 0.1280, "INRUSD=X": 0.0120}
        base_price = defaults[ticker]
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1825, freq="D")
        return pd.Series(base_price, index=dates)

    return pd.Series(dtype=float)

@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_vol(ticker: str, target_date: pd.Timestamp) -> float:
    """Fetch the closest historical volatility index value before or on the target date."""
    try:
        s = load_fx_history(ticker, period="5y")
        historical_data = s.loc[:target_date]
        if not historical_data.empty:
            return float(historical_data.iloc[-1])
    except Exception:
        pass
    return VOL_BENCHMARKS.get(ticker, 20.0)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_headlines(query: str, n: int) -> list:
    """Latest headlines from Google News RSS for one query."""
    url = "https://google.com" + urllib.parse.quote(query) + "&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [e.title for e in feed.entries[:n]]

# ---------------------------------------------------------------------------
# Hugging Face pipelines (cached resources)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the forecasting model (Chronos-Bolt) ...")
def load_forecaster():
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(CHRONOS_MODEL_ID, device_map="cpu")

@st.cache_resource(show_spinner="Loading the fine-tuned sentiment model ...")
def load_sentiment_model(model_id: str):
    from transformers import pipeline
    return pipeline("text-classification", model=model_id, truncation=True, max_length=128)

@st.cache_resource(show_spinner="Loading the fine-tuned briefing model ...")
def load_summary_model(model_id: str):
    from transformers import pipeline
    return pipeline("summarization", model=model_id)

# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------
def chronos_forecast(pipe, history: pd.Series, days: int = FORECAST_DAYS):
    import torch
    values = pd.to_numeric(history, errors="coerce").dropna().to_numpy(dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("No closes available for Chronos.")
    last_spot = float(values[-1])

    min_length = 32
    if values.size < min_length:
        pad = np.full(min_length - values.size, np.nan, dtype=np.float32)
        values = np.concatenate([pad, values])
    context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    try:
        quantiles, _ = pipe.predict_quantiles(context, prediction_length=days, quantile_levels=[0.1, 0.5, 0.9])
        q = quantiles.detach().float().cpu().numpy()
        p10, median, p90 = q[:, 0], q[:, 1], q[:, 2]
    except Exception as exc:
        st.warning(f"Chronos fallback used ({type(exc).__name__}).")
        drift = float(np.log(values[-1] / values[-2])) if np.isfinite(values[-2]) else 0.0
        steps = np.arange(1, days + 1)
        median = last_spot * np.exp(drift * steps)
        p10, p90 = median * 0.99, median * 1.01

    fill = lambda a: np.nan_to_num(np.asarray(a, dtype=float), nan=last_spot)
    return fill(median), fill(p10), fill(p90)

def score_headlines(clf, headlines: list):
    if not headlines:
        return 0.0, {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}, []
    dists = clf(headlines, top_k=None, batch_size=16)
    signed, counts, rows = [], {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}, []
    for headline, dist in zip(headlines, dists):
        d = {x["label"].upper(): x["score"] for x in dist}
        signed.append(d.get("POSITIVE", 0.0) - d.get("NEGATIVE", 0.0))
        top = max(dist, key=lambda x: x["score"])
        counts[top["label"].upper()] += 1
        rows.append({"Headline": headline, "Sentiment": top["label"].title(), "Confidence": round(float(top["score"]), 3)})
    return float(np.mean(signed)), counts, rows

def generate_briefing(summarizer, headlines: list, max_items: int = 8) -> str:
    if not headlines:
        return "No headlines available."
    text = " ".join(headlines[:max_items])[:3000]
    return summarizer(text, max_new_tokens=64, min_length=15, truncation=True)[0]["summary_text"]

# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------
def compute_daily_vol_adjustment(pair: str, vol_vals: dict, esi: float) -> float:
    adj = 0.0
    vix = vol_vals.get("^VIX")
    if vix is not None:
        adj += -VOL_WEIGHT * (vix - VOL_BENCHMARKS["^VIX"]) / VOL_BENCHMARKS["^VIX"]
    if pair == "EUR/USD":
        v2tx = vol_vals.get("^V2TX")
        if v2tx is not None:
            adj += -VOL_WEIGHT * (v2tx - VOL_BENCHMARKS["^V2TX"]) / VOL_BENCHMARKS["^V2TX"]
        adj += ESI_WEIGHT * (esi - ESI_BASELINE) / ESI_BASELINE
    elif pair == "GBP/USD":
        vftse = vol_vals.get("^VFTSE")
        if vftse is not None:
            adj += -VOL_WEIGHT * (vftse - VOL_BENCHMARKS["^VFTSE"]) / VOL_BENCHMARKS["^VFTSE"]
    elif pair == "HKD/USD":
        adj *= 0.2
    return adj

def build_projection(start_date, median, p10, p90, sentiment_score, daily_vol_adj,
                     pair, event_on, weekend_fee_on, seed=7) -> pd.DataFrame:
    dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
    rng = np.random.default_rng(seed)

    rows, cum_adj = [], 0.0
    for t in range(FORECAST_DAYS):
        day_no = t + 1
        cum_adj += SENTIMENT_WEIGHT * sentiment_score * (0.9 ** t)
        cum_adj += daily_vol_adj
        band_widen = 0.0
        if event_on and pair in EVENT_PAIRS and day_no >= 2:
            band_widen = EVENT_SPIKE
            cum_adj += rng.normal(0.0, EVENT_SPIKE / 4)

        rate = median[t] * (1 + cum_adj)
        low = p10[t] * (1 + cum_adj) * (1 - band_widen)
        high = p90[t] * (1 + cum_adj) * (1 + band_widen)

        note = ""
        if weekend_fee_on and dates[t].weekday() >= 5:
            rate, low, high = rate * WEEKEND_FACTOR, low * WEEKEND_FACTOR, high * WEEKEND_FACTOR
            note = "Weekend fee applied"

        rows.append({"Date": dates[t].date(), pair: round(float(rate), 4),
                     "Low (P10)": round(float(low), 4), "High (P90)": round(float(high), 4), "Note": note})
