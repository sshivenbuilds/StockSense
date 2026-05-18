import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="StockSense",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0a0a; color: #f0f0f0; }
#MainMenu, footer, header { visibility: hidden !important; }
.stApp { background: #0a0a0a; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 0.5rem !important; max-width: 1200px; }

.stButton > button {
    background: #ffffff; color: #0a0a0a;
    font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.9rem;
    border: none; border-radius: 8px; padding: 0.65rem 1.5rem; width: 100%;
    transition: all 0.15s; letter-spacing: 0.5px;
}
.stButton > button:hover { background: #e0e0e0; }

.price-card {
    background: #111; border: 1px solid #333; border-radius: 12px;
    padding: 1.5rem; text-align: center; margin-bottom: 0.8rem;
}
.price-card .label { font-size: 0.7rem; letter-spacing: 2.5px; color: #666; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; }
.price-card .price { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #fff; line-height: 1.2; margin-top: 0.3rem; }
.price-card .change { margin-top: 0.4rem; font-size: 0.95rem; font-weight: 500; }
.price-card .change.up   { color: #4caf50; }
.price-card .change.down { color: #f44336; }

.stat-card {
    background: #111; border: 1px solid #222; border-radius: 8px;
    padding: 0.7rem 1rem; margin-bottom: 0.4rem;
    display: flex; justify-content: space-between; align-items: center;
}
.stat-card .stat-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1px; font-family: 'IBM Plex Mono', monospace; }
.stat-card .stat-value  { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #f0f0f0; }

.indic-card {
    background: #0d0d0d; border: 1px solid #1a1a1a; border-radius: 8px;
    padding: 0.7rem 1rem; margin-bottom: 0.4rem;
    display: flex; flex-direction: column; gap: 0.25rem;
}
.indic-label { font-size: 0.65rem; color: #444; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'IBM Plex Mono', monospace; }

.mode-badge-intraday {
    display: inline-block; background: #0a2a0a; border: 1px solid #4caf50;
    color: #4caf50; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    border-radius: 4px; padding: 0.2rem 0.6rem; letter-spacing: 1.5px; margin-bottom: 0.6rem;
}
.mode-badge-daily {
    display: inline-block; background: #1a1200; border: 1px solid #ff8c00;
    color: #ff8c00; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    border-radius: 4px; padding: 0.2rem 0.6rem; letter-spacing: 1.5px; margin-bottom: 0.6rem;
}

.info-box {
    background: #111; border-left: 2px solid #333; border-radius: 0 6px 6px 0;
    padding: 0.7rem 1rem; font-size: 0.78rem; color: #555; line-height: 1.6; margin-top: 0.6rem;
}
.badge-stationary    { border: 1px solid #4caf50; color: #4caf50; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; border-radius: 4px; padding: 0.2rem 0.5rem; }
.badge-nonstationary { border: 1px solid #f44336; color: #f44336; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; border-radius: 4px; padding: 0.2rem 0.5rem; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 2px; color: #444; text-transform: uppercase; margin-bottom: 0.3rem; }
.divider-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 2px; color: #2a2a2a; text-transform: uppercase; margin: 0.5rem 0 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ────────────────────────────────────────────────
INDIAN_INDICES = {"^NSEI", "^BSESN", "^NSEBANK", "^CNXIT"}


# ── MARKET MODE DETECTION ────────────────────────────────────
def get_market_mode(ticker):
    """
    Returns 'intraday' if the market for this ticker is currently open,
    'daily' otherwise. Based purely on real-time UTC clock + exchange hours.
    """
    now_utc  = datetime.now(timezone.utc)
    is_india = ticker.endswith('.NS') or ticker.endswith('.BO') or ticker in INDIAN_INDICES

    if is_india:
        local_tz = timezone(timedelta(hours=5, minutes=30))   # IST
        oh, om, ch, cm = 9, 15, 15, 30
    else:
        # US EDT Mar-Nov = UTC-4, EST Nov-Mar = UTC-5
        offset   = -4 if 3 <= now_utc.month <= 11 else -5
        local_tz = timezone(timedelta(hours=offset))
        oh, om, ch, cm = 9, 30, 16, 0

    now_l  = now_utc.astimezone(local_tz)
    t_open = now_l.replace(hour=oh, minute=om, second=0, microsecond=0)
    t_shut = now_l.replace(hour=ch, minute=cm, second=0, microsecond=0)

    return 'intraday' if (now_l.weekday() < 5 and t_open <= now_l <= t_shut) else 'daily'


# ── DATE / TIME HELPERS ──────────────────────────────────────
def next_trading_day(last_date):
    """Walk forward day-by-day skipping Sat/Sun. No BDay, no today()."""
    cand = last_date + pd.Timedelta(days=1)
    while cand.weekday() >= 5:
        cand += pd.Timedelta(days=1)
    return cand

def to_local(ts, is_india):
    """
    Convert a yfinance hourly timestamp (UTC or tz-naive treated as UTC)
    to the correct local time for display.
    Indian stocks/indices → IST (UTC+5:30)
    US stocks            → EDT (UTC-4) Mar-Nov, EST (UTC-5) Nov-Mar
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')          # yfinance sometimes returns tz-naive UTC
    else:
        ts = ts.tz_convert('UTC')

    if is_india:
        tz = timezone(timedelta(hours=5, minutes=30))
    else:
        now_utc = datetime.now(timezone.utc)
        offset  = -4 if 3 <= now_utc.month <= 11 else -5
        tz      = timezone(timedelta(hours=offset))

    return ts.astimezone(tz)

def fmt_time(ts, is_india):
    """Format a UTC/tz-naive yfinance timestamp as local time with AM/PM."""
    local = to_local(ts, is_india)
    return local.strftime("%d %B %Y, %I:%M %p")

def next_hour_str(last_ts, is_india):
    """Return the next-hour timestamp string in local time with AM/PM."""
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize('UTC')
    next_ts = last_ts + pd.Timedelta(hours=1)
    return fmt_time(next_ts, is_india)


# ── INDICATOR COMPUTATION ────────────────────────────────────
def compute_indicators(df, include_rsi=True):
    df = df.copy()

    # ── ATR% (volatility, needs High/Low/Close)
    if all(c in df.columns for c in ['High', 'Low', 'Close']):
        pc  = df['Close'].shift(1)
        tr  = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - pc).abs(),
            (df['Low']  - pc).abs()
        ], axis=1).max(axis=1)
        df['ATR']     = tr.rolling(14).mean()
        # Express as % of price so it's scale-independent across stocks
        df['ATR_pct'] = (df['ATR'] / df['Close'] * 100).replace([np.inf, -np.inf], np.nan)
    else:
        df['ATR'] = df['ATR_pct'] = np.nan

    # ── Volume Ratio (today vs 20-day avg)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vm = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = (df['Volume'] / vm).replace([np.inf, -np.inf], np.nan)
    else:
        df['vol_ratio'] = np.nan

    # ── RSI (daily only; stored raw 0-100, exog uses /100 normalised)
    if include_rsi:
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df['RSI']      = 100 - (100 / (1 + rs))
        df['RSI_norm'] = df['RSI'] / 100          # feed this to ARIMAX
    else:
        df['RSI'] = df['RSI_norm'] = np.nan

    return df


def get_exog_cols(df, include_rsi=True):
    """Return exog column names that have at least 20 valid rows."""
    cands = ['vol_ratio', 'ATR_pct']
    if include_rsi:
        cands.append('RSI_norm')
    return [c for c in cands if c in df.columns and df[c].notna().sum() > 20]


def prepare_aligned(df, exog_cols):
    """Drop rows where price OR any exog is NaN so series + exog stay in sync."""
    series = df['Close'].copy()
    if not exog_cols:
        return series.dropna(), None
    exog  = df[exog_cols].copy()
    valid = series.notna() & exog.notna().all(axis=1)
    return series[valid], exog[valid]


# ── DATA FETCH ───────────────────────────────────────────────
def _clean_df(df, min_rows=30):
    """Shared cleaning for OHLCV dataframes. Returns None if insufficient rows."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Rename 'Adj Close' to 'Close' if needed
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})
    if 'Close' not in df.columns:
        return None
    cols = [c for c in ['Close', 'High', 'Low', 'Volume'] if c in df.columns]
    df   = df[cols].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)   # strip tz for daily
    df   = df[df.index.weekday < 5].sort_index()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce').ffill()
    for c in ['High', 'Low', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').ffill()
    df = df.dropna(subset=['Close'])
    return df if len(df) >= min_rows else None


def _try_download(t, start):
    """
    Tries yf.Ticker().history() first (more reliable for NSE),
    falls back to yf.download() if that fails.
    """
    # Method 1: Ticker.history — more reliable for NSE/BSE tickers
    try:
        tk  = yf.Ticker(t)
        df  = tk.history(start=start, interval="1d", auto_adjust=True, timeout=30)
        if df is not None and not df.empty and len(df) >= 10:
            return df
    except Exception:
        pass

    # Method 2: yf.download — classic fallback
    try:
        df = yf.download(t, start=start, interval="1d",
                         progress=False, timeout=30, auto_adjust=True)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Method 3: yf.download without auto_adjust (some tickers need this)
    try:
        df = yf.download(t, start=start, interval="1d",
                         progress=False, timeout=30, auto_adjust=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data_daily(ticker):
    """
    Robust daily OHLCV fetch with multiple strategies:
    - Tries .NS ticker with start dates 2019, 2021, 2022 (covers recent IPOs)
    - Falls back to .BO (BSE) if .NS fails
    - Each strategy tries Ticker.history() then download() then download(no-adjust)
    - ttl=3600 prevents stale None from being cached forever
    """
    starts = ["2019-01-01", "2021-01-01", "2022-01-01"]
    tickers_to_try = [ticker]
    if ticker.endswith('.NS'):
        tickers_to_try.append(ticker.replace('.NS', '.BO'))

    for t in tickers_to_try:
        for start in starts:
            raw = _try_download(t, start)
            if raw is not None:
                cleaned = _clean_df(raw, min_rows=30)
                if cleaned is not None:
                    return compute_indicators(cleaned, include_rsi=True)
    return None


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_data_intraday(ticker):
    """
    Tries Ticker.history() then yf.download() for intraday hourly data.
    ttl=1800 so cache refreshes every 30 mins during market hours.
    """
    tickers_to_try = [ticker]
    if ticker.endswith('.NS'):
        tickers_to_try.append(ticker.replace('.NS', '.BO'))

    for t in tickers_to_try:
        # Method 1: Ticker.history
        try:
            tk = yf.Ticker(t)
            df = tk.history(period="60d", interval="1h",
                            auto_adjust=True, timeout=30)
            if df is not None and not df.empty:
                cleaned = _clean_df(df, min_rows=20)
                if cleaned is not None:
                    return compute_indicators(cleaned, include_rsi=False)
        except Exception:
            pass

        # Method 2: yf.download
        try:
            df = yf.download(t, period="60d", interval="1h",
                             progress=False, timeout=30, auto_adjust=True)
            if df is not None and not df.empty:
                cleaned = _clean_df(df, min_rows=20)
                if cleaned is not None:
                    return compute_indicators(cleaned, include_rsi=False)
        except Exception:
            pass

    return None




# ── ARIMA GRID SEARCH ────────────────────────────────────────
def find_best_order(series, exog=None):
    best_aic, best_order = np.inf, (1, 1, 1)
    for p in range(0, 3):
        for q in range(0, 3):
            if p == 0 and q == 0:
                continue
            try:
                r = ARIMA(series, exog=exog, order=(p, 1, q)).fit()
                if r.aic < best_aic:
                    best_aic, best_order = r.aic, (p, 1, q)
            except Exception:
                continue
    return best_order, best_aic


# ── RUN DAILY AUTO-ARIMAX ────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_arimax_daily(df):
    exog_cols          = get_exog_cols(df, include_rsi=True)
    series, exog       = prepare_aligned(df, exog_cols)

    train_s, test_s    = series[:-30], series[-30:]
    train_x            = exog[:-30]   if exog is not None else None
    test_x             = exog[-30:]   if exog is not None else None

    best_order, best_aic = find_best_order(train_s, train_x)

    # 30-day backtest
    preds  = ARIMA(train_s, exog=train_x, order=best_order).fit() \
                   .forecast(steps=30, exog=test_x)
    rmse   = np.sqrt(mean_squared_error(test_s.values[:len(preds)], preds.values))

    # Final 1-step forecast on full series
    full_fit   = ARIMA(series, exog=exog, order=best_order).fit()
    next_exog  = exog.iloc[[-1]] if exog is not None else None
    next_price = float(full_fit.forecast(steps=1, exog=next_exog).iloc[0])

    adf_result = adfuller(series.diff().dropna())

    def lv(col):
        return float(df[col].dropna().iloc[-1]) if col in df.columns and df[col].notna().any() else None

    indicators = {'vol_ratio': lv('vol_ratio'), 'ATR_pct': lv('ATR_pct'), 'RSI': lv('RSI')}
    return next_price, rmse, adf_result, best_order, best_aic, indicators, exog_cols


# ── RUN INTRADAY ARIMA(1,1,1) + exog ────────────────────────
@st.cache_data(show_spinner=False)
def run_arima_intraday(df):
    exog_cols    = get_exog_cols(df, include_rsi=False)
    series, exog = prepare_aligned(df, exog_cols)
    order        = (1, 1, 1)

    # Backtest on last N candles
    n_back  = max(5, min(10, len(series) // 5))
    train_s, test_s = series[:-n_back], series[-n_back:]
    train_x = exog[:-n_back] if exog is not None else None
    test_x  = exog[-n_back:] if exog is not None else None

    preds  = ARIMA(train_s, exog=train_x, order=order).fit() \
                   .forecast(steps=n_back, exog=test_x)
    rmse   = np.sqrt(mean_squared_error(test_s.values[:len(preds)], preds.values))

    full_fit   = ARIMA(series, exog=exog, order=order).fit()
    next_exog  = exog.iloc[[-1]] if exog is not None else None
    next_price = float(full_fit.forecast(steps=1, exog=next_exog).iloc[0])

    def lv(col):
        return float(df[col].dropna().iloc[-1]) if col in df.columns and df[col].notna().any() else None

    indicators = {'vol_ratio': lv('vol_ratio'), 'ATR_pct': lv('ATR_pct'), 'RSI': None}
    return next_price, rmse, order, indicators, exog_cols


# ── CHARTS ───────────────────────────────────────────────────
def make_chart_daily(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    fig.subplots_adjust(bottom=0.14, top=0.92, left=0.08, right=0.99)
    recent = df['Close'][df.index >= df.index[-1] - pd.DateOffset(days=730)]
    vals   = np.array(recent.values).flatten()
    ax.plot(recent.index, vals, color='#ffffff', linewidth=1.4, alpha=0.9)
    ax.fill_between(recent.index, vals, vals.min() * 0.97, alpha=0.05, color='#ffffff')
    ma30 = pd.Series(vals, index=recent.index).rolling(30).mean()
    ax.plot(ma30.index, ma30.values, color='#ff8c00', linewidth=1.4, linestyle='--', label='30D MA')
    ax.set_ylim(bottom=vals.min() * 0.97, top=vals.max() * 1.02)
    ax.set_xlim(recent.index[0], recent.index[-1])
    ax.set_title(f"{ticker.upper()} — Last 2 Years (Daily)", color='#888',
                 fontsize=10, pad=6, fontfamily='monospace')
    ax.tick_params(colors='#444', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=25)
    for spine in ax.spines.values(): spine.set_edgecolor('#1e1e1e')
    ax.grid(axis='y', color='#1a1a1a', linewidth=0.5, linestyle='--')
    ax.legend(facecolor='#111', edgecolor='#222', labelcolor='#ff8c00', fontsize=8)
    return fig


def make_chart_intraday(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    fig.subplots_adjust(bottom=0.18, top=0.92, left=0.08, right=0.99)
    recent = df['Close'].tail(40)
    vals   = np.array(recent.values).flatten()
    ax.plot(recent.index, vals, color='#ffffff', linewidth=1.4, alpha=0.9)
    ax.fill_between(recent.index, vals, vals.min() * 0.97, alpha=0.05, color='#ffffff')
    ma5 = pd.Series(vals, index=recent.index).rolling(5).mean()
    ax.plot(ma5.index, ma5.values, color='#ff8c00', linewidth=1.4, linestyle='--', label='5H MA')
    ax.set_ylim(bottom=vals.min() * 0.97, top=vals.max() * 1.02)
    ax.set_xlim(recent.index[0], recent.index[-1])
    ax.set_title(f"{ticker.upper()} — Intraday · Last 40 Hourly Candles", color='#888',
                 fontsize=10, pad=6, fontfamily='monospace')
    ax.tick_params(colors='#444', labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=0)
    for spine in ax.spines.values(): spine.set_edgecolor('#1e1e1e')
    ax.grid(axis='y', color='#1a1a1a', linewidth=0.5, linestyle='--')
    ax.legend(facecolor='#111', edgecolor='#222', labelcolor='#ff8c00', fontsize=8)
    return fig


# ── INDICATOR BADGE RENDERERS ────────────────────────────────
def vol_badge(v):
    if v is None:
        return '<span style="color:#444;font-size:0.75rem;font-family:IBM Plex Mono,monospace;">N/A</span>'
    if v > 1.5:
        clr, tag = '#4caf50', 'HIGH'
    elif v < 0.7:
        clr, tag = '#f44336', 'LOW'
    else:
        clr, tag = '#f0f0f0', 'NORMAL'
    return (f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.88rem;font-weight:600;color:{clr};">'
            f'{v:.2f}x &nbsp;<span style="font-size:0.6rem;letter-spacing:1px;">{tag}</span></span>')

def atr_badge(v):
    if v is None:
        return '<span style="color:#444;font-size:0.75rem;font-family:IBM Plex Mono,monospace;">N/A</span>'
    if v > 2.0:
        clr, tag = '#ff8c00', 'HIGH VOL'
    elif v < 0.8:
        clr, tag = '#4caf50', 'CALM'
    else:
        clr, tag = '#f0f0f0', 'NORMAL'
    return (f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.88rem;font-weight:600;color:{clr};">'
            f'{v:.2f}% &nbsp;<span style="font-size:0.6rem;letter-spacing:1px;">{tag}</span></span>')

def rsi_badge(v):
    if v is None:
        return '<span style="color:#444;font-size:0.75rem;font-family:IBM Plex Mono,monospace;">N/A</span>'
    if v > 70:
        clr, tag = '#f44336', 'OVERBOUGHT'
    elif v < 30:
        clr, tag = '#4caf50', 'OVERSOLD'
    else:
        clr, tag = '#f0f0f0', 'NEUTRAL'
    return (f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.88rem;font-weight:600;color:{clr};">'
            f'{v:.1f} &nbsp;<span style="font-size:0.6rem;letter-spacing:1px;">{tag}</span></span>')


# ── STOCKS LIST ──────────────────────────────────────────────
STOCKS = {
    "── Select a stock ──": None,
    "📊 Nifty 50": "^NSEI",
    "📊 Sensex (BSE)": "^BSESN",
    "📊 Nifty Bank": "^NSEBANK",
    "📊 Nifty IT": "^CNXIT",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India (SBI)": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Punjab National Bank (PNB)": "PNB.NS",
    "Canara Bank": "CANBK.NS",
    "Union Bank of India": "UNIONBANK.NS",
    "Federal Bank": "FEDERALBNK.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "RBL Bank": "RBLBANK.NS",
    "AU Small Finance Bank": "AUBANK.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Cholamandalam Finance": "CHOLAFIN.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "Max Financial Services": "MFSL.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",
    "Mphasis": "MPHASIS.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "Coforge": "COFORGE.NS",
    "KPIT Technologies": "KPITTECH.NS",
    "Tata Elxsi": "TATAELXSI.NS",
    "Reliance Industries": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "Indian Oil (IOC)": "IOC.NS",
    "BPCL": "BPCL.NS",
    "Hindustan Petroleum (HPCL)": "HPCL.NS",
    "Petronet LNG": "PETRONET.NS",
    "Tata Power": "TATAPOWER.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Total Gas": "ATGL.NS",
    "NTPC": "NTPC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Coal India": "COALINDIA.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro (L&T)": "LT.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Grasim Industries": "GRASIM.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Hindustan Unilever (HUL)": "HINDUNILVR.NS",
    "Nestle India": "NESTLEIND.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Varun Beverages": "VBL.NS",
    "Jubilant FoodWorks": "JUBLFOOD.NS",
    "Trent": "TRENT.NS",
    "Titan Company": "TITAN.NS",
    "Bata India": "BATAINDIA.NS",
    "Page Industries": "PAGEIND.NS",
    "Relaxo Footwears": "RELAXO.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Lupin": "LUPIN.NS",
    "Biocon": "BIOCON.NS",
    "Torrent Pharmaceuticals": "TORNTPHARM.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS",
    "Divis Laboratories": "DIVISLAB.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "MM.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "TVS Motor Company": "TVSMOTOR.NS",
    "Motherson Sumi": "MOTHERSON.NS",
    "Balkrishna Industries (BKT)": "BALKRISIND.NS",
    "MRF": "MRF.NS",
    "Apollo Tyres": "APOLLOTYRE.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Vedanta": "VEDL.NS",
    "SAIL": "SAIL.NS",
    "NMDC": "NMDC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BEL (Bharat Electronics)": "BEL.NS",
    "HAL (Hindustan Aeronautics)": "HAL.NS",
    "Mazagon Dock": "MAZDOCK.NS",
    "BHEL": "BHEL.NS",
    "IRCTC": "IRCTC.NS",
    "Indian Railway Finance (IRFC)": "IRFC.NS",
    "Tata Communications": "TATACOMM.NS",
    "DLF": "DLF.NS",
    "Godrej Properties": "GODREJPROP.NS",
    "Prestige Estates": "PRESTIGE.NS",
    "Oberoi Realty": "OBEROIRLTY.NS",
    "Dixon Technologies": "DIXON.NS",
    "Havells India": "HAVELLS.NS",
    "Voltas": "VOLTAS.NS",
    "Crompton Greaves Consumer": "CROMPTON.NS",
    "Amber Enterprises": "AMBER.NS",
    "Kaynes Technology": "KAYNES.NS",
    "Nykaa (FSN E-Commerce)": "NYKAA.NS",
    "Delhivery": "DELHIVERY.NS",
    "PB Fintech (Policybazaar)": "POLICYBZR.NS",
    "Paytm (One97 Communications)": "PAYTM.NS",
    "IndiGo (InterGlobe Aviation)": "INDIGO.NS",
    "Indian Hotels (Taj)": "INDHOTEL.NS",
    "Lemon Tree Hotels": "LEMONTREE.NS",
    "PVR Inox": "PVRINOX.NS",
    "Zee Entertainment": "ZEEL.NS",
    "Sun TV Network": "SUNTV.NS",
    "Apple (NASDAQ)": "AAPL",
    "Tesla (NASDAQ)": "TSLA",
    "Google / Alphabet (NASDAQ)": "GOOGL",
    "Microsoft (NASDAQ)": "MSFT",
    "Amazon (NASDAQ)": "AMZN",
    "Meta (NASDAQ)": "META",
    "Nvidia (NASDAQ)": "NVDA",
    "S&P 500 Index": "^GSPC",
}


# ── UI HEADER ────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center;font-family:IBM Plex Mono,monospace;color:#fff;"
    "margin:0 0 0.2rem 0;'>Stock<span style='color:#ff8c00;'>Sense</span></h2>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#444;font-size:0.83rem;margin:0 0 1.2rem 0;'>"
    "Next-hour · Next-day forecasting &nbsp;·&nbsp; Auto ARIMAX &nbsp;·&nbsp; Volume · ATR · RSI</p>",
    unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid #1e1e1e;margin-bottom:1.2rem;'>",
            unsafe_allow_html=True)

col_drop, col_btn = st.columns([4, 1])
with col_drop:
    st.markdown("<div class='section-label'>Select a Stock</div>", unsafe_allow_html=True)
    selected_name = st.selectbox("", options=list(STOCKS.keys()), label_visibility="collapsed")
with col_btn:
    st.markdown("<div style='margin-top:1.45rem'></div>", unsafe_allow_html=True)
    run = st.button("FORECAST →")

ticker = STOCKS.get(selected_name) or None
st.markdown("<hr style='border:none;border-top:1px solid #1e1e1e;margin:1rem 0;'>",
            unsafe_allow_html=True)


# ── FORECAST LOGIC ───────────────────────────────────────────
if run and ticker:

    mode     = get_market_mode(ticker)
    currency = "₹" if (ticker.endswith(".NS") or ticker.endswith(".BO")
                       or ticker in INDIAN_INDICES) else "$"

    # ── MODE BADGE ──
    if mode == 'intraday':
        st.markdown("<span class='mode-badge-intraday'>⚡ INTRADAY MODE — Market Open</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<span class='mode-badge-daily'>📅 DAILY MODE — Market Closed</span>",
                    unsafe_allow_html=True)

    # ════════════════ INTRADAY ════════════════
    if mode == 'intraday':
        with st.status(f"📡 Fetching hourly data for {ticker}...", expanded=True) as status:
            st.write("📡 Fetching 60-day hourly OHLCV data...")
            df_h = fetch_data_intraday(ticker)
            if df_h is not None and len(df_h) >= 20:
                st.write("⚙️ Running ARIMA(1,1,1) with Volume Ratio + ATR%...")
                next_price, rmse, order, indicators, exog_cols = run_arima_intraday(df_h)
                st.write(f"✅ Model: ARIMA{order}  |  Exog: {exog_cols or 'none (index — no volume)'}")
                status.update(label="✅ Intraday forecast ready!", state="complete", expanded=False)
            else:
                st.write("⚠️ Hourly data unavailable — yfinance does not carry intraday data for this stock.")
                st.write("📅 Falling back to Daily Auto ARIMAX mode...")
                mode = 'fallback_daily'
                status.update(label="📅 Switched to daily forecast (no hourly data available)", state="complete", expanded=False)
        if mode == 'intraday' and df_h is not None and len(df_h) >= 20:
            last_price    = float(df_h['Close'].dropna().iloc[-1])
            last_ts       = df_h['Close'].dropna().index[-1]
            change        = next_price - last_price
            change_pct    = (change / last_price) * 100
            direction     = "▲" if change >= 0 else "▼"
            change_cls    = "up" if change >= 0 else "down"
            is_india      = (ticker.endswith('.NS') or ticker.endswith('.BO')
                             or ticker in INDIAN_INDICES)
            forecast_time = next_hour_str(last_ts, is_india)
            last_time_str = fmt_time(last_ts, is_india)

            left_col, right_col = st.columns([1, 2])
            with left_col:
                st.markdown(f"""
                <div class="price-card">
                    <div class="label">Next Hour Forecast</div>
                    <div class="price">{currency}{next_price:,.2f}</div>
                    <div class="change {change_cls}">{direction} {currency}{abs(change):.2f} &nbsp;|&nbsp; {change_pct:+.2f}%</div>
                    <div style="margin-top:0.8rem;font-size:0.74rem;color:#555;font-family:'IBM Plex Mono',monospace;">
                        {forecast_time}<br>
                        <span style="color:#333;">last candle: {last_time_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with right_col:
                st.markdown("<div class='section-label'>Intraday Price + 5H Moving Average</div>",
                            unsafe_allow_html=True)
                fig = make_chart_intraday(df_h, ticker)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:0.8rem 0;'>",
                        unsafe_allow_html=True)

            # Stats row 1
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">Last Price</div>'
                            f'<div class="stat-value">{currency}{last_price:,.2f}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">Model</div>'
                            f'<div class="stat-value">ARIMA{order}</div></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">RMSE (backtest)</div>'
                            f'<div class="stat-value">{currency}{rmse:.2f}</div></div>',
                            unsafe_allow_html=True)

            # Indicators row
            st.markdown("<div class='divider-label'>Live Indicators</div>", unsafe_allow_html=True)
            i1, i2, i3 = st.columns(3)
            with i1:
                st.markdown(f'<div class="indic-card"><div class="indic-label">Volume Ratio</div>'
                            f'{vol_badge(indicators["vol_ratio"])}</div>', unsafe_allow_html=True)
            with i2:
                st.markdown(f'<div class="indic-card"><div class="indic-label">ATR% (14-period)</div>'
                            f'{atr_badge(indicators["ATR_pct"])}</div>', unsafe_allow_html=True)
            with i3:
                st.markdown(f'<div class="indic-card"><div class="indic-label">RSI</div>'
                            f'<span style="color:#333;font-size:0.75rem;font-family:IBM Plex Mono,monospace;">'
                            f'Not used intraday (needs 14 daily candles)</span></div>',
                            unsafe_allow_html=True)

            st.markdown(
                '<div class="info-box">⚡ Intraday mode: ARIMA(1,1,1) fixed order on 60-day hourly data. '
                'Exogenous: Volume Ratio (today vs 20-period avg) + ATR% (14-period). '
                'RSI excluded — needs minimum 14 daily candles to be meaningful.</div>',
                unsafe_allow_html=True)

    # ════════════════ DAILY (+ fallback from intraday) ════════════════
    if mode in ('daily', 'fallback_daily'):
        if mode == 'fallback_daily':
            st.markdown(
                "<span class='mode-badge-daily'>📅 DAILY MODE — Intraday data unavailable for this stock</span>",
                unsafe_allow_html=True)
        with st.status(f"📡 Fetching data for {ticker}...", expanded=True) as status:
            st.write("📡 Fetching daily OHLCV data from 2019...")
            df_d = fetch_data_daily(ticker)
            if df_d is not None and len(df_d) >= 30:
                st.write("🔍 Running ADF stationarity test...")
                st.write("⚙️ Grid searching best ARIMAX (p,d,q) with Volume + ATR + RSI...")
                next_price, rmse, adf_result, best_order, best_aic, indicators, exog_cols = run_arimax_daily(df_d)
                st.write(f"✅ Best model: ARIMAX{best_order}  |  AIC: {best_aic:,.1f}  |  Exog: {exog_cols or 'none'}")
                status.update(label=f"✅ Forecast ready for {ticker}!", state="complete", expanded=False)
            else:
                status.update(label="❌ Failed to fetch data", state="error", expanded=False)

        if df_d is None or len(df_d) < 30:
            st.error(
                f"Could not fetch data for **{ticker}**. "
                "This can happen due to yfinance rate limits or missing data for this ticker. "
                "**Try clicking Forecast again** — if it fails 3 times, the stock may not be "
                "supported on yfinance for daily data."
            )
            st.info("💡 Tip: If you recently ran this stock and it failed, "
                    "click the menu (⋮) → **Clear cache** then try again.")
        else:
            last_price     = float(df_d['Close'].dropna().iloc[-1])
            last_date      = df_d['Close'].dropna().index[-1]
            forecast_date  = next_trading_day(last_date)
            change         = next_price - last_price
            change_pct     = (change / last_price) * 100
            direction      = "▲" if change >= 0 else "▼"
            change_cls     = "up" if change >= 0 else "down"
            last_date_str  = last_date.strftime("%d %B %Y")
            fcast_date_str = forecast_date.strftime("%A, %d %B %Y")

            left_col, right_col = st.columns([1, 2])
            with left_col:
                st.markdown(f"""
                <div class="price-card">
                    <div class="label">Next Day Forecast</div>
                    <div class="price">{currency}{next_price:,.2f}</div>
                    <div class="change {change_cls}">{direction} {currency}{abs(change):.2f} &nbsp;|&nbsp; {change_pct:+.2f}%</div>
                    <div style="margin-top:0.8rem;font-size:0.74rem;color:#555;font-family:'IBM Plex Mono',monospace;">
                        {fcast_date_str}<br>
                        <span style="color:#333;">data up to {last_date_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with right_col:
                st.markdown("<div class='section-label'>Price History + 30D Moving Average</div>",
                            unsafe_allow_html=True)
                fig = make_chart_daily(df_d, ticker)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:0.8rem 0;'>",
                        unsafe_allow_html=True)

            # Stats row 1 — model stats
            adf_p = adf_result[1]
            badge = ('<span class="badge-stationary">✔ STATIONARY (p={:.4f})</span>'.format(adf_p)
                     if adf_p < 0.05 else
                     '<span class="badge-nonstationary">✘ NON-STATIONARY (p={:.4f})</span>'.format(adf_p))

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">Last Close</div>'
                            f'<div class="stat-value">{currency}{last_price:,.2f}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">Best Model</div>'
                            f'<div class="stat-value">ARIMAX{best_order}</div></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">AIC Score</div>'
                            f'<div class="stat-value">{best_aic:,.1f}</div></div>',
                            unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">RMSE (30-day)</div>'
                            f'<div class="stat-value">{currency}{rmse:.2f}</div></div>',
                            unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;">'
                            f'<div class="stat-label">ADF Test</div>{badge}</div>',
                            unsafe_allow_html=True)

            # Indicators row
            st.markdown("<div class='divider-label'>Live Indicators (used as exogenous inputs)</div>",
                        unsafe_allow_html=True)
            i1, i2, i3 = st.columns(3)
            with i1:
                st.markdown(f'<div class="indic-card"><div class="indic-label">Volume Ratio</div>'
                            f'{vol_badge(indicators["vol_ratio"])}</div>', unsafe_allow_html=True)
            with i2:
                st.markdown(f'<div class="indic-card"><div class="indic-label">ATR% (14-day)</div>'
                            f'{atr_badge(indicators["ATR_pct"])}</div>', unsafe_allow_html=True)
            with i3:
                st.markdown(f'<div class="indic-card"><div class="indic-label">RSI (14-day)</div>'
                            f'{rsi_badge(indicators["RSI"])}</div>', unsafe_allow_html=True)

            exog_info = (f"Exogenous variables used: {', '.join(exog_cols)}"
                         if exog_cols else
                         "No exogenous variables (index has no volume — price-only ARIMA)")
            st.markdown(
                f'<div class="info-box">📅 Daily mode: Auto ARIMAX selects best p,d,q by lowest AIC across a 3×3 grid. '
                f'd=1 fixed. RMSE on 30-day backtest. {exog_info}. '
                f'Forecast date is the next weekday after the last real trading day.</div>',
                unsafe_allow_html=True)

elif run and not ticker:
    st.warning("Please select a stock first.")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#333;">
        <div style="font-size:2.5rem;margin-bottom:0.6rem;">📊</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;letter-spacing:2px;color:#333;">
            SELECT A STOCK AND HIT FORECAST
        </div>
        <div style="margin-top:0.8rem;font-size:0.72rem;color:#222;line-height:2.2;">
            During market hours → Next-hour ARIMA(1,1,1) + Volume + ATR<br>
            After market close &nbsp;→ Next-day Auto ARIMAX + Volume + ATR + RSI
        </div>
        <div style="margin-top:0.8rem;font-size:0.72rem;color:#1e1e1e;line-height:2;">
            Reliance · TCS · HDFC Bank · Zomato · Apple · Tesla · Nifty 50
        </div>
    </div>
    """, unsafe_allow_html=True)

