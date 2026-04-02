import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import BDay
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
.stButton > button { background: #ffffff; color: #0a0a0a; font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.9rem; border: none; border-radius: 8px; padding: 0.65rem 1.5rem; width: 100%; transition: all 0.15s; letter-spacing: 0.5px; }
.stButton > button:hover { background: #e0e0e0; }
.price-card { background: #111; border: 1px solid #333; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 0.8rem; }
.price-card .label { font-size: 0.7rem; letter-spacing: 2.5px; color: #666; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; }
.price-card .price { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #fff; line-height: 1.2; margin-top: 0.3rem; }
.price-card .change { margin-top: 0.4rem; font-size: 0.95rem; font-weight: 500; }
.price-card .change.up { color: #4caf50; }
.price-card .change.down { color: #f44336; }
.stat-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.4rem; display: flex; justify-content: space-between; align-items: center; }
.stat-card .stat-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1px; font-family: 'IBM Plex Mono', monospace; }
.stat-card .stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #f0f0f0; }
.info-box { background: #111; border-left: 2px solid #333; border-radius: 0 6px 6px 0; padding: 0.7rem 1rem; font-size: 0.78rem; color: #555; line-height: 1.6; margin-top: 0.6rem; }
.badge-stationary { border: 1px solid #4caf50; color: #4caf50; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; border-radius: 4px; padding: 0.2rem 0.5rem; }
.badge-nonstationary { border: 1px solid #f44336; color: #f44336; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; border-radius: 4px; padding: 0.2rem 0.5rem; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 2px; color: #444; text-transform: uppercase; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def fetch_data(ticker):
    try:
        df = yf.download(ticker, start="2015-01-01", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Close' not in df.columns:
            return None
        df = df[['Close']].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Close'] = df['Close'].ffill()
        df = df.asfreq('B')
        df['Close'] = df['Close'].ffill()
        if df['Close'].dropna().shape[0] < 60:
            return None
        return df
    except Exception:
        return None


def find_best_order(series):
    best_aic, best_order = np.inf, (1, 1, 1)
    for p in range(0, 4):
        for q in range(0, 4):
            if p == 0 and q == 0:
                continue
            try:
                r = ARIMA(series, order=(p, 1, q)).fit()
                if r.aic < best_aic:
                    best_aic, best_order = r.aic, (p, 1, q)
            except Exception:
                continue
    return best_order, best_aic


def run_arima(df):
    series = df['Close'].dropna()
    train, test = series[:-30], series[-30:]
    best_order, best_aic = find_best_order(train)
    preds = ARIMA(train, order=best_order).fit().forecast(steps=30)
    rmse = np.sqrt(mean_squared_error(test.values[:len(preds)], preds.values))
    next_price = float(ARIMA(series, order=best_order).fit().forecast(steps=1).iloc[0])
    adf_result = adfuller(series.diff().dropna())
    return next_price, rmse, adf_result, best_order, best_aic


def make_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    fig.subplots_adjust(bottom=0.14, top=0.92, left=0.08, right=0.99)
    cutoff = df.index[-1] - pd.DateOffset(days=730)
    recent = df['Close'][df.index >= cutoff].dropna()
    vals = np.array(recent.values).flatten()
    ax.plot(recent.index, vals, color='#ffffff', linewidth=1.4, alpha=0.9)
    ax.fill_between(recent.index, vals, vals.min() * 0.97, alpha=0.05, color='#ffffff')
    ma30 = pd.Series(vals, index=recent.index).rolling(30).mean()
    ax.plot(ma30.index, ma30.values, color='#ff8c00', linewidth=1.4, linestyle='--', label='30D MA')
    ax.set_ylim(bottom=vals.min() * 0.97, top=vals.max() * 1.02)
    ax.set_xlim(recent.index[0], recent.index[-1])
    ax.set_title(f"{ticker.upper()} — Last 2 Years", color='#888', fontsize=10, pad=6, fontfamily='monospace')
    ax.tick_params(colors='#444', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=25)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e1e1e')
    ax.grid(axis='y', color='#1a1a1a', linewidth=0.5, linestyle='--')
    ax.legend(facecolor='#111', edgecolor='#222', labelcolor='#ff8c00', fontsize=8)
    return fig


STOCKS = {
    "── Select a stock ──": None,
    "📊 Nifty 50 (NIFTYBEES ETF)": "NIFTYBEES.NS",
    "📊 Nifty Bank (BANKBEES ETF)": "BANKBEES.NS",
    "📊 Nifty IT (ITETF)": "ITETF.NS",
    "📊 S&P 500": "^GSPC",
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
    "Blue Star": "BLUESTAR.NS",
    "Crompton Greaves Consumer": "CROMPTON.NS",
    "Amber Enterprises": "AMBER.NS",
    "Kaynes Technology": "KAYNES.NS",
    "Zomato": "ZOMATO.NS",
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
}

INDIAN_TICKERS = {"NIFTYBEES.NS", "BANKBEES.NS", "ITETF.NS"}

st.markdown("<h2 style='text-align:center;font-family:IBM Plex Mono,monospace;color:#fff;margin:0 0 0.2rem 0;'>Stock<span style='color:#ff8c00;'>Sense</span></h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#444;font-size:0.83rem;margin:0 0 1.2rem 0;'>Next-day price forecasting · Auto ARIMA</p>", unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid #1e1e1e;margin-bottom:1.2rem;'>", unsafe_allow_html=True)

col_drop, col_btn = st.columns([4, 1])
with col_drop:
    st.markdown("<div class='section-label'>Select a Stock</div>", unsafe_allow_html=True)
    selected_name = st.selectbox("", options=list(STOCKS.keys()), label_visibility="collapsed")
with col_btn:
    st.markdown("<div style='margin-top:1.45rem'></div>", unsafe_allow_html=True)
    run = st.button("FORECAST →")

ticker = STOCKS.get(selected_name) or None
st.markdown("<hr style='border:none;border-top:1px solid #1e1e1e;margin:1rem 0;'>", unsafe_allow_html=True)

if run and ticker:
    with st.status(f"📡 Fetching data for {ticker}...", expanded=True) as status:
        st.write("📡 Fetching historical price data...")
        df = fetch_data(ticker)
        if df is not None:
            st.write("🔍 Running ADF stationarity test...")
            st.write("⚙️ Grid searching best ARIMA (p,d,q) — takes ~15 seconds...")
            next_price, rmse, adf_result, best_order, best_aic = run_arima(df)
            st.write(f"✅ Best model: ARIMA{best_order}  |  AIC: {best_aic:,.1f}")
            status.update(label=f"✅ Forecast ready for {selected_name}!", state="complete", expanded=False)
        else:
            status.update(label="❌ Failed to fetch data", state="error", expanded=False)

    if df is None:
        st.error(f"Could not fetch data for **{selected_name}**. Try a different stock.")
    else:
        last_price = float(df['Close'].dropna().iloc[-1])
        change     = next_price - last_price
        change_pct = (change / last_price) * 100
        direction  = "▲" if change >= 0 else "▼"
        change_cls = "up" if change >= 0 else "down"
        currency   = "₹" if (ticker.endswith(".NS") or ticker.endswith(".BO") or ticker in INDIAN_TICKERS) else "$"
        last_date     = df['Close'].dropna().index[-1]
        forecast_date = (last_date + BDay(1)).strftime("%A, %d %B %Y")
        last_date_str = last_date.strftime("%d %B %Y")

        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.markdown(f"""
            <div class="price-card">
                <div class="label">Next Day Forecast</div>
                <div class="price">{currency}{next_price:,.2f}</div>
                <div class="change {change_cls}">{direction} {currency}{abs(change):.2f} &nbsp;|&nbsp; {change_pct:+.2f}%</div>
                <div style="margin-top:0.8rem;font-size:0.74rem;color:#555;font-family:'IBM Plex Mono',monospace;">
                    {forecast_date}<br><span style="color:#333;">data up to {last_date_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right_col:
            st.markdown("<div class='section-label'>Price History + 30D Moving Average</div>", unsafe_allow_html=True)
            fig = make_chart(df, ticker)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:0.8rem 0;'>", unsafe_allow_html=True)
        adf_p = adf_result[1]
        badge = '<span class="badge-stationary">✔ STATIONARY (p={:.4f})</span>'.format(adf_p) if adf_p < 0.05 else '<span class="badge-nonstationary">✘ NON-STATIONARY (p={:.4f})</span>'.format(adf_p)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;"><div class="stat-label">Last Close</div><div class="stat-value">{currency}{last_price:,.2f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;"><div class="stat-label">Best Model</div><div class="stat-value">ARIMA{best_order}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;"><div class="stat-label">AIC Score</div><div class="stat-value">{best_aic:,.1f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;"><div class="stat-label">RMSE (30-day)</div><div class="stat-value">{currency}{rmse:.2f}</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="stat-card" style="flex-direction:column;align-items:flex-start;gap:0.3rem;"><div class="stat-label">ADF Test</div>{badge}</div>', unsafe_allow_html=True)

        st.markdown('<div class="info-box">🤖 Auto ARIMA selects best p,d,q by lowest AIC across a 4×4 grid. d=1 fixed. RMSE on 30-day backtest.</div>', unsafe_allow_html=True)

elif run and not ticker:
    st.warning("Please select a stock first.")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#333;">
        <div style="font-size:2.5rem;margin-bottom:0.6rem;">📊</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;letter-spacing:2px;color:#333;">SELECT A STOCK AND HIT FORECAST</div>
        <div style="margin-top:1rem;font-size:0.75rem;color:#2a2a2a;line-height:2;">
            Reliance · TCS · HDFC Bank · Zomato · Apple · Tesla · Nifty 50
        </div>
    </div>
    """, unsafe_allow_html=True)
