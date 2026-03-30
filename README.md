# StockSense — Next-Day Stock Price Forecaster using Auto ARIMA

StockSense is an interactive web application built with Streamlit that forecasts the next-day closing price of any publicly listed stock using time-series analysis. The app is designed to be simple, transparent, and honest about what the model can and cannot do.

## How It Works

The forecasting engine is built entirely on ARIMA (AutoRegressive Integrated Moving Average), a classical statistical model for time-series data. The app fetches historical daily closing price data from Yahoo Finance using the yfinance library, going back to 2015, and generates a next-day price prediction based on that data.

One important thing to understand about this model — StockSense forecasts purely based on past price history. It does not incorporate any external or exogenous variables such as trading volume, news sentiment, earnings reports, interest rates, or macroeconomic indicators. The model only learns from the historical pattern of the closing price series itself. This is a deliberate and honest design choice. ARIMA is a univariate model, meaning it takes one variable as input, and it is most reliable over very short horizons like next-day prediction rather than multi-week forecasts.

## Auto ARIMA — Adapting to Each Stock

Rather than using a fixed ARIMA order, the app runs an automated grid search across all combinations of p and q ranging from 0 to 3, with d fixed at 1 since stock prices are non-stationary by nature. The combination that produces the lowest AIC score is automatically selected. This means the model adapts to each stock's unique price behaviour instead of applying a one-size-fits-all order.

## Key Features
- 130+ Indian (NSE/BSE) and US stocks available via dropdown
- Auto ARIMA with AIC-based order selection per stock
- ADF stationarity test displayed for statistical transparency
- RMSE measured on a 30-day backtest to validate model accuracy
- 2-year interactive price chart with 30-day moving average overlay
- Next business day forecast date calculated automatically

## Tech Stack
Python · Streamlit · yfinance · Statsmodels · Scikit-learn · Matplotlib · Pandas · NumPy
