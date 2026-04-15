import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from predict import analyze_stock_history

USD_TO_INR = 83
COMPANIES = {
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "Wipro (WIPRO.NS)": "WIPRO.NS",
    "HCL Technologies (HCLTECH.NS)": "HCLTECH.NS",
    "Tech Mahindra (TECHM.NS)": "TECHM.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "Axis Bank (AXISBANK.NS)": "AXISBANK.NS",
    "Kotak Mahindra Bank (KOTAKBANK.NS)": "KOTAKBANK.NS",
}


st.set_page_config(page_title="Stock Market Trend Prediction", layout="wide")


@st.cache_data(ttl=900)
def load_stock_history(ticker, period):
    history = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    return history.fillna(0)


def get_signal(prediction, confidence):
    if prediction == "UP":
        return "Strong Buy" if confidence >= 0.75 else "Buy"
    return "Strong Sell" if confidence >= 0.75 else "Sell"


def get_risk_label(volatility):
    if volatility >= 0.03:
        return "High"
    if volatility >= 0.015:
        return "Medium"
    return "Low"


def safe_float(value, default=0.0):
    if value is None or pd.isna(value):
        return default
    return float(value)


def format_currency_pair(value):
    safe_value = safe_float(value, default=0.0)
    inr_value = safe_value * USD_TO_INR
    return f"₹{inr_value:,.2f} (converted from ${safe_value:,.2f})"


def build_projection_chart(close_series, predicted_price):
    projection_df = close_series.tail(30).to_frame(name="Close")
    next_index = projection_df.index[-1] + pd.tseries.offsets.BDay(1)
    projection_df["Projected Close"] = np.nan
    projection_df.loc[next_index, "Close"] = np.nan
    projection_df.loc[next_index, "Projected Close"] = safe_float(predicted_price)
    return projection_df


def apply_manual_inputs(history, open_price, high_price, low_price, close_price, volume):
    updated_history = history.copy()
    if updated_history.empty:
        return updated_history

    latest_index = updated_history.index[-1]
    updated_history.loc[latest_index, "Open"] = safe_float(open_price)
    updated_history.loc[latest_index, "High"] = safe_float(high_price)
    updated_history.loc[latest_index, "Low"] = safe_float(low_price)
    updated_history.loc[latest_index, "Close"] = safe_float(close_price)
    updated_history.loc[latest_index, "Volume"] = int(volume if volume is not None else 0)
    return updated_history.fillna(0)


st.title("Stock Market Trend Prediction")
st.write(
    "Analyze a stock by ticker, compute technical indicators automatically, "
    "and view model-driven trend forecasts with confidence, charts, and trading signals."
)

st.sidebar.header("Stock Analysis")
selected_company = st.sidebar.selectbox("Select Company", options=list(COMPANIES.keys()))
ticker = COMPANIES[selected_company]
period = st.sidebar.selectbox("Historical Range", ["6mo", "1y", "2y", "5y"], index=1)
st.sidebar.caption(f"Selected ticker: `{ticker}`")

st.sidebar.subheader("Manual Market Inputs")
open_price = st.sidebar.number_input("Open Price", min_value=0.0, value=0.0, step=0.01)
high_price = st.sidebar.number_input("High Price", min_value=0.0, value=0.0, step=0.01)
low_price = st.sidebar.number_input("Low Price", min_value=0.0, value=0.0, step=0.01)
close_price = st.sidebar.number_input("Close Price", min_value=0.0, value=0.0, step=0.01)
volume = st.sidebar.number_input("Volume", min_value=0, value=0, step=1000)
analyze_clicked = st.sidebar.button("Analyze Stock")

if analyze_clicked:
    with st.spinner(f"Fetching market data for {ticker} and generating insights..."):
        try:
            history = load_stock_history(ticker, period)
            if history.empty:
                st.error("No market data returned for this ticker. Please check the symbol and try again.")
            else:
                if any([open_price > 0, high_price > 0, low_price > 0, close_price > 0, volume > 0]):
                    history = apply_manual_inputs(history, open_price, high_price, low_price, close_price, volume)

                analysis = analyze_stock_history(history)
                current_price = safe_float(analysis["current_price"])
                predicted_price = safe_float(analysis["predicted_price"])
                confidence = safe_float(analysis["confidence"], default=0.0)
                prediction = analysis["prediction"]
                signal = get_signal(prediction, confidence)
                volatility = safe_float(analysis["feature_frame"]["Volatility"].iloc[-1], default=0.0)
                risk_label = get_risk_label(volatility)

                st.subheader(f"{selected_company} Market Snapshot")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Current Price", f"${current_price:,.2f}")
                metric_cols[1].metric("Predicted Future Price", f"${predicted_price:,.2f}")
                metric_cols[2].metric("Trend Prediction", prediction)
                metric_cols[3].metric("Confidence", f"{confidence * 100:.1f}%")

                st.write(f"Current Price in INR: {format_currency_pair(current_price)}")
                st.write(f"Predicted Future Price in INR: {format_currency_pair(predicted_price)}")

                signal_col, risk_col = st.columns(2)
                with signal_col:
                    if prediction == "UP":
                        st.success(f"Signal: {signal}")
                    else:
                        st.error(f"Signal: {signal}")
                with risk_col:
                    st.info(f"Risk Level: {risk_label}")

                st.write(
                    f"Model view: `{ticker}` currently looks **{prediction.lower()}**, with an approximate "
                    f"next-price projection of **${predicted_price:,.2f}** based on the latest technical setup."
                )

                st.subheader("Price Chart")
                st.line_chart(history[["Close"]].tail(120).fillna(0))

                st.subheader("Moving Average Chart")
                ma_chart = analysis["feature_frame"][["Close", "MA_10", "MA_50"]].tail(120).fillna(0)
                st.line_chart(ma_chart)

                st.subheader("Prediction Trend Chart")
                projection_chart = build_projection_chart(history["Close"], predicted_price)
                st.line_chart(projection_chart)

                st.subheader("Latest Technical Indicators")
                latest_features = pd.DataFrame([analysis["latest_features"]]).T
                latest_features.columns = ["Latest Value"]
                st.dataframe(latest_features.fillna(0), use_container_width=True)

                st.subheader("Decision Support")
                if prediction == "UP":
                    st.write("- Buy/Sell Signal: Prefer buy or hold setups when price action confirms strength.")
                    st.write("- Risk Note: Watch resistance levels and earnings/news catalysts before entering.")
                else:
                    st.write("- Buy/Sell Signal: Prefer sell, reduce exposure, or wait for reversal confirmation.")
                    st.write("- Risk Note: Use stop-loss discipline because weak trends can accelerate quickly.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
else:
    st.info("Select a company, optionally enter manual OHLCV values, and click `Analyze Stock` to generate charts, prediction confidence, and trade guidance.")

st.divider()
st.caption("This app is for educational purposes only and does not constitute financial advice.")
