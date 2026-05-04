import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import timedelta
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    time TEXT,
    symbol TEXT,
    predicted REAL
)
""")

def save_prediction(time, symbol, price):
    cursor.execute(
        "INSERT INTO predictions VALUES (?, ?, ?)",
        (str(time), symbol, float(price))
    )
    conn.commit()

# =========================
# GBP RATE
# =========================
def get_gbp_rate():
    try:
        data = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()
        return data["rates"]["GBP"]
    except:
        return 0.79

# =========================
# DATA (COINGECKO)
# =========================
def get_crypto(symbol="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": "30", "interval": "hourly"}

    data = requests.get(url, params=params).json()
    prices = data["prices"]

    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # fake volume (stable)
    df["volume"] = np.linspace(100, 200, len(df))

    return df

# =========================
# FEATURES
# =========================
def add_features(df):
    df['ma10'] = df['close'].rolling(10).mean()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(5).std()

    df = df.bfill().ffill()
    return df

# =========================
# MODEL
# =========================
def train_model(df):
    features = ['close','volume','ma10','volatility']

    X = df[features]
    y = df['close'].shift(-1).ffill()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=150)
    model.fit(X_scaled, y)

    return model, scaler

# =========================
# BETTER PREDICTION ENGINE
# =========================
def predict_future(model, scaler, df, steps):
    features = ['close','volume','ma10','volatility']

    preds = []
    temp_df = df.copy()

    for _ in range(steps):
        temp_df = temp_df.bfill().ffill()

        current = temp_df.iloc[-1:]

        X = current[features]
        X_scaled = scaler.transform(X)

        base_pred = model.predict(X_scaled)[0]

        # === IMPROVEMENTS ===

        # trend (momentum)
        trend = temp_df['close'].pct_change().tail(5).mean()

        # volatility influence
        volatility = temp_df['volatility'].iloc[-1]

        # combine
        pred = base_pred * (1 + trend)

        # add realistic noise
        pred += np.random.normal(0, volatility * pred * 0.5)

        preds.append(pred)

        # append new row
        new_row = current.copy()
        new_row['close'] = pred

        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

        # recalc features
        temp_df['ma10'] = temp_df['close'].rolling(10).mean()
        temp_df['returns'] = temp_df['close'].pct_change()
        temp_df['volatility'] = temp_df['returns'].rolling(5).std()

    return preds

# =========================
# SIGNAL ENGINE
# =========================
def get_signal(df, preds):
    last_price = df['close'].iloc[-1]
    future_price = preds[-1]

    change = (future_price - last_price) / last_price

    if change > 0.01:
        return "🟢 BUY"
    elif change < -0.01:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"

# =========================
# UI
# =========================
st.set_page_config(page_title="Crypto Price Predictor", layout="wide")

st.title("Crypto Price Predictor")

coin_map = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin"
}

col1, col2 = st.columns(2)

coin = col1.selectbox("Crypto", list(coin_map.keys()))
steps = col2.slider("Hours Ahead", 1, 24, 8)

if st.button("Run Prediction"):

    with st.spinner("Analyzing market..."):

        df = get_crypto(coin_map[coin])
        df = add_features(df)

        model, scaler = train_model(df)
        preds = predict_future(model, scaler, df, steps)

        gbp_rate = get_gbp_rate()

    # TIMES
    future_times = [
        df['timestamp'].iloc[-1] + timedelta(hours=i+1)
        for i in range(steps)
    ]

    result_df = pd.DataFrame({
        "Time": future_times,
        "USD": preds,
        "GBP": np.array(preds) * gbp_rate
    })

    # SAVE
    for t, p in zip(future_times, preds):
        save_prediction(t, coin, p)

    # SIGNAL
    signal = get_signal(df, preds)

    st.subheader(f"📢 Signal: {signal}")

    # TABLE
    st.subheader("📊 Predictions")
    st.dataframe(result_df)

    # CHART
    fig, ax = plt.subplots()

    ax.plot(df['timestamp'], df['close'], label="History")
    ax.plot(future_times, preds, linestyle='dashed', label="Prediction")

    ax.set_title(f"{coin} Price Forecast")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # HISTORY
    st.subheader("💾 History")
    history = pd.read_sql("SELECT * FROM predictions ORDER BY time DESC LIMIT 20", conn)
    st.dataframe(history)

    st.success("Analysis complete 🚀")
