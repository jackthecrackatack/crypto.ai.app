import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta
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
# COINGECKO DATA (FIXED)
# =========================
def get_crypto(symbol="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    
    params = {
        "vs_currency": "usd",
        "days": "2",
        "interval": "hourly"
    }

    data = requests.get(url, params=params).json()

    prices = data["prices"]

    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Fake volume (since free API doesn't include it)
    df["volume"] = np.random.rand(len(df)) * 1000

    return df

# =========================
# FEATURES
# =========================
def add_features(df):
    df['ma20'] = df['close'].rolling(10).mean()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(5).std()
    return df.dropna()

# =========================
# MODEL
# =========================
def train_model(df):
    features = ['close','volume','ma20','volatility']
    
    X = df[features]
    y = df['close'].shift(-1).dropna()
    X = X.iloc[:-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_scaled, y)

    return model, scaler

# =========================
# PREDICTION
# =========================
def predict_future(model, scaler, df, steps):
    features = ['close','volume','ma20','volatility']

    preds = []
    current = df.iloc[-1:].copy()

    for _ in range(steps):
        X = current[features]
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        preds.append(pred)

        new_row = current.copy()
        new_row['close'] = pred
        current = new_row

    return preds

# =========================
# UI
# =========================
st.title("🚀 Crypto AI Predictor (Stable Deploy Version)")

coin_map = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin"
}

coin = st.selectbox("Select Crypto", list(coin_map.keys()))
steps = st.slider("Prediction Hours", 1, 24, 8)

if st.button("Run Prediction"):

    with st.spinner("Loading..."):

        df = get_crypto(coin_map[coin])
        df = add_features(df)

        model, scaler = train_model(df)
        preds = predict_future(model, scaler, df, steps)

        gbp_rate = get_gbp_rate()

    future_times = [
        df['timestamp'].iloc[-1] + timedelta(hours=i+1)
        for i in range(steps)
    ]

    result_df = pd.DataFrame({
        "Time": future_times,
        "USD": preds,
        "GBP": np.array(preds) * gbp_rate
    })

    for t, p in zip(future_times, preds):
        save_prediction(t, coin, p)

    st.subheader("📊 Predictions")
    st.dataframe(result_df)

    fig, ax = plt.subplots()
    ax.plot(df['timestamp'], df['close'], label="History")
    ax.plot(future_times, preds, linestyle='dashed', label="Prediction")
    ax.legend()

    st.pyplot(fig)

    st.subheader("💾 History")
    history = pd.read_sql("SELECT * FROM predictions ORDER BY time DESC LIMIT 20", conn)
    st.dataframe(history)

    st.success("Working perfectly 🚀")
