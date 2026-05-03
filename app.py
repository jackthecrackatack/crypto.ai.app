import streamlit as st
import ccxt
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
# DATA
# =========================
def get_crypto(symbol):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=300)

    df = pd.DataFrame(bars, columns=[
        'timestamp','open','high','low','close','volume'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

# =========================
# FEATURES (THIS IS YOUR "AI")
# =========================
def add_features(df):
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()

    df = df.dropna()
    return df

# =========================
# TRAIN MODEL
# =========================
def train_model(df):
    features = ['close','volume','ma20','ma50','volatility']
    
    X = df[features]
    y = df['close'].shift(-1).dropna()
    X = X.iloc[:-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_scaled, y)

    return model, scaler

# =========================
# MULTI STEP PREDICTION
# =========================
def predict_future(model, scaler, df, steps):
    features = ['close','volume','ma20','ma50','volatility']

    preds = []
    current = df.iloc[-1:].copy()

    for _ in range(steps):
        X = current[features]
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        preds.append(pred)

        # update fake future row
        new_row = current.copy()
        new_row['close'] = pred
        current = new_row

    return preds

# =========================
# UI
# =========================
st.title("🚀 Crypto AI Predictor (Deployable Version)")

symbol = st.text_input("Crypto Pair", "BTC/USDT")
steps = st.slider("Prediction Steps (hours)", 1, 24, 8)

if st.button("Run Prediction"):

    with st.spinner("Processing..."):

        df = get_crypto(symbol)
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
        "GBP": np.array(preds)*gbp_rate
    })

    # SAVE
    for t, p in zip(future_times, preds):
        save_prediction(t, symbol, p)

    # TABLE
    st.subheader("📊 Predictions")
    st.dataframe(result_df)

    # CHART
    fig, ax = plt.subplots()

    ax.plot(df['timestamp'].tail(100), df['close'].tail(100), label="History")
    ax.plot(future_times, preds, linestyle='dashed', label="Prediction")

    ax.legend()
    ax.set_title(symbol)

    st.pyplot(fig)

    # HISTORY
    st.subheader("💾 Previous Predictions")
    history = pd.read_sql(
        "SELECT * FROM predictions ORDER BY time DESC LIMIT 20", conn
    )
    st.dataframe(history)

    st.success("Done 🚀")
