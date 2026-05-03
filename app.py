import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import timedelta
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import ta  # technical indicators

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
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=500)

    df = pd.DataFrame(bars, columns=[
        'timestamp','open','high','low','close','volume'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

# =========================
# FEATURES
# =========================
def add_indicators(df):
    df['ma20'] = df['close'].rolling(20).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df = df.dropna()
    return df

# =========================
# LSTM PREP
# =========================
def prepare_lstm(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']])

    X, y = [], []
    window = 20

    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler

# =========================
# LSTM MODEL
# =========================
def build_lstm(shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# =========================
# GB MODEL
# =========================
def train_gb(df):
    X = df[['close','volume','ma20','rsi']]
    y = df['close'].shift(-1).dropna()
    X = X.iloc[:-1]

    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

# =========================
# ENSEMBLE PREDICT
# =========================
def predict_ensemble(lstm, gb, df, scaler, steps):
    preds = []

    lstm_input = df[['close']].values[-20:]
    lstm_input = scaler.transform(lstm_input).reshape(1,20,1)

    gb_input = df[['close','volume','ma20','rsi']].iloc[-1].values.reshape(1,-1)

    for _ in range(steps):

        lstm_pred = lstm.predict(lstm_input, verbose=0)[0][0]
        lstm_pred = scaler.inverse_transform([[lstm_pred]])[0][0]

        gb_pred = gb.predict(gb_input)[0]

        final_pred = (lstm_pred + gb_pred) / 2
        preds.append(final_pred)

        # update inputs
        lstm_input = np.append(lstm_input[:,1:,:], [[[scaler.transform([[final_pred]])[0][0]]]], axis=1)
        gb_input[0][0] = final_pred

    return preds

# =========================
# BACKTEST
# =========================
def backtest(df):
    y_true = df['close'].shift(-1).dropna()
    y_pred = df['close'].iloc[:-1]

    return mean_squared_error(y_true, y_pred)

# =========================
# STREAMLIT UI
# =========================
st.title("🚀 Ultimate AI Crypto Predictor")

symbol = st.text_input("Crypto Pair", "BTC/USDT")
steps = st.slider("Prediction Steps", 1, 24, 8)

if st.button("Run Ultimate Prediction"):

    with st.spinner("Building models..."):
        df = get_crypto(symbol)
        df = add_indicators(df)

        gb_model = train_gb(df)

        X, y, scaler = prepare_lstm(df)
        lstm_model = build_lstm((X.shape[1],1))
        lstm_model.fit(X, y, epochs=5, batch_size=16, verbose=0)

        preds = predict_ensemble(lstm_model, gb_model, df, scaler, steps)

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
    st.pyplot(fig)

    # BACKTEST
    st.subheader("🧪 Model Error (baseline)")
    st.write(backtest(df))

    # HISTORY
    st.subheader("💾 Stored Predictions")
    history = pd.read_sql("SELECT * FROM predictions ORDER BY time DESC LIMIT 20", conn)
    st.dataframe(history)

    st.success("Done 🚀")
