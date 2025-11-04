import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date, timedelta

# ---------------------------------------
# Streamlit UI Setup
# ---------------------------------------
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("📈 Stock Market Forecasting (India)")

st.write("This app uses an **LSTM neural network** to predict the next 5 days of stock closing prices using data from Yahoo Finance.")

# ---------------------------------------
# 1. User Input
# ---------------------------------------
stocks = {
    "Nestle India": "NESTLEIND.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Finance": "BAJFINANCE.NS"
}

stock_name = st.selectbox("Select a Stock:", list(stocks.keys()))
ticker = stocks[stock_name]

st.write(f"📊 Showing forecast for: **{stock_name}** ({ticker})")

# ---------------------------------------
# 2. Load Data
# ---------------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end=date.today() + timedelta(days=1))
    return data

data = load_data(ticker)

if data.empty:
    st.error("❌ No data found. Please check your internet or ticker symbol.")
    st.stop()

st.subheader("📅 Historical Stock Data (last 10 rows)")
st.dataframe(data.tail(10))

# ---------------------------------------
# 3. Prepare Data
# ---------------------------------------
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

def create_dataset(dataset):
    X, y = [], []
    for i in range(60, len(dataset)):
        X.append(dataset[i - 60:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ---------------------------------------
# 4. Build and Train Model
# ---------------------------------------
with st.spinner("🔧 Training LSTM model... (this may take a minute)"):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# ---------------------------------------
# 5. Make Predictions
# ---------------------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# ---------------------------------------
# 6. Plot Actual vs Predicted
# ---------------------------------------
st.subheader(f"📉 Actual vs Predicted Prices — {stock_name}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(real_prices, label='Actual Price', color='blue')
ax.plot(predictions, label='Predicted Price', color='red')
ax.set_title(f'{stock_name} Stock Price Prediction')
ax.set_xlabel('Days')
ax.set_ylabel('Price (INR)')
ax.legend()
st.pyplot(fig)

# ---------------------------------------
# 7. Predict Next 5 Days
# ---------------------------------------
future_days = 5
last_60_days = scaled_data[-60:].copy()
predicted_prices = []

for _ in range(future_days):
    X_future = last_60_days[-60:].reshape(1, 60, 1)
    future_price = model.predict(X_future)
    predicted_prices.append(future_price[0, 0])
    last_60_days = np.append(last_60_days, future_price)[-60:]

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
forecast_dates = pd.date_range(start=date.today() + timedelta(days=1), periods=future_days)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price (INR)': predicted_prices.flatten()})

# ---------------------------------------
# 8. Display Forecast
# ---------------------------------------
st.subheader(f"🔮 Predicted Prices for Next 5 Days — {stock_name}")
st.dataframe(forecast_df.style.format({'Predicted Price (INR)': '₹{:.2f}'}))

# Forecast Plot
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(forecast_df['Date'], forecast_df['Predicted Price (INR)'], marker='o', color='green')
ax2.set_title(f'Predicted Next 5 Days Prices — {stock_name}')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (INR)')
st.pyplot(fig2)

st.success("✅ Forecast generation complete!")
