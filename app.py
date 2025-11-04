# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

st.title("📈 Stock Price Prediction App (LSTM)")
st.write("Predict future stock prices using LSTM Neural Networks.")

# ---------------------------------------
# 1. User Input
# ---------------------------------------
ticker = st.text_input("Enter Stock Symbol (e.g., NESTLEIND.NS):", "NESTLEIND.NS")

start_date = st.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.date_input("End Date", datetime.today())

if st.button("Run Prediction"):
    with st.spinner("Downloading stock data..."):
        data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for this ticker or date range. Try again.")
    else:
        st.success("✅ Data successfully downloaded!")
        st.dataframe(data.tail())

        # ---------------------------------------
        # 2. Prepare Data
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
                X.append(dataset[i-60:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data)
        X_test, y_test = create_dataset(test_data)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # ---------------------------------------
        # 3. Build Model
        # ---------------------------------------
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # ---------------------------------------
        # 4. Train Model
        # ---------------------------------------
        with st.spinner("Training LSTM model (may take a few minutes)..."):
            model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        st.success("✅ Model training complete!")

        # ---------------------------------------
        # 5. Predict
        # ---------------------------------------
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # ---------------------------------------
        # 6. Visualize
        # ---------------------------------------
        st.subheader("📊 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(real_prices, label="Actual Price", color='blue')
        ax.plot(predictions, label="Predicted Price", color='red')
        ax.legend()
        st.pyplot(fig)

        # ---------------------------------------
        # 7. Predict Next 5 Days
        # ---------------------------------------
        st.subheader("🔮 Next 5-Day Forecast")
        last_60_days = scaled_data[-60:]
        future_predictions = []
        future_dates = []

        for i in range(5):
            X_future = np.reshape(last_60_days, (1, 60, 1))
            next_price = model.predict(X_future)[0][0]
            future_predictions.append(next_price)
            last_60_days = np.append(last_60_days[1:], next_price)
            last_60_days = last_60_days.reshape(-1, 1)
            future_dates.append(datetime.today() + timedelta(days=i+1))

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        forecast_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Predicted Price (INR)': np.round(future_predictions.flatten(), 2)
        })

        st.table(forecast_df)
