import streamlit as st
import pandas as pd
import torch
from model import LSTMModel
import joblib

scaler = joblib.load("scaler.save")

input_size = 5
hidden_size = 64
num_layers = 2
output_size = 2

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("lstm_model_weights.pth", map_location=torch.device("cpu")))
model.eval()

st.title("Stock Price Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:", df.head())

    # Ensure Ticker column exists
    if "Ticker" in df.columns:
        ticker_options = df["Ticker"].unique().tolist()
        selected_ticker = st.selectbox("Select Ticker:", ticker_options)

        # Filter data for that ticker
        df_ticker = df[df["Ticker"] == selected_ticker]

        st.write(f"Data for {selected_ticker}:", df_ticker.head())

        # Prediction logic using df_ticker
        if len(df_ticker) >= 30:
            data_scaled = scaler.transform(df_ticker[["Open", "High", "Low", "Close", "Volume"]])
            last_30_days = torch.tensor(data_scaled[-30:], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(last_30_days).numpy()
            st.success(f"Predicted Next Day Open: {prediction[0][0]:.2f}, Close: {prediction[0][1]:.2f}")
        else:
            st.warning("Not enough data for prediction (need at least 30 days).")
    else:
        st.error("Uploaded CSV must have a 'Ticker' column.")
