import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from lstm_model import LSTMModel

# Load model
model = LSTMModel(input_size=5, hidden_size=64, num_layers=2, output_size=2)
model.load_state_dict(torch.load("lstm_model_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Load scaler
scaler = joblib.load("scaler.save")

st.title("📈 LSTM Stock Price Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload stock data (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.tail())

    # Ensure last 30 days available
    if len(df) < 30:
        st.warning("Need at least 30 rows of historical data.")
    else:
        recent_data = df[-30:][['Open', 'High', 'Low', 'Close', 'Volume']].values
        scaled_input = scaler.transform(recent_data)
        X = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 30, 5)

        with torch.no_grad():
            prediction = model(X).numpy()

        # Fill dummy values to inverse transform
        dummy = np.zeros((1, 5))
        dummy[0, 0] = prediction[0, 0]  # Open
        dummy[0, 3] = prediction[0, 1]  # Close
        inv = scaler.inverse_transform(dummy)
        pred_open, pred_close = inv[0, 0], inv[0, 3]

        st.subheader("📊 Predicted Next Day Prices:")
        st.write(f"🔹 Predicted Open: **${pred_open:.2f}**")
        st.write(f"🔹 Predicted Close: **${pred_close:.2f}**")
