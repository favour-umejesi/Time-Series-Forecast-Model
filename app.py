import streamlit as st
import pandas as pd
import torch
from model import LSTMModel
import joblib

# Load the same scaler used during training
scaler = joblib.load("scaler.save")

# Model parameters (must match training)
input_size = 5
hidden_size = 64
num_layers = 2
output_size = 2

# Load model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("lstm_model_weights.pth", map_location=torch.device("cpu")))
model.eval()

st.title("Apple Stock Price Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV with Apple data)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:", df.head())

    # Filter for Apple only
    if "Ticker" in df.columns:
        df_apple = df[df["Ticker"] == "AAPL"]

        if df_apple.empty:
            st.error("No Apple (AAPL) data found in the file.")
        else:
            st.write("Filtered Apple Data:", df_apple.tail())

            # Ensure we have enough history
            if len(df_apple) >= 30:
                # Scale the data
                features = df_apple[["Open", "High", "Low", "Close", "Volume"]]
                data_scaled = scaler.transform(features)

                # Get last 30 days for prediction
                last_30_days = torch.tensor(data_scaled[-30:], dtype=torch.float32).unsqueeze(0)

                # Debug: Show what the model sees
                st.write("Last 30 days (scaled) fed into model:", last_30_days.numpy())

                # Predict
                with torch.no_grad():
                    prediction_scaled = model(last_30_days).numpy()

                # Inverse transform to original scale
                dummy_full = [[0, 0, 0, 0, 0]] * (scaler.n_features_in_ - output_size)  # padding
                inverse_input = []
                for pred in prediction_scaled:
                    row = list(pred) + [0] * (scaler.n_features_in_ - output_size)
                    inverse_input.append(row)
                prediction_original = scaler.inverse_transform(inverse_input)

                pred_open = prediction_original[0][0]
                pred_close = prediction_original[0][3]  # Close is 4th in features

                st.success(f"Predicted Next Day Open: {pred_open:.2f}, Close: {pred_close:.2f}")
            else:
                st.warning("Not enough Apple data for prediction (need at least 30 days).")
    else:
        st.error("Uploaded CSV must have a 'Ticker' column.")
