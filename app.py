import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel  # your LSTM model class

# App UI
st.title("Stock Price Prediction (LSTM)")

uploaded_file = st.file_uploader("Upload stock CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Detect ticker (assuming there's a column or metadata)
    if "Ticker" in df.columns:
        ticker = df["Ticker"].iloc[0]
    else:
        ticker = st.text_input("Enter ticker manually:")

    st.write(f"Detected ticker: **{ticker}**")

    # Model params
    input_size = 5
    hidden_size = 64
    num_layers = 2
    output_size = 2

    # If ticker matches trained model ticker
    if ticker.upper() == "AAPL":  
        st.write("Using pre-trained LSTM model for Apple...")
        scaler = joblib.load("scaler.pkl")
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        model.load_state_dict(torch.load("lstm_model_weights.pth"))
        model.eval()
    else:
        st.write("Training new LSTM model for uploaded ticker...")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]].values)

        # Here: create sequences, train new LSTM on the fly
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        # train_model(model, scaled_data)  # your training function
        model.eval()

    # Prediction button
    if st.button("Predict Next Day"):
        last_30_days = scaler.transform(df[["Open", "High", "Low", "Close", "Volume"]].values[-30:])
        input_tensor = torch.tensor(last_30_days, dtype=torch.float32).unsqueeze(0)
        pred = model(input_tensor).detach().numpy().flatten()

        # Rescale prediction
        dummy = np.zeros((1, 5))
        dummy[0, 0] = pred[0]  # Open
        dummy[0, 3] = pred[1]  # Close
        rescaled_pred = scaler.inverse_transform(dummy)[0, [0, 3]]

        st.write(f"Predicted Open: {rescaled_pred[0]:.2f}")
        st.write(f"Predicted Close: {rescaled_pred[1]:.2f}")
