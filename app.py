import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Train function
def train_model(df, epochs=20, seq_len=30):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, [0, 3]])  # Open & Close
    X, y = np.array(X), np.array(y)

    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel(input_size=5, hidden_size=64, num_layers=2, output_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model, scaler

# --- Streamlit UI ---
st.title("📈 Stock Price Prediction (Train from Uploaded Data)")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    # Optional: detect ticker if in dataset
    ticker = st.text_input("Enter Ticker Symbol", "Unknown")

    if st.button("Train Model"):
        model, scaler = train_model(df)
        st.success(f"Model trained successfully for {ticker}!")

        # Predict next day
        last_30 = scaler.transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])[-30:]
        X_input = torch.tensor(last_30, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_input).numpy()

        dummy = np.zeros((pred_scaled.shape[0], 3))
        merged = np.hstack([pred_scaled[:, 0:1], dummy[:, 0:1], dummy[:, 1:2], pred_scaled[:, 1:2], dummy[:, 2:3]])
        pred_rescaled = scaler.inverse_transform(merged)[:, [0, 3]]

        st.subheader("📊 Predicted Next Day Prices")
        st.write(f"**Open:** {pred_rescaled[0,0]:.2f}")
        st.write(f"**Close:** {pred_rescaled[0,1]:.2f}")
