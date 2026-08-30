# src/train_timeseries.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import os
import joblib

os.makedirs('models', exist_ok=True)

def create_sequences(data, lookback=24):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :-1])  # all cols except last (label)
        y.append(data[i-1, -1])            # label at previous step (predict next hour)
    return np.array(X), np.array(y)

def run_timeseries_training():
    print("="*50)
    print("STAGE II - TIME-SERIES LSTM (Failure Prediction)")
    print("="*50)

    # 1. Load raw sensor data
    sensor = pd.read_csv('data/raw/sensors.csv', parse_dates=['timestamp'])
    sensor = sensor.sort_values(['machine_id', 'timestamp'])

    # 2. Create failure label (if vibration > 2.0 or pressure > 110 or rpm < 2800)
    sensor['failure'] = 0
    for machine in sensor['machine_id'].unique():
        mask = sensor['machine_id'] == machine
        # Rolling max of next 6 timesteps (1 hour). Shift backwards to label current time.
        future_anomaly = ((sensor.loc[mask, 'vibration'] > 2.0) |
                          (sensor.loc[mask, 'pressure'] > 110) |
                          (sensor.loc[mask, 'rpm'] < 2800)).astype(int)
        # If any anomaly in next 6 steps, current label = 1
        shifted = future_anomaly.rolling(6, min_periods=1).max().shift(-6).fillna(0).astype(int)
        sensor.loc[mask, 'failure'] = shifted.values

    # 3. Scale features
    features = ['vibration', 'pressure', 'rpm']
    scaler = StandardScaler()
    sensor[features] = scaler.fit_transform(sensor[features])
    joblib.dump(scaler, 'models/ts_scaler.pkl')

    # 4. Prepare sequences per machine and split temporally
    lookback = 24
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for machine in sensor['machine_id'].unique():
        machine_data = sensor[sensor['machine_id'] == machine][features + ['failure']].values
        n = len(machine_data)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        for split_name, start, end in [('train', 0, train_end), ('val', train_end, val_end), ('test', val_end, n)]:
            data_seg = machine_data[start:end]
            if len(data_seg) > lookback:
                Xs, ys = create_sequences(data_seg, lookback)
                if split_name == 'train':
                    X_train.append(Xs); y_train.append(ys)
                elif split_name == 'val':
                    X_val.append(Xs); y_val.append(ys)
                else:
                    X_test.append(Xs); y_test.append(ys)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 5. Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

    # 6. Define LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return self.sigmoid(out)

    model = LSTM(input_size=len(features), hidden_size=64, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Train
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    for epoch in range(30):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss.item():.4f}")

    # 8. Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy().flatten()
        test_binary = (test_pred >= 0.5).astype(int)
    print("\nLSTM Performance:")
    print(classification_report(y_test, test_binary))
    print(f"ROC-AUC: {roc_auc_score(y_test, test_pred):.4f}")

    torch.save(model.state_dict(), 'models/lstm_failure.pth')
    print("LSTM model saved to 'models/'")

if __name__ == "__main__":
    run_timeseries_training()