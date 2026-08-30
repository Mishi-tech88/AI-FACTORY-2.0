# src/train_timeseries.py - Sklearn Logistic Regression with data cleaning
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

os.makedirs('models', exist_ok=True)

def create_sequences(data, lookback=24):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :-1].flatten())
        y.append(data[i-1, -1])
    return np.array(X), np.array(y)

def run_timeseries_training():
    print("="*50)
    print("STAGE II - TIME-SERIES LOGISTIC REGRESSION (Failure Prediction)")
    print("="*50)

    # 1. Load raw sensor data
    sensor = pd.read_csv('data/raw/sensors.csv', parse_dates=['timestamp'])
    sensor = sensor.sort_values(['machine_id', 'timestamp'])

    # 2. Clean missing values (forward-fill per machine, then backward)
    sensor[['vibration', 'pressure', 'rpm']] = sensor.groupby('machine_id')[['vibration', 'pressure', 'rpm']].ffill().bfill()
    sensor = sensor.dropna(subset=['vibration', 'pressure', 'rpm'])  # remove any remaining NaN

    # 3. Create failure label (anomaly in next 6 timesteps = 1 hour)
    sensor['failure'] = 0
    for machine in sensor['machine_id'].unique():
        mask = sensor['machine_id'] == machine
        future_anomaly = ((sensor.loc[mask, 'vibration'] > 2.0) |
                          (sensor.loc[mask, 'pressure'] > 110) |
                          (sensor.loc[mask, 'rpm'] < 2800)).astype(int)
        shifted = future_anomaly.rolling(6, min_periods=1).max().shift(-6).fillna(0).astype(int)
        sensor.loc[mask, 'failure'] = shifted.values

    # 4. Scale features
    features = ['vibration', 'pressure', 'rpm']
    scaler = StandardScaler()
    sensor[features] = scaler.fit_transform(sensor[features])
    joblib.dump(scaler, 'models/ts_scaler.pkl')

    # 5. Create sequences and split by time per machine
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

    # 6. Train Logistic Regression (handles class imbalance)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nLogistic Regression Performance:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if len(np.unique(y_proba)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    else:
        print("ROC-AUC: N/A (only one class predicted)")

    # 8. Save model
    joblib.dump(clf, 'models/timeseries_model.pkl')
    print("Time-series model saved to 'models/timeseries_model.pkl'")

if __name__ == "__main__":
    run_timeseries_training()