# src/train_tabular.py
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import os

os.makedirs('models', exist_ok=True)

def run_tabular_training():
    print("="*50)
    print("STAGE II - TABULAR MODELS (Defect Classification)")
    print("="*50)

    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')

    train['defect'] = (train['quality_score'] < 0.2).astype(int)
    val['defect'] = (val['quality_score'] < 0.2).astype(int)
    test['defect'] = (test['quality_score'] < 0.2).astype(int)

    features = [col for col in train.columns if col not in ['timestamp', 'quality_score', 'defect']]
    X_train = train[features]
    y_train = train['defect']
    X_val = val[features]
    y_val = val['defect']
    X_test = test[features]
    y_test = test['defect']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/tabular_scaler.pkl')

    # ---- Random Forest ----
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    print(classification_report(y_test, rf_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
    joblib.dump(rf, 'models/rf_model.pkl')

    # ---- XGBoost ----
    print("\n--- XGBoost ---")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    print(classification_report(y_test, xgb_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
    joblib.dump(xgb, 'models/xgb_model.pkl')

    # ---- ANN (Deep Learning) ----
    print("\n--- ANN (Deep Learning) ---")
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    class ANN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    model = ANN(X_train_t.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    for epoch in range(30):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            # Clamp to avoid numerical instability
            pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_pred = torch.clamp(val_pred, 1e-7, 1 - 1e-7)
                val_loss = criterion(val_pred, y_val_t)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy().flatten()
        test_binary = (test_pred >= 0.5).astype(int)
    print(classification_report(y_test, test_binary))
    print(f"ROC-AUC: {roc_auc_score(y_test, test_pred):.4f}")

    torch.save(model.state_dict(), 'models/ann_tabular.pth')
    print("Tabular models saved to 'models/'")

if __name__ == "__main__":
    run_tabular_training()