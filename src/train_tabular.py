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
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

def run_tabular_training():
    print("="*50)
    print("STAGE II - TABULAR MODELS (Defect Classification)")
    print("="*50)

    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')

    # Create target (defect if quality_score < 0.2)
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

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/tabular_scaler.pkl')

    # ---- Random Forest (with class_weight) ----
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    print(classification_report(y_test, rf_pred))
    if len(np.unique(rf_pred)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
    else:
        print("ROC-AUC: N/A (only one class predicted)")

    # ---- XGBoost (with scale_pos_weight) ----
    print("\n--- XGBoost ---")
    # Compute ratio of negative to positive
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    scale = neg / pos if pos > 0 else 1
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=scale, random_state=42, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    print(classification_report(y_test, xgb_pred))
    if len(np.unique(xgb_pred)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
    else:
        print("ROC-AUC: N/A (only one class predicted)")

    # ---- ANN (Deep Learning) with class weight ----
    print("\n--- ANN (Deep Learning) ---")
    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Calculate positive weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([neg / pos]) if pos > 0 else torch.tensor([1.0])

    class ANN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)  # no sigmoid – we use BCEWithLogitsLoss
            )
        def forward(self, x):
            return self.net(x)

    model = ANN(X_train_t.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    for epoch in range(50):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.sigmoid(test_logits).numpy().flatten()
        test_binary = (test_probs >= 0.5).astype(int)

    print(classification_report(y_test, test_binary))
    if len(np.unique(test_binary)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
    else:
        print("ROC-AUC: N/A (only one class predicted)")

    torch.save(model.state_dict(), 'models/ann_tabular.pth')
    print("Tabular models saved to 'models/'")

if __name__ == "__main__":
    run_tabular_training()