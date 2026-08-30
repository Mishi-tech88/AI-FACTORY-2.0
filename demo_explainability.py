# demo_explainability.py
import os
import numpy as np
import pandas as pd
import joblib
import torch
from src.explainability import explain_tabular, explain_image, explain_timeseries
from src.vision_advanced import load_resnet_model
from src.train_tabular import load_xgb_model  # you may need to add this or just load
from sklearn.preprocessing import StandardScaler
import cv2
import matplotlib.pyplot as plt

def demo():
    print("="*50)
    print("Explainability Demo")
    print("="*50)

    # 1. Tabular
    print("\n--- Tabular Explanation ---")
    model = joblib.load('models/xgb_model.pkl')
    test_df = pd.read_csv('data/processed/test.csv')
    feature_names = ['production_count', 'temperature', 'vibration', 'pressure', 'rpm',
                     'vibration_rolling_mean_24', 'vibration_rolling_std_24',
                     'pressure_rolling_mean_24', 'pressure_rolling_std_24',
                     'rpm_rolling_mean_24', 'rpm_rolling_std_24',
                     'vibration_lag1', 'vibration_lag24',
                     'pressure_lag1', 'pressure_lag24',
                     'rpm_lag1', 'rpm_lag24',
                     'machine_id_M2', 'machine_id_M3', 'machine_id_M4',
                     'operator_OpB', 'operator_OpC']
    X_sample = test_df.iloc[0][feature_names].values.reshape(1, -1)
    shap_vals, expected, tab_exp = explain_tabular(model, X_sample, feature_names)
    print(tab_exp)

    # 2. Image
    print("\n--- Image Explanation ---")
    img_path = 'data/images/prod_0.png'
    if os.path.exists(img_path):
        model_img = load_resnet_model()
        heatmap, img_exp = explain_image(model_img, img_path)
        print(img_exp)
        # Save heatmap
        if heatmap is not None:
            cv2.imwrite('gradcam_output.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            print("Heatmap saved as gradcam_output.png")
    else:
        print("No image found.")

    # 3. Time-series
    print("\n--- Time-Series Explanation ---")
    ts_model = joblib.load('models/timeseries_model.pkl')
    scaler = joblib.load('models/ts_scaler.pkl')
    # Create a dummy sequence
    seq = np.random.randn(24, 3)
    seq_flat = seq.flatten().reshape(1, -1)
    imp, ts_exp = explain_timeseries(ts_model, seq_flat, scaler, [f"t{i}_{j}" for i in range(24) for j in ['vib','pres','rpm']])
    print(ts_exp)

if __name__ == "__main__":
    demo()