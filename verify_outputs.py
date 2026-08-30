# verify_outputs.py
import os
import pandas as pd
import joblib
import torch

def check_files():
    required = [
        'data/raw/production.csv',
        'data/raw/sensors.csv',
        'data/raw/image_metadata.csv',
        'data/raw/maintenance_notes.csv',
        'data/raw/M1_manual.pdf',
        'data/raw/SOP_general.pdf',
        'data/processed/train.csv',
        'data/processed/val.csv',
        'data/processed/test.csv',
        'models/rf_model.pkl',
        'models/xgb_model.pkl',
        'models/ann_tabular.pth',
        'models/lstm_failure.pth',
        'models/resnet_defect.pth',
        'data/processed/maintenance_enhanced.csv'
    ]
    missing = []
    for f in required:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        print("❌ Missing files:")
        for m in missing:
            print(f"   {m}")
        return False
    else:
        print("✅ All expected files exist.")
        return True

def quick_model_test():
    try:
        # Test XGBoost
        xgb = joblib.load('models/xgb_model.pkl')
        print("✅ XGBoost model loaded.")
        # Test ResNet
        model = torch.load('models/resnet_defect.pth', map_location='cpu')
        print("✅ ResNet model loaded.")
        # Load LSTM
        lstm = torch.load('models/lstm_failure.pth', map_location='cpu')
        print("✅ LSTM model loaded.")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

if __name__ == "__main__":
    print("Verifying Stage I - IV outputs...")
    files_ok = check_files()
    models_ok = quick_model_test()
    if files_ok and models_ok:
        print("\n🎉 All stages appear to be complete!")
    else:
        print("\n⚠️ Some stages may need to be rerun.")