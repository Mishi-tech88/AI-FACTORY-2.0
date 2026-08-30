# src/explainability.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import cv2
import joblib
import os

# Try to import SHAP and LIME; if missing, set to None
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

# ---- Tabular Explainability (SHAP with fallback) ----
def explain_tabular(model, X_sample, feature_names, background_data=None):
    """
    Explain a tabular model prediction using SHAP if available, else fallback.
    Returns: (shap_values, expected_value, explanation_text)
    """
    if not SHAP_AVAILABLE:
        return None, None, "SHAP not installed. Please install shap: pip install shap"
    try:
        if background_data is None:
            background_data = X_sample
        if hasattr(model, 'predict_proba'):
            # For tree-based models
            explainer = shap.TreeExplainer(model, background_data)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1] if len(expected_value) == 2 else expected_value
        else:
            # Fallback to KernelExplainer (model-agnostic)
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            shap_values = explainer.shap_values(X_sample)
            expected_value = explainer.expected_value
        shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)
        feature_importance = dict(zip(feature_names, shap_vals[0]))
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:3]
        explanation_text = f"Top features influencing prediction: "
        explanation_text += ", ".join([f"{feat} (SHAP={val:.3f})" for feat, val in top_features])
        return shap_values, expected_value, explanation_text
    except Exception as e:
        return None, None, f"SHAP explanation failed: {e}"

# ---- Image Explainability (Grad-CAM) ----
def explain_image(model, image_path, target_layer=None):
    """
    Generate Grad-CAM heatmap for an image.
    Returns: heatmap image (numpy array) and a description.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        device = next(model.parameters()).device
        if target_layer is None:
            target_layer = model.layer4[-1]
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        img_pil = Image.open(image_path).convert('L')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        img_resized = img_pil.resize((224, 224))
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_np = np.stack([img_np]*3, axis=-1)
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        explanation_text = "Grad-CAM highlights regions most influential for the decision (red = high influence)."
        return visualization, explanation_text
    except ImportError:
        print("pytorch-grad-cam not installed. Returning original image.")
        img = cv2.imread(image_path)
        return img, "Grad-CAM not available (library missing)."
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        img = cv2.imread(image_path)
        return img, f"Grad-CAM error: {e}"

# ---- Time-Series Explainability ----
def explain_timeseries(model, X_sequence, scaler, feature_names=None):
    """
    Explain a time-series prediction using logistic regression coefficients if possible.
    Returns: feature importance dict and explanation text.
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_sequence.shape[1])]
    try:
        if hasattr(model, 'coef_'):
            coef = model.coef_[0]
            importance = dict(zip(feature_names, coef))
            sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top = sorted_imp[:5]
            explanation_text = f"Top time-series features: "
            explanation_text += ", ".join([f"{feat} (coef={val:.3f})" for feat, val in top])
            return importance, explanation_text
        else:
            # Fallback: try SHAP if available
            if SHAP_AVAILABLE:
                explainer = shap.KernelExplainer(model.predict_proba, X_sequence)
                shap_values = explainer.shap_values(X_sequence)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                if shap_values.ndim == 1:
                    shap_values = shap_values.reshape(1, -1)
                importance = dict(zip(feature_names, shap_values[0]))
                sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
                top = sorted_imp[:5]
                explanation_text = f"Top SHAP features: "
                explanation_text += ", ".join([f"{feat} (SHAP={val:.3f})" for feat, val in top])
                return importance, explanation_text
            else:
                return {}, "Explanation not available for this model."
    except Exception as e:
        return {}, f"Explanation error: {e}"

def save_heatmap(heatmap, filename='gradcam.png'):
    if heatmap is not None:
        cv2.imwrite(filename, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))