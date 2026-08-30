# src/vision_advanced.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

# ---- Load ResNet model ----
def load_resnet_model(model_path='models/resnet_defect.pth'):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ---- Grad-CAM (simplified; requires pytorch-grad-cam if used) ----
def generate_gradcam(image_path, model, target_layer=None):
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
        return visualization
    except ImportError:
        print("pytorch-grad-cam not installed; returning None")
        return None

# ---- Severity estimation (count defective pixels) ----
def estimate_severity(image_path, threshold=200):
    """Returns ratio of pixels above threshold (defect area)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    defect_pixels = np.sum(binary == 255)
    total_pixels = img.shape[0] * img.shape[1]
    return defect_pixels / total_pixels

# ---- Autoencoder (optional, kept from earlier) ----
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ---- Training autoencoder (not used in agent, but kept) ----
def train_autoencoder(normal_images, epochs=20, batch_size=32):
    from torch.utils.data import DataLoader, TensorDataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = []
    for path in normal_images:
        img = Image.open(path).convert('L')
        dataset.append(transform(img))
    dataset = torch.stack(dataset)
    loader = DataLoader(TensorDataset(dataset), batch_size=batch_size, shuffle=True)
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"AE Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    return model

def anomaly_score(model, image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img = transform(Image.open(image_path).convert('L')).unsqueeze(0)
    with torch.no_grad():
        recon = model(img)
        mse = torch.mean((recon - img) ** 2).item()
    return mse