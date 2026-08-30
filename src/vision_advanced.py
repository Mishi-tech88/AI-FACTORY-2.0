# src/vision_advanced.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import pandas as pd
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_resnet_model(model_path='models/resnet_defect.pth'):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def generate_gradcam(image_path, model, target_layer=None):
    """Generate Grad-CAM heatmap for a single image."""
    if target_layer is None:
        target_layer = model.layer4[-1]  # last conv layer

    # Preprocess image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    img_pil = Image.open(image_path).convert('L')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Run Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)  # None = use highest predicted class
    grayscale_cam = grayscale_cam[0, :]

    # Overlay on original image (resized to 224x224)
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_np = np.stack([img_np]*3, axis=-1)  # make 3-channel for visualization
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

# Example usage (when run standalone)
if __name__ == "__main__":
    model = load_resnet_model()
    # Pick a test image (say the first defect image)
    meta = pd.read_csv('data/raw/image_metadata.csv')
    defect_img = meta[meta['defect_flag']==1].iloc[0]['image_path']
    img_path = f"data/images/{defect_img}"
    heatmap = generate_gradcam(img_path, model)
    plt.imshow(heatmap)
    plt.title(f"Grad-CAM for {defect_img}")
    plt.show()


# Autoencoder for anomaly detection
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
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

def train_autoencoder(normal_images, epochs=20, batch_size=32):
    """normal_images: list of image paths (only class 0)"""
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