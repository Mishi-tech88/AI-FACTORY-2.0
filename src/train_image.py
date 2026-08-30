# src/train_image.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import os
import joblib

os.makedirs('models', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def run_image_training():
    print("="*50)
    print("STAGE II - IMAGE CLASSIFICATION (ResNet Transfer Learning)")
    print("="*50)

    # 1. Load metadata
    img_meta = pd.read_csv('data/raw/image_metadata.csv')

    # 2. Split (random, since no time dependency)
    train_ids, temp_ids = train_test_split(img_meta.index, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    # 3. Custom Dataset
    class FactoryImageDataset(Dataset):
        def __init__(self, indices, transform):
            self.indices = indices
            self.transform = transform
            self.meta = img_meta
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            idx = self.indices[i]
            row = self.meta.loc[idx]
            img = Image.open(f"data/images/{row['image_path']}").convert('L')  # grayscale
            img = self.transform(img)
            label = row['defect_flag']
            return img, label

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    train_ds = FactoryImageDataset(train_ids, transform_train)
    val_ds = FactoryImageDataset(val_ids, transform_val)
    test_ds = FactoryImageDataset(test_ids, transform_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # 4. Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1)  # binary classification
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training
    print("Training ResNet...")
    for epoch in range(10):  # only 10 epochs for speed
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1,1)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # 6. Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = torch.sigmoid(model(imgs)).cpu().numpy().flatten()
            all_probs.extend(preds)
            all_labels.extend(labels.numpy())
            all_preds.extend((preds >= 0.5).astype(int))

    print("\nResNet-18 Performance:")
    print(classification_report(all_labels, all_preds))
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")

    torch.save(model.state_dict(), 'models/resnet_defect.pth')
    print("ResNet model saved to 'models/'")

if __name__ == "__main__":
    run_image_training()