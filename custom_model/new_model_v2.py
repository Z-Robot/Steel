import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler  # 改为随机过采样
import os

# 参数配置
DATA_PATH = "../data/images"
CSV_PATH = "../data/processed_cold_erosion_report.csv"
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强（保持不变）
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
    transforms.RandomCrop(IMG_SIZE, padding=16),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 修正后的过采样实现
class BalancedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = self._oversample(df)
        self.transform = transform

    def _oversample(self, df):
        ros = RandomOverSampler(random_state=42)
        # 创建虚拟特征矩阵
        X_dummy = np.zeros((len(df), 1))  # 特征维度为1
        y = df['主要缺陷'].values

        # 执行过采样
        X_res, y_res = ros.fit_resample(X_dummy, y)

        # 根据采样结果索引原始数据
        resampled_indices = X_res[:, 0].astype(int)
        return df.iloc[resampled_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0] + ".jpg"
        img_path = os.path.join(DATA_PATH, img_name)
        image = Image.open(img_path).convert("RGB")
        label = 0 if self.df.iloc[idx, 1] == "中心偏析" else 1
        if self.transform:
            image = self.transform(image)
        return image, label


# 模型及其他部分保持不变
class DualStreamNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()

        self.aux_stream = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fusion = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(576 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat_dl = self.backbone(x).squeeze()
        feat_trad = self.aux_stream(x).squeeze()
        fused = torch.cat([feat_dl, feat_trad], dim=1)
        return self.fusion(fused)


# 初始化数据
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["主要缺陷"], random_state=42)

train_dataset = BalancedDataset(train_df, train_transform)
val_dataset = BalancedDataset(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = DualStreamNet().to(DEVICE)
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
    {'params': model.aux_stream.parameters(), 'lr': LEARNING_RATE},
    {'params': model.fusion.parameters(), 'lr': LEARNING_RATE * 2}
], weight_decay=1e-3)

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_acc = 0
confidence_window = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算置信度
            probs = torch.softmax(outputs, 1)
            confidence = torch.max(probs, 1)[0].mean().item()
            confidence_window.append(confidence)

    acc = 100 * correct / total
    scheduler.step()

    # 早停逻辑
    if len(confidence_window) > 10:
        if np.mean(confidence_window[-10:]) < 0.65:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {acc:.2f}%")

print("Training completed.")