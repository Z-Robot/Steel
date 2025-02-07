import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import RandomOverSampler  # 新增过采样

# 参数配置
DATA_PATH = "../data/images"
CSV_PATH = "../data/processed_cold_erosion_report.csv"
BATCH_SIZE = 32  # 增大批量大小
IMG_SIZE = 224
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4  # 调整初始学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 7  # 放宽早停条件

# 优化数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),  # 替换Resize+Crop组合
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 减小旋转幅度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)  # 添加随机擦除
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 增强数据集（添加过采样）
class BalancedSteelDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = self._oversample(df)  # 过采样
        self.transform = transform

    def _oversample(self, df):
        ros = RandomOverSampler(random_state=42)
        X = df[['试样代号']].values
        y = df['主要缺陷'].values
        X_res, y_res = ros.fit_resample(X, y)
        return pd.DataFrame({
            '试样代号': X_res[:, 0],
            '主要缺陷': y_res
        })

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


# 数据准备
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["主要缺陷"], random_state=42)

# 使用平衡数据集
train_dataset = BalancedSteelDataset(train_df, transform=train_transform)
val_dataset = BalancedSteelDataset(val_df, transform=val_transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 修正：添加 train_loader
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 修改模型架构
class RobustEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)  # 使用更大的B1版本

        # 冻结前40%的层
        total_layers = len(list(self.base.children()))
        freeze_idx = int(total_layers * 0.4)
        for idx, child in enumerate(self.base.children()):
            if idx < freeze_idx:
                for param in child.parameters():
                    param.requires_grad = False

        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # 降低Dropout率
            nn.Linear(in_features, 512),
            nn.SiLU(),  # 使用Swish激活函数
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base(x)


model = RobustEfficientNet(num_classes=2).to(DEVICE)


# 改进损失函数
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        weights = self.alpha * (1 - pt) ** self.gamma
        return (weights * ce_loss).mean()


# 优化器配置
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)  # 根据准确率调整
criterion = BalancedFocalLoss(alpha=0.8, gamma=1.5)

# 改进训练循环
best_val_acc = 0
no_improve_epochs = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in train_loader:  # 修正：使用 train_loader
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    # 验证阶段
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total if total != 0 else 0
    scheduler.step(val_acc)  # 根据准确率调整学习率

    # 动态早停策略
    if val_acc > best_val_acc + 0.5:  # 要求至少有0.5%的提升
        best_val_acc = val_acc
        torch.save(model.state_dict(), "../data/results/best_model.pth")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss / len(train_loader):.4f} | "
          f"Val Loss: {val_loss / len(val_loader):.4f} | "
          f"Val Acc: {val_acc:.2f}% | "
          f"Best Acc: {best_val_acc:.2f}%")

    # 早停触发条件
    if no_improve_epochs >= PATIENCE and epoch > 15:  # 至少训练15个epoch
        print(f"Early Stopping at Epoch {epoch + 1}")
        break

print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

'''
Epoch [1/100] | Train Loss: 0.2342 | Val Loss: 0.1910 | Val Acc: 63.83% | Best Acc: 63.83%
Epoch [2/100] | Train Loss: 0.1976 | Val Loss: 0.1856 | Val Acc: 52.13% | Best Acc: 63.83%
Epoch [3/100] | Train Loss: 0.2085 | Val Loss: 0.1489 | Val Acc: 73.40% | Best Acc: 73.40%
Epoch [4/100] | Train Loss: 0.1775 | Val Loss: 0.1547 | Val Acc: 60.64% | Best Acc: 73.40%
Epoch [5/100] | Train Loss: 0.1650 | Val Loss: 0.1987 | Val Acc: 51.06% | Best Acc: 73.40%
Epoch [6/100] | Train Loss: 0.1609 | Val Loss: 0.1669 | Val Acc: 58.51% | Best Acc: 73.40%
Epoch [7/100] | Train Loss: 0.1249 | Val Loss: 0.1981 | Val Acc: 63.83% | Best Acc: 73.40%
Epoch [8/100] | Train Loss: 0.1251 | Val Loss: 0.1741 | Val Acc: 57.45% | Best Acc: 73.40%
Epoch [9/100] | Train Loss: 0.1289 | Val Loss: 0.1549 | Val Acc: 77.66% | Best Acc: 77.66%
Epoch [10/100] | Train Loss: 0.1144 | Val Loss: 0.1518 | Val Acc: 70.21% | Best Acc: 77.66%
Epoch [11/100] | Train Loss: 0.1397 | Val Loss: 0.1289 | Val Acc: 78.72% | Best Acc: 78.72%
Epoch [12/100] | Train Loss: 0.1326 | Val Loss: 0.1337 | Val Acc: 77.66% | Best Acc: 78.72%
Epoch [13/100] | Train Loss: 0.1039 | Val Loss: 0.1307 | Val Acc: 86.17% | Best Acc: 86.17%
Epoch [14/100] | Train Loss: 0.0968 | Val Loss: 0.1428 | Val Acc: 63.83% | Best Acc: 86.17%
Epoch [15/100] | Train Loss: 0.1324 | Val Loss: 0.1370 | Val Acc: 82.98% | Best Acc: 86.17%
Epoch [16/100] | Train Loss: 0.1222 | Val Loss: 0.1268 | Val Acc: 82.98% | Best Acc: 86.17%
Epoch [17/100] | Train Loss: 0.1188 | Val Loss: 0.1292 | Val Acc: 74.47% | Best Acc: 86.17%
Epoch [18/100] | Train Loss: 0.1017 | Val Loss: 0.1184 | Val Acc: 93.62% | Best Acc: 93.62%
Epoch [19/100] | Train Loss: 0.1072 | Val Loss: 0.1279 | Val Acc: 81.91% | Best Acc: 93.62%
Epoch [20/100] | Train Loss: 0.0993 | Val Loss: 0.1276 | Val Acc: 84.04% | Best Acc: 93.62%
Epoch [21/100] | Train Loss: 0.1023 | Val Loss: 0.1288 | Val Acc: 73.40% | Best Acc: 93.62%
Epoch [22/100] | Train Loss: 0.1012 | Val Loss: 0.1290 | Val Acc: 75.53% | Best Acc: 93.62%
Epoch [23/100] | Train Loss: 0.0911 | Val Loss: 0.1306 | Val Acc: 86.17% | Best Acc: 93.62%
Epoch [24/100] | Train Loss: 0.0861 | Val Loss: 0.1242 | Val Acc: 86.17% | Best Acc: 93.62%
Epoch [25/100] | Train Loss: 0.0817 | Val Loss: 0.1298 | Val Acc: 86.17% | Best Acc: 93.62%
Early Stopping at Epoch 25
Best Validation Accuracy: 93.62%

进程已结束,退出代码0
'''