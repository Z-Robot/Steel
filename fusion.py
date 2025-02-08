import os
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# ==================== 配置模块 ====================
class Config:
    # 数据配置
    DATA_PATH = "./data/images"
    CSV_PATH = "data/processed_cold_erosion_report.csv"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练参数
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 5e-4
    PATIENCE = 7
    ALPHA = 0.8  # Focal Loss参数
    GAMMA = 1.5

    # 特征融合配置
    USE_EDGE_FEATURE = True  # 是否使用边缘特征
    FUSION_TYPE = 'attention'  # 可选：concat, attention

    # 预处理参数
    CROP_STRATEGY = {
        'center': (0.3, 0.7),  # 中心裁剪区域 (x_start, x_end)
        'left': (0.0, 0.4)  # 左侧裁剪区域
    }


# ==================== 数据处理模块 ====================
# ==================== 数据处理模块修正版 ====================
class DefectDataset(Dataset):
    def __init__(self, df, transform=None, edge_detection=True):
        self.original_df = df  # 保留原始数据引用
        self.df = self._oversample(df)  # 过采样后的数据
        self.transform = transform
        self.edge_detection = edge_detection
        self.crop_strategy = Config.CROP_STRATEGY

    def _oversample(self, df):
        """修正过采样实现"""
        # 分离特征和标签
        X = df[['试样代号']].values.reshape(-1, 1)  # 必须为二维数组
        y = df['主要缺陷'].values

        # 执行过采样
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)

        # 重建DataFrame
        return pd.DataFrame({
            '试样代号': X_res[:, 0],  # 保持列名一致
            '主要缺陷': y_res
        })

    def __len__(self):
        """关键修正：正确返回数据集长度"""
        return len(self.df)

    def __getitem__(self, idx):
        """修正数据获取逻辑"""
        # 确保使用过采样后的数据
        row = self.df.iloc[idx]
        img_name = row['试样代号'] + ".jpg"  # 使用正确的列名
        label = 0 if row['主要缺陷'] == "中心偏析" else 1

        # 原始图像处理
        img_path = os.path.join(Config.DATA_PATH, img_name)
        # 在__getitem__中添加路径验证
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件缺失：{img_path}")
        image = Image.open(img_path).convert("RGB")
        image = self._adaptive_crop(image, label)

        # 边缘特征生成
        edge = self._generate_edge(image) if Config.USE_EDGE_FEATURE else None

        # 应用变换
        if self.transform:
            image = self.transform(image)
            if edge is not None:
                # 边缘图需要单独处理（单通道）
                edge_transform = transforms.Compose([
                    transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
                    transforms.RandomCrop(Config.IMG_SIZE),
                    transforms.ToTensor()
                ])
                edge = edge_transform(edge)

        return (image, edge), label

    def _adaptive_crop(self, image, label):
        """修正裁剪逻辑"""
        img_array = np.array(image)
        h, w, _ = img_array.shape  # 修正为三维形状

        if label == 0:  # 多数类（中心偏析）
            x_start = int(w * self.crop_strategy['center'][0])
            x_end = int(w * self.crop_strategy['center'][1])
        else:  # 少数类（左侧缺陷）
            x_start = int(w * self.crop_strategy['left'][0])
            x_end = int(w * self.crop_strategy['left'][1])

        return image.crop((x_start, 0, x_end, h))

    def _generate_edge(self, image):
        """修正边缘检测"""
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # 自适应Canny阈值
        v = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(gray, lower, upper)
        return Image.fromarray(edges)

# ==================== 模型架构模块 ====================
class EdgeFeatureExtractor(nn.Module):
    """升级版边缘特征提取器"""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 输入：1通道边缘图
            nn.Conv2d(1, 64, 3, padding=1),  # 增加初始通道数
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 1280, 1),  # 1x1卷积对齐通道数
            nn.BatchNorm2d(1280),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x)


class AttentionFusion(nn.Module):
    """通道自适应的注意力融合"""

    def __init__(self):
        super().__init__()
        # 输入通道数应为1280*2（RGB+Edge）
        self.channel_attn = nn.Sequential(
            nn.Conv2d(1280 * 2, 1280, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, edge_feat):
        # 拼接特征作为注意力输入
        mixed = torch.cat([rgb_feat, edge_feat], dim=1)
        attn = self.channel_attn(mixed)
        return attn * rgb_feat + (1 - attn) * edge_feat


class HybridEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 主干网络
        self.base = models.efficientnet_b1(pretrained=True)
        self._freeze_layers()

        # 边缘特征分支
        self.edge_branch = EdgeFeatureExtractor()

        # 融合模块
        self.fusion = AttentionFusion() if Config.FUSION_TYPE == 'attention' else None

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)  # 保持输出维度一致
        )

    def _freeze_layers(self):
        # 冻结前60%的层以提升训练稳定性
        total_layers = len(list(self.base.features.children()))
        freeze_idx = int(total_layers * 0.6)
        for idx, child in enumerate(self.base.features.children()):
            if idx < freeze_idx:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, rgb, edge=None):
        # RGB特征提取
        rgb_feat = self.base.features(rgb)

        # 边缘特征处理
        if Config.USE_EDGE_FEATURE and edge is not None:
            edge_feat = self.edge_branch(edge)

            # 空间对齐（双线性插值）
            edge_feat = nn.functional.interpolate(
                edge_feat,
                size=rgb_feat.shape[2:],  # H, W
                mode='bilinear',
                align_corners=False
            )

            # 特征融合
            if Config.FUSION_TYPE == 'attention':
                fused = self.fusion(rgb_feat, edge_feat)
            else:
                fused = rgb_feat + edge_feat  # 直接相加融合
        else:
            fused = rgb_feat

        return self.classifier(fused)


# ==================== 训练模块 ====================
class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=Config.LEARNING_RATE,
                                     weight_decay=Config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5)
        self.criterion = self._get_loss()

    def _get_loss(self):
        class_counts = [230, 30]
        weights = torch.tensor([sum(class_counts) / c for c in class_counts]).float()
        return nn.CrossEntropyLoss(weight=weights.to(Config.DEVICE))

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for (rgb, edge), labels in self.train_loader:
            rgb = rgb.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            edge = edge.to(Config.DEVICE) if edge is not None else None

            self.optimizer.zero_grad()
            outputs = self.model(rgb, edge)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for (rgb, edge), labels in self.val_loader:
                rgb = rgb.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                edge = edge.to(Config.DEVICE) if edge is not None else None

                outputs = self.model(rgb, edge)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        acc = 100 * correct / len(self.val_loader.dataset)
        return total_loss / len(self.val_loader), acc

    def run(self):
        best_acc = 0
        no_improve = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step(val_acc)

            # 早停机制
            if val_acc > best_acc + 0.5:
                best_acc = val_acc
                no_improve = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                no_improve += 1

            print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")

            if no_improve >= Config.PATIENCE and epoch > 15:
                print("Early stopping triggered")
                break


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    df = pd.read_csv(Config.CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2,
                                        stratify=df["主要缺陷"],
                                        random_state=42)

    train_dataset = DefectDataset(train_df, train_transform)
    val_dataset = DefectDataset(val_df, val_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=Config.BATCH_SIZE,
                            shuffle=False)


    # 维度验证函数
    def validate_dimensions():
        dummy_rgb = torch.randn(2, 3, 224, 224).to(Config.DEVICE)
        dummy_edge = torch.randn(2, 1, 224, 224).to(Config.DEVICE)

        model = HybridEfficientNet().to(Config.DEVICE)
        rgb_feat = model.base.features(dummy_rgb)
        edge_feat = model.edge_branch(dummy_edge)

        print(f"RGB特征尺寸: {rgb_feat.shape}")  # 应为 [2,1280,H,W]
        print(f"边缘特征尺寸: {edge_feat.shape}")  # 应为 [2,1280,H,W]

        edge_feat = nn.functional.interpolate(edge_feat, size=rgb_feat.shape[2:])
        fused = model.fusion(rgb_feat, edge_feat)
        print(f"融合后特征尺寸: {fused.shape}")  # 应与RGB特征一致


    validate_dimensions()
    # 初始化模型
    model = HybridEfficientNet(num_classes=2)

    # 开始训练
    trainer = Trainer(model, train_loader, val_loader)
    trainer.run()