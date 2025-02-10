import os
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns  # 新增可视化库


# ==================== 配置模块 ====================
class Config:
    # 数据配置
    DATA_PATH = "./data/images"
    CSV_PATH = "data/processed_cold_erosion_report.csv"
    IMG_SIZE = 224
    BATCH_SIZE = 8  # 减小批量大小
    NUM_EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_FOCAL_LOSS = True
    current_stage = 'pretrain'  # 训练阶段控制
    stage_patience = 3         # 阶段切换耐心值
    stage_threshold = 0.01     # 1%提升阈值

    # 训练参数
    LEARNING_RATE = 1e-5  # 降低学习率
    WEIGHT_DECAY = 3e-4  # 增大权重衰减
    PATIENCE = 10  # 放宽早停条件
    ALPHA = 0.8
    GAMMA = 1.5

    # 特征融合配置
    USE_EDGE_FEATURE = True
    FUSION_TYPE = 'attention'  # 可选：concat, attention, add

    # 预处理参数
    CROP_STRATEGY = {
        'center': (0.3, 0.7),
        'left': (0.0, 0.4)
    }


# ==================== 数据处理模块 ====================
class DefectDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.original_df = df
        self.df = self._safe_oversample(df, is_train)
        self.transform = transform
        self.is_train = is_train
        self.crop_strategy = Config.CROP_STRATEGY

    def _safe_oversample(self, df, is_train):
        """防止数据泄露的过采样"""
        if not is_train:
            return df

        X = df[['试样代号']].values.reshape(-1, 1)
        y = df['主要缺陷'].values

        # 检查是否需要过采样
        if (y == 1).sum() < 10:  # 少数类样本不足时进行过采样
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X, y)
            return pd.DataFrame({
                '试样代号': X_res[:, 0],
                '主要缺陷': y_res
            })
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['试样代号'] + ".jpg"
        label = 0 if row['主要缺陷'] == "中心偏析" else 1

        # 加载图像
        img_path = os.path.join(Config.DATA_PATH, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件缺失：{img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self._adaptive_crop(image, label)

        # 边缘检测
        edge = self._generate_edge(image) if Config.USE_EDGE_FEATURE else torch.zeros(0)  # 空张量
        # 应用变换
        if self.transform:
            image = self.transform(image)
            if edge is not None:
                edge = self._edge_transform(edge)

        return (image, edge), torch.tensor(label, dtype=torch.long)

    def _adaptive_crop(self, image, label):
        """改进的智能裁剪"""
        img_array = np.array(image)
        h, w, _ = img_array.shape

        # 添加随机偏移防止过拟合
        if self.is_train:
            offset = np.random.randint(-10, 10)
        else:
            offset = 0

        if label == 0:
            x_start = int(w * Config.CROP_STRATEGY['center'][0]) + offset
            x_end = int(w * Config.CROP_STRATEGY['center'][1]) + offset
        else:
            x_start = int(w * Config.CROP_STRATEGY['left'][0]) + offset
            x_end = int(w * Config.CROP_STRATEGY['left'][1]) + offset

        return image.crop((max(0, x_start), 0, min(w, x_end), h))

    def _generate_edge(self, image):
        """改进的自适应边缘检测"""
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 自适应Canny
        sigma = 0.33
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        return Image.fromarray(edges)

    def _edge_transform(self, edge):
        """边缘图专用变换"""
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE + 64, Config.IMG_SIZE + 64)),
            transforms.RandomCrop(Config.IMG_SIZE),
            transforms.ToTensor()
        ])(edge)


# ==================== 模型架构模块 ====================
class LightEdgeExtractor(nn.Module):
    """轻量级边缘特征提取器"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 输入：单通道边缘图
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 1280, 1)  # 通道对齐
        )

    def forward(self, x):
        return self.layers(x)


class FocalLoss(nn.Module):
    """Focal Loss 替代交叉熵"""
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()



class RobustFusion(nn.Module):
    """鲁棒的注意力融合模块"""

    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(1280 * 2, 1280, 1),  # 输入：RGB+Edge特征拼接
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(1280)

    def forward(self, rgb, edge):
        mixed = torch.cat([rgb, edge], dim=1)
        attn = self.attention(mixed)
        return self.bn(attn * rgb + (1 - attn) * edge)


class OptimizedEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 主干网络（使用B0版本）
        self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self._freeze_backbone()

        # 边缘特征分支
        self.edge_branch = LightEdgeExtractor()

        # 融合模块
        self.fusion = None
        if Config.FUSION_TYPE == 'attention':
            self.fusion = RobustFusion()
        elif Config.FUSION_TYPE == 'add':
            self.fusion = lambda x, y: x + y

        # 增强的分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _freeze_backbone(self):
        children = list(self.base.features.children())
        freeze_idx = 0
        if Config.current_stage == 'pretrain':
            freeze_idx = int(len(children) * 0.9)  # 冻结前90%
        elif Config.current_stage == 'finetune':
            freeze_idx = int(len(children) * 0.5)  # 冻结前50%

        # 冻结或解冻层
        for idx, layer in enumerate(children):
            if idx < freeze_idx:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # 日志输出逻辑
        total_params = sum(p.numel() for p in self.base.parameters())
        trainable_params = sum(p.numel() for p in self.base.parameters() if p.requires_grad)

        # 初始化时打印日志
        if not hasattr(self, 'last_stage'):
            self.last_stage = Config.current_stage
            print(f"\n初始化阶段: {Config.current_stage}")
            print(f"冻结层数: {freeze_idx}/{len(children)}")
            print(f"可训练参数比例: {trainable_params / total_params:.1%}")
        # 阶段切换时打印日志
        elif self.last_stage != Config.current_stage:
            self.last_stage = Config.current_stage
            print(f"\n切换到 {Config.current_stage} 阶段")
            print(f"新冻结层数: {freeze_idx}/{len(children)}")
            print(f"可训练参数比例: {trainable_params / total_params:.1%}")

    def forward(self, rgb, edge=None):
        assert edge is None or isinstance(edge, torch.Tensor), "Edge must be Tensor or None"

        # RGB特征提取
        rgb_feat = self.base.features(rgb)

        # 边缘特征处理
        if Config.USE_EDGE_FEATURE and edge is not None:
            edge_feat = self.edge_branch(edge)

            # 空间对齐（最近邻插值）
            edge_feat = nn.functional.interpolate(
                edge_feat,
                size=rgb_feat.shape[2:],
                mode='nearest'
            )

            # 特征融合
            if self.fusion is not None:
                fused = self.fusion(rgb_feat, edge_feat)
            else:
                fused = rgb_feat
        else:
            fused = rgb_feat

        return self.classifier(fused)


# ==================== 训练模块 ====================
class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 新增训练记录
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.current_epoch = 0

        # 优化器和学习率调度
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=Config.NUM_EPOCHS)

        # 损失函数
        class_counts = self._get_class_counts()
        weights = torch.tensor(
            [1.0, class_counts[0] / class_counts[1]],
            dtype=torch.float32
        ).to(Config.DEVICE)
        self.criterion = FocalLoss(alpha=0.75, gamma=2) if Config.USE_FOCAL_LOSS \
            else nn.CrossEntropyLoss(weight=weights)
        self.best_acc = 0
        self.no_improve = 0

    def _get_class_counts(self):
        return [
            (self.train_loader.dataset.original_df['主要缺陷'] == "中心偏析").sum(),
            (self.train_loader.dataset.original_df['主要缺陷'] != "中心偏析").sum()
        ]

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for (rgb, edge), labels in self.train_loader:
            rgb = rgb.to(Config.DEVICE)
            edge = edge.to(Config.DEVICE) if edge is not None else None
            labels = labels.to(Config.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(rgb, edge)
            loss = self.criterion(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        train_acc = 100 * correct / total
        self.train_accs.append(train_acc)
        self.train_losses.append(total_loss / len(self.train_loader))
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for (rgb, edge), labels in self.val_loader:
                rgb = rgb.to(Config.DEVICE)
                edge = edge.to(Config.DEVICE) if edge is not None else None
                labels = labels.to(Config.DEVICE)

                outputs = self.model(rgb, edge)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        return total_loss / len(self.val_loader), acc

    def run(self):
        stage_changed = False  # 标记阶段是否已切换
        best_stage_acc = 0  # 记录当前阶段最佳准确率
        no_improve_epochs = 0  # 未提升计数器

        for epoch in range(Config.NUM_EPOCHS):

            self.current_epoch = epoch + 1
            # 动态调整学习率
            if Config.current_stage == 'finetune' and not stage_changed:
                new_lr = Config.LEARNING_RATE * 0.1  # finetune阶段使用更小学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            # --- 阶段切换逻辑 ---
            self.model._freeze_backbone()  # 强制更新冻结状态
            if stage_changed:
                # 重新初始化优化器以包含新解冻的参数
                self.optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=Config.LEARNING_RATE * 0.1 if Config.current_stage == 'finetune' else Config.LEARNING_RATE,
                    weight_decay=Config.WEIGHT_DECAY
                )
                stage_changed = False
                print(f"\n切换到 {Config.current_stage} 阶段，重置优化器")

            # --- 训练与验证 ---
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            # --- 阶段切换判断 ---
            if Config.current_stage == 'pretrain':
                if val_acc > best_stage_acc + Config.stage_threshold:
                    best_stage_acc = val_acc
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                # 满足切换条件
                if no_improve_epochs >= Config.stage_patience and self.current_epoch > 10:
                    Config.current_stage = 'finetune'
                    stage_changed = True
                    no_improve_epochs = 0
                    best_stage_acc = 0  # 重置阶段最佳值

            # 记录验证数据
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            self.scheduler.step(val_acc)

            # 早停机制
            if val_acc > self.best_acc + 0.1:
                self.best_acc = val_acc
                self.no_improve = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                self.no_improve += 1

            # 打印训练信息
            print(f"Epoch {self.current_epoch}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {self.train_accs[-1]:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {self.best_acc:.2f}%\n")

            # 性能分析
            if (self.current_epoch) % 10 == 0:
                self._analyze_performance()

            if self.no_improve >= Config.PATIENCE and self.current_epoch > 20:
                print(f"Early stopping at epoch {self.current_epoch}")
                break

        # 训练结束可视化
        self._visualize_training()

    def _analyze_performance(self):
        """增强的性能分析可视化"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for (rgb, edge), labels in self.val_loader:
                outputs = self.model(
                    rgb.to(Config.DEVICE),
                    edge.to(Config.DEVICE) if edge is not None else None
                )
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Epoch {self.current_epoch})')
        plt.savefig(f"confusion_matrix_epoch{self.current_epoch}.png")
        plt.close()

        print("\n性能分析:")
        print(f"混淆矩阵:\n{cm}")
        if cm.shape[0] > 1:
            recall = cm[1, 1] / cm[1].sum()
            print(f"少数类召回率: {recall:.2%}")

    def _visualize_training(self):
        """训练过程可视化"""
        plt.figure(figsize=(12, 5))

        # Loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_curves.png")
        plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 数据增强配置
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 64, Config.IMG_SIZE + 64)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 数据加载
    df = pd.read_csv(Config.CSV_PATH)
    train_df_raw, val_df_raw = train_test_split(
        df,
        test_size=0.2,
        stratify=df["主要缺陷"],
        random_state=42
    )

    train_dataset = DefectDataset(train_df_raw, train_transform, is_train=True)
    val_dataset = DefectDataset(val_df_raw, val_transform, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # 初始化模型
    model = OptimizedEfficientNet(num_classes=2)

    # 开始训练
    trainer = AdvancedTrainer(model, train_loader, val_loader)
    trainer.run()

"""

    
    
        # 数据增强配置
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 64, Config.IMG_SIZE + 64)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomApply([
            transforms.RandomResizedCrop(Config.IMG_SIZE,
                                        scale=(0.8, 1.0),
                                        ratio=(0.8, 1.2))  # 增强少数类多样性
        ], p=0.5 if Config.current_stage == 'pretrain' else 0.8),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.7, scale=(0.05, 0.3))  # 更强遮挡
"""


'''
Epoch 1/100
Train Loss: 0.5113 | Val Loss: 0.6329
Val Acc: 83.02% | Best Acc: 83.02%
Epoch 2/100
Train Loss: 0.2891 | Val Loss: 0.4411
Val Acc: 79.25% | Best Acc: 83.02%
Epoch 3/100
Train Loss: 0.2017 | Val Loss: 0.2384
Val Acc: 88.68% | Best Acc: 88.68%
Epoch 4/100
Train Loss: 0.1875 | Val Loss: 0.3423
Val Acc: 86.79% | Best Acc: 88.68%
Epoch 5/100
Train Loss: 0.1830 | Val Loss: 0.2562
Val Acc: 86.79% | Best Acc: 88.68%
Epoch 6/100
Train Loss: 0.1909 | Val Loss: 0.1784
Val Acc: 90.57% | Best Acc: 90.57%
Epoch 7/100
Train Loss: 0.1900 | Val Loss: 0.2614
Val Acc: 84.91% | Best Acc: 90.57%
Epoch 8/100
Train Loss: 0.1727 | Val Loss: 0.1411
Val Acc: 94.34% | Best Acc: 94.34%
Epoch 9/100
Train Loss: 0.1531 | Val Loss: 0.2943
Val Acc: 86.79% | Best Acc: 94.34%
Epoch 10/100
Train Loss: 0.1536 | Val Loss: 0.2961
Val Acc: 86.79% | Best Acc: 94.34%

性能分析:
混淆矩阵:
[[41  6]
 [ 1  5]]
少数类召回率: 83.33%
Epoch 11/100
Train Loss: 0.1596 | Val Loss: 0.1356
Val Acc: 92.45% | Best Acc: 94.34%
Epoch 12/100
Train Loss: 0.1774 | Val Loss: 0.2180
Val Acc: 92.45% | Best Acc: 94.34%
Epoch 13/100
Train Loss: 0.1166 | Val Loss: 0.1156
Val Acc: 94.34% | Best Acc: 94.34%
Epoch 14/100
Train Loss: 0.1384 | Val Loss: 0.1725
Val Acc: 92.45% | Best Acc: 94.34%
Epoch 15/100
Train Loss: 0.1624 | Val Loss: 0.2258
Val Acc: 94.34% | Best Acc: 94.34%
Epoch 16/100
Train Loss: 0.1177 | Val Loss: 0.1210
Val Acc: 94.34% | Best Acc: 94.34%
Epoch 17/100
Train Loss: 0.1459 | Val Loss: 0.2184
Val Acc: 90.57% | Best Acc: 94.34%
Epoch 18/100
Train Loss: 0.1258 | Val Loss: 0.1729
Val Acc: 92.45% | Best Acc: 94.34%
Epoch 19/100
Train Loss: 0.1479 | Val Loss: 0.2180
Val Acc: 88.68% | Best Acc: 94.34%
Epoch 20/100
Train Loss: 0.1089 | Val Loss: 0.3125
Val Acc: 92.45% | Best Acc: 94.34%

性能分析:
混淆矩阵:
[[42  5]
 [ 1  5]]
少数类召回率: 83.33%
Epoch 21/100
Train Loss: 0.1432 | Val Loss: 0.2767
Val Acc: 92.45% | Best Acc: 94.34%
Epoch 22/100
Train Loss: 0.1196 | Val Loss: 0.1855
Val Acc: 90.57% | Best Acc: 94.34%
Early stopping at epoch 22

进程已结束，退出代码为 0

'''