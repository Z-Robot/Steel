import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 超参数配置
config = {
    'lr': 1e-3,
    'epochs': 30,
    'batch_size': 32,
    'weight_decay': 1e-3,
    'num_folds': 5,
    'patience': 5,
    'dropout_rate': 0.7,
    'class_weights': [1.0, 10.0]  # 手动设置类别权重
}


# 数据增强策略（增强少数类）
class AdaptiveTransform:
    def __init__(self, label):
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.label = label

    def __call__(self, x):
        x = self.base_transform(x)
        if self.label == 1:  # 对少数类进行额外增强
            x = transforms.functional.adjust_sharpness(x, 3.0)
            if np.random.rand() > 0.5:
                x = transforms.RandomErasing(p=1.0, scale=(0.2, 0.4))(x)
        return x


# 数据集类（添加样本验证和分层支持）
class ImbalanceDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.label_map = {'中心偏析': 0, '内裂': 1}
        self._validate_files()

        # 计算分层分组ID
        self.df['stratify_group'] = self.df['主要缺陷'].apply(lambda x: x + str(np.random.randint(0, 3)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['试样代号']}.jpg"
        label = self.label_map[row['主要缺陷']]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        transform = AdaptiveTransform(label)
        return transform(img), label

    def _validate_files(self):
        missing = []
        for idx in range(len(self.df)):
            img_name = f"{self.df.iloc[idx]['试样代号']}.jpg"
            if not os.path.exists(os.path.join(self.img_dir, img_name)):
                missing.append(img_name)
        if missing:
            print(f"警告：缺失{len(missing)}个图像文件，首5个示例：{missing[:5]}")


# 轻量化ResNet模型（增加正则化）
class LiteResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            self._make_layer(32, 64, 1),
            self._make_layer(64, 128, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(config['dropout_rate'])
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 2)
        )

    def _make_layer(self, in_c, out_c, blocks):
        layers = [ResBlock(in_c, out_c)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# 训练验证流程（集成Focal Loss）
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class Trainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss(alpha=0.75, gamma=2.0)
        self.optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                                     weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(loader, desc='训练进度', leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        return total_loss / total, correct / total

    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc='验证进度', leave=False):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 计算AUC和分类报告
        auc = roc_auc_score(all_labels, all_probs)
        report = classification_report(all_labels, np.round(all_probs),
                                       target_names=['中心偏析', '内裂'],
                                       zero_division=0)
        cm = confusion_matrix(all_labels, np.round(all_probs))

        return {
            'loss': total_loss / total,
            'acc': correct / total,
            'auc': auc,
            'report': report,
            'cm': cm,
            'probs': all_probs,
            'labels': all_labels
        }


# K折交叉验证主流程（修复分类报告问题）
def kfold_training(dataset, config):
    skf = StratifiedKFold(n_splits=config['num_folds'], shuffle=True)
    labels = [label for _, label in dataset]

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n{'=' * 40}")
        print(f" Fold {fold + 1}/{config['num_folds']} ")
        print(f"{'=' * 40}")

        # 数据划分
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        # 加权采样（平衡类别）
        weights = [config['class_weights'][label] for _, label in train_sub]
        sampler = WeightedRandomSampler(weights, len(train_sub) * 3, replacement=True)

        train_loader = DataLoader(train_sub, batch_size=config['batch_size'],
                                  sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_sub, batch_size=config['batch_size'],
                                shuffle=False, num_workers=4)

        # 模型初始化
        model = LiteResNet()
        trainer = Trainer(model, device, config)

        best_auc = 0
        no_improve = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

        for epoch in range(config['epochs']):
            # 训练阶段
            train_loss, train_acc = trainer.train_epoch(train_loader)

            # 验证阶段
            val_results = trainer.evaluate(val_loader)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_results['loss'])
            history['val_acc'].append(val_results['acc'])
            history['val_auc'].append(val_results['auc'])

            # 学习率调整
            trainer.scheduler.step(val_results['auc'])

            # 打印进度
            print(f"Epoch {epoch + 1:02d}/{config['epochs']} | "
                  f"训练损失: {train_loss:.4f} | "
                  f"验证损失: {val_results['loss']:.4f} | "
                  f"准确率: {val_results['acc']:.2%} | "
                  f"AUC: {val_results['auc']:.4f}")

            # 早停判断
            if val_results['auc'] > best_auc:
                best_auc = val_results['auc']
                no_improve = 0
                torch.save(model.state_dict(), f'best_fold{fold}.pth')
                best_results = val_results
            else:
                no_improve += 1
                if no_improve >= config['patience']:
                    print(f"-> 早停触发于第 {epoch + 1} 轮")
                    break

        # Fold结果汇总
        print(f"\nFold {fold + 1} 最佳结果:")
        print(f"- 最高AUC: {best_auc:.4f}")
        print(f"- 最终准确率: {history['val_acc'][-1]:.2%}")
        print("混淆矩阵:")
        print(best_results['cm'])
        print("\n分类报告:")
        print(best_results['report'])

        # 绘制AUC曲线
        fpr, tpr, _ = roc_curve(best_results['labels'], best_results['probs'])
        plt.figure()
        plt.plot(fpr, tpr, label=f'Fold {fold + 1} (AUC={best_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f'auc_fold{fold}.png')

        fold_results.append({
            'history': history,
            'best_auc': best_auc
        })

    # 最终汇总
    avg_auc = np.mean([res['best_auc'] for res in fold_results])
    std_auc = np.std([res['best_auc'] for res in fold_results])
    print(f"\n{'=' * 40}")
    print(f" 最终报告: 平均AUC = {avg_auc:.4f} (±{std_auc:.4f}) ")
    print(f"{'=' * 40}")

    # 可视化结果
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    for res in fold_results:
        plt.plot(res['history']['val_auc'], alpha=0.3)
    plt.title('各Fold验证AUC变化')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')

    plt.subplot(122)
    plt.boxplot([res['history']['val_auc'] for res in fold_results])
    plt.title('AUC分布箱线图')
    plt.savefig('cross_val_results.png', dpi=300, bbox_inches='tight')

    return fold_results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化数据集
    dataset = ImbalanceDataset('../data/processed_cold_erosion_report.csv', 'data/images/')
    print(f"数据集加载完成，总样本数: {len(dataset)}")
    print(f"类别分布: {dict(dataset.df['主要缺陷'].value_counts())}")

    # 执行交叉验证
    results = kfold_training(dataset, config)