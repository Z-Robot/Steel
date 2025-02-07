from torch.nn import functional as F
from d2l import torch as d2l
from torch import nn
import torch
import os
from PIL import Image
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 超参数管理
class Hyperparameters:
    def __init__(self, num_epochs=10, lr=0.001, batch_size=16):
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

# 数据预处理和加载
class DataPreprocessor:
    def __init__(self, csv_file_path, image_dir, transform, batch_size):
        self.csv_file_path = csv_file_path
        self.image_dir = image_dir
        self.transform = transform
        self.batch_size = batch_size

    def load_data(self):
        data = pd.read_csv(self.csv_file_path, header=0, names=['filename', 'label'])
        logging.info("Unique labels: %s", data['label'].unique())
        dataset = CustomImageDataset(csv_file=self.csv_file_path, img_dir=self.image_dir, transform=self.transform)
        logging.info("Dataset size: %d", len(dataset))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        logging.info("Train dataset size: %d", len(train_dataset))
        logging.info("Test dataset size: %d", len(test_dataset))
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        return train_loader, test_loader

# 自定义数据集类
class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file, header=0, names=['filename', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_to_int = {label: idx for idx, label in enumerate(self.img_labels['label'].unique())}
        logging.info("Label to int mapping: %s", self.label_to_int)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            logging.error("文件不存在: %s", img_path)
            return None, None
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = self.label_to_int[label]
        return image, label

# 模型结构
class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def build_model():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = resnet_block(64, 64, 2, first_block=True)
    b3 = resnet_block(64, 128, 2)
    b4 = resnet_block(128, 256, 2)
    b5 = resnet_block(256, 512, 2)
    net = nn.Sequential(b1, *b2, *b3, *b4, *b5, nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 2))
    return net

# 训练和评估函数
def evaluate_accuracy(net, data_loader, device):
    net.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total != 0 else 0

def train_epoch(net, train_loader, optimizer, criterion, device):
    net.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()  # 梯度清零
        outputs = net(X)  # 前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # # 打印部分批次的训练信息（可选）
        # if (i + 1) % (max(1, len(train_loader) // 5)) == 0 or i == len(train_loader) - 1:
        #     logging.info(f'Epoch [{epoch + 1}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    return running_loss / len(train_loader), correct / total

def train_ch6(net, train_loader, test_loader, num_epochs, lr, device):
    net.to(device)  # 将模型移动到指定设备
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 定义优化器

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        # 训练一个 epoch
        train_loss, train_acc = train_epoch(net, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试模型
        test_acc = evaluate_accuracy(net, test_loader, device)
        test_accs.append(test_acc)

        # 输出每个 epoch 的结果
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # 绘制损失和准确率图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='train loss')
    plt.plot(range(1, num_epochs + 1), train_accs, label='train acc', linestyle='--')
    plt.plot(range(1, num_epochs + 1), test_accs, label='test acc', linestyle='-.')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training and Testing Performance')
    plt.legend()
    plt.ylim(0, 1)  # 设置纵轴范围
    plt.savefig('loss_accuracy.png')  # 保存图表为图片文件
    plt.close()

# 主函数
def main():
    # 超参数设置
    hyperparams = Hyperparameters(num_epochs=10, lr=0.001, batch_size=16)

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_preprocessor = DataPreprocessor(csv_file_path='../data/processed_cold_erosion_report.csv',
                                         image_dir='../data/images',
                                         transform=transform,
                                         batch_size=hyperparams.batch_size)
    train_loader, test_loader = data_preprocessor.load_data()

    # 模型构建
    net = build_model()

    # 训练和评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ch6(net, train_loader, test_loader, hyperparams.num_epochs, hyperparams.lr, device)

if __name__ == "__main__":
    main()

