from torch.nn import functional as F
from d2l import torch as d2l
from torch import nn
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from Preprocess.animator import Animator
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

lr, num_epochs, batch_size = 0.001, 10, 16

class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1, dropout_rate=0.5):
        """
        初始化Residual块

        参数:
        input_channels (int): 输入通道数
        num_channels (int): 输出通道数
        use_1x1conv (bool): 是否使用1x1卷积层进行通道数匹配
        strides (int): 卷积层的步幅
        dropout_rate (float): Dropout层的丢弃概率
        """
        super().__init__()
        # 第一个卷积层，可能进行下采样
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               stride=strides, padding=1)
        # 第二个卷积层，保持尺寸不变
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               stride=1, padding=1)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # 如果需要使用1x1卷积层进行通道数匹配
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        """
        前向传播

        参数:
        X (Tensor): 输入张量
        """
        # 第一个卷积层和批量归一化后应用ReLU激活
        Y = F.relu(self.bn1(self.conv1(X)))
        # 第二个卷积层和批量归一化
        Y = self.bn2(self.conv2(Y))
        # 添加Dropout层
        Y = self.dropout(Y)
        # 如果有1x1卷积层，则对输入进行变换以匹配通道数
        if self.conv3:
            X = self.conv3(X)
        # 输出前应用ReLU激活
        return F.relu(Y + X)

# 创建一个Residual块实例
# blk = Residual(3, 6, use_1x1conv=True, strides=2)
# 打印Residual块的形状，这里应改为先构建模型再输入以查看输出形状
# print(blk.shape)

# 第一个卷积层和最大池化层序列
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """
    构建ResNet的一个阶段

    参数:
    input_channels (int): 输入通道数
    num_channels (int): 输出通道数
    num_residuals (int): Residual块的数量
    first_block (bool): 是否是第一个阶段（不进行下采样）
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 除第一个阶段外，其他阶段的第一个块进行下采样
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # 其他块保持通道数和尺寸不变
            blk.append(Residual(num_channels, num_channels))
    # 返回构建的阶段
    return blk

# 构建ResNet的各个阶段
b2 = resnet_block(64, 64, 1, first_block=True)
b3 = resnet_block(64, 128, 1)
b4 = resnet_block(128, 256, 1)
# 构建整个ResNet模型
net = nn.Sequential(b1, *b2, *b3, *b4, nn.AdaptiveAvgPool2d((1, 1)),
                     nn.Flatten(), nn.Linear(256, 2))  # 修改: 将线性层的输入通道数从512改为256

# 随机生成输入数据，通过模型查看每层输出形状
# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape: \t', X.shape)


# 添加数据集加载和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),  # 随机旋转±30度
    transforms.RandomHorizontalFlip(p=0.5),  # 50%的概率进行水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




# 读取CSV文件，跳过第一行
csv_file_path = '../data/processed_cold_erosion_report.csv'
data = pd.read_csv(csv_file_path, header=0, names=['filename', 'label'])

# 假设图片存储在一个文件夹中，路径为 'F:\\PythonProjects\\TorchLearning\\steel\\images'
image_dir = '../data/images'

# 自定义数据集类
class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        # 读取CSV文件，跳过第一行
        self.img_labels = pd.read_csv(csv_file, header=0, names=['filename', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # 创建一个标签到整数的映射
        self.label_to_int = {label: idx for idx, label in enumerate(self.img_labels['label'].unique())}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")  # 添加 .jpg 扩展名
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            # 记录缺失的文件
            self.missing_files.append(img_path)
            # 选择跳过该条记录
            return None, None
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # 将标签转换为整数类型
        label = self.label_to_int[label]
        return image, label

# 创建数据集实例
dataset = CustomImageDataset(csv_file=csv_file_path, img_dir=image_dir, transform=transform)

# 创建数据加载器
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


def custom_collate_fn(batch):
    # 过滤掉包含 None 的样本对
    batch = [sample for sample in batch if sample[0] is not None and sample[1] is not None]

    images, labels = zip(*batch)

    # 检查并转换图像数据为张量
    images = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img for img in images]

    # 检查并转换标签数据为张量
    labels = [torch.tensor(label) if not isinstance(label, torch.Tensor) else label for label in labels]

    return torch.stack(images), torch.tensor(labels)


# 创建数据加载器时使用自定义 collate_fn
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


# 在调用 d2l.train_ch6 之前添加检查
# def train_ch6(net, train_loader, test_loader, num_epochs, lr, device):
#     net.to(device)  # 将模型移动到指定设备
#     criterion = nn.CrossEntropyLoss()  # 定义损失函数
#     optimizer = optim.Adam(net.parameters(), lr=lr)  # 定义优化器
#
#     train_losses = []
#     train_accs = []
#     test_accs = []
#
#     for epoch in range(num_epochs):
#         # 获取训练集和测试集的批次数量
#         num_train_batches = len(train_loader)
#         num_test_batches = len(test_loader)
#
#         # 检查批次数量是否为零
#         if num_train_batches == 0:
#             print("Error: 训练数据加载器未生成任何批次数据，请检查训练数据集和数据加载器配置。")
#             return
#         if num_test_batches == 0:
#             print("警告: 测试数据集为空，无法进行测试。")
#             return
#     # 调用 d2l.train_ch6 函数
# train_ch6(net, train_loader, test_loader, num_epochs=10, lr=0.001, device=device)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device  # 看网络层第一个参数在哪个设备
    metrics = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):  # list 就一个个挪
            X = [x.to(device) for x in X]
        else:  # tensor就直接挪
            X = X.to(device)
        y = y.to(device)
        metrics.add(d2l.accuracy(net(X), y), y.numel())

def train_resnet(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch学习率减小为原来的0.1

    # 计算类别权重
    train_labels = [y for _, y in train_iter.dataset]
    class_counts = torch.tensor([train_labels.count(i) for i in range(len(set(train_labels)))])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()

    loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    best_test_acc = 0.0
    no_improve_epochs = 0
    patience = 3  # 定义耐心值

    for epoch in range(num_epochs):
        # 训练
        plt.ylim(0, 1)  # 设置纵轴范围
        metric = d2l.Accumulator(3)  # 训练损失之和, 训练准确率之和, 样本数
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

        # 测试
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        # 打印调试信息
        print(
            f'Epoch {epoch + 1}, train_l: {train_l:.3f}, train_acc: {train_acc:.3f}, test_acc: {test_acc:.3f}')
        # 更新动画
        animator.add(epoch + 1, (train_l, train_acc, test_acc))

        # 确保图像刷新
        plt.pause(0.001)

        # 更新学习率
        scheduler.step()

        # 早停法
        if test_acc >= 0.7:  # 修改条件判断，只有当准确率大于等于0.7时才考虑早停
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    plt.savefig('loss_accuracy1.png')  # 保存图表为图片文件



# 使用自定义的训练函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_resnet(net, train_loader, test_loader, num_epochs, lr, device)

