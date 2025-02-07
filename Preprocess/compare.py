# 添加必要的导入语句
import os
from PIL import Image
import pandas as pd
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

# 添加数据集加载和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义数据集类
class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        # 读取CSV文件，跳过第一行
        self.img_labels = pd.read_csv(csv_file, header=0, names=['filename', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")  # 添加 .jpg 扩展名
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            # 记录缺失的文件
            missing_files.append(img_path)
            # 选择跳过该条记录
            return None, None
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

csv_file_path = '../data/processed_cold_erosion_report.csv'


# 假设图片存储在一个文件夹中，路径为 'F:\\PythonProjects\\TorchLearning\\steel\\images'
image_dir = '../data/images'
# 创建数据集实例
missing_files = []  # 初始化缺失文件列表
dataset = CustomImageDataset(csv_file=csv_file_path, img_dir=image_dir, transform=transform)

# 检查CSV文件中的文件名是否在图片文件夹中存在
csv_filenames = set(dataset.img_labels['filename'])
image_filenames = set([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')])
missing_in_csv = csv_filenames - image_filenames
missing_in_images = image_filenames - csv_filenames

# 删除CSV文件中缺失的图片文件对应的记录
filtered_img_labels = dataset.img_labels[~dataset.img_labels['filename'].isin(missing_in_csv)]

# 更新dataset的img_labels
dataset.img_labels = filtered_img_labels

print(f"CSV文件中缺失的图片文件: {missing_in_csv}")
print(f"图片文件夹中缺失的CSV记录: {missing_in_images}")
