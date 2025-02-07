import pandas as pd
import re

# 读取XLS文件
df = pd.read_excel('冷蚀报告2011-04减列.xls', sheet_name=0)

# 定义缺陷优先级顺序
defect_priority = ['内裂', '角裂', '夹杂', '中心偏析', '中心疏松', '针孔']


# 自定义函数，提取数值部分
def extract_number(value):
    if pd.isna(value):
        return None
    match = re.search(r'\d+(\.\d+)?', str(value))
    return float(match.group()) if match else None


# 自定义函数，根据缺陷数值和优先级顺序确定主要缺陷
def determine_main_defect(row):
    # 获取缺陷数值
    defect_values = {
        '内裂': extract_number(row['内裂']),
        '角裂': extract_number(row['角裂']),
        '夹杂': extract_number(row['夹杂']),
        '中心偏析': extract_number(row['中心偏析']),
        '中心疏松': extract_number(row['中心疏松']),
        '针孔': extract_number(row['针孔'])
    }

    # 过滤掉非数值和NaN值
    defect_values = {k: v for k, v in defect_values.items() if pd.notna(v) and isinstance(v, (int, float))}

    # 如果所有缺陷数值都是NaN，返回None
    if not defect_values:
        return None

    # 找出数值最大的缺陷
    max_value = max(defect_values.values())
    max_defects = [defect for defect, value in defect_values.items() if value == max_value]

    # 如果有多个缺陷数值相同，根据优先级顺序选择
    if len(max_defects) > 1:
        for priority_defect in defect_priority:
            if priority_defect in max_defects:
                return priority_defect
    else:
        return max_defects[0]


# 应用自定义函数，创建新列'main_defect'
df['主要缺陷'] = df.apply(determine_main_defect, axis=1)

# 只保留图片名字和主要缺陷列
result_df = df[['name', '主要缺陷']]

# 重命名列名
result_df.columns = ['试样代号', '主要缺陷']

defect_counts = result_df['主要缺陷'].value_counts()

# # 打印统计结果
# print("主要缺陷的数量统计：")
# print(defect_counts)

# 保存处理后的数据到新的CSV文件
result_df.to_csv('processed_cold_erosion_report.csv', index=False)

print("处理完成，结果已保存到'processed_cold_erosion_report.csv'")

'''
中心偏析    230
内裂       32
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),  # 随机旋转±30度
    transforms.RandomHorizontalFlip(p=0.5),  # 50%的概率进行水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 50%的概率进行垂直翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.RandomCrop(224, padding=4),  # 随机裁剪
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # 随机裁剪和缩放
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''