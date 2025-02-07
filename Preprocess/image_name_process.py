import os

# 指定包含图片的目录
directory = 'images'

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 检查文件名是否以.jpg结尾
    if filename.endswith('.jpg'):
        # 获取新的文件名，去掉前九个字符
        new_filename = filename[9:]
        # 构建完整的文件路径
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"文件 {filename} 已重命名为 {new_filename}")

print("所有文件处理完成")
