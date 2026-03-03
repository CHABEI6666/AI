import os
import shutil
import random

# 源数据集路径
source_dir = 'dataset'

# 新的训练集和验证集路径
train_dir = 'dataset_peach/train'
valid_dir = 'dataset_peach/valid'

prop_train = 0.8

# 创建训练集和验证集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# 列出所有类别
categories = ['half-ripe', 'raw', 'ripe']

# 遍历每个类别
for category in categories:
    source_category_dir = os.path.join(source_dir, category)
    train_category_dir = os.path.join(train_dir, category)
    valid_category_dir = os.path.join(valid_dir, category)

    # 创建每个类别的训练和验证子目录
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(valid_category_dir, exist_ok=True)

    # 获取所有图片文件名
    images = [f for f in os.listdir(source_category_dir) if os.path.isfile(os.path.join(source_category_dir, f))]

    # 打乱图片顺序
    random.shuffle(images)

    # 计算分割点
    split_point = int(len(images) * prop_train)


    # 分割图片到训练集和验证集
    train_images = images[:split_point]
    valid_images = images[split_point:]
    print(len(train_images), len(valid_images))

    # 复制图片到相应的目录
    for img in train_images:
        shutil.copy(os.path.join(source_category_dir, img), os.path.join(train_category_dir, img))

    for img in valid_images:
        shutil.copy(os.path.join(source_category_dir, img), os.path.join(valid_category_dir, img))

print("数据集划分完成")

