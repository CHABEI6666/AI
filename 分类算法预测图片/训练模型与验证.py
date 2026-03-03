# _*_ coding: utf-8 _*_
# 导入所需的库
import os
import random
import sys

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np

# 导入深度学习框架 PyTorch 相关库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ResNet import resnet18
# from ResNet import resnet34
# from ResNet import resnet50
# import torchvision.models as models
from tqdm import tqdm

import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置随机种子以保证结果的可重复性
def setup_seed(seed):
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True  # 设置 cudnn 为确定性算法


# 设置随机种子
setup_seed(0)
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(size=224),  # 随机裁剪和缩放
                                 transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                 # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色变换,亮度、对比度、饱和度和色调
                                 # transforms.RandomRotation(15),  # 随机旋转
                                 transforms.ToTensor(),  # 转换为张量
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "valid": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

train_dataset = datasets.ImageFolder("./dataset_peach/train", transform=transform["train"])
valid_dataset = datasets.ImageFolder("./dataset_peach/valid", transform=transform["valid"])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# 打印一下图片
examples = enumerate(valid_dataloader)
batch_idx, (imgs, labels) = next(examples)
for i in range(4):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = imgs[i].numpy() * std[:, None, None] + mean[:, None, None]
    # 将图片转成numpy数组，主要是转换通道和宽高位置
    image = np.transpose(image, (1, 2, 0))

    image = (image * 255).astype(np.uint8)
    plt.subplot(2, 2, i+1)
    plt.imshow(image)
    plt.title(f"Truth: {labels[i]}")
plt.show()

model = resnet18().to(device)
# model = resnet34().to(device)
# model = resnet50().to(device)
# model = models.resnet18(pretrained=True).to(device)  # 使用ResNet-18

model.load_state_dict(torch.load('./pre_model/resnet18-f37072fd.pth'))
# model.load_state_dict(torch.load('./pre_model/resnet34-333f7ec4.pth'))
# model.load_state_dict(torch.load('./pre_model/resnet50-19c8e357.pth'))

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 3),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_num = len(train_dataset)
valid_num = len(valid_dataset)

epochs = 10
for epoch in range(epochs):
    most_acc = 0.0
    # 训练
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    train_bar = tqdm(train_dataloader, file=sys.stdout)
    # 计算空格数量
    spaces = " " * (len(f"epoch[{epoch + 1}/{epochs}]") + 1)
    for step, data in enumerate(train_bar):
        images, labels = data
        # 将数据移动到设备上
        images = images.to(device)
        labels = labels.to(device)

        train_out = model(images)
        # print(train_out.shape)
        loss = criterion(train_out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()

        predict_y = torch.max(train_out, dim=1)[1]
        # print("---", predict_y)
        train_acc += torch.eq(predict_y, labels).sum().item()

        train_bar.desc = f"epoch[{epoch + 1}/{epochs}] train_loss:{loss:.3f}"
    # print(train_acc)
    # print(train_num)
    train_loss /= train_num
    train_acc /= train_num
    print(f'{spaces}train_loss: {train_loss:.3f}  train_acc: {train_acc:.3f}')

    # 验证
    model.eval()
    val_acc = 0.0  # accumulate accurate number / epoch
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(valid_dataloader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            # 将数据移动到设备上
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_out = model(val_images)
            loss = criterion(val_out, val_labels)
            val_loss += loss.item()

            predict_y = torch.max(val_out, dim=1)[1]
            val_acc += torch.eq(predict_y, val_labels).sum().item()

            val_bar.desc = f"{spaces}val_loss:{loss:.3f}"

    val_loss /= valid_num
    val_acc /= valid_num
    print(f'{spaces}val_loss: {val_loss:.3f}  val_acc: {val_acc:.3f}')

    if val_acc > most_acc:
        most_acc = val_acc
        torch.save(model.state_dict(), './model/best.pth')
    torch.save(model.state_dict(), './model/last.pth')


model.load_state_dict(torch.load('./model/best.pth'))
# 评估模型
correct = 0
total = 0
predicted_labels = []
true_labels = []
model.eval()
with torch.no_grad():
    for images, labels in valid_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# 生成混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

