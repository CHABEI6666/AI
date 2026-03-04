import torch
from torch import nn

import cv2
from torchvision import transforms
from ResNet import resnet18


# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

classes = {0: "half-ripe", 1: "raw", 2: "ripe"}

# 1、加载模型
# 假设你的模型定义在 `resnet18` 函数中并已保存最优模型为 './model/best.pth'
model = resnet18().to(device)  # 根据你的模型定义调整 num_classes

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 3),
).to(device)

model.load_state_dict(torch.load('./model/best.pth'))

model.eval()  # 设置模型为评估模式

# 2、预热,确保在不计算梯度的情况下进行推断
with torch.no_grad():
    model(torch.zeros(size=(1, 3, 224, 224)).to(device))

# 3、推理
# 定义图像预处理步骤,torchvision.transforms，通常都需要PIL图像对象作为输入。
preprocess = transforms.Compose([
    transforms.ToPILImage(),  # 先转换为PIL图像
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_resnet(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
    # 预处理图像
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # 创建一个批次 添加为BCHW
    # 确保在不计算梯度的情况下进行推断
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # 获取预测类别
    _, predicted_class = torch.max(probabilities, dim=0)
    return classes[predicted_class.item()]

