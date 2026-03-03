import torch
from torch import nn

import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from ResNet import resnet18

import contextlib
import time


class Profile(contextlib.ContextDecorator):  # 测量代码段的执行时间
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """
        当使用 with 语句或作为装饰器时，__enter__ 方法会在代码块执行前调用。
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        当使用 with 语句或作为装饰器时，__exit__ 方法会在代码块执行后调用。
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        提供了获取当前时间戳的接口，如果 self.cuda 为真，会先调用 torch.cuda.synchronize() 来确保 GPU 的所有操作完成后再获取时间，以得到准确的时间测量。
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


dt = (Profile(), Profile(), Profile(), Profile())

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

classes = {0: "half-ripe", 1: "raw", 2: "ripe"}

# 1、加载模型
with dt[0]:
    # 假设你的模型定义在 `resnet18` 函数中并已保存最优模型为 './model/best.pth'
    model = resnet18().to(device)  # 根据你的模型定义调整 num_classes

    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 3),
    ).to(device)

    model.load_state_dict(torch.load('./model/best.pth'))

model.eval()  # 设置模型为评估模式

# 2、预处理
with dt[1]:
    # 定义图像预处理步骤,torchvision.transforms，通常都需要PIL图像对象作为输入。
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # 先转换为PIL图像
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载并预处理图像
    img_path = './dataset_peach/valid/half-ripe/img41.jpg'  # 替换为你的图像路径
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
    # 预处理图像
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # 创建一个批次 添加为BCHW

# 3、预热，
with dt[2]:
    # 预热的作用：
    # 内存分配：深度学习模型在第一次推理时需要进行大量的内存分配，包括模型参数和中间结果的存储。预热阶段会完成这些内存分配，从而在后续的推理中减少延迟。
    # 编译优化：一些深度学习框架（例如TensorFlow、PyTorch等）会在第一次执行模型时进行编译和优化，包括图优化和算子融合等。这些优化在预热阶段完成后，可以显著提高后续推理的效率。
    # 缓存机制：硬件（如CPU和GPU）的缓存机制可以在预热过程中加载必要的数据和指令，从而提高后续推理的速度。例如，GPU在预热阶段会将模型参数和计算内核加载到显存中，以减少后续计算时的数据传输延迟。
    # 系统调度：操作系统和硬件资源的调度也会影响推理时间。预热阶段可以让操作系统和硬件资源在实际推理前做好准备，从而减少调度延迟。
    with torch.no_grad():
        model(torch.zeros(size=(1, 3, 224, 224)).to(device))

# 4、推理
with dt[3]:
    # 确保在不计算梯度的情况下进行推断
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # 获取预测类别
    _, predicted_class = torch.max(probabilities, dim=0)

    pre_class = classes[predicted_class.item()]

# 打印预测结果
print(f'Predicted class: {pre_class}, probability {probabilities[predicted_class].item():.4f}')

# 可视化输入图像
plt.imshow(image)
plt.title(f'Predicted class: {pre_class}, probability: {probabilities[predicted_class].item():.4f}')

t = tuple(x.t / 1 * 1E3 for x in dt)  # speeds per image  1E3是1000，用来将时间单位从秒转换为毫秒。
print(f'图片尺寸: {(1, 3, 224, 224)}, 加载模型: %.1fms, 预处理: %.1fms ,预热: %.1fms ,推理: %.1fms' % t)

plt.show()
