import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

import torch
# 重写 torch.load，强制关闭 weights_only
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# 加载预训练模型
model = YOLO("./yolov8n.pt")

if __name__ == '__main__':
    # 单GPU训练
    model.train(data="../数据的采集/dataset/mydata.yaml", epochs=10, imgsz=640,device ="cpu")  # 训练模型
    # 多GPU训练
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])
    metrics = model.val()  # 在验证集上评估模型性能
    success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
    results = model("../数据的采集/dataset/test/images/image5.jpg")  # 对图像进行预测
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen

