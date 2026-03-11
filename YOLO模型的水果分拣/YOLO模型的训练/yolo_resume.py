from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/last.pt")  # 加载预训练模型（建议用于训练）

if __name__ == '__main__':
    # 单GPU训练
    model.train(resume=True)  # 训练模型
    # 多GPU训练
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])
    metrics = model.val()  # 在验证集上评估模型性能
    success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
    results = model("./dataset/test/images/00002.png")  # 对图像进行预测
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen

