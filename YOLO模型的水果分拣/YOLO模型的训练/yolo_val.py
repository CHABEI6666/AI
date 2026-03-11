from ultralytics import YOLO

model = YOLO(r"C:\Users\dyvbr\Desktop\YOLO模型的水果分拣\YOLO模型的训练\train10\weights\best.pt")  # 加载预训练模型

if __name__ == '__main__':
    metrics = model.val()  # 在验证集上评估模型性能
    print("map50-95:", metrics.box.map)  # map50-95
    print("map50:", metrics.box.map50)  # map50
    print("map75:", metrics.box.map75)  # map75ww
    print("map50-95:", metrics.box.maps)  # a list contains map50-95 of each category

    results = model(r"C:\Users\dyvbr\Desktop\YOLO模型的水果分拣\数据的采集\dataset\test\images\img5.jpg")
    for result in results:
        # 提取检测框的类别索引
        cls_indices = result.boxes.cls.cpu().numpy()  # 转为numpy数组
        # 映射为类别名称
        labels = [model.names[int(cls)] for cls in cls_indices]
        print(type(labels))
        print("检测到的标签:", labels)




