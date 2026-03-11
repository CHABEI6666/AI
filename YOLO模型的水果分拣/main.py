import json
import time
import threading
import sys
import os
from hqyj_mqtt import Mqtt_Clt
import base64
import numpy as np
import cv2
from ultralytics import YOLO

# 定义光推杆序号，以免后续编程时混淆
load_rod = "first"
ripe_rod = "second"
half_ripe_rod = "third"
raw_rod = "fourth"
all_rod = "all"
# 定义光电对管序号，以免后续编程时混淆
ripe_switch = {"first_switch": False}
half_ripe_switch = {"second_switch": False}
raw_switch = {"third_switch": False}

exit_flag  = False


def push_pull_load_rod(rod):
    mqtt_client.send_json_msg(json.dumps({"rod_control": f"{rod}_push"}))
    time.sleep(1)
    mqtt_client.send_json_msg(json.dumps({"rod_control": f"{rod}_pull"}))


def push_pull(rod):
    def run_rod():
        mqtt_client.send_json_msg(json.dumps({"rod_control": f"{rod}_push"}))
        time.sleep(0.4)
        mqtt_client.send_json_msg(json.dumps({"rod_control": f"{rod}_pull"}))
    # 每个推杆开独立线程，不互相阻塞
    threading.Thread(target=run_rod).start()

def load_rod_loop():
    time.sleep(0.5)
    while not exit_flag:
        push_pull_load_rod(load_rod)
        time.sleep(0.5)



if __name__ == '__main__':
    mqtt_client = Mqtt_Clt("127.0.0.1", 21883, "bb", "aa", 60)
    print("开始控制")
    mqtt_client.control_device("conveyor", 'run') #控制传送带运行
    mqtt_client.control_device("rod_control", "all_pull") #将所有推杆拉回，初始化

    model = YOLO(r"C:\Users\dyvbr\Desktop\YOLO模型的水果分拣\YOLO模型的训练\train10\weights\best.pt")

    load_thread = threading.Thread(target = load_rod_loop)
    load_thread.daemon=True
    load_thread.start()

    results = []  #

    ripe_num = []  #里面的数字代表被检测为熟的桃子的第几个，如[2,4,6],就代表第2，4，6个桃子为熟
    half_ripe_num =[]
    raw_num = []

    cross_ripe_num  = 0  # 穿过熟的推杆的个数
    cross_half_ripe_num = 0
    cross_raw_num  = 0

    completed_ripe_count = 0  #当执行ripe的推杆时的数量
    completed_half_ripe_count = 0


    all_count = 0
    try:
        while True:
            msg = mqtt_client.mqtt_queue.get()
            # 如果获取的是图像数据
            if 'image' in msg:
                # 将 Base64 编码的字符串解码为原始二进制数据。
                image_data = base64.b64decode(msg['image'])
                # 将二进制数据转换为一个 np.uint8 类型的 NumPy 数组。
                image_array = np.frombuffer(image_data, np.uint8)
                # 将 NumPy 数组解码为 OpenCV 图像对象
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                # 识别
                content = model(image)
                labels_cla = [] #每次检测一张后，进入循环为空

                for result in content:
                    # 提取检测框的类别索引
                    cls_indices = result.boxes.cls.cpu().numpy()  # 转为numpy数组
                    # 映射为类别名称
                    labels = [model.names[int(cls)] for cls in cls_indices]
                    labels_cla.extend(labels)
                    print("检测到的标签:", labels)
                    results.extend(labels)
                all_count += 1
                if labels_cla[0] == "ripe":
                    ripe_num.append(all_count)
                elif  labels_cla[0] == "half-ripe":
                    # half_ripe_num.append(all_count)
                    half_ripe_num.append(all_count-results.count("ripe"))
                elif labels_cla[0] == "raw":
                    raw_num.append(all_count)

                "content的值有3种，ripe,half_ripe,raw"
                # print(f"result: {results}")
            else:
                if not results:
                    continue  # 列表为空，跳过分拣
                rod_control = None

                if  ripe_switch == msg:  # 过第一个对管
                    cross_ripe_num += 1   #经过第一个光线对管的数量
                    if ripe_num and ripe_num[0] == cross_ripe_num:
                        push_pull(ripe_rod)
                        del ripe_num[0]
                        completed_ripe_count += 1

                elif  half_ripe_switch == msg:  #水果经过第二个对管
                      cross_half_ripe_num += 1
                      if half_ripe_num and half_ripe_num[0] == cross_half_ripe_num:
                          push_pull(half_ripe_rod)
                          del half_ripe_num[0]
                          completed_half_ripe_count += 1

                elif  raw_switch == msg:  #过第三个对管
                    push_pull(raw_rod)


    except KeyboardInterrupt:
        exit_flag = True
        load_thread.join()


