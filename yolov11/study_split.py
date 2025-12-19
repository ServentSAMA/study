"""
实例分割
"""
from ultralytics import YOLO
import numpy as np
import cv2

# Load a model
# model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)

metrics = model.val()

results = model("https://ultralytics.com/images/bus.jpg")

# 可视化掩码
masks = results[0].masks.data.cpu().numpy()
img = results[0].orig_img
# 可视化第一个掩码
if len(masks) > 0:
    mask0 = masks[0]  # H x W 的二值图（0 或 1）

    # 将掩码转为三通道用于叠加
    colored_mask = np.zeros_like(img)
    colored_mask[mask0 == 1] = [0, 255, 0]  # 绿色

    # 叠加到原图
    overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)


# Access the results
# for result in results:
#     xy = result.masks.xy  # mask in polygon format
#     xyn = result.masks.xyn  # normalized
#     masks = result.masks.data  # mask in matrix format (num_objects x H x W)

# model.export(format="onnx")
