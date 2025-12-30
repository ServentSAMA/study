"""
实例分割
"""
from ultralytics import YOLO
import numpy as np
import cv2

# Load a model
# model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)

metrics = model.val()

results = model("https://ultralytics.com/images/bus.jpg")

# 可视化掩码
masks = results[0].masks.data.cpu().numpy()
img = results[0].orig_img
# 可视化第一个掩码
annotated_frame = results[0].plot()  # plot() 自动叠加掩码、框、标签
cv2.imshow("Segmentation", annotated_frame)
cv2.waitKey(0)
cv2.imwrite("output.jpg", annotated_frame)

image = cv2.imread("bus.jpg")

if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()  # (N, H, W) 的二值掩码
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy()

    # 创建一个与原图同尺寸的彩色掩码图（用于叠加）
    overlay = image.copy()
    for i, mask in enumerate(masks):
        # 将 mask 调整到原始图像尺寸（YOLO 输出的 mask 是低分辨率的，需 resize）
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 随机颜色（或按类别分配固定颜色）
        color = np.random.randint(0, 255, size=3).tolist()
        overlay[mask_resized == 1] = color  # 给掩码区域上色

    # 叠加原图和掩码（透明度可调）
    alpha = 0.5
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    cv2.imshow('Masks Overlay', output)
    cv2.waitKey(0)
    cv2.imwrite('mask_overlay.jpg', output)

# Access the results
# for result in results:
#     xy = result.masks.xy  # mask in polygon format
#     xyn = result.masks.xyn  # normalized
#     masks = result.masks.data  # mask in matrix format (num_objects x H x W)

# model.export(format="onnx")
