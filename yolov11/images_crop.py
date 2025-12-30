import os
import cv2
import numpy as np

# 配置
IMG_WIDTH = 2678
IMG_HEIGHT = 1786
OUTPUT_SIZE = 640
INPUT_IMAGE_DIR = "datasets\\lock\\images"          # 原始图像目录
INPUT_LABEL_DIR = "datasets\\lock\\labels"          # 原始标签目录
OUTPUT_IMAGE_DIR = "output\\images\\val"
OUTPUT_LABEL_DIR = "output\\labels\\val"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def crop_and_resize_with_label(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return

    h, w = img.shape[:2]
    assert (w, h) == (IMG_WIDTH, IMG_HEIGHT), f"图像尺寸不匹配: {img_path} -> ({w}, {h})"

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        cx_norm, cy_norm, bw_norm, bh_norm = map(float, parts[1:5])

        # 转为像素坐标
        cx = cx_norm * w
        cy = cy_norm * h
        bw = bw_norm * w
        bh = bh_norm * h

        # 确定裁剪区域：以(cx, cy)为中心的正方形
        side = max(w, h)  # 用最大边确保覆盖，但实际会受边界限制
        half = min(side // 2, max(w, h))  # 安全半长

        # 实际裁剪区域（不能越界）
        left = int(max(0, cx - half))
        right = int(min(w, cx + half))
        top = int(max(0, cy - half))
        bottom = int(min(h, cy + half))

        # 如果不是正方形，补成正方形（用边缘像素或黑色填充）
        crop_h = bottom - top
        crop_w = right - left
        max_side = max(crop_h, crop_w)

        # 创建正方形画布（这里用黑色填充，也可用边缘复制）
        square_crop = np.zeros((max_side, max_side, 3), dtype=img.dtype)

        # 计算在正方形中的偏移
        y_offset = (max_side - crop_h) // 2
        x_offset = (max_side - crop_w) // 2

        cropped = img[top:bottom, left:right]
        square_crop[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped

        # 缩放到 640x640
        resized = cv2.resize(square_crop, (OUTPUT_SIZE, OUTPUT_SIZE))

        # 转换标签坐标到新图像
        # 原始框在裁剪图中的位置（未缩放前）
        box_left_orig = cx - bw / 2
        box_right_orig = cx + bw / 2
        box_top_orig = cy - bh / 2
        box_bottom_orig = cy + bh / 2

        # 映射到正方形裁剪图（考虑填充偏移）
        new_left = (box_left_orig - left) + x_offset
        new_right = (box_right_orig - left) + x_offset
        new_top = (box_top_orig - top) + y_offset
        new_bottom = (box_bottom_orig - top) + y_offset

        # 缩放比例
        scale = OUTPUT_SIZE / max_side

        # 缩放后坐标
        new_left *= scale
        new_right *= scale
        new_top *= scale
        new_bottom *= scale

        # 转为 YOLO 格式（归一化）
        new_cx = (new_left + new_right) / 2 / OUTPUT_SIZE
        new_cy = (new_top + new_bottom) / 2 / OUTPUT_SIZE
        new_bw = (new_right - new_left) / OUTPUT_SIZE
        new_bh = (new_bottom - new_top) / OUTPUT_SIZE

        # 限制在 [0,1]
        new_cx = np.clip(new_cx, 0, 1)
        new_cy = np.clip(new_cy, 0, 1)
        new_bw = np.clip(new_bw, 0, 1)
        new_bh = np.clip(new_bh, 0, 1)

        # 保存新图像和标签
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_img_name = f"{base_name}_obj{idx}.jpg"
        out_label_name = f"{base_name}_obj{idx}.txt"

        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, out_img_name), resized)

        with open(os.path.join(OUTPUT_LABEL_DIR, out_label_name), 'w') as lf:
            lf.write(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}\n")

        print(f"已处理: {out_img_name}")

# 批量处理
for label_file in os.listdir(INPUT_LABEL_DIR):
    if not label_file.endswith('.txt'):
        continue
    img_file = label_file.replace('.txt', '.jpg')
    img_path = os.path.join(INPUT_IMAGE_DIR, img_file)
    label_path = os.path.join(INPUT_LABEL_DIR, label_file)

    if os.path.exists(img_path):
        crop_and_resize_with_label(img_path, label_path)
    else:
        print(f"图像不存在: {img_path}")

print("处理完成！")