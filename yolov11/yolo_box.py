from ultralytics import YOLO
from PIL import Image

#     input_image = "datasets\\lock\\images\\train\\0acfad2c-2025-12-03_16-05-17.mp4_20251219_115848.163.jpg"
#     input_label = "datasets\\lock\\labels\\train\\0acfad2c-2025-12-03_16-05-17.mp4_20251219_115848.163.txt"
# Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("box.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# model.train(data="lock.yml", epochs=50)

im1 = Image.open("D:\\工作资料\\箱锁识别\\数据集\\2025-10-29_16-16-06.mp4_20251230_101650.699.jpg")

results = model.predict(source="D:\\RecordFile\\2025-11-18\\2025-11-19_15-22-11.mp4", show=True)

# success = model.export()
