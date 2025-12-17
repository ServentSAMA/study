"""
实例分割
"""
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)

metrics = model.val()

results = model("https://ultralytics.com/images/bus.jpg")

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)

model.export(format="onnx")
