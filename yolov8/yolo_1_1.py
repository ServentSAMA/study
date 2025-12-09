from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

