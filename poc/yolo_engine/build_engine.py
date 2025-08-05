from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11m.pt")

# Export the model to TensorRT format
model.export(format="onnx",
             half=True,
             imgsz=640,
             batch=8,
             workspace=4,
             dynamic=True,
             nms=True,
             device=0)  # creates 'yolo11n.engine'