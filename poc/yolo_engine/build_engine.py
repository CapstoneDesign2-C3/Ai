from ultralytics import YOLO
import onnx

model = YOLO("yolo11m.pt")  # load an official model
model.export(format="enigne",
             imgsz=(1920,1080),
             half=True
             ) 