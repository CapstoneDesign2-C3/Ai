from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.export(
    format="onnx",
    half=False,          # FP32 사용
    imgsz=640,
    batch=1,
    dynamic=False,
    nms=False,
    device=0
)