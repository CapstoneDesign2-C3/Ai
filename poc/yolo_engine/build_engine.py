from ultralytics import YOLO
model = YOLO("best.pt")

model.export(
    format="engine",   # TensorRT plan 직접 생성
    imgsz=640,
    dynamic=False,     # 우리 파이프라인이 고정 640 가정
    half=True,         # FP16
    nms=True,          # EfficientNMS 포함!
    simplify=True,
    device=0
)