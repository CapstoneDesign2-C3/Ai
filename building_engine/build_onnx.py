from ultralytics import YOLO
# dynamic=True 로 설정
YOLO('yolo11m.pt').export(
  format='onnx',
  imgsz=640,
  dynamic=True,      # dynamic axes 활성화
  batch=1,
  verbose=True
)