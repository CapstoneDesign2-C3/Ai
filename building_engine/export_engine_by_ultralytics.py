from ultralytics import YOLO

model = YOLO('yolo11m.pt')

# 동적 크기 대신 고정 크기 사용
model.export(
    format='engine',
    imgsz=640,  # 고정 크기
    dynamic=False,  # 동적 크기 비활성화
    batch=1,  # 고정 배치 크기
    workspace=4,  # workspace 크기 조정 (GB)
    verbose=True
)