import torch
from ultralytics import YOLO

# 1) YOLO 모델 로드
model = YOLO("yolo11m.pt")
model.model.cuda().eval()

# 2) 더미 입력 생성 (batch=1, C=3, H=640, W=640)
dummy = torch.zeros(1, 3, 640, 640, device="cuda")

# 3) ONNX로 내보내기 (opset=12 권장, dynamic_axes=None → static shape)
torch.onnx.export(
    model.model, dummy, 
    "yolo11m.onnx",
    opset_version=12,
    input_names=["images"],
    output_names=["output0"],
    dynamic_axes=None,
    do_constant_folding=True
)
print("✅ ONNX export completed.")
