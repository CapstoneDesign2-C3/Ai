import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# 1) 엔진 로드
with open("yolov8n_fp16.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 2) 컨텍스트 생성
context = engine.create_execution_context()
context.set_binding_shape(0, (1, 3, 320, 320))  # dynamic 배치 크기

# 3) 메모리 할당
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
for binding in engine:
    size = trt.volume(context.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # GPU 메모리 할당
    buf = cuda.mem_alloc(size * dtype().nbytes)
    bindings.append(int(buf))
    if engine.binding_is_input(binding):
        inputs.append(buf)
    else:
        outputs.append(buf)

# 4) 전처리된 이미지(NumPy) 업로드
img = np.random.random((1,3,320,320)).astype(np.float32)
cuda.memcpy_htod_async(inputs[0], img, stream)

# 5) 추론 실행
context.execute_async_v2(bindings, stream.handle)

# 6) 결과 다운로드
output_shape = context.get_binding_shape(1)
output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output, outputs[0], stream)
stream.synchronize()

# 7) Post-processing (NMS, decode 등)
# ... (YOLO 디코더 적용)
