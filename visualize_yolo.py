import cv2
import numpy as np
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class YoloTRT:
    def __init__(self, engine_path='poc/yolo_engine/yolo11m_fp16.engine', input_size=(640, 640)):
        self.input_size = input_size

        # TensorRT 엔진 로드
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # 입력 크기 설정
        h, w = self.input_size
        # 엔진에 따라 첫 번째 바인딩 이름을 사용
        try:
            self.context.set_input_shape('images', (1, 3, h, w))
        except Exception:
            name = self.engine.get_tensor_name(0)
            self.context.set_input_shape(name, (1, 3, h, w))

        # 엔진 정보 출력 (디버깅용)
        print(f"Engine inputs/outputs:")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            print(f"  {name}: {shape}, {dtype}, {mode}")

        # 버퍼 할당
        self.bindings, self.inputs, self.outputs = [], [], []
        self.stream = cuda.Stream()
        for name in self.engine:
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            dev_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(dev_mem))
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append((name, dev_mem, size, dtype))
            else:
                self.outputs.append((name, dev_mem, size, dtype))

    def preprocess(self, frame):
        h, w = self.input_size
        # 원본 이미지 크기 저장
        self.orig_h, self.orig_w = frame.shape[:2]
        
        # 비율 유지하면서 리사이즈
        img = cv2.resize(frame, (w, h)).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)

    def infer(self, frame):
        tensor = self.preprocess(frame)
        in_name, in_ptr, _, _ = self.inputs[0]
        out_name, out_ptr, out_size, out_dtype = self.outputs[0]

        cuda.memcpy_htod_async(in_ptr, tensor.ravel(), self.stream)
        self.context.set_tensor_address(in_name, in_ptr)
        self.context.set_tensor_address(out_name, out_ptr)
        self.context.execute_async_v3(self.stream.handle)

        host_out = np.empty(out_size, dtype=out_dtype)
        cuda.memcpy_dtoh_async(host_out, out_ptr, self.stream)
        self.stream.synchronize()
        
        # 출력 형태 디버깅
        print(f"Output shape: {host_out.shape}")
        print(f"Output size: {out_size}")
        
        return host_out

    def postprocess(self, output, frame_shape, conf_thresh=0.25):
        h_img, w_img = frame_shape[:2]
        
        # YOLO v8/v11 출력 형태에 맞게 조정
        try:
            # 일반적인 YOLO v8/v11 출력: (1, 84, 8400) 또는 (84, 8400)
            if len(output.shape) == 1:
                # 1차원인 경우 재구성 시도
                if output.size % 84 == 0:
                    num_boxes = output.size // 84
                    data = output.reshape(84, num_boxes).T
                else:
                    print(f"Unexpected output size: {output.size}")
                    return []
            elif len(output.shape) == 3:
                # (1, 84, N) 형태
                data = output[0].T  # (N, 84)
            elif len(output.shape) == 2:
                # (84, N) 형태
                data = output.T  # (N, 84)
            else:
                print(f"Unexpected output shape: {output.shape}")
                return []
                
            print(f"Processed data shape: {data.shape}")
            
            # YOLO v8/v11: [x, y, w, h, conf0, conf1, ...]
            boxes = data[:, :4]  # x, y, w, h (center format, normalized)
            scores = data[:, 4:]  # class confidences
            
            # 각 박스의 최대 클래스 점수와 클래스 ID
            cls_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)
            
            # 임계값 적용
            keep = max_scores > conf_thresh
            
            print(f"Total detections: {len(data)}, After threshold: {np.sum(keep)}")
            
            results = []
            for i in np.where(keep)[0]:
                cx, cy, bw, bh = boxes[i]
                score = max_scores[i]
                cls_id = cls_ids[i]
                
                print(f"Detection {i}: cx={cx:.3f}, cy={cy:.3f}, w={bw:.3f}, h={bh:.3f}, score={score:.3f}, cls={cls_id}")
                
                # YOLO v11 출력이 640x640 기준 픽셀 좌표인 경우
                # 입력 이미지 크기로 스케일링
                input_h, input_w = self.input_size
                scale_x = w_img / input_w
                scale_y = h_img / input_h
                
                # 중심점과 크기를 원본 이미지 크기로 변환
                cx_scaled = cx * scale_x
                cy_scaled = cy * scale_y
                bw_scaled = bw * scale_x
                bh_scaled = bh * scale_y
                
                x1 = int(cx_scaled - bw_scaled/2)
                y1 = int(cy_scaled - bh_scaled/2)
                x2 = int(cx_scaled + bw_scaled/2)
                y2 = int(cy_scaled + bh_scaled/2)
                
                print(f"  Scaled: cx={cx_scaled:.1f}, cy={cy_scaled:.1f}, w={bw_scaled:.1f}, h={bh_scaled:.1f}")
                print(f"  Converted: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                # 경계 체크
                x1 = max(0, min(x1, w_img-1))
                y1 = max(0, min(y1, h_img-1))
                x2 = max(0, min(x2, w_img-1))
                y2 = max(0, min(y2, h_img-1))
                
                # 박스 크기 검증 (최소 5x5 픽셀)
                if x2 > x1 + 5 and y2 > y1 + 5:
                    results.append((x1, y1, x2, y2, float(score), int(cls_id)))
                    print(f"  ✓ Valid box added")
                else:
                    print(f"  ✗ Box too small: w={x2-x1}, h={y2-y1}")
            
            print(f"Valid results: {len(results)}")
            return results
            
        except Exception as e:
            print(f"Postprocessing error: {e}")
            return []


def visualize_image(path, yolo):
    frame = cv2.imread(path)
    if frame is None:
        print(f"✖️ 이미지 로딩 실패: {path}")
        return
    
    print(f"Image shape: {frame.shape}")
    output = yolo.infer(frame)
    dets = yolo.postprocess(output, frame.shape)
    
    print(f"Found {len(dets)} detections")
    
    for x1, y1, x2, y2, score, cls in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls}:{score:.2f}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.imshow("YOLO Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_video(path, yolo):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"✖️ 비디오 로딩 실패: {path}")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        output = yolo.infer(frame)
        dets = yolo.postprocess(output, frame.shape)
        
        if frame_count % 30 == 0:  # 30프레임마다 출력
            print(f"Frame {frame_count}: {len(dets)} detections")
        
        for x1, y1, x2, y2, score, cls in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls}:{score:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO-TRT Sample Visualizer")
    parser.add_argument("input", help="Path to sample.jpg or sample.mp4")
    parser.add_argument("--engine", default="yolo11m_fp16.engine",
                        help="TensorRT engine path")
    args = parser.parse_args()

    # 엔진 경로 수정
    yolo = YoloTRT(engine_path=args.engine, input_size=(640, 640))
    
    infile = args.input.lower()
    if infile.endswith(('.jpg', '.jpeg', '.png')):
        visualize_image(args.input, yolo)
    elif infile.endswith(('.mp4', '.avi', '.mov')):
        visualize_video(args.input, yolo)
    else:
        print("지원되지 않는 파일 형식입니다. jpg/png 또는 mp4/avi/mov 만 가능합니다.")

if __name__ == '__main__':
    main()