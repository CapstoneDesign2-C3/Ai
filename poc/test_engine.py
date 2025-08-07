import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import json
from pathlib import Path

class TensorRTYOLOv11:
    def __init__(self, engine_path, class_names_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        TensorRT YOLOv11 엔진 초기화
        
        Args:
            engine_path: .engine 파일 경로
            class_names_path: 클래스 이름 파일 경로 (선택사항)
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값
        """
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 클래스 이름 로드
        self.class_names = self._load_class_names(class_names_path)
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        
        # TensorRT 엔진 로드
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # 입출력 바인딩 설정
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # 모델 정보 출력
        self._print_engine_info()
    
    def _load_class_names(self, class_names_path):
        """클래스 이름 로드"""
        if class_names_path and Path(class_names_path).exists():
            try:
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    if class_names_path.endswith('.json'):
                        data = json.load(f)
                        return data.get('names', list(data.values()))
                    else:
                        return [line.strip() for line in f.readlines() if line.strip()]
            except Exception as e:
                print(f"⚠️  클래스 파일 로드 실패: {e}")
        
        # COCO 80개 클래스 (YOLOv11 기본)
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def _load_engine(self):
        """TensorRT 엔진 로드"""
        try:
            with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            print(f"✅ TensorRT 엔진 로드 완료: {self.engine_path}")
            return engine
        except Exception as e:
            print(f"❌ 엔진 로드 실패: {e}")
            raise
    
    def _allocate_buffers(self):
        """GPU 메모리 할당"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        # TensorRT 버전에 따른 처리
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 8.5+ 새로운 API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                shape = self.context.get_tensor_shape(tensor_name)
                size = trt.volume(shape)
                
                # GPU 메모리 할당
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.input_shape = shape
                    self.input_name = tensor_name
                    inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                else:
                    self.output_name = tensor_name
                    outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': tensor_name})
        else:
            # TensorRT 7.x/8.x 이전 API
            for binding in self.engine:
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                shape = self.context.get_binding_shape(binding)
                size = trt.volume(shape)
                
                # GPU 메모리 할당
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.input_shape = shape
                    self.input_name = binding
                    inputs.append({'host': host_mem, 'device': device_mem, 'name': binding})
                else:
                    self.output_name = binding
                    outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': binding})
        
        return inputs, outputs, bindings, stream
    
    def _print_engine_info(self):
        """엔진 정보 출력"""
        print(f"🔧 TensorRT 엔진 정보:")
        print(f"   - TensorRT 버전: {trt.__version__}")
        print(f"   - 입력 크기: {self.input_shape}")
        print(f"   - 클래스 수: {len(self.class_names)}")
        print(f"   - 신뢰도 임계값: {self.conf_threshold}")
        print(f"   - IoU 임계값: {self.iou_threshold}")
        
        # 사용 가능한 메서드 확인
        methods = []
        if hasattr(self.context, 'execute_async_v3'):
            methods.append("execute_async_v3")
        if hasattr(self.context, 'execute_async_v2'):
            methods.append("execute_async_v2")
        if hasattr(self.context, 'execute_async'):
            methods.append("execute_async")
        if hasattr(self.context, 'execute_v2'):
            methods.append("execute_v2")
        if hasattr(self.context, 'execute'):
            methods.append("execute")
        
        print(f"   - 사용 가능한 실행 메서드: {', '.join(methods)}")
    
    def preprocess(self, image):
        """이미지 전처리"""
        # YOLOv11 입력 크기 (보통 640x640)
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        
        # 이미지 리사이즈 (비율 유지)
        img_h, img_w = image.shape[:2]
        scale = min(input_w / img_w, input_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        # 리사이즈 및 패딩
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        
        # 중앙 배치
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # 정규화 및 차원 변경 (HWC -> CHW)
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, pad_x, pad_y
    
    def postprocess(self, outputs, scale, pad_x, pad_y, debug=False):
        """후처리 - NMS 적용하여 최종 검출 결과 생성"""
        if debug:
            print(f"🔍 후처리 디버그:")
            print(f"   - 출력 개수: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"   - 출력 {i} 형태: {output.shape}")
        
        # YOLOv11 출력 형식 확인 및 처리
        output = outputs[0]
        
        # YOLOv11은 보통 (1, 84, 8400) 형식으로 출력됨
        if len(output.shape) == 3:
            output = output[0]  # 배치 차원 제거 -> (84, 8400)
            if output.shape[0] < output.shape[1]:  # (84, 8400) -> (8400, 84)로 전치
                output = output.T
        
        if debug:
            print(f"   - 처리된 출력 형태: {output.shape}")
            print(f"   - 신뢰도 임계값: {self.conf_threshold}")
        
        boxes = []
        scores = []
        class_ids = []
        
        # YOLOv11 출력 파싱
        for i, detection in enumerate(output):
            # YOLOv11 형식: [x_center, y_center, width, height, class0_conf, class1_conf, ...]
            x_center, y_center, width, height = detection[:4]
            class_confs = detection[4:]
            
            # 최대 클래스 신뢰도 찾기
            max_conf = np.max(class_confs)
            class_id = np.argmax(class_confs)
            
            if debug and i < 5:  # 처음 5개만 디버그 출력
                print(f"   - 검출 {i}: conf={max_conf:.3f}, class={class_id}, pos=({x_center:.1f},{y_center:.1f})")
            
            if max_conf > self.conf_threshold:
                # 좌표 변환 (모델 입력 크기 기준 -> 원본 이미지 기준)
                input_h, input_w = self.input_shape[2], self.input_shape[3]
                
                # 정규화된 좌표를 픽셀 좌표로 변환
                x_center = x_center
                y_center = y_center
                width = width
                height = height
                
                # 패딩 보정 및 스케일링
                x_center = (x_center - pad_x) / scale
                y_center = (y_center - pad_y) / scale
                width = width / scale
                height = height / scale
                
                # 중심점을 좌상단 좌표로 변환
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                
                boxes.append([x1, y1, width, height])
                scores.append(float(max_conf))
                class_ids.append(int(class_id))
        
        if debug:
            print(f"   - 임계값 통과 객체: {len(boxes)}개")
        
        if len(boxes) == 0:
            return [], [], []
        
        # NMS 적용
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                  self.conf_threshold, self.iou_threshold)
        
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0] if len(indices) > 0 else []
            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                
                if debug:
                    print(f"   - NMS 후 최종 객체: {len(indices)}개")
                
                return boxes[indices], scores[indices], class_ids[indices]
        
        if debug:
            print(f"   - NMS 후 최종 객체: 0개")
        
        return [], [], []
    
    def infer(self, image, debug=False):
        """추론 실행"""
        # 전처리
        start_time = time.time()
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)
        preprocess_time = time.time() - start_time
        
        if debug:
            print(f"🔍 추론 디버그:")
            print(f"   - 입력 이미지 크기: {image.shape}")
            print(f"   - 전처리된 텐서 크기: {input_tensor.shape}")
            print(f"   - 스케일: {scale:.3f}, 패딩: ({pad_x}, {pad_y})")
        
        # GPU로 데이터 복사
        start_time = time.time()
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # TensorRT 버전에 따른 추론 실행 (최적화된 버전)
        try:
            # execute_async_v3 우선 사용 (가장 빠름)
            if hasattr(self.context, 'execute_async_v3'):
                # 텐서 주소 설정
                self.context.set_tensor_address(self.input_name, self.inputs[0]['device'])
                for output in self.outputs:
                    self.context.set_tensor_address(output['name'], output['device'])
                # 비동기 실행
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                # 폴백: execute_v2 사용 (동기)
                self.context.execute_v2(bindings=self.bindings)
        except Exception as e:
            print(f"추론 실행 오류: {e}")
            # 최종 폴백
            self.context.execute_v2(bindings=self.bindings)
        
        # 결과를 CPU로 복사
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        self.stream.synchronize()
        inference_time = time.time() - start_time
        
        # 후처리
        start_time = time.time()
        output_data = [output['host'].reshape(output['shape']) for output in self.outputs]
        
        if debug:
            print(f"   - 추론 시간: {inference_time*1000:.1f}ms")
            for i, data in enumerate(output_data):
                print(f"   - 출력 {i} 통계: min={np.min(data):.3f}, max={np.max(data):.3f}, mean={np.mean(data):.3f}")
        
        boxes, scores, class_ids = self.postprocess(output_data, scale, pad_x, pad_y, debug)
        postprocess_time = time.time() - start_time
        
        return boxes, scores, class_ids, {
            'preprocess': preprocess_time,
            'inference': inference_time,
            'postprocess': postprocess_time,
            'total': preprocess_time + inference_time + postprocess_time
        }
    
    def draw_detections(self, image, boxes, scores, class_ids, debug=False):
        """검출 결과를 이미지에 그리기"""
        if debug:
            print(f"🎨 그리기 디버그:")
            print(f"   - 이미지 크기: {image.shape}")
            print(f"   - 박스 개수: {len(boxes)}")
        
        if len(boxes) == 0:
            if debug:
                print("   - 그릴 박스가 없음")
            return image
        
        result_image = image.copy()
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            try:
                x1, y1, w, h = box.astype(int)
                x2, y2 = x1 + w, y1 + h
                
                # 이미지 경계 내로 제한
                img_h, img_w = image.shape[:2]
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                if debug and i < 3:
                    print(f"   - 박스 {i}: ({x1},{y1})-({x2},{y2}), 클래스={class_id}, 점수={score:.3f}")
                
                # 박스가 너무 작으면 스킵
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                
                # 클래스 이름과 색상
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                color = self.colors[class_id % len(self.colors)]
                
                # 바운딩 박스 그리기 (더 두껍게)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                
                # 라벨 텍스트
                label = f"{class_name}: {score:.2f}"
                
                # 텍스트 크기 계산
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # 라벨 배경 (더 큰 패딩)
                bg_x1 = x1
                bg_y1 = max(0, y1 - text_h - baseline - 10)
                bg_x2 = min(img_w, x1 + text_w + 10)
                bg_y2 = y1
                
                cv2.rectangle(result_image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                
                # 라벨 텍스트 (흰색으로 더 선명하게)
                text_x = x1 + 5
                text_y = y1 - 5
                cv2.putText(result_image, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
            except Exception as e:
                if debug:
                    print(f"   - 박스 {i} 그리기 오류: {e}")
                continue
        
        return result_image

def test_image(engine_path, image_path, class_names_path=None, debug=False):
    """이미지 테스트"""
    print(f"🖼️  이미지 테스트: {image_path}")
    
    # 모델 로드
    detector = TensorRTYOLOv11(engine_path, class_names_path)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
        return
    
    print(f"📷 원본 이미지 크기: {image.shape}")
    
    # 디버그 모드 확인
    if not debug:
        debug_input = input("디버그 모드 사용? (y/N): ").strip().lower()
        debug = debug_input == 'y'
    
    # 추론 실행
    boxes, scores, class_ids, timing = detector.infer(image, debug=debug)
    
    # 결과 출력
    print(f"\n⏱️  처리 시간:")
    print(f"   - 전처리: {timing['preprocess']*1000:.1f}ms")
    print(f"   - 추론: {timing['inference']*1000:.1f}ms")
    print(f"   - 후처리: {timing['postprocess']*1000:.1f}ms")
    print(f"   - 총 시간: {timing['total']*1000:.1f}ms")
    print(f"🎯 검출된 객체: {len(boxes)}개")
    
    # 검출된 객체 정보 출력
    if len(boxes) > 0:
        print(f"\n📋 검출된 객체 목록:")
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            class_name = detector.class_names[class_id] if class_id < len(detector.class_names) else f"Class_{class_id}"
            x, y, w, h = box
            print(f"   {i+1}. {class_name}: {score:.3f} - 위치({x:.1f},{y:.1f}), 크기({w:.1f}x{h:.1f})")
    
    # 검출 결과 시각화
    result_image = detector.draw_detections(image, boxes, scores, class_ids, debug=debug)
    
    # 결과 표시
    cv2.imshow('TensorRT YOLOv11 Detection', result_image)
    print(f"\n👁️  결과 이미지를 표시합니다. 아무 키나 누르면 종료됩니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 결과 저장 옵션
    save_result = input("결과 이미지를 저장하시겠습니까? (y/N): ").strip().lower()
    if save_result == 'y':
        output_path = f"{Path(image_path).stem}_detected.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"💾 결과 저장됨: {output_path}")

def debug_engine_output(engine_path, image_path=None):
    """엔진 출력 형식 디버깅"""
    print("🔍 TensorRT 엔진 출력 디버깅")
    print("=" * 50)
    
    # 모델 로드
    detector = TensorRTYOLOv11(engine_path)
    
    # 테스트 이미지 준비
    if image_path and Path(image_path).exists():
        image = cv2.imread(image_path)
    else:
        print("📷 테스트용 더미 이미지 생성")
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print(f"🖼️  입력 이미지 크기: {image.shape}")
    
    # 전처리
    input_tensor, scale, pad_x, pad_y = detector.preprocess(image)
    print(f"🔧 전처리 결과:")
    print(f"   - 입력 텐서 크기: {input_tensor.shape}")
    print(f"   - 스케일 팩터: {scale}")
    print(f"   - 패딩: x={pad_x}, y={pad_y}")
    
    # GPU로 데이터 복사 및 추론
    np.copyto(detector.inputs[0]['host'], input_tensor.ravel())
    cuda.memcpy_htod_async(detector.inputs[0]['device'], detector.inputs[0]['host'], detector.stream)
    
    # 추론 실행
    if hasattr(detector.context, 'execute_async_v3'):
        detector.context.set_tensor_address(detector.input_name, detector.inputs[0]['device'])
        for output in detector.outputs:
            detector.context.set_tensor_address(output['name'], output['device'])
        detector.context.execute_async_v3(stream_handle=detector.stream.handle)
    else:
        detector.context.execute_v2(bindings=detector.bindings)
    
    # 결과 복사
    for output in detector.outputs:
        cuda.memcpy_dtoh_async(output['host'], output['device'], detector.stream)
    detector.stream.synchronize()
    
    # 출력 분석
    print(f"\n📊 모델 출력 분석:")
    for i, output in enumerate(detector.outputs):
        data = output['host'].reshape(output['shape'])
        print(f"   출력 {i}:")
        print(f"     - 형태: {data.shape}")
        print(f"     - 데이터 타입: {data.dtype}")
        print(f"     - 값 범위: {np.min(data):.3f} ~ {np.max(data):.3f}")
        print(f"     - 평균: {np.mean(data):.3f}")
        print(f"     - 표준편차: {np.std(data):.3f}")
        
        # 샘플 데이터 출력
        if len(data.shape) == 2:  # 2D 출력인 경우
            print(f"     - 샘플 (첫 5행, 첫 10열):")
            sample = data[:5, :10] if data.shape[1] >= 10 else data[:5, :]
            for row in sample:
                print(f"       {[f'{x:.3f}' for x in row]}")
        elif len(data.shape) == 3:  # 3D 출력인 경우
            print(f"     - 샘플 (첫 번째 배치, 첫 5행, 첫 10열):")
            sample = data[0, :5, :10] if data.shape[2] >= 10 else data[0, :5, :]
            for row in sample:
                print(f"       {[f'{x:.3f}' for x in row]}")
    
    print(f"\n🎯 신뢰도 임계값: {detector.conf_threshold}")
    print(f"🔄 IoU 임계값: {detector.iou_threshold}")
    
    # 다양한 임계값으로 테스트
    print(f"\n🧪 다양한 신뢰도 임계값 테스트:")
    thresholds = [0.1, 0.25, 0.5, 0.7, 0.9]
    
    for thresh in thresholds:
        original_thresh = detector.conf_threshold
        detector.conf_threshold = thresh
        
        output_data = [output['host'].reshape(output['shape']) for output in detector.outputs]
        boxes, scores, class_ids = detector.postprocess(output_data, scale, pad_x, pad_y, debug=False)
        
        print(f"   임계값 {thresh}: {len(boxes)}개 객체 검출")
        if len(boxes) > 0:
            max_score = np.max(scores)
            min_score = np.min(scores)
            print(f"     점수 범위: {min_score:.3f} ~ {max_score:.3f}")
        
        detector.conf_threshold = original_thresh

def test_webcam(engine_path, class_names_path=None):
    """웹캠 테스트"""
    print("📹 웹캠 테스트 시작 (ESC 키로 종료)")
    
    # 모델 로드
    detector = TensorRTYOLOv11(engine_path, class_names_path)
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return
    
    # 성능 측정 변수
    frame_count = 0
    total_time = 0
    fps_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 추론 실행
        boxes, scores, class_ids, timing = detector.infer(frame)
        
        # FPS 계산
        frame_count += 1
        total_time += timing['total']
        current_fps = 1.0 / timing['total'] if timing['total'] > 0 else 0
        fps_history.append(current_fps)
        
        # 최근 30프레임 평균 FPS
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        # 검출 결과 그리기
        result_frame = detector.draw_detections(frame, boxes, scores, class_ids)
        
        # 성능 정보 표시
        info_text = [
            f"FPS: {avg_fps:.1f}",
            f"Objects: {len(boxes)}",
            f"Inference: {timing['inference']*1000:.1f}ms",
            f"Total: {timing['total']*1000:.1f}ms"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('TensorRT YOLOv11 Webcam', result_frame)
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"📊 테스트 완료:")
    print(f"   - 총 프레임: {frame_count}")
    print(f"   - 평균 FPS: {total_time/frame_count if frame_count > 0 else 0:.1f}")

def test_video(engine_path, video_path, class_names_path=None, save_output=False):
    """비디오 테스트"""
    print(f"🎬 비디오 테스트: {video_path}")
    
    # 모델 로드
    detector = TensorRTYOLOv11(engine_path, class_names_path)
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📺 비디오 정보:")
    print(f"   - 해상도: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - 총 프레임: {total_frames}")
    print(f"   - 길이: {total_frames/fps:.1f}초")
    
    # 출력 비디오 설정 (선택사항)
    out = None
    if save_output:
        output_path = f"{Path(video_path).stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 출력 파일: {output_path}")
    
    # 성능 측정 변수
    frame_count = 0
    total_inference_time = 0
    total_processing_time = 0
    fps_history = []
    detection_counts = []
    
    print("\n🎥 비디오 처리 시작... (스페이스바: 일시정지, ESC: 종료)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # 추론 실행
            boxes, scores, class_ids, timing = detector.infer(frame)
            
            # 통계 업데이트
            frame_count += 1
            total_inference_time += timing['inference']
            current_fps = 1.0 / timing['total'] if timing['total'] > 0 else 0
            fps_history.append(current_fps)
            detection_counts.append(len(boxes))
            
            # 최근 30프레임 평균 FPS
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # 검출 결과 그리기
            result_frame = detector.draw_detections(frame, boxes, scores, class_ids)
            
            # 진행률 계산
            progress = (frame_count / total_frames) * 100
            
            # 정보 오버레이
            info_overlay = [
                f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)",
                f"FPS: {avg_fps:.1f}",
                f"Objects: {len(boxes)}",
                f"Inference: {timing['inference']*1000:.1f}ms",
                f"Total: {timing['total']*1000:.1f}ms"
            ]
            
            # 진행률 바 그리기
            bar_width = width - 40
            bar_height = 10
            bar_x, bar_y = 20, height - 50
            
            # 배경
            cv2.rectangle(result_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # 진행률
            progress_width = int(bar_width * progress / 100)
            cv2.rectangle(result_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # 텍스트 정보
            for i, text in enumerate(info_overlay):
                y_pos = 30 + i * 25
                # 배경 박스
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(result_frame, (10, y_pos - 20), (20 + text_size[0], y_pos + 5), (0, 0, 0), -1)
                # 텍스트
                cv2.putText(result_frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 화면 출력
            cv2.imshow('TensorRT YOLOv11 Video Analysis', result_frame)
            
            # 출력 파일 저장
            if out is not None:
                out.write(result_frame)
            
            total_processing_time += time.time() - frame_start_time
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⏹️  사용자가 중단했습니다.")
                break
            elif key == 32:  # Space
                print("⏸️  일시정지 (아무 키나 누르면 계속)")
                cv2.waitKey(0)
            
            # 진행률 출력 (10% 단위)
            if frame_count % (total_frames // 10 + 1) == 0:
                print(f"📊 진행률: {progress:.1f}% - FPS: {avg_fps:.1f} - 객체: {len(boxes)}개")
    
    except KeyboardInterrupt:
        print("\n⏹️  처리가 중단되었습니다.")
    
    finally:
        # 리소스 정리
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # 최종 통계
        if frame_count > 0:
            avg_inference_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            avg_processing_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
            avg_detections = sum(detection_counts) / len(detection_counts) if detection_counts else 0
            
            print(f"\n📈 비디오 분석 완료:")
            print(f"   - 처리된 프레임: {frame_count}/{total_frames}")
            print(f"   - 평균 추론 FPS: {avg_inference_fps:.1f}")
            print(f"   - 평균 전체 FPS: {avg_processing_fps:.1f}")
            print(f"   - 평균 검출 객체: {avg_detections:.1f}개")
            print(f"   - 총 처리 시간: {total_processing_time:.1f}초")
            
            if save_output:
                print(f"   - 출력 파일 저장됨: {output_path}")

def test_rtsp_stream(engine_path, rtsp_url, class_names_path=None):
    """RTSP 스트림 테스트"""
    print(f"📡 RTSP 스트림 테스트: {rtsp_url}")
    
    # 모델 로드
    detector = TensorRTYOLOv11(engine_path, class_names_path)
    
    # RTSP 스트림 열기
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화 (지연 감소)
    
    if not cap.isOpened():
        print(f"❌ RTSP 스트림을 열 수 없습니다: {rtsp_url}")
        return
    
    print("📺 RTSP 스트림 연결됨. ESC로 종료...")
    
    # 성능 측정
    frame_count = 0
    fps_history = []
    skip_frames = 0  # 프레임 드롭 카운트
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📡 스트림 연결 끊어짐. 재연결 시도...")
                time.sleep(1)
                continue
            
            frame_count += 1
            
            # 실시간 처리를 위한 프레임 스킵 (선택사항)
            if frame_count % 2 == 0:  # 매 2번째 프레임만 처리
                skip_frames += 1
                continue
            
            # 추론 실행
            boxes, scores, class_ids, timing = detector.infer(frame)
            
            # FPS 계산
            current_fps = 1.0 / timing['total'] if timing['total'] > 0 else 0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # 검출 결과 그리기
            result_frame = detector.draw_detections(frame, boxes, scores, class_ids)
            
            # 스트림 정보 표시
            stream_info = [
                f"RTSP Stream - LIVE",
                f"FPS: {avg_fps:.1f}",
                f"Objects: {len(boxes)}",
                f"Inference: {timing['inference']*1000:.1f}ms",
                f"Skipped: {skip_frames}"
            ]
            
            for i, text in enumerate(stream_info):
                cv2.putText(result_frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('TensorRT YOLOv11 RTSP Stream', result_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  스트림 처리가 중단되었습니다.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 처리된 프레임: {frame_count - skip_frames}, 스킵된 프레임: {skip_frames}")

def batch_video_analysis(engine_path, video_dir, class_names_path=None, save_results=True):
    """배치 비디오 분석"""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {video_dir}")
        return
    
    # 지원하는 비디오 확장자
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_dir}")
        return
    
    print(f"🎬 배치 비디오 분석 시작: {len(video_files)}개 파일")
    
    # 모델 로드
    detector = TensorRTYOLOv11(engine_path, class_names_path)
    
    results_summary = []
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n📹 [{i}/{len(video_files)}] 처리 중: {video_file.name}")
        
        try:
            # 각 비디오에 대해 간단한 분석
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"❌ 파일을 열 수 없습니다: {video_file}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 샘플 프레임들만 분석 (매 30프레임마다)
            sample_frames = list(range(0, total_frames, 30))
            detections_per_frame = []
            
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                boxes, scores, class_ids, timing = detector.infer(frame)
                detections_per_frame.append(len(boxes))
            
            cap.release()
            
            # 결과 요약
            avg_detections = sum(detections_per_frame) / len(detections_per_frame) if detections_per_frame else 0
            max_detections = max(detections_per_frame) if detections_per_frame else 0
            
            result = {
                'filename': video_file.name,
                'total_frames': total_frames,
                'fps': fps,
                'duration': total_frames / fps if fps > 0 else 0,
                'sample_frames': len(sample_frames),
                'avg_objects': avg_detections,
                'max_objects': max_detections
            }
            
            results_summary.append(result)
            
            print(f"   - 총 프레임: {total_frames}")
            print(f"   - 평균 객체: {avg_detections:.1f}개")
            print(f"   - 최대 객체: {max_detections}개")
            
        except Exception as e:
            print(f"❌ 처리 중 오류 발생: {e}")
            continue
    
    # 결과 저장
    if save_results and results_summary:
        results_file = video_dir / 'analysis_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 분석 결과 저장됨: {results_file}")
    
    # 전체 요약
    if results_summary:
        total_duration = sum(r['duration'] for r in results_summary)
        avg_objects_overall = sum(r['avg_objects'] for r in results_summary) / len(results_summary)
        
        print(f"\n📊 배치 분석 완료:")
        print(f"   - 처리된 파일: {len(results_summary)}개")
        print(f"   - 총 영상 길이: {total_duration/60:.1f}분")
        print(f"   - 전체 평균 객체: {avg_objects_overall:.1f}개")

def test_benchmark(engine_path, iterations=100):
    """벤치마크 테스트"""
    print(f"⚡ 벤치마크 테스트 ({iterations}회 반복)")
    
    # 모델 로드
    detector = TensorRTYOLOv11(engine_path)
    
    # 더미 이미지 생성
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 워밍업 (GPU 초기화)
    print("🔥 워밍업 중...")
    for _ in range(10):
        detector.infer(dummy_image)
    
    # 벤치마크 실행
    print("📊 벤치마크 실행 중...")
    times = []
    
    for i in range(iterations):
        _, _, _, timing = detector.infer(dummy_image)
        times.append(timing['inference'] * 1000)  # ms 단위
        
        if (i + 1) % 20 == 0:
            print(f"   진행률: {i + 1}/{iterations}")
    
    # 결과 분석
    times = np.array(times)
    print(f"\n📈 벤치마크 결과:")
    print(f"   - 평균 추론 시간: {np.mean(times):.2f}ms")
    print(f"   - 최소 추론 시간: {np.min(times):.2f}ms")
    print(f"   - 최대 추론 시간: {np.max(times):.2f}ms")
    print(f"   - 표준편차: {np.std(times):.2f}ms")
    print(f"   - 평균 FPS: {1000/np.mean(times):.1f}")

def main():
    """메인 함수"""
    print("🚀 TensorRT YOLOv11 엔진 테스트")
    print("=" * 50)
    
    # 엔진 파일 경로 입력
    engine_path = input("엔진 파일 경로를 입력하세요 (.engine): ").strip()
    if not engine_path:
        engine_path = "yolov11.engine"
    
    if not Path(engine_path).exists():
        print(f"❌ 엔진 파일을 찾을 수 없습니다: {engine_path}")
        return
    
    # 클래스 이름 파일 (선택사항)
    class_names_path = input("클래스 이름 파일 경로 (선택사항, Enter로 기본값): ").strip()
    if class_names_path and not Path(class_names_path).exists():
        print(f"⚠️  클래스 파일을 찾을 수 없어 기본값을 사용합니다: {class_names_path}")
        class_names_path = None
    
    # 테스트 옵션
    print("\n테스트 옵션:")
    print("1. 이미지 테스트")
    print("2. 웹캠 테스트")
    print("3. 비디오 테스트")
    print("4. RTSP 스트림 테스트")
    print("5. 배치 비디오 분석")
    print("6. 벤치마크 테스트")
    
    choice = input("선택 (1-6): ").strip()
    
    try:
        if choice == '1':
            image_path = input("이미지 경로: ").strip()
            test_image(engine_path, image_path, class_names_path)
            
        elif choice == '2':
            test_webcam(engine_path, class_names_path)
            
        elif choice == '3':
            video_path = input("비디오 파일 경로: ").strip()
            save_output = input("결과 비디오 저장? (y/N): ").strip().lower() == 'y'
            test_video(engine_path, video_path, class_names_path, save_output)
            
        elif choice == '4':
            rtsp_url = input("RTSP URL (예: rtsp://192.168.1.100:554/stream): ").strip()
            test_rtsp_stream(engine_path, rtsp_url, class_names_path)
            
        elif choice == '5':
            video_dir = input("비디오 폴더 경로: ").strip()
            save_results = input("분석 결과 JSON 저장? (Y/n): ").strip().lower() != 'n'
            batch_video_analysis(engine_path, video_dir, class_names_path, save_results)
            
        elif choice == '6':
            iterations = input("반복 횟수 (기본: 100): ").strip()
            iterations = int(iterations) if iterations.isdigit() else 100
            test_benchmark(engine_path, iterations)
            
        else:
            print("잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        print("\n⏹️  테스트가 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()