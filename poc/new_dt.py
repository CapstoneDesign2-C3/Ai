import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import json
import os
import io
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image

from dotenv import load_dotenv
from deep_sort_realtime.deepsort_tracker import DeepSort

# 수정된 import - 개선된 producers 사용
from kafka_util.improved_kafka_producers import (
    create_detected_result_producer,
    create_track_result_producer
)

class DetectorAndTracker:
    """
    YOLOv11 TensorRT 기반 객체 검출 및 DeepSort 추적 클래스
    Kafka와 완전 연동 가능
    """
    
    # 상수 정의
    DEFAULT_PADDING_COLOR = 114
    NORMALIZATION_FACTOR = 255.0
    DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    DEFAULT_FONT_SCALE = 0.7
    DEFAULT_THICKNESS = 2
    
    # COCO 클래스 이름들
    COCO_CLASSES = [
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

    def __init__(self, 
                 camera_id: str,
                 engine_path: Optional[str] = None,
                 class_names_path: Optional[str] = None, 
                 conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45,
                 enable_kafka: bool = True,
                 enable_tracking: bool = True,
                 send_detection_results: bool = True,
                 send_tracking_results: bool = True):
        """
        초기화
        
        Args:
            camera_id: 카메라 ID (Kafka 메시지 키로 사용)
            engine_path: TensorRT 엔진 파일 경로
            class_names_path: 클래스 이름 파일 경로
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값
            enable_kafka: Kafka 사용 여부
            enable_tracking: 객체 추적 사용 여부
            send_detection_results: 검출 결과 Kafka 전송 여부
            send_tracking_results: 추적 결과 Kafka 전송 여부
        """
        # 기본 설정
        self.camera_id = str(camera_id)
        self.engine_path = engine_path or os.getenv('ENGINE_PATH')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_kafka = enable_kafka
        self.enable_tracking = enable_tracking
        self.send_detection_results = send_detection_results
        self.send_tracking_results = send_tracking_results
        
        # 유효성 검사
        self._validate_config()
        
        # 컴포넌트 초기화
        self._initialize_kafka()
        self._initialize_class_names(class_names_path)
        self._initialize_tensorrt()
        self._initialize_tracking()
        
        # 상태 변수
        self.local_id_set = set()
        self.frame_count = 0
        
        # 마지막 검출 결과 저장 (draw_detections용)
        self._last_boxes = np.array([])
        self._last_scores = np.array([])
        self._last_class_ids = np.array([])
        
        print(f"✅ DetectorAndTracker 초기화 완료 - Camera ID: {self.camera_id}")

    def _validate_config(self) -> None:
        """설정 유효성 검사"""
        if not self.camera_id:
            raise ValueError("camera_id는 필수입니다.")
            
        if not self.engine_path or not Path(self.engine_path).exists():
            raise ValueError(f"TensorRT 엔진 파일을 찾을 수 없습니다: {self.engine_path}")
        
        if not (0.0 <= self.conf_threshold <= 1.0):
            raise ValueError(f"conf_threshold는 0.0-1.0 사이여야 합니다: {self.conf_threshold}")
        
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold는 0.0-1.0 사이여야 합니다: {self.iou_threshold}")

    def _initialize_kafka(self) -> None:
        """Kafka 컴포넌트 초기화"""
        self.result_producer = None
        self.track_result_producer = None
        
        if self.enable_kafka:
            try:
                # 검출 결과 Producer (camera_id 불필요)
                if self.send_detection_results:
                    self.result_producer = create_detected_result_producer()
                    print("✅ DetectedResultProducer 초기화 완료")
                
                # 추적 결과 Producer (camera_id 필요)
                if self.send_tracking_results and self.enable_tracking:
                    self.track_result_producer = create_track_result_producer(self.camera_id)
                    print("✅ TrackResultProducer 초기화 완료")
                    
            except Exception as e:
                print(f"⚠️ Kafka 초기화 실패: {e}")
                self.enable_kafka = False

    def _initialize_class_names(self, class_names_path: Optional[str]) -> None:
        """클래스 이름 초기화"""
        self.class_names = self._load_class_names(class_names_path)
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

    def _initialize_tensorrt(self) -> None:
        """TensorRT 엔진 초기화"""
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.engine = self._load_engine()
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
            self._print_engine_info()
            
            # 🔥 실행 메서드 우선순위 설정 (안정성 우선)
            self.execution_method = self._determine_execution_method()
            
        except Exception as e:
            raise RuntimeError(f"TensorRT 초기화 실패: {e}")

    def _determine_execution_method(self) -> str:
        """최적의 실행 메서드 결정"""
        methods = []
        
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
        # execute_async_v3는 문제가 있으므로 제외
        
        print(f"   - 사용 가능한 실행 메서드: {', '.join(methods)}")
        
        # 안정성 우선 순서로 선택
        if "execute_async_v3" in methods:
            print("   - 선택된 실행 메서드: execute_v3 (동기, 안정)")
            return "execute_async_v3"
        elif "execute" in methods:
            print("   - 선택된 실행 메서드: execute (동기, 레거시)")
            return "execute"
        elif "execute_async_v2" in methods:
            print("   - 선택된 실행 메서드: execute_async_v2 (비동기)")
            return "execute_async_v2"
        elif "execute_async" in methods:
            print("   - 선택된 실행 메서드: execute_async (비동기, 레거시)")
            return "execute_async"
        else:
            raise RuntimeError("사용 가능한 TensorRT 실행 메서드가 없습니다.")

    def _initialize_tracking(self) -> None:
        """추적 시스템 초기화"""
        if self.enable_tracking:
            try:
                self.tracker = DeepSort(max_age=5)
                print("✅ 추적 시스템 초기화 완료")
            except Exception as e:
                print(f"⚠️ 추적 시스템 초기화 실패: {e}")
                self.enable_tracking = False

    def _load_class_names(self, class_names_path: Optional[str]) -> List[str]:
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
                print(f"⚠️ 클래스 파일 로드 실패: {e}")
        
        print("📋 기본 COCO 클래스 사용")
        return self.COCO_CLASSES.copy()

    def _load_engine(self) -> trt.ICudaEngine:
        """TensorRT 엔진 로드"""
        try:
            with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            print(f"✅ TensorRT 엔진 로드 완료: {self.engine_path}")
            return engine
        except Exception as e:
            raise RuntimeError(f"엔진 로드 실패: {e}")

    def detect_and_track(self, frame: np.ndarray, debug: bool = False) -> Tuple[List, Dict]:
        """
        객체 검출 및 추적 (Kafka 연동 포함)
        
        Args:
            frame: 입력 프레임
            debug: 디버그 모드
            
        Returns:
            tracks: 추적 결과 리스트
            timing_info: 타이밍 정보
        """
        self.frame_count += 1
        
        # 1. 추론 실행
        boxes, scores, class_ids, timing_info = self.infer(frame, debug)
        
        # 마지막 검출 결과 저장 (draw_detections용)
        self._last_boxes = boxes
        self._last_scores = scores
        self._last_class_ids = class_ids
        
        # 2. 검출 결과 Kafka 전송
        if self.enable_kafka and self.send_detection_results and self.result_producer:
            self._send_detection_results(boxes, scores, class_ids, timing_info, debug=debug)
        
        # 3. 추적이 비활성화되어 있으면 검출 결과만 반환
        if not self.enable_tracking or len(boxes) == 0:
            return [], timing_info
        
        # 4. DeepSort 형식으로 변환
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, w, h = box.astype(int)
            detections.append(([x1, y1, w, h], float(score), int(class_id)))
        
        # 5. 추적 업데이트 (embedder 없이)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # 6. 신규 확정 트랙 처리 및 Kafka 전송
        if self.enable_kafka and self.send_tracking_results and self.track_result_producer:
            self._process_new_tracks(tracks, frame, debug=debug)
        
        return tracks, timing_info

    def _send_detection_results(self, boxes: np.ndarray, scores: np.ndarray, 
                               class_ids: np.ndarray, timing_info: Dict, debug: bool = False) -> None:
        """검출 결과를 Kafka로 전송"""
        try:
            # 검출 결과를 JSON 형태로 구성
            detections = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, w, h = box.astype(int)
                class_name = (self.class_names[class_id] if class_id < len(self.class_names) 
                            else f"Class_{class_id}")
                
                detection = {
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "confidence": float(score),
                    "class_id": int(class_id),
                    "class_name": class_name
                }
                detections.append(detection)
            
            # 전체 메시지 구성
            message = {
                "camera_id": self.camera_id,
                "frame_count": self.frame_count,
                "timestamp": int(time.time() * 1000),
                "detections": detections,
                "timing_info": timing_info,
                "detection_count": len(detections)
            }
            
            # Kafka 전송
            result = self.result_producer.send_message(self.camera_id, message)
            
            if result.get('status_code') != 200:
                if debug:
                    print(f"⚠️ 검출 결과 전송 실패: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ 검출 결과 Kafka 전송 오류: {e}")

    def _process_new_tracks(self, tracks: List, frame: np.ndarray, debug: bool = False) -> None:
        """신규 트랙 처리 및 Kafka 전송"""
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            if track_id in self.local_id_set:
                continue
            
            try:
                # 바운딩 박스 추출
                l, t, r, b = track.to_ltrb()
                
                # 경계 확인
                h, w = frame.shape[:2]
                l = max(0, min(l, w-1))
                t = max(0, min(t, h-1))
                r = max(0, min(r, w-1))
                b = max(0, min(b, h-1))
                
                # 유효한 크기인지 확인
                if r - l < 10 or b - t < 10:
                    continue
                
                # 크롭 추출
                crop = frame[t:b, l:r]
                
                # 트랙의 추가 정보 추출
                bbox = [l, t, r-l, b-t]  # [x, y, w, h] 형식
                
                # detection 정보가 있다면 활용
                detection = track.get_det() if hasattr(track, 'get_det') else None
                confidence = detection[1] if detection else None
                class_id = detection[2] if detection else None
                class_name = (self.class_names[class_id] if class_id is not None and class_id < len(self.class_names) 
                            else None)
                
                # Kafka로 추적 결과 전송
                result = self.track_result_producer.send_message(
                    crop=crop,
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_name=class_name,
                    encoding='base64'  # 또는 'binary'
                )
                
                if result.get('status_code') == 200:
                    self.local_id_set.add(track_id)
                    if debug:
                        print(f"✅ 새로운 트랙 전송 완료: ID={track_id}, Class={class_name}")
                else:
                    if debug:
                        print(f"⚠️ 트랙 결과 전송 실패: {result.get('error')}")
                    
            except Exception as e:
                if debug:
                    print(f"❌ 트랙 {track_id} 처리 중 오류: {e}")

    def infer(self, image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """추론 실행"""
        try:
            # 전처리
            start_time = time.time()
            input_tensor, scale, pad_x, pad_y = self.preprocess(image)
            preprocess_time = time.time() - start_time
            
            # GPU 추론
            start_time = time.time()
            self._run_inference(input_tensor)
            inference_time = time.time() - start_time
            
            # 후처리
            start_time = time.time()
            output_data = [output['host'].reshape(output['shape']) for output in self.outputs]
            boxes, scores, class_ids = self.postprocess(output_data, scale, pad_x, pad_y, debug)
            postprocess_time = time.time() - start_time
            
            timing_info = {
                'preprocess': preprocess_time * 1000,  # ms 단위로 변환
                'inference': inference_time * 1000,
                'postprocess': postprocess_time * 1000,
                'total': (preprocess_time + inference_time + postprocess_time) * 1000
            }
            
            return boxes, scores, class_ids, timing_info
            
        except Exception as e:
            print(f"❌ 추론 실행 오류: {e}")
            return np.array([]), np.array([]), np.array([]), {}
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
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """이미지 전처리"""
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        img_h, img_w = image.shape[:2]
        
        # 스케일 계산
        scale = min(input_w / img_w, input_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        # 리사이즈 및 패딩
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_h, input_w, 3), self.DEFAULT_PADDING_COLOR, dtype=np.uint8)
        
        # 중앙 배치
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # 정규화 및 차원 변경
        input_tensor = padded.astype(np.float32) / self.NORMALIZATION_FACTOR
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, pad_x, pad_y

    def postprocess(self, outputs: List[np.ndarray], scale: float, 
                   pad_x: int, pad_y: int, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """후처리"""
        if debug:
            print(f"🔍 후처리 디버그: 출력 개수: {len(outputs)}")
        
        output = outputs[0]
        
        # 형태 정규화
        if len(output.shape) == 3:
            output = output[0]
            if output.shape[0] < output.shape[1]:
                output = output.T
        
        boxes, scores, class_ids = self._parse_detections(output, scale, pad_x, pad_y, debug)
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # NMS 적용
        boxes_array = np.array(boxes)
        scores_array = np.array(scores)
        class_ids_array = np.array(class_ids)
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0] if len(indices) > 0 else []
            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                return boxes_array[indices], scores_array[indices], class_ids_array[indices]
        
        return np.array([]), np.array([]), np.array([])

    def _parse_detections(self, output: np.ndarray, scale: float, 
                         pad_x: int, pad_y: int, debug: bool) -> Tuple[List, List, List]:
        """검출 결과 파싱"""
        boxes = []
        scores = []
        class_ids = []
        
        for i, detection in enumerate(output):
            x_center, y_center, width, height = detection[:4]
            class_confs = detection[4:]
            
            max_conf = np.max(class_confs)
            class_id = np.argmax(class_confs)
            
            if debug and i < 5:
                print(f"   - 검출 {i}: conf={max_conf:.3f}, class={class_id}")
            
            if max_conf > self.conf_threshold:
                # 좌표 변환
                x_center = (x_center - pad_x) / scale
                y_center = (y_center - pad_y) / scale
                width = width / scale
                height = height / scale
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                
                boxes.append([x1, y1, width, height])
                scores.append(float(max_conf))
                class_ids.append(int(class_id))
        
        return boxes, scores, class_ids

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
    

    def _run_inference(self, input_tensor: np.ndarray) -> None:
        """🔥 안정적인 GPU 추론 실행"""
        # GPU로 데이터 복사
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 선택된 실행 메서드에 따라 추론 실행
        try:
            if self.execution_method == "execute_v2":
                success = self.context.execute_v2(bindings=self.bindings)
                if not success:
                    raise RuntimeError("execute_v2 실패")
            
            elif self.execution_method == "execute":
                batch_size = 1
                success = self.context.execute(batch_size=batch_size, bindings=self.bindings)
                if not success:
                    raise RuntimeError("execute 실패")
            
            elif self.execution_method == "execute_async_v2":
                success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                if not success:
                    raise RuntimeError("execute_async_v2 실패")
            
            elif self.execution_method == "execute_async":
                batch_size = 1
                success = self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
                if not success:
                    raise RuntimeError("execute_async 실패")
            
            else:
                raise RuntimeError(f"알 수 없는 실행 메서드: {self.execution_method}")
        
        except Exception as e:
            print(f"❌ {self.execution_method} 실행 중 오류: {e}")
            # 폴백: execute_v2 시도
            try:
                print("   🔄 execute_v2로 폴백 시도...")
                success = self.context.execute_v2(bindings=self.bindings)
                if not success:
                    raise RuntimeError("폴백 execute_v2도 실패")
            except Exception as fallback_e:
                print(f"   ❌ 폴백도 실패: {fallback_e}")
                raise e  # 원본 오류를 다시 발생
        
        # 결과를 CPU로 복사
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        self.stream.synchronize()

    def _print_engine_info(self) -> None:
        """엔진 정보 출력"""
        print(f"🔧 TensorRT 엔진 정보:")
        print(f"   - TensorRT 버전: {trt.__version__}")
        print(f"   - 입력 크기: {self.input_shape}")
        print(f"   - 클래스 수: {len(self.class_names)}")
        print(f"   - 신뢰도 임계값: {self.conf_threshold}")
        print(f"   - IoU 임계값: {self.iou_threshold}")

    def extract_crop_from_frame(self, frame: np.ndarray, bbox: List[int]) -> str:
        """프레임에서 crop 추출하여 base64로 인코딩"""
        try:
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
            
            # PIL Image로 변환
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # base64 인코딩
            buffer = io.BytesIO()
            crop_pil.save(buffer, format='JPEG')
            crop_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return crop_b64
        except Exception as e:
            print(f"❌ Crop 추출 실패: {e}")
            return ""

    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, 
                       scores: np.ndarray, class_ids: np.ndarray, 
                       debug: bool = False) -> np.ndarray:
        """검출 결과 시각화"""
        if len(boxes) == 0:
            return image
        
        result_image = image.copy()
        img_h, img_w = image.shape[:2]
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            try:
                x1, y1, w, h = box.astype(int)
                x2, y2 = x1 + w, y1 + h
                
                # 경계 확인
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                # 최소 크기 확인
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                
                # 그리기
                self._draw_single_detection(result_image, x1, y1, x2, y2, score, class_id)
                
            except Exception as e:
                if debug:
                    print(f"⚠️ 박스 {i} 그리기 오류: {e}")
                continue
        
        return result_image

    def _draw_single_detection(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                              score: float, class_id: int) -> None:
        """단일 검출 결과 그리기"""
        class_name = (self.class_names[class_id] if class_id < len(self.class_names) 
                     else f"Class_{class_id}")
        color = tuple(map(int, self.colors[class_id % len(self.colors)]))
        
        # 바운딩 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # 라벨
        label = f"{class_name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.DEFAULT_FONT, self.DEFAULT_FONT_SCALE, self.DEFAULT_THICKNESS)
        
        # 배경
        bg_x1 = x1
        bg_y1 = max(0, y1 - text_h - baseline - 10)
        bg_x2 = min(image.shape[1], x1 + text_w + 10)
        bg_y2 = y1
        
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # 텍스트
        cv2.putText(image, label, (x1 + 5, y1 - 5), 
                   self.DEFAULT_FONT, self.DEFAULT_FONT_SCALE, 
                   (255, 255, 255), self.DEFAULT_THICKNESS)

    def get_statistics(self) -> Dict:
        """현재 상태 통계 반환"""
        return {
            "camera_id": self.camera_id,
            "frame_count": self.frame_count,
            "active_tracks": len(self.local_id_set),
            "kafka_enabled": self.enable_kafka,
            "tracking_enabled": self.enable_tracking,
            "detection_results_sending": self.send_detection_results,
            "tracking_results_sending": self.send_tracking_results,
            "execution_method": getattr(self, 'execution_method', 'unknown')
        }

    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # CUDA 스트림 동기화
            if hasattr(self, 'stream'):
                self.stream.synchronize()
            
            # Kafka Producer 종료
            if self.result_producer:
                self.result_producer.close()
                print("✅ DetectedResultProducer 종료 완료")
            
            if self.track_result_producer:
                self.track_result_producer.close()
                print("✅ TrackResultProducer 종료 완료")
            
            print("✅ DetectorAndTracker 리소스 정리 완료")
            
        except Exception as e:
            print(f"⚠️ 리소스 정리 중 오류: {e}")

    def __del__(self):
        """소멸자"""
        self.cleanup()


# 사용 예시
def main():
    """사용 예시"""
    try:
        # DetectorAndTracker 초기화
        detector = DetectorAndTracker(
            camera_id="camera_001",
            engine_path="/home/hiperwall/Ai_modules/Ai/poc/yolo_engine/yolo11m_fp16.engine",
            conf_threshold=0.25,
            iou_threshold=0.45,
            enable_kafka=True,
            enable_tracking=True,
            send_detection_results=True,
            send_tracking_results=True
        )
        
        # 더미 프레임으로 테스트
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 검출 및 추적 실행
        tracks, timing = detector.detect_and_track(test_frame, debug=True)
        
        print(f"📊 통계 정보: {detector.get_statistics()}")
        print(f"⏱️ 타이밍 정보: {timing}")
        
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
    finally:
        if 'detector' in locals():
            detector.cleanup()

if __name__ == "__main__":
    main()