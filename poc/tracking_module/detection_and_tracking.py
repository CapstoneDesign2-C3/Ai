import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import json

import os
import json
import cv2
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from kafka_util import consumers, producers
from deep_sort_realtime.deepsort_tracker import DeepSort

class Detector_and_tracker(self):
    def __init__(self, class_names_path=None, conf_threshold=0.25, iou_threshold=0.45, cameraID=None):
        """
        TensorRT YOLOv11 엔진 초기화
        
        Args:
            class_names_path: 클래스 이름 파일 경로 (선택사항)
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값
        """
        # initialize kafka module
        self.frame_consumer = consumers.FrameConsumer()
        self.result_producer = producers.DetectedResultProducer()   
        self.track_result_producer = producers.TrackResultProducer()

        # initialize local id assignment
        self.local_id_set = set()

        # load engine
        self.engine_path = os.getenv('ENGINE_PATH')
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
        
        # initialize tracker 
        self.tracker = DeepSort(max_age=5)

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
        # 입출력 크기를 맞추면 preprocessing이 필요한가?
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
    
    def extract_crop_from_frame(self, frame: np.ndarray, bbox: list) -> str:
        """Extract person crop from frame and encode to base64"""
        try:
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
            
            # Convert to PIL Image
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # Encode to base64
            buffer = io.BytesIO()
            crop_pil.save(buffer, format='JPEG')
            crop_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return crop_b64
        except Exception as e:
            self.logger.error(f"Failed to extract crop: {e}")
            return ""
        
    
    def detect_and_track(self, frame, debug=False):
        # 0) Inference
        boxes, scores, class_ids, timing_info = self.infer(frame, debug)
        # timing_info 활용 예시
        if debug:
            print(f"Inference timing: {timing_info}")

        # 1) DeepSort 형식으로 변환
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, w, h = box.astype(int)
            detections.append(
                ([x1, y1, w, h], float(score), int(class_id))
            )

        # 2) Crop & Embedding (한 번만 수행)
        object_chips = [
            frame[y:y+h, x:x+w]
            for (x, y, w, h), _, _ in detections
        ]
        embeds = self.embedder(object_chips)

        # 3) Track 업데이트
        tracks = self.tracker.update_tracks(
            detections,
            embeds=embeds,
            frame=frame   # 시각화 용도
        )

        # 4) Confirmed된 신규 track만 ReID 요청
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            if track_id in self.local_id_set:
                continue

            # bbox 좌표 (Left, Top, Right, Bottom)
            l, t, r, b = track.to_ltrb()
            crop = frame[t:b, l:r]

            # ReID 요청 (track_id와 crop 이미지 전달)
            self.track_result_producer.send_message(crop)

            # 처리된 ID 기록
            self.local_id_set.add(track_id)

        
    
    # TODO: 일정 주기로 draw 하고 프레임 자체를 전송하도록  수정해야함.
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