import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import json
import threading
import queue
from collections import deque
from pathlib import Path

from kafka_util import consumers, producers
import os

class OptimizedSingleDetector: 
    def __init__(self, class_names_path=None, conf_threshold=0.25, iou_threshold=0.45,
                 app_batch_size=None, max_wait_time=0.05):
        """
        Args:
            app_batch_size: 애플리케이션 레벨 배치 크기 (None이면 엔진 배치 크기 사용)
            max_wait_time: 최대 대기 시간 (초)
        """
        # Kafka 설정
        self.frame_consumer = consumers.FrameConsumer()
        self.result_producer = producers.DetectedResultProducer()
        
        # 모델 설정
        self.engine_path = os.getenv('ENGINE_PATH')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_wait_time = max_wait_time
        
        # 클래스 이름 로드
        self.class_names = self._load_class_names(class_names_path)
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        
        # TensorRT 엔진 로드
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # 엔진의 실제 배치 크기 확인
        self.engine_batch_size, self.input_shape = self._get_engine_info()
        
        # 애플리케이션 배치 크기 결정
        if app_batch_size is None:
            self.app_batch_size = self.engine_batch_size
        else:
            self.app_batch_size = min(app_batch_size, self.engine_batch_size)
        
        # 버퍼 할당
        self._setup_buffers()
        
        # 프레임 큐
        self.frame_queue = queue.Queue(maxsize=self.app_batch_size * 4)
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'total_inferences': 0,
            'total_inference_time': 0,
            'lock': threading.Lock()
        }
        
        print(f"✅ 최적화된 Detector 초기화 완료")
        print(f"   - 엔진 배치 크기: {self.engine_batch_size}")
        print(f"   - 애플리케이션 배치 크기: {self.app_batch_size}")
        print(f"   - 입력 형태: {self.input_shape}")
        print(f"   - 최대 대기 시간: {max_wait_time*1000:.1f}ms")
    
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
    
    def _get_engine_info(self):
        """엔진의 실제 배치 크기와 입력 형태 확인"""
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 8.5+ API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    shape = self.context.get_tensor_shape(tensor_name)
                    batch_size = shape[0] if shape[0] > 0 else 1  # -1이면 동적, 양수면 고정
                    return batch_size, shape
        else:
            # TensorRT 7.x/8.x API
            for binding in self.engine:
                if self.engine.binding_is_input(binding):
                    shape = self.context.get_binding_shape(binding)
                    batch_size = shape[0] if shape[0] > 0 else 1
                    return batch_size, shape
        
        return 1, None  # 기본값
    
    def _setup_buffers(self):
        """GPU 버퍼 설정"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        if hasattr(self.engine, 'num_io_tensors'):
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    shape = self.input_shape
                    size = trt.volume(shape) if shape[0] > 0 else trt.volume(shape[1:]) * self.engine_batch_size
                    
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.bindings.append(int(device_mem))
                    self.inputs.append({'host': host_mem, 'device': device_mem, 
                                      'shape': shape, 'name': tensor_name})
                    self.input_name = tensor_name
                else:
                    shape = self.context.get_tensor_shape(tensor_name)
                    # 출력 크기 계산 (배치 크기 고려)
                    if shape[0] <= 0:  # 동적 배치
                        actual_shape = (self.engine_batch_size,) + shape[1:]
                        size = trt.volume(actual_shape)
                    else:
                        size = trt.volume(shape)
                        actual_shape = shape
                    
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.bindings.append(int(device_mem))
                    self.outputs.append({'host': host_mem, 'device': device_mem, 
                                       'shape': actual_shape, 'name': tensor_name})
                    self.output_name = tensor_name
        
        print(f"🔧 버퍼 설정 완료:")
        print(f"   - 입력 버퍼 크기: {self.inputs[0]['host'].nbytes / 1024 / 1024:.1f} MB")
        print(f"   - 출력 버퍼 크기: {self.outputs[0]['host'].nbytes / 1024 / 1024:.1f} MB")
    
    def preprocess_images(self, images):
        """이미지 전처리 (배치 또는 단일)"""
        if self.engine_batch_size == 1:
            # 엔진이 단일 배치만 지원하는 경우
            return self._preprocess_single_batch(images)
        else:
            # 엔진이 진짜 배치를 지원하는 경우
            return self._preprocess_true_batch(images)
    
    def _preprocess_single_batch(self, images):
        """단일 이미지씩 전처리 (엔진 배치 크기 = 1)"""
        preprocessed = []
        metadata = []
        
        _, input_h, input_w = self.input_shape[1], self.input_shape[2], self.input_shape[3]
        
        for image in images:
            img_h, img_w = image.shape[:2]
            scale = min(input_w / img_w, input_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
            pad_x = (input_w - new_w) // 2
            pad_y = (input_h - new_h) // 2
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            
            tensor = padded.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)  # [1, 3, H, W]
            
            preprocessed.append(tensor)
            metadata.append({'scale': scale, 'pad_x': pad_x, 'pad_y': pad_y})
        
        return preprocessed, metadata
    
    def _preprocess_true_batch(self, images):
        """진짜 배치 전처리 (엔진이 배치 지원)"""
        batch_tensors = []
        metadata = []
        
        _, input_h, input_w = self.input_shape[1], self.input_shape[2], self.input_shape[3]
        
        for image in images:
            img_h, img_w = image.shape[:2]
            scale = min(input_w / img_w, input_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
            pad_x = (input_w - new_w) // 2
            pad_y = (input_h - new_h) // 2
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            
            tensor = padded.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            
            batch_tensors.append(tensor)
            metadata.append({'scale': scale, 'pad_x': pad_x, 'pad_y': pad_y})
        
        # 진짜 배치로 결합
        batch_tensor = np.stack(batch_tensors, axis=0)  # [N, 3, H, W]
        
        # 엔진 배치 크기에 맞춰 패딩
        if len(batch_tensors) < self.engine_batch_size:
            padding_count = self.engine_batch_size - len(batch_tensors)
            padding = np.zeros((padding_count,) + batch_tensor.shape[1:], dtype=np.float32)
            batch_tensor = np.concatenate([batch_tensor, padding], axis=0)
        
        return batch_tensor, metadata
    
    def postprocess_results(self, outputs, metadata_list, actual_batch_size):
        """후처리 결과"""
        results = []
        
        if self.engine_batch_size == 1:
            # 단일 추론 결과들
            for i, (output, metadata) in enumerate(zip(outputs, metadata_list)):
                boxes, scores, class_ids = self._postprocess_single(output, metadata)
                results.append((boxes, scores, class_ids))
        else:
            # 진짜 배치 결과
            batch_output = outputs[0]  # [batch_size, ...]
            for i in range(actual_batch_size):
                single_output = batch_output[i]  # i번째 이미지 결과
                metadata = metadata_list[i]
                boxes, scores, class_ids = self._postprocess_single(single_output, metadata)
                results.append((boxes, scores, class_ids))
        
        return results
    
    def _postprocess_single(self, output, metadata):
        """단일 이미지 후처리"""
        if len(output.shape) == 3:
            output = output[0]
        if len(output.shape) == 2 and output.shape[0] < output.shape[1]:
            output = output.T
        
        x_centers = output[:, 0]
        y_centers = output[:, 1]
        widths = output[:, 2]
        heights = output[:, 3]
        class_confs = output[:, 4:]
        
        max_confs = np.max(class_confs, axis=1)
        class_ids = np.argmax(class_confs, axis=1)
        
        valid_mask = max_confs > self.conf_threshold
        if not np.any(valid_mask):
            return np.array([]), np.array([]), np.array([])
        
        # 좌표 변환
        scale = metadata['scale']
        pad_x = metadata['pad_x']
        pad_y = metadata['pad_y']
        
        x_centers = (x_centers[valid_mask] - pad_x) / scale
        y_centers = (y_centers[valid_mask] - pad_y) / scale
        widths = widths[valid_mask] / scale
        heights = heights[valid_mask] / scale
        max_confs = max_confs[valid_mask]
        class_ids = class_ids[valid_mask]
        
        x1s = x_centers - widths / 2
        y1s = y_centers - heights / 2
        boxes = np.column_stack([x1s, y1s, widths, heights])
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_confs.tolist(),
                                      self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0] if len(indices) > 0 else []
                if len(indices) > 0:
                    indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                    return boxes[indices], max_confs[indices], class_ids[indices]
        
        return np.array([]), np.array([]), np.array([])
    
    def infer_batch(self, images, camera_ids):
        """배치 추론 실행"""
        if not images:
            return []
        
        actual_batch_size = len(images)
        start_time = time.time()
        
        # 전처리
        preprocessed, metadata = self.preprocess_images(images)
        
        if self.engine_batch_size == 1:
            # 엔진이 단일 배치만 지원 - 순차 실행
            all_outputs = []
            for tensor in preprocessed:
                np.copyto(self.inputs[0]['host'], tensor.ravel())
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
                
                # 동적 형태 설정 (필요시)
                if hasattr(self.context, 'set_tensor_shape'):
                    self.context.set_tensor_shape(self.input_name, tensor.shape)
                
                # 추론 실행
                if hasattr(self.context, 'execute_async_v3'):
                    self.context.set_tensor_address(self.input_name, self.inputs[0]['device'])
                    self.context.set_tensor_address(self.output_name, self.outputs[0]['device'])
                    self.context.execute_async_v3(stream_handle=self.stream.handle)
                else:
                    self.context.execute_v2(bindings=self.bindings)
                
                # 결과 복사
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
                self.stream.synchronize()
                
                # 출력 저장
                output_data = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
                all_outputs.append(output_data)
        else:
            # 엔진이 진짜 배치 지원 - 한번에 실행
            batch_tensor = preprocessed
            np.copyto(self.inputs[0]['host'], batch_tensor.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # 동적 형태 설정
            if hasattr(self.context, 'set_tensor_shape'):
                self.context.set_tensor_shape(self.input_name, batch_tensor.shape)
            
            # 추론 실행
            if hasattr(self.context, 'execute_async_v3'):
                self.context.set_tensor_address(self.input_name, self.inputs[0]['device'])
                self.context.set_tensor_address(self.output_name, self.outputs[0]['device'])
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                self.context.execute_v2(bindings=self.bindings)
            
            # 결과 복사
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            
            all_outputs = [self.outputs[0]['host'].reshape(self.outputs[0]['shape'])]
        
        inference_time = time.time() - start_time
        
        # 후처리
        results = self.postprocess_results(all_outputs, metadata, actual_batch_size)
        
        # Kafka 결과 전송
        for i, (camera_id, (boxes, scores, class_ids)) in enumerate(zip(camera_ids, results)):
            detections = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, w, h = box.astype(int)
                detections.append({
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'confidence': float(score),
                    'class': int(class_id),
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                })
            
            payload = {
                'camera_id': camera_id,
                'timestamp': time.time(),
                'detections': detections,
                'inference_time_ms': inference_time * 1000,
                'batch_size': actual_batch_size,
                'engine_batch_size': self.engine_batch_size
            }
            
            payload_bytes = json.dumps(payload).encode('utf-8')
            self.result_producer.send_message(
                key=camera_id.encode('utf-8'),
                value=payload_bytes
            )
        
        # 통계 업데이트
        with self.stats['lock']:
            self.stats['total_frames'] += actual_batch_size
            self.stats['total_inferences'] += 1
            self.stats['total_inference_time'] += inference_time
        
        return results
    
    def batch_processor(self):
        """배치 처리 워커"""
        batch_buffer = []
        last_process_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                timeout = max(0.001, self.max_wait_time - (time.time() - last_process_time))
                
                try:
                    item = self.frame_queue.get(timeout=timeout)
                    if item is None:
                        break
                    batch_buffer.append(item)
                except queue.Empty:
                    pass
                
                # 처리 조건
                should_process = (
                    len(batch_buffer) >= self.app_batch_size or
                    (len(batch_buffer) > 0 and 
                     time.time() - last_process_time >= self.max_wait_time)
                )
                
                if should_process and batch_buffer:
                    camera_ids, images = zip(*batch_buffer)
                    self.infer_batch(list(images), list(camera_ids))
                    self.result_producer.flush()
                    
                    batch_buffer.clear()
                    last_process_time = time.time()
                    
            except Exception as e:
                print(f"❌ 배치 처리 오류: {e}")
                batch_buffer.clear()
    
    def run(self):
        """메인 실행"""
        print(f"🚀 최적화된 검출 시작")
        
        self.processing_thread = threading.Thread(target=self.batch_processor)
        self.processing_thread.start()
        
        try:
            for msg in self.frame_consumer.consumer:
                if self.stop_event.is_set():
                    break
                
                camera_id = msg.key.decode('utf-8')
                frame_bytes = msg.value
                
                try:
                    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    
                    if not self.frame_queue.full():
                        self.frame_queue.put((camera_id, frame), block=False)
                    else:
                        print(f"⚠️ 큐 포화 - {camera_id} 프레임 드롭")
                        
                except Exception as e:
                    print(f"❌ 프레임 처리 오류 {camera_id}: {e}")
                
                if self.stats['total_frames'] % 50 == 0:
                    self.print_stats()
                    
        except KeyboardInterrupt:
            print("\n🛑 종료")
        finally:
            self.stop_event.set()
            self.frame_queue.put(None)
            if self.processing_thread:
                self.processing_thread.join()
            self.result_producer.producer.flush()
            self.print_stats()
    
    def print_stats(self):
        """통계 출력"""
        with self.stats['lock']:
            if self.stats['total_inferences'] > 0:
                avg_inference_time = (self.stats['total_inference_time'] / 
                                    self.stats['total_inferences']) * 1000
                total_fps = self.stats['total_frames'] / self.stats['total_inference_time']
                
                print(f"📊 처리 통계:")
                print(f"   - 총 프레임: {self.stats['total_frames']}")
                print(f"   - 총 추론: {self.stats['total_inferences']}")
                print(f"   - 평균 추론 시간: {avg_inference_time:.1f}ms")
                print(f"   - 총 FPS: {total_fps:.1f}")
                print(f"   - 프레임/추론: {self.stats['total_frames'] / self.stats['total_inferences']:.1f}")


# 사용 예시
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('/home/hiperwall/Ai_modules/Ai/env/aws.env')
    
    detector = OptimizedSingleDetector(
        app_batch_size=4,       # 애플리케이션 레벨 배치 크기
        max_wait_time=0.05,     # 50ms 최대 대기
        conf_threshold=0.3,
        iou_threshold=0.45
    )
    
    detector.run()