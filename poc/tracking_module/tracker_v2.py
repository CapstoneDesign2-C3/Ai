import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import time
from contextlib import contextmanager
from deep_sort_realtime.deepsort_tracker import DeepSort

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class OptimizedDetectorTracker:
    def __init__(self, engine_path: str, input_size=(640, 640), conf_thresh=0.25, 
                 max_age=5, max_iou_distance=0.4, nn_budget=50):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.engine = None
        self.context = None
        self.stream = None

        print(f"🚀 Initializing OptimizedDetectorTracker with engine: {engine_path}")

        # 컨텍스트 관리자로 초기화
        with self._cuda_context():
            self._load_engine(engine_path)
            self._setup_memory_buffers()
            self._setup_preprocessing_buffers()

        # 트래커 초기화
        self.tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, nn_budget=nn_budget)

        print("🛰️ OptimizedDetectorTracker fully initialized.")

    @contextmanager
    def _cuda_context(self):
        """CUDA 컨텍스트 자동 관리 (안정성 향상)"""
        try:
            # CUDA 컨텍스트 상태 확인
            current_context = cuda.Context.get_current()
            
            # 스트림 생성 (없는 경우만)
            if not hasattr(self, 'stream') or self.stream is None:
                self.stream = cuda.Stream()
            
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # CUDA 컨텍스트 동기화
            cuda.Context.synchronize()
            
            yield
            
        except Exception as e:
            print(f"⚠️  CUDA context error: {e}")
            # 컨텍스트 복구 시도
            try:
                cuda.Context.synchronize()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            raise
        finally:
            # 최종 동기화
            try:
                cuda.Context.synchronize()
            except:
                pass

    def _load_engine(self, engine_path: str):
        """TensorRT 엔진 로드"""
        try:
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                raise RuntimeError(f"❌ Failed to deserialize engine: {engine_path}")
            
            self.context = self.engine.create_execution_context()
            
            print("✅ TensorRT Engine loaded successfully")
            print("📐 Tensor I/O info:")
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                mode = self.engine.get_tensor_mode(name)
                print(f"   - {name} | shape={shape}, dtype={dtype}, mode={mode}")
                
        except Exception as e:
            print(f"❌ Failed to load TensorRT engine: {e}")
            raise

    def get_engine_info(self):
        """TensorRT 엔진 정보 출력용 (디버깅용)"""
        return {
            'input_size': self.input_size,
            'conf_thresh': self.conf_thresh,
            'num_inputs': len(self.inputs) if hasattr(self, 'inputs') else 0,
            'num_outputs': len(self.outputs) if hasattr(self, 'outputs') else 0
        }

    def _setup_memory_buffers(self):
        """메모리 버퍼 설정 최적화 (cuTensor 오류 방지)"""
        self.bindings, self.inputs, self.outputs = [], [], []

        for name in self.engine:
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            
            # 메모리 정렬을 위해 크기 조정 (512바이트 정렬)
            aligned_size = ((size * dtype().itemsize + 511) // 512) * 512
            dev_mem = cuda.mem_alloc(aligned_size)
            
            self.bindings.append(int(dev_mem))
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append((name, dev_mem, size, dtype))
                # 페이지 락 메모리도 정렬하여 할당
                self.input_host_mem = cuda.pagelocked_empty(size, dtype)
            else:
                self.outputs.append((name, dev_mem, size, dtype))
                self.output_host_mem = cuda.pagelocked_empty(size, dtype)

    def _setup_preprocessing_buffers(self):
        """전처리용 버퍼 사전 할당"""
        h, w = self.input_size
        # 재사용 가능한 버퍼들
        self.resized_buffer = np.empty((h, w, 3), dtype=np.uint8)
        self.normalized_buffer = np.empty((h, w, 3), dtype=np.float32)
        self.transposed_buffer = np.empty((1, 3, h, w), dtype=np.float32)

    def preprocess_optimized(self, frame: np.ndarray):
        """최적화된 전처리"""
        h, w = self.input_size
        self.orig_h, self.orig_w = frame.shape[:2]
        
        # 기존 버퍼 재사용
        cv2.resize(frame, (w, h), dst=self.resized_buffer)
        cv2.cvtColor(self.resized_buffer, cv2.COLOR_BGR2RGB, dst=self.resized_buffer)
        
        # 정규화 (in-place 연산)
        np.divide(self.resized_buffer, 255.0, out=self.normalized_buffer)
        
        # 차원 변경
        self.transposed_buffer[0] = np.transpose(self.normalized_buffer, (2, 0, 1))
        
        return self.transposed_buffer

    def postprocess_vectorized(self, output: np.ndarray, frame_shape):
        """벡터화된 후처리 (HALF 데이터 타입 지원)"""
        h_img, w_img = frame_shape[:2]
        try:
            # HALF 타입에서 FLOAT32로 변환
            if output.dtype == np.float16:
                output = output.astype(np.float32)
            
            # 출력 형태 정규화 (YOLO11m 출력: [1, 300, 6])
            if len(output.shape) == 1:
                # 1D 배열인 경우 reshaping
                expected_elements = 300 * 6  # YOLO11m 출력 크기
                if output.size >= expected_elements:
                    data = output[:expected_elements].reshape(300, 6)
                else:
                    return []
            elif len(output.shape) == 3 and output.shape[0] == 1:
                # [1, 300, 6] 형태
                data = output[0]  # [300, 6]
            elif len(output.shape) == 2:
                # [300, 6] 형태
                data = output
            else:
                print(f"[Warning] Unexpected output shape: {output.shape}")
                return []

            if data.size == 0 or data.shape[1] < 6:
                return []

            # YOLO11m 출력 형태: [x1, y1, x2, y2, conf, class]
            boxes = data[:, :4]  # bbox 좌표
            confidences = data[:, 4]  # confidence scores
            class_ids = data[:, 5].astype(int)  # class IDs
            
            # Person 클래스 (ID: 0)만 필터링하고 confidence 임계값 적용
            keep = (confidences > self.conf_thresh) & (class_ids == 0)

            if not np.any(keep):
                return []

            # 필터링된 데이터
            filtered_boxes = boxes[keep]
            filtered_scores = confidences[keep]
            filtered_cls_ids = class_ids[keep]

            # 좌표 스케일링 (YOLO11m은 이미 절대 좌표)
            input_h, input_w = self.input_size
            scale_x = w_img / input_w
            scale_y = h_img / input_h

            # 좌표 변환 및 스케일링
            x1, y1, x2, y2 = filtered_boxes.T
            x1_scaled = np.clip((x1 * scale_x).astype(int), 0, w_img - 1)
            y1_scaled = np.clip((y1 * scale_y).astype(int), 0, h_img - 1)
            x2_scaled = np.clip((x2 * scale_x).astype(int), 0, w_img - 1)
            y2_scaled = np.clip((y2 * scale_y).astype(int), 0, h_img - 1)

            # 유효한 박스 필터링
            valid_mask = (x2_scaled > x1_scaled + 5) & (y2_scaled > y1_scaled + 5)
            
            # 결과 생성
            results = []
            valid_indices = np.where(valid_mask)[0]
            
            for i in valid_indices:
                results.append({
                    'bbox': [x1_scaled[i], y1_scaled[i], x2_scaled[i], y2_scaled[i]],
                    'score': float(filtered_scores[i]),
                    'class': int(filtered_cls_ids[i])
                })

            return results

        except Exception as e:
            print(f"[Error] postprocess_vectorized failed: {e}")
            print(f"[Debug] Output shape: {output.shape}, dtype: {output.dtype}")
            return []

    def detect_and_track(self, frame: np.ndarray):
        """최적화된 검출 및 추적 (cuTensor 오류 방지)"""
        try:
            with self._cuda_context():
                # 최적화된 전처리
                tensor = self.preprocess_optimized(frame)
                np.copyto(self.input_host_mem, tensor.ravel().astype(np.float16))  # HALF 타입으로 변환

                # 동기 방식으로 변경 (안정성 향상)
                in_name, in_ptr, _, _ = self.inputs[0]
                out_name, out_ptr, _, _ = self.outputs[0]

                # 동기 메모리 복사
                cuda.memcpy_htod(in_ptr, self.input_host_mem)
                
                # 텐서 주소 설정
                self.context.set_tensor_address(in_name, in_ptr)
                self.context.set_tensor_address(out_name, out_ptr)
                
                # CUDA 컨텍스트 동기화
                cuda.Context.synchronize()
                
                # 동기 실행 시도
                try:
                    success = self.context.execute_v2(self.bindings)
                    if not success:
                        print("[Warning] TensorRT execution failed, trying alternative method")
                        # 대안: 비동기 실행
                        self.context.execute_async_v3(self.stream.handle)
                        self.stream.synchronize()
                except Exception as exec_error:
                    print(f"[Warning] Primary execution failed: {exec_error}")
                    # 대안: 비동기 실행
                    self.context.execute_async_v3(self.stream.handle)
                    self.stream.synchronize()
                
                # 결과 복사
                cuda.memcpy_dtoh(self.output_host_mem, out_ptr)
                
                # 후처리 (HALF 타입 처리)
                output_data = self.output_host_mem.astype(np.float32)  # HALF -> FLOAT32 변환
                detections = self.postprocess_vectorized(output_data, frame.shape)

                # 추적 업데이트
                det_list = [([*d['bbox']], d['score'], d['class']) for d in detections]
                tracks = self.tracker.update_tracks(det_list, frame=frame)

                # 결과 생성
                results = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    l, t, r, b = track.to_ltrb()
                    results.append({
                        'local_id': getattr(track, 'track_id', None),
                        'bbox': [int(l), int(t), int(r), int(b)],
                        'score': getattr(track, 'det_conf', 0.5),
                        'class': getattr(track, 'det_class', 0),
                        'is_new': getattr(track, 'age', 0) == 1
                    })
                
                return results

        except Exception as e:
            print(f"[Error] detect_and_track failed: {e}")
            # CUDA 컨텍스트 재설정 시도
            try:
                cuda.Context.synchronize()
            except:
                pass
            return []

    def visualize_results(self, frame: np.ndarray, results: list):
        """
        결과를 프레임에 시각화 (향상된 버전)
        """
        annotated_frame = frame.copy()

        # 전체 정보 표시
        info_text = f"Objects: {len(results)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        for result in results:
            try:
                bbox = result.get('bbox', [])
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = bbox
                
                # 새 객체와 기존 객체 구분
                is_new = result.get('is_new', False)
                color = (0, 255, 255) if is_new else (0, 255, 0)  # 노란색: 새 객체, 초록색: 기존 객체
                thickness = 3 if is_new else 2
                
                # 바운딩 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨 텍스트 구성
                label_parts = ["Person"]
                
                # ID 추가
                local_id = result.get('local_id')
                if local_id is not None:
                    label_parts.append(f"ID:{local_id}")
                
                # 점수 추가
                score = result.get('score')
                if score is not None:
                    label_parts.append(f"{score:.2f}")
                
                # NEW 표시
                if is_new:
                    label_parts.append("NEW")
                
                label = " ".join(label_parts)
                
                # 라벨 배경 (더 넓게)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                bg_color = (0, 255, 255) if is_new else (0, 255, 0)
                
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - label_size[1] - 12), 
                             (x1 + label_size[0] + 10, y1), 
                             bg_color, -1)
                
                # 라벨 텍스트 (검은색으로 선명하게)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 중심점 표시 (선택적)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
                
            except Exception as viz_error:
                print(f"⚠️  Visualization error for result: {viz_error}")
                continue
        
        # 타임스탬프 추가
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp, 
                   (10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def benchmark_performance(self, frame: np.ndarray, iterations=100):
        """성능 벤치마크 (컨텍스트 관리 포함)"""
        print(f"🔥 Performance Benchmark ({iterations} iterations)")
        
        # 워밍업
        for _ in range(10):
            self.detect_and_track(frame)
        
        # 벤치마크
        start_time = time.perf_counter()
        for _ in range(iterations):
            results = self.detect_and_track(frame)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / iterations
        fps = 1.0 / avg_time
        
        print(f"⚡ Average inference time: {avg_time*1000:.2f} ms")
        print(f"🎯 Average FPS: {fps:.1f}")
        
        return avg_time, fps

    def profile_pipeline(self, frame: np.ndarray):
        """파이프라인 프로파일링 (컨텍스트 관리 포함)"""
        print("🔍 Pipeline Profiling")
        
        with self._cuda_context():
            # 전처리 측정
            start = time.perf_counter()
            tensor = self.preprocess_optimized(frame)
            preprocess_time = time.perf_counter() - start
            
            # 추론 측정
            start = time.perf_counter()
            np.copyto(self.input_host_mem, tensor.ravel())
            in_name, in_ptr, _, _ = self.inputs[0]
            out_name, out_ptr, _, _ = self.outputs[0]
            cuda.memcpy_htod_async(in_ptr, self.input_host_mem, self.stream)
            self.context.set_tensor_address(in_name, in_ptr)
            self.context.set_tensor_address(out_name, out_ptr)
            self.context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()
            inference_time = time.perf_counter() - start
            
            # 후처리 측정
            start = time.perf_counter()
            cuda.memcpy_dtoh(self.output_host_mem, out_ptr)
            detections = self.postprocess_vectorized(self.output_host_mem, frame.shape)
            postprocess_time = time.perf_counter() - start
        
        print(f"📊 Preprocessing: {preprocess_time*1000:.2f} ms")
        print(f"🚀 Inference: {inference_time*1000:.2f} ms") 
        print(f"⚙️  Postprocessing: {postprocess_time*1000:.2f} ms")
        print(f"🎯 Total: {(preprocess_time + inference_time + postprocess_time)*1000:.2f} ms")

    def cleanup_resources(self):
        """리소스 정리 메서드"""
        try:
            print("🧹 Cleaning up OptimizedDetectorTracker resources...")
            
            # GPU 메모리 해제
            if hasattr(self, 'inputs'):
                for _, dev_mem, _, _ in self.inputs:
                    if dev_mem:
                        dev_mem.free()
            
            if hasattr(self, 'outputs'):
                for _, dev_mem, _, _ in self.outputs:
                    if dev_mem:
                        dev_mem.free()
            
            # 스트림 정리
            if hasattr(self, 'stream') and self.stream:
                self.stream.synchronize()
            
            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("✅ OptimizedDetectorTracker cleanup completed")
            
        except Exception as e:
            print(f"[Warning] Cleanup error: {e}")

    def __del__(self):
        """메모리 정리"""
        self.cleanup_resources()

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.cleanup_resources()
        return False