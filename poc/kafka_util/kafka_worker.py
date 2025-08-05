import os
import cv2
import numpy as np
import torch
import gc
import time
import json
import base64
import threading
from contextlib import contextmanager
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

# 프로젝트 경로 추가
import sys
sys.path.append('/home/hiperwall/Ai_modules/Ai')

from poc.tracking_module.tracker_v2 import OptimizedDetectorTracker

class KafkaWorker:
    """
    개선된 Kafka Worker:
    - 자동 컨텍스트 관리
    - 향상된 에러 처리
    - 리소스 자동 정리
    - 성능 모니터링
    """
    
    def __init__(self, engine_path: str):
        print(f"[*] Initializing Enhanced KafkaWorker with engine: {engine_path}")
        
        # 환경 변수 로드
        self._load_environment()
        
        # Kafka 컴포넌트 초기화
        self._init_kafka_components()
        
        # DetectorTracker 초기화 (컨텍스트 관리 포함)
        self._init_detector_tracker(engine_path)
        
        # 성능 모니터링 변수
        self.stats = {
            'frames_processed': 0,
            'detections_sent': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        print("[*] Enhanced KafkaWorker initialization completed")

    def _load_environment(self):
        """환경 변수 로드"""
        try:
            load_dotenv(dotenv_path="/home/hiperwall/Ai_modules/Ai/env/aws.env")
            
            self.broker = os.getenv('BROKER')
            self.in_topic = os.getenv('FRAME_TOPIC', 'camera-frames')
            self.out_topic = os.getenv('OUTPUT_TOPIC', 'detection-results')
            self.reid_topic = os.getenv('REID_REQUEST_TOPIC', 'reid-requests')
            
            if not self.broker:
                raise ValueError("BROKER environment variable not set")
            
            print(f"[*] Environment loaded - Broker: {self.broker}")
            print(f"[*] Topics - Input: {self.in_topic}, Output: {self.out_topic}, ReID: {self.reid_topic}")
            
        except Exception as e:
            print(f"[Error] Failed to load environment: {e}")
            raise

    def _init_kafka_components(self):
        """Kafka 컴포넌트 초기화"""
        try:
            # Consumer 초기화 (향상된 설정)
            self.consumer = KafkaConsumer(
                self.in_topic,
                bootstrap_servers=[self.broker],
                key_deserializer=lambda b: b.decode('utf-8') if b else None,
                value_deserializer=lambda v: v,
                auto_offset_reset='latest',  # 최신 메시지부터 시작
                enable_auto_commit=True,
                group_id='enhanced-det-track',
                consumer_timeout_ms=5000,  # 5초 타임아웃
                max_poll_records=10,  # 배치 처리를 위해 증가
                fetch_min_bytes=1024,  # 최소 페치 크기
                fetch_max_wait_ms=500  # 최대 대기 시간
            )
            
            # Producers 초기화 (향상된 설정)
            producer_config = {
                'bootstrap_servers': [self.broker],
                'batch_size': 32768,  # 배치 크기 증가
                'linger_ms': 5,  # 응답성 향상
                'compression_type': 'lz4',  # 압축 추가
                'acks': 1,  # 응답성과 신뢰성 균형
                'retries': 3,
                'retry_backoff_ms': 100
            }
            
            self.producer = KafkaProducer(
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                **producer_config
            )
            
            self.reid_producer = KafkaProducer(
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                **producer_config
            )
            
            print("[*] Kafka components initialized successfully")
            
        except Exception as e:
            print(f"[Error] Failed to initialize Kafka components: {e}")
            raise

    def _init_detector_tracker(self, engine_path: str):
        """DetectorTracker 초기화"""
        try:
            # 경로 검증
            if not os.path.isabs(engine_path):
                engine_path = os.path.abspath(engine_path)
            
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
            
            print(f"[*] Loading TensorRT engine: {engine_path}")
            
            # CUDA 메모리 정리
            self._cleanup_gpu_memory()
            
            # DetectorTracker 초기화 (컨텍스트 관리 포함)
            self.detector_tracker = OptimizedDetectorTracker(engine_path=engine_path)
            
            print("[*] DetectorTracker initialized successfully")
            print(f"[*] Engine info: {self.detector_tracker.get_engine_info()}")
            
        except Exception as e:
            print(f"[Error] Failed to initialize DetectorTracker: {e}")
            raise

    @contextmanager
    def _performance_monitor(self, operation_name: str):
        """성능 모니터링 컨텍스트 매니저"""
        start_time = time.perf_counter()
        try:
            yield
        except Exception as e:
            self.stats['errors'] += 1
            print(f"[Error] {operation_name} failed: {e}")
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            if elapsed > 0.1:  # 100ms 이상 걸린 작업만 로그
                print(f"[Perf] {operation_name}: {elapsed*1000:.1f}ms")

    def _cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

    def _process_frame(self, camera_id: str, frame_data: bytes, timestamp: float):
        """프레임 처리 (컨텍스트 관리 포함)"""
        try:
            with self._performance_monitor("Frame Processing"):
                # 프레임 디코딩
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR
                )
                
                if frame is None:
                    print(f"[Warning] Failed to decode frame from camera {camera_id}")
                    return None
                
                # Detection & Tracking 수행
                with self._performance_monitor("Detection & Tracking"):
                    tracks = self.detector_tracker.detect_and_track(frame)
                
                # 새로운 객체 처리
                new_objects_count = 0
                with self._performance_monitor("ReID Processing"):
                    for track in tracks:
                        if track.get('is_new', False):
                            success = self._send_reid_request(camera_id, track, frame, timestamp)
                            if success:
                                new_objects_count += 1
                
                # 결과 발행
                result = {
                    'camera_id': camera_id,
                    'timestamp': timestamp,
                    'tracks': tracks,
                    'frame_shape': frame.shape
                }
                
                with self._performance_monitor("Result Publishing"):
                    self.producer.send(self.out_topic, result)
                
                self.stats['frames_processed'] += 1
                self.stats['detections_sent'] += len(tracks)
                
                return {
                    'tracks_count': len(tracks),
                    'new_objects': new_objects_count
                }
                
        except Exception as e:
            print(f"[Error] Frame processing failed: {e}")
            self.stats['errors'] += 1
            return None

    def _send_reid_request(self, camera_id: str, track: dict, frame: np.ndarray, timestamp: float):
        """ReID 요청 전송"""
        try:
            bbox = track.get('bbox', [])
            if len(bbox) != 4:
                return False
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 경계 체크
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            # Crop 추출
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return False
            
            # JPEG 인코딩
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
            success, buffer = cv2.imencode('.jpg', crop, encode_param)
            
            if not success:
                return False
            
            # ReID 요청 생성
            reid_request = {
                'camera_id': camera_id,
                'local_id': track['local_id'],
                'timestamp': timestamp,
                'bbox': bbox,
                'score': track.get('score', 0.0),
                'crop_jpg': base64.b64encode(buffer.tobytes()).decode('ascii')
            }
            
            # 전송
            self.reid_producer.send(self.reid_topic, reid_request)
            return True
            
        except Exception as e:
            print(f"[Error] ReID request failed: {e}")
            return False

    def _print_stats(self):
        """성능 통계 출력"""
        elapsed = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        print(f"[Stats] Processed: {self.stats['frames_processed']} frames, "
              f"FPS: {fps:.1f}, "
              f"Detections: {self.stats['detections_sent']}, "
              f"Errors: {self.stats['errors']}")
        
        # GPU 메모리 상태
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[GPU] Memory - Allocated: {memory_allocated:.1f}MB, "
                  f"Reserved: {memory_reserved:.1f}MB")

    def run(self):
        """메인 실행 루프 (개선된 버전)"""
        print(f"[*] Enhanced KafkaWorker started, listening on '{self.in_topic}'...")
        
        retry_count = 0
        max_retries = 3
        stats_interval = 100  # 100프레임마다 통계 출력
        cleanup_interval = 500  # 500프레임마다 GPU 메모리 정리
        
        try:
            with self.detector_tracker:  # 컨텍스트 관리자 사용
                
                while retry_count < max_retries:
                    try:
                        # 토픽 존재 확인
                        partitions = self.consumer.partitions_for_topic(self.in_topic)
                        if partitions is None:
                            print(f"[Warning] Topic '{self.in_topic}' not found, waiting...")
                            time.sleep(5)
                            continue
                        
                        print(f"[*] Topic '{self.in_topic}' ready with {len(partitions)} partitions")
                        
                        # 메시지 처리 루프
                        for message in self.consumer:
                            camera_id = message.key or "unknown"
                            timestamp = time.time()
                            
                            # 프레임 처리
                            result = self._process_frame(camera_id, message.value, timestamp)
                            
                            # 통계 출력
                            if self.stats['frames_processed'] % stats_interval == 0:
                                self._print_stats()
                            
                            # 주기적 GPU 메모리 정리
                            if self.stats['frames_processed'] % cleanup_interval == 0:
                                self._cleanup_gpu_memory()
                            
                            # 결과 로깅
                            if result and self.stats['frames_processed'] % 50 == 1:
                                print(f"[Process] Frame {self.stats['frames_processed']}: "
                                      f"{result['tracks_count']} tracks "
                                      f"({result['new_objects']} new)")
                        
                        # 정상 종료
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"[Error] Worker error (attempt {retry_count}/{max_retries}): {e}")
                        
                        if retry_count < max_retries:
                            print(f"[*] Retrying in {retry_count * 5} seconds...")
                            time.sleep(retry_count * 5)
                        else:
                            print("[Error] Max retries reached")
                            raise
                
        except KeyboardInterrupt:
            print("\n[*] Received interrupt signal")
        except Exception as e:
            print(f"[Error] Worker failed: {e}")
            raise
        finally:
            self._print_stats()
            print("[*] Worker execution completed")

    def cleanup(self):
        """리소스 정리"""
        print("[*] Cleaning up KafkaWorker resources...")
        
        try:
            # Kafka 컴포넌트 정리
            if hasattr(self, 'consumer'):
                self.consumer.close()
                print("[*] Kafka consumer closed")
            
            if hasattr(self, 'producer'):
                self.producer.flush(timeout=5)
                self.producer.close()
                print("[*] Kafka producer closed")
            
            if hasattr(self, 'reid_producer'):
                self.reid_producer.flush(timeout=5)
                self.reid_producer.close()
                print("[*] ReID producer closed")
            
            # GPU 메모리 정리
            self._cleanup_gpu_memory()
            
            print("[*] KafkaWorker cleanup completed")
            
        except Exception as e:
            print(f"[Warning] Cleanup error: {e}")

    def __del__(self):
        """소멸자"""
        self.cleanup()

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.cleanup()
        return False


def main():
    """메인 함수"""
    import sys
    
    # 엔진 경로 설정
    engine_path = 'poc/yolo_engine/yolo11m.engine'
    if len(sys.argv) > 1:
        engine_path = sys.argv[1]
    
    print(f"[*] Starting Enhanced KafkaWorker with engine: {engine_path}")
    
    try:
        # 컨텍스트 관리자로 워커 실행
        with KafkaWorker(engine_path=engine_path) as worker:
            worker.run()
            
    except KeyboardInterrupt:
        print("\n[*] Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"[Error] KafkaWorker failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("[*] KafkaWorker shutdown completed")
    return 0


if __name__ == '__main__':
    sys.exit(main())