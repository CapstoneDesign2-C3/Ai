import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv
from deep_sort_realtime.deepsort_tracker import DeepSort
# 현재 디렉토리 구조에 맞는 올바른 import
from poc.tracking_module import tracker
import base64
import time 
import json

class KafkaWorker:
    """
    Kafka I/O: consumes frames, applies DetectorTracker,
    sends new-object crops to OSNet module, and publishes all tracks.
    """
    def __init__(self, engine_path: str):
        print(f"[*] Initializing KafkaWorker with engine: {engine_path}")
        
        # 환경 변수 로드
        try:
            load_dotenv('env/aws.env')
            self.broker = os.getenv('BROKER')
            self.in_topic = os.getenv('FRAME_TOPIC', 'camera-frames')  # 기본값 설정
            self.out_topic = os.getenv('OUTPUT_TOPIC', 'detection-results')
            self.reid_topic = os.getenv('REID_REQUEST_TOPIC', 'reid-requests')
            
            print(f"[*] Kafka Config - Broker: {self.broker}")
            print(f"[*] Topics - Input: {self.in_topic}, Output: {self.out_topic}, ReID: {self.reid_topic}")
            
            if not self.broker:
                raise ValueError("BROKER environment variable not set")
                
        except Exception as e:
            print(f"[Error] Failed to load environment variables: {e}")
            raise
        
        # Kafka consumer 설정
        try:
            self.consumer = KafkaConsumer(
                self.in_topic,
                bootstrap_servers=[self.broker],
                key_deserializer=lambda b: b.decode('utf-8') if b else None,
                value_deserializer=lambda v: v,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='det-track',
                consumer_timeout_ms=30000,  # 30초 타임아웃
                max_poll_records=1  # 한 번에 하나씩 처리
            )
            print(f"[*] Kafka consumer initialized for topic: {self.in_topic}")
        except Exception as e:
            print(f"[Error] Failed to initialize Kafka consumer: {e}")
            raise
        
        # Kafka producers 설정
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=16384,
                linger_ms=10
            )
            self.reid_producer = KafkaProducer(
                bootstrap_servers=[self.broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=16384,
                linger_ms=10
            )
            print(f"[*] Kafka producers initialized")
        except Exception as e:
            print(f"[Error] Failed to initialize Kafka producers: {e}")
            raise
        
        # DetectorTracker 초기화
        try:
            # engine_path가 상대경로인 경우 절대경로로 변환
            if not os.path.isabs(engine_path):
                engine_path = os.path.abspath(engine_path)
            
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
            
            print(f"[*] Loading TensorRT engine from: {engine_path}")
            self.dt = tracker.DetectorTracker(engine_path)
            print(f"[*] DetectorTracker initialized successfully")
        except Exception as e:
            print(f"[Error] Failed to initialize DetectorTracker: {e}")
            raise

    def run(self):
        print(f"[*] KafkaWorker started, listening for messages on topic '{self.in_topic}'...")
        
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                message_count = 0
                
                # 컨슈머 상태 확인
                partitions = self.consumer.partitions_for_topic(self.in_topic)
                if partitions is None:
                    print(f"[Warning] Topic '{self.in_topic}' not found, waiting...")
                    time.sleep(5)
                    continue
                
                print(f"[*] Topic '{self.in_topic}' has {len(partitions)} partitions")
                
                for msg in self.consumer:
                    message_count += 1
                    camera_id = msg.key if msg.key else "unknown"
                    
                    if message_count % 5 == 1:  # 5개마다 로그 (첫 번째와 이후 5의 배수)
                        print(f"[KafkaWorker] Processing frame #{message_count} from camera {camera_id}")
                    
                    # 프레임 디코딩
                    try:
                        frame = cv2.imdecode(
                            np.frombuffer(msg.value, np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is None:
                            print(f"[Warning] Failed to decode frame from camera {camera_id}")
                            continue
                        
                        if message_count == 1:  # 첫 번째 프레임만 로그
                            print(f"[*] Frame decoded successfully: {frame.shape}")
                            
                    except Exception as e:
                        print(f"[Error] Frame decoding failed: {e}")
                        continue
                    
                    timestamp = time.time()
                    
                    # Detection & Tracking 수행
                    try:
                        tracks = self.dt.detect_and_track(frame)
                        
                        if tracks and message_count % 10 == 1:  # 10개마다 로그
                            print(f"[*] Detected {len(tracks)} tracks")
                        
                        # 새로운 객체의 crop을 ReID 모듈로 전송
                        new_objects = 0
                        for tr in tracks:
                            if tr.get('is_new', False):
                                new_objects += 1
                                try:
                                    x1, y1, x2, y2 = map(int, tr['bbox'])
                                    # 경계 체크
                                    h, w = frame.shape[:2]
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w, x2), min(h, y2)
                                    
                                    if x2 > x1 and y2 > y1:  # 유효한 bbox인지 확인
                                        crop = frame[y1:y2, x1:x2]
                                        if crop.size > 0:  # 빈 crop 체크
                                            _, buf = cv2.imencode('.jpg', crop)
                                            crop_bytes = buf.tobytes()
                                            
                                            req = {
                                                'camera_id': camera_id,
                                                'local_id': tr['local_id'],
                                                'timestamp': timestamp,
                                                'crop_jpg': base64.b64encode(crop_bytes).decode('ascii')
                                            }
                                            self.reid_producer.send(self.reid_topic, req)
                                except Exception as e:
                                    print(f"[Error] Failed to process new object crop: {e}")
                        
                        if new_objects > 0:
                            print(f"[*] Sent {new_objects} new object crops to ReID service")
                        
                        # 모든 트랙 결과 발행
                        out = {
                            'camera_id': camera_id, 
                            'timestamp': timestamp, 
                            'tracks': tracks
                        }
                        self.producer.send(self.out_topic, out)
                        
                        if message_count % 50 == 0:  # 50개마다 상태 로그
                            print(f"[*] Processed {message_count} frames successfully")
                            
                    except Exception as e:
                        print(f"[Error] Detection/tracking failed: {e}")
                        continue
                
                # 정상적으로 루프를 빠져나온 경우 (consumer timeout)
                print("[*] Consumer timeout reached, retrying...")
                break
                
            except Exception as e:
                retry_count += 1
                print(f"[Error] Kafka worker error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print(f"[*] Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"[Error] Max retries reached, stopping worker")
                    raise

    def __del__(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'consumer'):
                self.consumer.close()
                print("[*] Kafka consumer closed")
            if hasattr(self, 'producer'):
                self.producer.flush()
                self.producer.close()
                print("[*] Kafka producer closed")
            if hasattr(self, 'reid_producer'):
                self.reid_producer.flush()
                self.reid_producer.close()
                print("[*] ReID producer closed")
            print("[*] KafkaWorker resources cleaned up")
        except Exception as e:
            print(f"[Warning] Error during cleanup: {e}")

if __name__ == '__main__':
    import sys
    
    engine_path = 'poc/yolo_engine/yolo11m_fp16.engine'
    if len(sys.argv) > 1:
        engine_path = sys.argv[1]
    
    print(f"[*] Starting KafkaWorker with engine: {engine_path}")
    
    try:
        worker = KafkaWorker(engine_path=engine_path)
        worker.run()
    except KeyboardInterrupt:
        print("\n[*] Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"[Error] KafkaWorker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)